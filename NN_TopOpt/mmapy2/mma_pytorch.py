import torch
import numpy as np
from torch.optim import Optimizer

try:
    from mma import MMAParams, init_mma, update_mma
except ImportError:
    print("Using the bundled mmapy2 MMA implementation")
    from mmapy2.mma import MMAParams, init_mma, update_mma

class MMA(Optimizer):
    """PyTorch Optimizer implementing the Method of Moving Asymptotes (MMA).

    The optimizer wraps the numpy-based MMA implementation in mma.py.
    closure should return a tuple (objective, constraints) where:
      - objective: a scalar Tensor
      - constraints: Tensor of shape (num_cons, 1)
    The optimization is run until convergence criteria in MMAParams are met.
    """
    def __init__(self,
                 params,
                 num_cons,
                 lower_bound,
                 upper_bound,
                 max_iter=400,
                 kkt_tol=1e-3,
                 step_tol=1e-3,
                 move_limit=1e-2,
                 a0=1.0,
                 a=0.0,
                 c=1000.0,
                 d=1.0):
        if lower_bound is None or upper_bound is None:
            raise ValueError("lower_bound and upper_bound must be provided as Tensors of shape (n,1)")
        params = list(params)
        super().__init__(params, {})
        # store parameter list
        self._params = params
        # flatten initial design variables to numpy
        x0_np = np.concatenate(
            [p.data.detach().cpu().numpy().reshape(-1, 1) for p in self._params],
            axis=0
        )
        n = x0_np.shape[0]
        # process bounds
        lb_np = lower_bound.detach().cpu().numpy().reshape((n, 1))
        ub_np = upper_bound.detach().cpu().numpy().reshape((n, 1))
        # initialize MMA parameters and state
        self.mma_params = MMAParams(
            max_iter=max_iter,
            kkt_tol=kkt_tol,
            step_tol=step_tol,
            move_limit=move_limit,
            num_design_var=n,
            num_cons=num_cons,
            lower_bound=lb_np,
            upper_bound=ub_np,
            a0_value=float(a0),
            a_value=float(a),
            c_value=float(c),
            d_value=float(d),
        )
        # store maximum iterations for external loop
        self.max_number_of_iterations = max_iter
        self.mma_state = init_mma(x0_np, self.mma_params)

    def step(self, obj, cons):
        """Perform a single MMA iteration."""
        # if closure is None:
        #     raise ValueError("closure must be provided and return (objective, constraints)")
        # evaluate objective and constraints
        # obj, cons = closure()
        if not torch.is_tensor(obj):
            raise ValueError("Objective must be a Tensor")
        if not torch.is_tensor(cons):
            raise ValueError("Constraints must be a Tensor")
        # compute gradient of objective
        need_retain = self.mma_params.num_cons > 0
        grads_obj = torch.autograd.grad(
            obj, self._params, retain_graph=need_retain, create_graph=False
        )
        df0dx = np.concatenate(
            [g.detach().cpu().numpy().reshape(-1, 1) for g in grads_obj],
            axis=0
        )
        # pack constraint values
        cons_tensor = cons.view(self.mma_params.num_cons, 1)
        fval = cons_tensor.detach().cpu().numpy()
        # compute gradients of each constraint
        dfdx_list = []
        for i in range(self.mma_params.num_cons):
            retain = (i < self.mma_params.num_cons - 1)
            grads_c = torch.autograd.grad(
                cons_tensor[i, 0], self._params,
                retain_graph=retain, create_graph=False
            )
            grad_c_np = np.concatenate(
                [gc.detach().cpu().numpy().reshape(-1, 1) for gc in grads_c],
                axis=0
            ).reshape((1, -1))
            dfdx_list.append(grad_c_np)
        if dfdx_list:
            dfdx = np.vstack(dfdx_list)
        else:
            dfdx = np.zeros((0, df0dx.shape[0]))
        # call MMA update (numpy)
        f0_np = np.array([obj.detach().cpu().numpy().item()])
        self.mma_state = update_mma(
            self.mma_state,
            self.mma_params,
            f0_np,
            df0dx,
            fval,
            dfdx
        )
        # write back new design variables into torch parameters
        xmma_np = self.mma_state.x  # shape (n,1)
        xmma_tensor = torch.from_numpy(xmma_np).to(
            device=self._params[0].device,
            dtype=self._params[0].dtype
        )
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.data.copy_(
                xmma_tensor[offset:offset + numel].view_as(p.data)
            )
            offset += numel
        return obj

    def sync_accepted_design(self, accepted_design=None):
        """Commit an externally corrected accepted point to MMA history.

        `update_mma()` has already advanced the MMA history from

            (x_k, x_{k-1}, x_{k-2})

        to

            (x_raw, x_k, x_{k-1})

        and has already computed the new moving asymptotes `low` and `upp`
        from the pre-step accepted history.

        If an external feasibility mechanism subsequently replaces `x_raw`
        by an accepted point `x_acc`, the correct post-step MMA state is

            x       = x_acc
            x_old_1 = x_k
            x_old_2 = x_{k-1}

        while preserving the just-computed `low`, `upp`, and `epoch`.

        The raw-candidate KKT norm is no longer valid at `x_acc`, so this
        routine invalidates it and clears the convergence flag.  The next
        normal MMA iteration recomputes all physics, gradients and KKT data
        at the synchronized accepted point.

        Args:
            accepted_design:
                Optional numpy array / Tensor containing the accepted flattened
                design.  If omitted, the current wrapped torch parameters are
                treated as the accepted design.

        Returns:
            A dictionary of synchronization diagnostics.
        """
        if accepted_design is None:
            x_acc = np.concatenate(
                [
                    p.data.detach().cpu().numpy().reshape(-1, 1)
                    for p in self._params
                ],
                axis=0,
            )
        elif torch.is_tensor(accepted_design):
            x_acc = (
                accepted_design.detach()
                .cpu()
                .numpy()
                .reshape(-1, 1)
                .copy()
            )
        else:
            x_acc = np.asarray(
                accepted_design,
                dtype=float,
            ).reshape(-1, 1).copy()

        expected_shape = self.mma_state.x.shape
        if x_acc.shape != expected_shape:
            raise ValueError(
                "accepted design has incompatible shape: "
                f"{x_acc.shape}, expected {expected_shape}"
            )

        lower = np.asarray(
            self.mma_params.lower_bound,
            dtype=float,
        ).reshape(expected_shape)
        upper = np.asarray(
            self.mma_params.upper_bound,
            dtype=float,
        ).reshape(expected_shape)
        if np.any(x_acc < lower - 1.0e-12) or np.any(
            x_acc > upper + 1.0e-12
        ):
            raise ValueError(
                "accepted design lies outside MMA box bounds"
            )

        raw_x = self.mma_state.x.copy()
        old_1 = self.mma_state.x_old_1.copy()
        old_2 = self.mma_state.x_old_2.copy()
        low_before = self.mma_state.low.copy()
        upp_before = self.mma_state.upp.copy()
        epoch_before = int(self.mma_state.epoch)

        # The only design-history field that should change after an external
        # correction is the current point. x_old_1 / x_old_2 and the current
        # asymptotes already represent the accepted pre-step trajectory.
        self.mma_state.x = x_acc.copy()
        self.mma_state.change_design_var = float(
            np.linalg.norm(
                self.mma_state.x
                - self.mma_state.x_old_1
            )
        )

        # Raw-candidate convergence/KKT information is not valid at x_acc.
        self.mma_state.is_converged = False
        self.mma_state.kkt_norm = np.inf

        # Keep torch parameters exactly synchronized with MMAState.x.
        x_tensor = torch.from_numpy(self.mma_state.x).to(
            device=self._params[0].device,
            dtype=self._params[0].dtype,
        )
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.data.copy_(
                x_tensor[
                    offset:offset + numel
                ].view_as(p.data)
            )
            offset += numel

        # These assertions document and protect the intended commit semantics.
        if int(self.mma_state.epoch) != epoch_before:
            raise AssertionError(
                "MMA epoch changed during accepted-design sync"
            )
        if not np.array_equal(
            self.mma_state.x_old_1,
            old_1,
        ):
            raise AssertionError(
                "x_old_1 changed during accepted-design sync"
            )
        if not np.array_equal(
            self.mma_state.x_old_2,
            old_2,
        ):
            raise AssertionError(
                "x_old_2 changed during accepted-design sync"
            )
        if not np.array_equal(
            self.mma_state.low,
            low_before,
        ):
            raise AssertionError(
                "lower asymptotes changed during accepted-design sync"
            )
        if not np.array_equal(
            self.mma_state.upp,
            upp_before,
        ):
            raise AssertionError(
                "upper asymptotes changed during accepted-design sync"
            )

        return {
            "raw_to_accepted_l2": float(
                np.linalg.norm(x_acc - raw_x)
            ),
            "accepted_step_l2": float(
                self.mma_state.change_design_var
            ),
            "epoch": int(self.mma_state.epoch),
        }

# Example unit tests
import unittest

class TestMMAOptimizer(unittest.TestCase):
    def test_single_variable_unconstrained(self):
        # Objective: (x-50)^2+25, bounds [0,100], active constraint x <= 50
        x = torch.nn.Parameter(torch.tensor([[0.0]], dtype=torch.float64))
        max_iter = 200
        optimizer = MMA(
            [x],
            num_cons=1,
            lower_bound=torch.tensor([[0.0]], dtype=torch.float64),
            upper_bound=torch.tensor([[100.0]], dtype=torch.float64),
            max_iter=400,
            kkt_tol=1e-6,
            step_tol=1e-6,
            move_limit=0.1
        )
        # def closure():
        #     cons = x - 50.0
        #     loss = (x - 50.0)**2 + 25.0
        #     return loss, cons
        for _ in range(max_iter):
            loss = (x - 50.0)**2 + 25.0
            cons = x - 50.0
            print(f"loss: {loss}, cons: {cons}")
            optimizer.step(loss, cons)
        self.assertAlmostEqual(x.item(), 50.0, places=2)

    def test_two_variable_unconstrained(self):
        # Objective: (x-30)^2+(y-1)^2+1000, bounds [0,1000], active constraint x+y <= 31
        p1 = torch.nn.Parameter(torch.tensor([[0.0]], dtype=torch.float64))
        p2 = torch.nn.Parameter(torch.tensor([[0.0]], dtype=torch.float64))
        max_iter = 200
        optimizer = MMA(
            [p1, p2],
            num_cons=1,
            lower_bound=torch.tensor([[0.0], [0.0]], dtype=torch.float64),
            upper_bound=torch.tensor([[1000.0], [1000.0]], dtype=torch.float64),
            max_iter=600,
            kkt_tol=1e-6,
            step_tol=1e-6,
            move_limit=0.04
        )
        # def closure():
        #     cons = p1 + p2 - 31.0
        #     loss = (p1 - 30.0)**2 + (p2 - 1.0)**2 + 1000.0
        #     return loss, cons
        for _ in range(max_iter):
            loss = (p1 - 30.0)**2 + (p2 - 1.0)**2 + 1000.0
            cons = p1 + p2 - 31.5
            print(f"loss: {loss}, cons: {cons}")
            optimizer.step(loss, cons)
        self.assertAlmostEqual(p1.item(), 30.0, places=1)
        self.assertAlmostEqual(p2.item(), 1.0, places=1)

if __name__ == "__main__":
    unittest.main()