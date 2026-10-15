#imports
import gmsh 
import numpy as np
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import torch

from scipy.sparse import linalg as sla
from scipy.sparse import coo_matrix
from scipy.spatial import distance_matrix
from scipy import sparse

from random import randint, randrange
import json

import yaml

# from NN_TopOpt.mesh_utils import LoadedMesh2D
from mesh_utils import LoadedMesh2D
import time

# from NN_TopOpt.TopOpt import SIMP_basic
# from NN_TopOpt.TopOpt import TopOptimizer2D
# from NN_TopOpt.TopOpt import fit_ellipsoid

from TopOpt import SIMP_basic, SIMP_MMA
from TopOpt import TopOptimizer2D
from TopOpt import fit_ellipsoid

from dataset_generation.utils_generation import extract_geometry, get_rounded_polygon

from models.sdf_models import AE_DeepSDF, AE, VAE, VAE_DeepSDF, MMD_VAE, MMD_VAE_DeepSDF
models = {'AE_DeepSDF': AE_DeepSDF,
          'AE': AE, 
          'VAE': VAE,
          'VAE_DeepSDF': VAE_DeepSDF,
          'MMD_VAE': MMD_VAE,
          'MMD_VAE_DeepSDF': MMD_VAE_DeepSDF}


def if_nan(tensor):
    return torch.isnan(tensor).any()

def _points_to_grid(points, z_values, grid_size=40, half_side=1):
    grid = torch.zeros((1, grid_size, grid_size), dtype=torch.float32)
    
    # Normalize coordinates to [0, 1] range
    # points_norm = points.clone()
    # points_min = points.min(axis=0)
    # points_max = points.max(axis=0)
    points_min = torch.tensor(-half_side)
    points_max = torch.tensor(half_side)
    points_norm = (points - points_min) / (points_max - points_min)

    
    # Map to grid indices
    indices = (points_norm * (grid_size - 1)).type(torch.int32)
    
    # Create a sparse matrix to handle multiple points at same location
    sparse_grid = {}
    for idx, z in zip(indices, z_values):
        i, j = tuple(idx)
        if (i, j) in sparse_grid:
            sparse_grid[(i, j)].append(z)
        else:
            sparse_grid[(i, j)] = [z]
            
    # Fill grid with average z values
    for (i, j), z_values in sparse_grid.items():
        grid[0, i, j] = sum(z_values) / len(z_values)
        
    return grid


class FeatureMappingTopOpt:
    def __init__(self, args) -> None:

        top_opt_methods = {
            "FM_AE_DeepSDF": FeatureMappingDecSDF,
            "GaussianSplattingCompliance": GaussianSplattingCompliance,
            "CombinedMappingDecoderSDF": CombinedMappingDecoderSDF
        }

        self.max_iter = args["args"]["max_iter"]
        # self.rs_loss_start_iter = args["args"]["rs_loss_start_iter"]
        self.volfrac = args["args"]["volfrac"]
        self.N_g_x = args["args"]["N_g_x"]
        self.N_g_y = args["args"]["N_g_y"]
        self.Emin = args["Emin"]
        self.Emax = args["Emax"]
        self.penal = args["penal"]
        self.shape_combination = args["args"]["shape_combination"]

        top_opt_method = top_opt_methods[args["args"]["top_opt_method"]]

        self.problem_config = args["problem_config"]

        self.Th = args["Th"]
        self.nme = self.Th.me.shape[0]
        self.x = self.volfrac * np.ones(self.Th.me.shape[0],dtype=float)
        self.x_old = self.x.copy()
        self.coords = torch.tensor(self.Th.centroids)
        self.volumes = torch.tensor(self.Th.areas)
        self.H = torch.from_numpy(self.x)

        self.dc = np.ones(self.Th.me.shape[0])

        # self.gaussian_core = GaussianMixCompliance(dist_means, dist_stdves, self.coords, self.Emin, self.Emax, self.penal)        
        # self.gaussian_core = GaussianSplattingCompliance(dist_means, self.coords, self.Emin, self.Emax, self.penal, num_gaussian, args) 
        self.gaussian_core = top_opt_method(self.coords, self.volumes, args)
        # self.gaussian_core.load_state_dict(torch.load('test_problems/checkpoints/gaussian_core_iter160.pt'))

        self.stop_flag = False
        self.obj = 0

        self.global_i = 0
        self.meta = {'x': self.x.copy().tolist(), 'dc': self.dc.copy().tolist(), 'stop_flag': self.stop_flag}

        ## Optimizer
        # TODO: check for different vars
        w_scale_lr = args["args"].get("w_scale_lr", 5e-2)
        w_shape_var_lr = args["args"].get("w_shape_var_lr", 5e-2)
        w_rotation_lr = args["args"].get("w_rotation_lr", 1e-1)
        w_offsets_lr = args["args"].get("w_offsets_lr", 1e-1)
        self.optim = torch.optim.Adam([
            {'params': self.gaussian_core.W_scale, 'lr': w_scale_lr},
            {'params': self.gaussian_core.W_shape_var, 'lr': w_shape_var_lr},
            {'params': self.gaussian_core.W_rotation, 'lr': w_rotation_lr},
            {'params': self.gaussian_core.W_offsets, 'lr': w_offsets_lr}
        ], maximize=False, eps=1e-8)

    def parameter_opt_step(self, ce):
        # for id in range(2):
        #     _ = self.gaussian_core.get_x(self.global_i)
        self.optim.zero_grad()
        _ = self.gaussian_core.get_x(self.global_i)
        loss, splitted_loss = self.gaussian_core(
            torch.tensor(ce), self.global_i
        )
        # Check for NaN before backward pass
        if torch.isnan(loss):
            print("Warning: NaN detected in loss")
        # print(f"Loss: {loss.item()}")
        loss.backward(retain_graph=True)  # Add retain_graph=True to keep computation graph

        # Check for NaN gradients
        for name, param in self.gaussian_core.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN gradient detected in {name}")
                param.grad = torch.where(torch.isnan(param.grad), 
                                    torch.zeros_like(param.grad), 
                                    param.grad)
                
        self.gaussian_core.prepare_grads(self.global_i)

        # TODO: check for different vars
        # torch.nn.utils.clip_grad_norm_(self.gaussian_core.parameters(), max_norm=0.02)

        # Compute gradient norms for different parameters
        grad_norms = {}
        for name, param in self.gaussian_core.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                grad_norms[name] = grad_norm

        # Combine splitted_loss with grad_norms
        combined_metrics = {**splitted_loss, **grad_norms}

        self.optim.step()

        if self.shape_combination:
            self.gaussian_core.update_shape_combination(self.global_i)

        return loss, combined_metrics
    
    def plot_H_gradients(self, ce):
        """Plot gradients of H field as a 2D heatmap"""
        # Get H tensor and compute gradients
        self.optim.zero_grad()
        loss, splitted_loss = self.gaussian_core(
            torch.tensor(ce), self.global_i
        )

    
        # Get gradients using autograd.grad instead of .grad attribute
        # if self.global_i >= self.rs_loss_start_iter+2:
        #     print("rs_loss: ", splitted_loss["rs_loss"])
        #     grad_H = torch.autograd.grad(splitted_loss["rs_loss"], self.gaussian_core.H_inverted, create_graph=False, retain_graph=True)[0]
        # else:
        grad_H = torch.autograd.grad(splitted_loss["compliance"], self.gaussian_core.H, create_graph=False, retain_graph=True)[0]

        # Convert to numpy and reshape to 2D grid
        grad_magnitude = torch.sqrt(grad_H**2).numpy()
        
        # Create figure
        fig = plt.figure(figsize=(12,4))
        
        # Plot H field with gradient magnitude overlay
        contour_h = plt.tricontourf(self.coords[:,0], self.coords[:,1],
                                   self.gaussian_core.H.detach().numpy(), cmap='viridis', 
                                   levels=50)
        plt.colorbar(contour_h, label='H Value')
        
        contour_grad = plt.tricontourf(self.coords[:,0], self.coords[:,1], 
                                      grad_magnitude, cmap='magma', alpha=0.3,
                                      levels=20) 
        plt.colorbar(contour_grad, label='Gradient Magnitude')
        
        plt.title('H Field with Gradient Magnitude Overlay')
        plt.xlabel('X')
        plt.ylabel('Y')
        
        # Set axis limits based on coordinate bounds
        x_min, x_max = self.coords[:,0].min(), self.coords[:,0].max()
        y_min, y_max = self.coords[:,1].min(), self.coords[:,1].max()
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        
        plt.axis('equal')
        # plt.tight_layout()
        
        plt.show()

    def get_x(self, args):
        
        if self.global_i != 0:
            ce = args["ce"]
            # self.plot_H_gradients(ce)
            obj, splitted_loss = self.parameter_opt_step(ce)

        # if self.global_i == 160:
            # Save model parameters at iteration 10
            # torch.save(self.gaussian_core.state_dict(), f'test_problems/checkpoints/gaussian_core_iter160.pt')

        self.xold = self.x.copy()
        self.x = np.ones_like(self.x_old)*self.Emax
   
        self.H = self.gaussian_core.get_x(self.global_i)
        self.x = self.H.detach().numpy()
        
        change=np.linalg.norm(self.x.reshape(self.nme,1)-self.xold.reshape(self.nme,1),np.inf).item()


        if self.global_i+1 == self.max_iter:
            self.stop_flag = True

        if self.global_i != 0:

            self.meta = {'x': self.x.copy().tolist(),
                        'ce': {k: v.copy().tolist() for k, v in ce.items()} if isinstance(ce, dict) else ce.copy().tolist(),
                        # 'obj_opt': obj.item(),
                        # 'obj': obj_real.item(),
                        # 'volfrac_loss_pre': volfrac_loss_pre.item(),
                        # 'gaussian_overlap': gaussian_overlap.item(),
                        # 'ff_loss': ff_loss.item(),
                        # 'rs_loss': rs_loss.item(),
                        # 'compliance': compliance.item(),
                        'dc': self.dc.copy().tolist(),
                        'change': change,
                        'stop_flag': self.stop_flag}
            
            for key, value in splitted_loss.items():
                self.meta[key] = value.item()
            

        self.global_i += 1
        
        return self.x.copy()
    
    def update_problem_config(self, new_problem_name):
        new_parameters = self.gaussian_core.get_final_x()

        self.problem_config.update(new_parameters)

        with open('test_problems/problems.json', 'r') as fp:
            problem_list = json.load(fp)

        problem_list[new_problem_name] = self.problem_config

        with open('test_problems/problems.json', 'w') as f:
            json.dump(problem_list, f)

from models.sdf_models import AE_DeepSDF
from models.rs_loss_models import RSLossPredictorResNet50

class FeatureMappingDecSDF(torch.nn.Module):
    def __init__(self, coords, volumes, args):
        super().__init__()

        self.compliance_w = args["args"]["compliance_w"]
        self.volfrac_w = args["args"]["volfrac_w"]
        self.gaussian_overlap_w = args["args"]["gaussian_overlap_w"]
        self.gaussian_overlap_scale = args["args"]["gaussian_overlap_scale"]
        self.ff_loss_w = args["args"]["ff_loss_w"]
        self.smooth_k = args["args"]["smooth_k"]
        self.saved_model_name = args["args"]["saved_model_name"]
        # self.decoder_radius_sum_name = args["args"]["decoder_radius_sum_name"]
        num_samples = args["args"]["N_g"]

        problem_config = args["problem_config"]

        self.coords = coords.to(torch.float32)


        init_offsets = torch.tensor(problem_config["offsets"]).to(torch.float32)
        init_scales = torch.tensor(problem_config["scale"])*3
        init_sigmas_ratios = torch.tensor(problem_config["sigmas_ratio"]).flatten().to(torch.float32)
        init_rotations = torch.tensor(problem_config["rotation"])

        self.Emin = args["Emin"]
        self.Emax = args["Emax"]
        self.penal = args["penal"]
        self.volfrac = args["args"]["volfrac"]

        # load stats for latent space
        latent_goal = args["args"]["latent_goal"]
        data = np.load(f"model_weights/{self.saved_model_name}_{latent_goal}_stats.npz")
        self.latent_goal_center = torch.tensor(data['center'], dtype=torch.float32)
        self.latent_goal_cov_inv = torch.tensor(data['class_cov_inv'], dtype=torch.float32)
        self.latent_mins = torch.tensor(data['latent_mins'], dtype=torch.float32) * 1.2
        self.latent_maxs = torch.tensor(data['latent_maxs'], dtype=torch.float32) * 1.2
        self.latent_dim = self.latent_mins.shape[0]

        # RS Loss
        if args["args"]["rs_loss"]:
            self.rs_loss = True
            self.rs_loss_w = args["args"]["rs_loss_w"]
            self.rs_loss_e = args["args"]["rs_loss_e"]
            self.rs_loss_convex_threshold = args["args"]["rs_loss_convex_threshold"]
            self.rs_loss_plato_threshold = args["args"]["rs_loss_plato_threshold"]
            self.rs_loss_start_iter = args["args"]["rs_loss_start_iter"]
            self.rs_loss_small_cos_dist_threshold = args["args"]["rs_loss_small_cos_dist_threshold"]

            print("computing nearest neighbors ...")

            # self.rs_loss_neighbors_indices, self.rs_loss_inverted_matrices = find_nearest_neighbors(self.coords[:, 0], self.coords[:, 1], self.rs_loss_e)
            self.rs_loss_vector = None

        # self.model = VAE(input_dim=10, latent_dim=3, hidden_dim=128)
        # self.model = AE_DeepSDF(
        #     input_dim=17, 
        #     latent_dim=self.latent_dim, 
        #     hidden_dim=128, 
        #     regularization='l2',   # Use 'l1', 'l2', or None
        #     reg_weight=1e-3        # Adjust the weight as needed
        # )

        self.model = AE_DeepSDF_explicit_radius(
            input_dim=17, 
            latent_dim=self.latent_dim, 
            hidden_dim=128, 
            rad_latent_dim=2,
            rad_loss_weight=0.1,
            regularization='l2',   # Use 'l1', 'l2', or None
            reg_weight=1e-3        # Adjust the weight as needed
        )
        self.model.load_state_dict(torch.load(f"model_weights/{self.saved_model_name}.pt"), strict=False)
        self.model.eval()

        # self.decoder_radius_sum = Decoder_loss(latent_dim=self.latent_dim, hidden_dim=128)
        # self.decoder_radius_sum.load_state_dict(torch.load(f"model_weights/{self.decoder_radius_sum_name}.pt"))
        # self.decoder_radius_sum.eval()

        # create model for rs_loss
        # checkpoint_path = 'model_weights/rs_loss_conv_combined.ckpt'
        checkpoint_path = 'model_weights/rs_loss_conv.ckpt'
        self.rs_loss_predictor = RSLossPredictorResNet50.load_from_checkpoint(checkpoint_path)
        self.rs_loss_predictor.eval()

        # Create input vector for encoder
        encoder_input = torch.zeros(init_offsets.shape[0], 17)
        encoder_input[:, 3] = init_sigmas_ratios  # Set scale values at index 1

        # get latent vectors
        _, _, z = self.model(encoder_input)

        self.sigma_min = 0.002
        self.sigma_max = 0.1

        self.scale_sigma = torch.tensor(0.06)/4
        self.scale_max = self.scale_sigma*10
        self.scale_min = self.scale_sigma*4

        self.rotation_min = -torch.pi/2
        self.rotation_max = torch.pi/2

        self.shape_var_mins = self.latent_mins
        self.shape_var_maxs = self.latent_maxs
        z = torch.clamp(z, self.shape_var_mins, self.shape_var_maxs)
        init_scales = torch.clamp(init_scales, self.scale_min, self.scale_max)



        shape_variables = torch.zeros((num_samples, z.shape[1])).to(torch.float32)
        offsets = torch.zeros((num_samples, 2)).to(torch.float32)
        scales = torch.ones(num_samples).to(torch.float32)
        rotations = torch.zeros(num_samples).to(torch.float32)


        init_num_samples = z.shape[0]

        offsets[:init_num_samples] = init_offsets
        scales[:init_num_samples] = init_scales
        rotations[:init_num_samples] = init_rotations
        shape_variables[:init_num_samples] = z

        x_min = coords[:, 0].min()
        x_max = coords[:, 0].max()
        y_min = coords[:, 1].min()
        y_max = coords[:, 1].max()

        self.coord_max = torch.tensor([x_max, y_max]).to(torch.float32)
        self.coord_min = torch.tensor([x_min, y_min]).to(torch.float32)

        offsets_values = torch.logit((offsets - self.coord_min)/(self.coord_max - self.coord_min)).type(torch.float32)
        scale_values = torch.logit((scales - self.scale_min)/(self.scale_max - self.scale_min)).type(torch.float32)
        rotation_values = torch.logit((rotations - self.rotation_min)/(self.rotation_max - self.rotation_min)).type(torch.float32)
        
        shape_var_values = torch.logit((shape_variables - self.shape_var_mins)/(self.shape_var_maxs - self.shape_var_mins)).type(torch.float32)

        self.W_scale = torch.nn.Parameter(scale_values)
        self.W_shape_var = torch.nn.Parameter(shape_var_values)
        self.W_rotation = torch.nn.Parameter(rotation_values)
        self.W_offsets = torch.nn.Parameter(offsets_values)

        left_over_size = num_samples - init_num_samples
        self.current_marker = init_num_samples

        self.persistent_mask = torch.cat([torch.ones(init_num_samples, dtype=bool),torch.zeros(left_over_size, dtype=bool)], dim=0)

    def get_x(self, global_i):

        W_scale = self.W_scale[self.persistent_mask]
        W_shape_var = self.W_shape_var[self.persistent_mask] 
        W_rotation = self.W_rotation[self.persistent_mask]
        W_offsets = self.W_offsets[self.persistent_mask]
        batch_size = W_scale.shape[0]

        base_scale = (self.scale_max - self.scale_min)*torch.sigmoid(W_scale) + self.scale_min

        shape_var = (self.shape_var_maxs - self.shape_var_mins)*torch.sigmoid(W_shape_var) + self.shape_var_mins    
        rotation = self.rotation_min + (self.rotation_max - self.rotation_min)*torch.sigmoid(W_rotation).view(batch_size)
        offsets = self.coord_min + (self.coord_max - self.coord_min)*torch.sigmoid(W_offsets)

        cos_rot = torch.cos(rotation)
        sin_rot = torch.sin(rotation)

        # rotation matrix
        R = torch.stack([
            torch.stack([cos_rot, -sin_rot], dim=-1),
            torch.stack([sin_rot, cos_rot], dim=-1)
        ], dim=-2)

        # Compute inverse covariance
        coords = self.coords.unsqueeze(0).expand(batch_size, -1, -1, -1)
        offsets = offsets[:, None, None, :]
        xy = coords - offsets

        scaled_xy = xy /((base_scale[:, None, None, None])+1e-8)
        scaled_xy_shifted = xy/(self.gaussian_overlap_scale*base_scale[:, None, None, None]+1e-8)

        # Apply rotation to coordinates
        rotated_scaled_xy = torch.einsum('bij,bxyj->bxyi', R, scaled_xy)
        rotated_scaled_xy_shifted = torch.einsum('bij,bxyj->bxyi', R, scaled_xy_shifted)
        
        rotated_scaled_xy = rotated_scaled_xy.view(batch_size, -1, 2)
        rotated_scaled_xy_shifted = rotated_scaled_xy_shifted.view(batch_size, -1, 2)

        # Calculate z using pre_inv_covariance (no rotation)
        # rot_scaled_xy = torch.einsum('bxyi,bij,bxyj->bxy', rotated_xy, -0.5 * pre_inv_covariance, rotated_xy)
        kernel = torch.zeros(self.coords.shape[0], batch_size)#.double()
        kernel_shifted = torch.zeros(self.coords.shape[0], batch_size)#.double()

        for i in range(batch_size):
            mask = (rotated_scaled_xy[i].abs().max(dim=-1)[0] < 1)
            mask_shifted = (rotated_scaled_xy_shifted[i].abs().max(dim=-1)[0] < 1)
            sdfs = self.model.sdf(shape_var[i], rotated_scaled_xy[i, mask, :])
            sdfs_shifted = self.model.sdf(shape_var[i], rotated_scaled_xy_shifted[i, mask_shifted, :])
            kernel[mask, i] = sdfs.squeeze()#.double()
            kernel_shifted[mask_shifted, i] = sdfs_shifted.squeeze()#.double()

        kernel_sum = kernel.sum(dim=1)+1e-8

        self.H = (1 - self.Emin)*torch.sigmoid(-self.smooth_k*(kernel_sum - 0.5)) + self.Emin
        self.H_splitted = kernel_shifted
        self.H_splitted_sum = self.H_splitted.sum(dim=1)
        self.H_inverted = kernel_sum*0.5

        grid_batch = []

        if global_i >= self.rs_loss_start_iter:
            for i in range(batch_size):
                half_side = 1.5
                mask = (rotated_scaled_xy[i].abs().max(dim=-1)[0] < half_side)
                rotated_scaled_xy_rs = rotated_scaled_xy[i, mask, :]

                mapped_grid = _points_to_grid(rotated_scaled_xy_rs, self.H_inverted[mask], half_side=half_side)
                grid_batch.append(mapped_grid)
                # rs_loss_pred = self.rs_loss_predictor(rotated_scaled_xy_rs)

                # Reshape grid for visualization
                # grid_size = 40
                # grid_2d = mapped_grid.reshape(grid_size, grid_size).detach().numpy()
                # grid_2d = mapped_grid[0].detach().numpy()
                
                # plt.figure(figsize=(6,6))
                # plt.imshow(grid_2d, cmap='viridis', origin='lower')
                # plt.colorbar()
                # plt.title('Mapped Grid')
                # plt.axis('equal')
                # plt.show()


            rs_loss_grid = torch.stack(grid_batch)
            self.rs_loss_vector = self.rs_loss_predictor(rs_loss_grid)

        # Create entropy plot
        # plt.figure(figsize=(12, 4))
        # plt.subplot(121)
        # plt.tricontourf(self.coords[:, 0], self.coords[:, 1], p.detach().numpy(), cmap='viridis', levels=20)
        # plt.colorbar()
        # plt.title(f'Normalized H_splitted_sum (Entropy: {entropy.item():.4f})')
        # plt.axis('equal')

        # Create scatter plot of H_splitted_sum using coords
        # plt.figure(figsize=(12, 4))
        # plt.tricontourf(self.coords[:, 0], self.coords[:, 1], self.H_splitted_sum.detach().numpy(), cmap='viridis', levels=20)
        # plt.colorbar()
        # plt.title('H_splitted_sum')
        # plt.axis('equal')

        self.H_splitted_sum_clipped = torch.nn.functional.relu(
            self.H_splitted_sum - 1
        )

        return self.H

    def prepare_grads(self):
        self.W_scale.grad.data[~self.persistent_mask] = 0.0
        self.W_shape_var.grad.data[~self.persistent_mask] = 0.0
        self.W_rotation.grad.data[~self.persistent_mask] = 0.0
        self.W_offsets.grad.data[~self.persistent_mask] = 0.0

    def compute_ff_loss1(self): # form factor loss
        # get reconstruction code from decoder_input
        W_shape_var = self.W_shape_var[self.persistent_mask]
        shape_var = (self.shape_var_maxs - self.shape_var_mins)*torch.sigmoid(W_shape_var) + self.shape_var_mins 
        shape_code = self.model.decoder_input(shape_var)

        q0 = shape_code[:, 0]

        ff_loss = torch.nn.functional.relu(0.5 - q0)
        
        return ff_loss.sum()
    
    def compute_ff_loss2(self): # form factor loss
        
        latent_center = self.latent_goal_center
        latent_cov_inv = self.latent_goal_cov_inv

        W_shape_var = self.W_shape_var[self.persistent_mask]
        shape_var = (self.shape_var_maxs - self.shape_var_mins)*torch.sigmoid(W_shape_var) + self.shape_var_mins 
        
        dist = torch.norm(shape_var - latent_center, dim=1) # TODO: check if this is correct
        maha_dist_sq = torch.sum((shape_var - latent_center) @ latent_cov_inv @ (shape_var - latent_center).T, dim=1)
        ff_loss = torch.nn.functional.relu(dist - 0.8)

        return ff_loss.sum()
    
    def compute_ff_loss(self, global_i): # form factor loss
        
        if global_i > 5:

            W_shape_var = self.W_shape_var[self.persistent_mask]
            shape_var = (self.shape_var_maxs - self.shape_var_mins)*torch.sigmoid(W_shape_var) + self.shape_var_mins 
            radius_sum = self.model.radius_sum(shape_var[:, :2])


            return radius_sum.sum()

        #######################################################################

        else:
            latent_center = self.latent_goal_center
            latent_cov_inv = self.latent_goal_cov_inv
            W_shape_var = self.W_shape_var[self.persistent_mask]
            shape_var = (self.shape_var_maxs - self.shape_var_mins)*torch.sigmoid(W_shape_var) + self.shape_var_mins 
            
            dist = torch.norm(shape_var - latent_center, dim=1) # TODO: check if this is correct
            maha_dist_sq = torch.sum((shape_var - latent_center) @ latent_cov_inv @ (shape_var - latent_center).T, dim=1)
            ff_loss = torch.nn.functional.relu(dist - 0.8)

            return ff_loss.sum()
    
    def forward(self, ce, global_i):

        ff_loss = self.compute_ff_loss(global_i)

        # Add numerical stability to compliance calculation
        H_vec = torch.clamp(self.H, min=self.Emin, max=self.Emax)**self.penal
        compliance = torch.dot(H_vec, ce.float())

        volfrac_goal = self.volfrac - 0.2*max(0, min(1, (global_i-20)/20))
        
        # Safeguard the loss calculations
        volfrac_loss_pre = torch.nn.functional.relu(
            self.H.mean() - volfrac_goal
        )
        
        gaussian_overlap = self.H_splitted_sum_clipped.mean()

        # check the sign
        # Start of Selection
        obj_ce = (
            -compliance * self.compliance_w
            + ff_loss * self.ff_loss_w
            + volfrac_loss_pre * self.volfrac_w
            + gaussian_overlap * self.gaussian_overlap_w
        )
        obj_real = (
            compliance * self.compliance_w
            + ff_loss * self.ff_loss_w
            + volfrac_loss_pre * self.volfrac_w
            + gaussian_overlap * self.gaussian_overlap_w
        )

        if self.rs_loss and global_i >= self.rs_loss_start_iter+1:
            # print("indices: ", len(self.rs_loss_neighbors_indices), self.rs_loss_neighbors_indices[0].shape)
            # rs_loss, _ = compute_rs_loss(self.coords[:, 0], self.coords[:, 1], self.H_splitted_sum,
            #                           self.rs_loss_neighbors_indices, self.rs_loss_inverted_matrices, 
            #                           self.rs_loss_e, self.rs_loss_convex_threshold,
            #                           self.rs_loss_plato_threshold, self.rs_loss_small_cos_dist_threshold,
            #                           visualize=True)
            rs_loss = self.rs_loss_vector.mean()
            rs_loss_weighted = rs_loss * self.rs_loss_w

            obj_real += rs_loss_weighted
            obj_ce += rs_loss_weighted

        else:

            rs_loss = torch.tensor(0.0)

        splitted_loss = {
            "gaussian_overlap": gaussian_overlap,
            "volfrac_loss_pre": volfrac_loss_pre,
            "compliance": compliance,
            "ff_loss": ff_loss,
            "rs_loss": rs_loss,
            "obj_ce": obj_ce,
            "obj_real": obj_real,
        }

        return obj_ce, splitted_loss


class GaussianSplattingCompliance(torch.nn.Module):
    def __init__(self, coords, volumes, args):
        super().__init__()

        num_samples = args["args"]["N_g"]
        dist_means = torch.zeros((num_samples, 2))

        x_min = coords[:, 0].min()
        x_max = coords[:, 0].max()
        y_min = coords[:, 1].min()
        y_max = coords[:, 1].max()

        try:
            x_centers = args["args"]["init_centers_x"]
            y_centers = args["args"]["init_centers_y"]
            grid_x = torch.tensor(x_centers)[:, None]
            grid_y = torch.tensor(y_centers)[:, None]
        except:
            # initialize the feature centers according to the grid
            center_x = (x_max - x_min)/2
            center_y = (y_max - y_min)/2
            x_grid_offset = (x_max - x_min)*0.05
            y_grid_offset = (y_max - y_min)*0.05
            x = torch.linspace(x_min + x_grid_offset, x_max - x_grid_offset, args["args"]["N_g_x"])
            y = torch.linspace(y_min + y_grid_offset, y_max - y_grid_offset, args["args"]["N_g_y"])
            x = x - (x.mean() - center_x)
            y = y - (y.mean() - center_y)
            grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
            grid_x = grid_x.reshape(-1, 1)
            grid_y = grid_y.reshape(-1, 1)


        init_num_samples = grid_x.shape[0]
        dist_means_init = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)[:init_num_samples]
        
        dist_means[:init_num_samples] = dist_means_init

        init_scale = args["args"]["init_scale"]
        self.scale_sigma = torch.tensor(init_scale)
        self.scale_max = torch.tensor(1.5)
        self.scale_min = torch.tensor(0.05)

        self.rotation_min = -torch.pi/2
        self.rotation_max = torch.pi/2

        self.coord_max = torch.tensor([x_max, y_max])
        self.coord_min = torch.tensor([x_min, y_min])

        self.coord_min -= self.coord_max * args["args"]["axis_offset"]
        self.coord_max += self.coord_max * args["args"]["axis_offset"]

        self.sigmas_ratio_max = 3.5
        self.sigmas_ratio_min = 0.5

        self.coords = coords
        self.volumes = volumes
        self.volumes_sum = self.volumes.sum()

        offsets_values = torch.logit(
            (dist_means - self.coord_min) / (self.coord_max - self.coord_min)
        ).type(torch.DoubleTensor)
        
        scale_values = torch.logit(
            (torch.ones((num_samples, 1)) * self.scale_sigma - self.scale_min) / 
            (self.scale_max - self.scale_min)
        ).type(torch.DoubleTensor)
        
        sigmas_ratio_values = torch.logit(
            (torch.ones((num_samples, 1)) - self.sigmas_ratio_min) / 
            (self.sigmas_ratio_max - self.sigmas_ratio_min)
        ).type(torch.DoubleTensor)
        
        rotation_values = torch.logit(
            (torch.zeros((num_samples, 1)) - self.rotation_min) / 
            (self.rotation_max - self.rotation_min)
        ).type(torch.DoubleTensor)

        self.W_scale = torch.nn.Parameter(scale_values)
        self.W_shape_var = torch.nn.Parameter(sigmas_ratio_values)
        self.W_rotation = torch.nn.Parameter(rotation_values)
        self.W_offsets = torch.nn.Parameter(offsets_values)

        left_over_size = num_samples - init_num_samples
        self.current_marker = init_num_samples

        self.persistent_mask = torch.cat(
            [torch.ones(init_num_samples, dtype=bool), torch.zeros(left_over_size, dtype=bool)], dim=0
        )
        self.new_gaussians_mask = torch.zeros_like(self.persistent_mask, dtype=bool)

        self.Emin = args["Emin"]
        self.Emax = args["Emax"]
        self.penal = args["penal"]

        self.volfrac = args["args"]["volfrac"]
        self.volfrac_increment = args["args"]["volfrac_increment"]
        self.volfrac_increment_marker = args["args"]["volfrac_increment_marker"]
        self.volfrac_increment_duration = args["args"]["volfrac_increment_duration"]

        self.compliance_w = args["args"]["compliance_w"]
        self.volfrac_w = args["args"]["volfrac_w"]
        self.gaussian_overlap_w = args["args"]["gaussian_overlap_w"]
        self.smooth_k = args["args"]["smooth_k"]

        self.merging_marker = args["args"]["merging_marker"]
        self.merging_adaptation_length = args["args"]["merging_adaptation_length"]

        self.points_inside_elements = {}

    def get_x(self, global_i):

        # TODO: rewrite becasu sclaing is used twice

        W_scale = self.W_scale[self.persistent_mask]
        W_shape_var = self.W_shape_var[self.persistent_mask] 
        W_rotation = self.W_rotation[self.persistent_mask]
        W_offsets = self.W_offsets[self.persistent_mask]
        batch_size = W_scale.shape[0]

        base_scale = (self.scale_max - self.scale_min)*torch.sigmoid(W_scale) + self.scale_min
        base_scale = base_scale.squeeze()
        sigmas_ratio = (self.sigmas_ratio_max - self.sigmas_ratio_min)*torch.sigmoid(W_shape_var) + self.sigmas_ratio_min    
        sigmas = torch.cat([torch.ones(batch_size, 1), torch.ones(batch_size, 1) * sigmas_ratio], dim=1)
        rotation = self.rotation_min + (self.rotation_max - self.rotation_min)*torch.sigmoid(W_rotation).view(batch_size)
        offsets = self.coord_min + (self.coord_max - self.coord_min)*torch.sigmoid(W_offsets)

        cos_rot = torch.cos(rotation)
        sin_rot = torch.sin(rotation)

        # rotation matrix
        R = torch.stack([
            torch.stack([cos_rot, -sin_rot], dim=-1),
            torch.stack([sin_rot, cos_rot], dim=-1)
        ], dim=-2)

        covariance_diag = sigmas**2
        pre_inv_covariance_diag = 1/(covariance_diag + 1e-6)

        pre_inv_covariance = torch.diag_embed(pre_inv_covariance_diag)
        # pre_inv_covariance = pre_inv_covariance

        # Compute inverse covariance
        coords = self.coords.unsqueeze(0).expand(batch_size, -1, -1, -1)
        offsets = offsets[:, None, None, :]
        xy = coords - offsets

        # TODO: check if this is correct (not multiplying by base_scale)
        xy = xy /((base_scale[:, None, None, None])+1e-8)

        # Apply rotation to coordinates
        rotated_xy = torch.einsum('bij,bxyj->bxyi', R, xy)
        
        # Calculate z using pre_inv_covariance (no rotation)
        z = torch.einsum('bxyi,bij,bxyj->bxy', rotated_xy, pre_inv_covariance, rotated_xy)
        z = torch.sqrt(z)

        # kernel = torch.exp(20*z)
        kernel = 1/(1 + torch.exp(10*(z-1)))
        kernel = torch.squeeze(kernel)
             
        kernel_sum = kernel.sum(dim=0)+1e-8

        if global_i == self.merging_marker-1:
            print("merging preparation")
            self.compute_merging_pairs(kernel)

        self.H = (1 - self.Emin)*torch.sigmoid(-self.smooth_k*(kernel_sum - 0.5)) + self.Emin

        # TODO: check if this is correct (splitted sum)
        self.H_splitted = (1 - self.Emin)*torch.sigmoid(1.8*self.smooth_k*(kernel - 0.1)) + self.Emin
        self.H_splitted_sum = self.H_splitted.sum(dim=0)
        # self.H_offset = (1 - self.Emin)*torch.sigmoid(-4*torch.log(kernel_sum*1.1)) + self.Emin

        return self.H

    def prepare_grads(self, global_i):
        if (global_i > self.merging_marker and 
            global_i < self.merging_marker + self.merging_adaptation_length and 
            self.new_gaussians_mask.any()):
            evolving_mask = self.new_gaussians_mask
        else:
            evolving_mask = self.persistent_mask

        self.W_scale.grad.data[~evolving_mask] = 0.0
        self.W_shape_var.grad.data[~evolving_mask] = 0.0
        self.W_rotation.grad.data[~evolving_mask] = 0.0
        self.W_offsets.grad.data[~evolving_mask] = 0.0

        # if (global_i > self.merging_marker and 
        #         global_i < self.merging_marker + self.merging_adaptation_length):
        torch.nn.utils.clip_grad_norm_(self.W_offsets, max_norm=0.001)
        torch.nn.utils.clip_grad_norm_(self.W_scale, max_norm=0.01)
        torch.nn.utils.clip_grad_norm_(self.W_shape_var, max_norm=0.01)
        torch.nn.utils.clip_grad_norm_(self.W_rotation, max_norm=0.01)

    def forward(self, ce, global_i):

        # Add numerical stability to compliance calculation
        H_vec = torch.clamp(self.H, min=self.Emin, max=self.Emax)**self.penal
        compliance = torch.dot(H_vec, ce)

        # volfrac_goal = self.volfrac - 0.1*max(0, min(1, (global_i-20)/20))
        volfrac_goal = self.volfrac - self.volfrac_increment * max(
            0, min(1, (global_i - self.volfrac_increment_marker) / self.volfrac_increment_duration)
        )
        volume_goal = self.volumes_sum * volfrac_goal
        # Safeguard the loss calculations
        # TODO: change to vectorproduct with volume vectors
        volfrac_loss_pre = torch.nn.functional.relu(
            self.H.T @ self.volumes/volume_goal - 1
        )
        
        gaussian_overlap = torch.nn.functional.relu(
            self.H_splitted_sum - 1.5
        ).mean()

        print(f"Iteration: {global_i} | current volfrac: {(self.H.T @ self.volumes/self.volumes.sum()).item():.6f} | compliance: {compliance.item():.6f}")

        obj_ce = (
            -compliance * self.compliance_w
            + volfrac_loss_pre * self.volfrac_w
            + gaussian_overlap * self.gaussian_overlap_w
        )
        obj_real = (
            compliance * self.compliance_w
            + volfrac_loss_pre * self.volfrac_w
            + gaussian_overlap * self.gaussian_overlap_w
        )

        splitted_loss = {
            "gaussian_overlap": gaussian_overlap,
            "volfrac_loss_pre": volfrac_loss_pre,
            "compliance": compliance,
            "obj_ce": obj_ce,
            "obj_real": obj_real,
        }

        return obj_ce, splitted_loss
    
    def compute_merging_pairs(self, kernel):
        # kernel_sigmoids = torch.sigmoid(kernel) 
        # Compute pairwise sums of kernel vectors
        n = kernel.shape[0]  # number of vectors
        pairwise_sums = torch.zeros((n, n), device=kernel.device)

        close_gauss_pairs = []
        close_pair_kernel_sums = []
        self.points_inside_elements = {}

        for i in range(n):
            for j in range(n):
                if j > i:
                    sum_ij = kernel[i] + kernel[j]
                    pairwise_sums[i,j] = sum_ij.max()
                    if sum_ij.max() > 1.5:
                        close_gauss_pairs.append((i, j))
                        close_pair_kernel_sums.append(sum_ij.max())

                        if i not in self.points_inside_elements.keys():
                            self.points_inside_elements[i] = self.coords[kernel[i] > 0.5]

                        if j not in self.points_inside_elements.keys():
                            self.points_inside_elements[j] = self.coords[kernel[j] > 0.5]

        # Convert pairs and sums to tensors and sort by sum values
        if close_gauss_pairs:  # Only process if there are pairs
            pairs_tensor = torch.tensor(close_gauss_pairs, device=pairwise_sums.device)
            sums_tensor = torch.tensor(close_pair_kernel_sums, device=pairwise_sums.device)
            
            # Sort by kernel sums in descending order
            sorted_indices = torch.argsort(sums_tensor, descending=True)
            pairs_tensor = pairs_tensor[sorted_indices]
            sums_tensor = sums_tensor[sorted_indices]
            
            close_gauss_pairs = pairs_tensor.tolist()
            close_pair_kernel_sums = sums_tensor.tolist()

        self.merging_pairs = close_gauss_pairs
        self.merging_kernel_sums = close_pair_kernel_sums

    
    def merging(self):
        # TODO: rotation is inncorrectly used (need to take into account sin). 
        # also check the result areas
        # check determinant before splitting
        W_scale = self.W_scale[self.persistent_mask]
        W_shape_var = self.W_shape_var[self.persistent_mask]
        # batch_size = W_scale.shape[0]

        base_scale = (self.scale_max - self.scale_min)*torch.sigmoid(W_scale) + self.scale_min
        sigmas_ratio = (self.sigmas_ratio_max - self.sigmas_ratio_min)*torch.sigmoid(W_shape_var) + self.sigmas_ratio_min
        scale = torch.cat([base_scale, base_scale * sigmas_ratio], dim=1)
        
        covariance_diag = scale.pow(2)
        det_covariance = torch.prod(covariance_diag, dim=1)

        ###############################################################################################################
        persistent_mask_indices = torch.arange(self.persistent_mask.shape[0])[self.persistent_mask]
        self.new_gaussians_mask = torch.zeros_like(self.persistent_mask, dtype=bool)

        areas = np.pi * np.sqrt(det_covariance.detach().cpu().numpy())

        while len(self.merging_pairs) > 0:
            process_pair = self.merging_pairs.pop(0)
            current_index = self.current_marker + 1

            x = torch.cat([self.points_inside_elements[process_pair[0]], self.points_inside_elements[process_pair[1]]])
            x = x.unique(dim=0)
            x_mean = x.mean(dim=0).clone()
            x = x - x_mean
            x = x.detach().cpu().numpy()

            target_area = areas[process_pair[0]] + areas[process_pair[1]]

            center, R, Sigma = fit_ellipsoid(x, target_area)
            center = x_mean - torch.tensor(center)
            R = torch.tensor(R)

            alpha = torch.arccos(R[0, 0])
            if alpha > torch.pi/2:
                alpha = alpha - torch.pi
            elif alpha < -torch.pi/2:
                alpha = alpha + torch.pi

            scale = torch.tensor(Sigma[0])
            ratio = torch.tensor(Sigma[1]/Sigma[0])

            scale = min(
                self.scale_max - 0.1 * (self.scale_max - self.scale_min),
                max(self.scale_min + 0.1 * (self.scale_max - self.scale_min), scale)
            )
            ratio = min(
                self.sigmas_ratio_max - 0.1 * (self.sigmas_ratio_max - self.sigmas_ratio_min),
                max(self.sigmas_ratio_min + 0.1 * (self.sigmas_ratio_max - self.sigmas_ratio_min), ratio)
            )

            scale = torch.tensor(scale)
            ratio = torch.tensor(ratio)

            self.W_rotation.data[current_index] = torch.logit((alpha - self.rotation_min)/(self.rotation_max - self.rotation_min))
            self.W_scale.data[current_index] = torch.logit((scale - self.scale_min)/(self.scale_max - self.scale_min))
            self.W_shape_var.data[current_index] = torch.logit((ratio - self.sigmas_ratio_min)/(self.sigmas_ratio_max - self.sigmas_ratio_min))
            self.W_offsets.data[current_index] = torch.logit((center - self.coord_min)/(self.coord_max - self.coord_min))

            self.persistent_mask[current_index] = True
            self.new_gaussians_mask[current_index] = True
            self.persistent_mask[persistent_mask_indices[process_pair[0]]] = False
            self.persistent_mask[persistent_mask_indices[process_pair[1]]] = False

            # Remove any remaining pairs that contain the merged indices
            self.merging_pairs = [pair for pair in self.merging_pairs if process_pair[0] not in pair and process_pair[1] not in pair]

            self.current_marker += 1
        ##########################################################################################################
        # check determinant before splitting
        # W_scale = self.W_scale[self.persistent_mask]
        # W_shape_var = self.W_shape_var[self.persistent_mask]
        # batch_size = W_scale.shape[0]

        # base_scale = (self.scale_max - self.scale_min)*torch.sigmoid(W_scale) + self.scale_min
        # sigmas_ratio = (self.sigmas_ratio_max - self.sigmas_ratio_min)*torch.sigmoid(W_shape_var) + self.sigmas_ratio_min    
        # scale = torch.cat([base_scale, base_scale * sigmas_ratio], dim=1)
        
        # covariance_diag = scale.pow(2)
        # det_covariance = torch.prod(covariance_diag, dim=1)
        # print("det_covariance: ", det_covariance)
        # print("det_covariance_sum: ", det_covariance.sum())

        ###############################################################################################################

    def update_shape_combination(self, global_i):
        if global_i == self.merging_marker:
            self.merging()

    def get_final_x(self):
        W_scale = self.W_scale[self.persistent_mask]
        W_shape_var = self.W_shape_var[self.persistent_mask] 
        W_rotation = self.W_rotation[self.persistent_mask]
        W_offsets = self.W_offsets[self.persistent_mask]
        batch_size = W_scale.shape[0]

        base_scale = (self.scale_max - self.scale_min)*torch.sigmoid(W_scale) + self.scale_min
        base_scale = base_scale.squeeze()
        sigmas_ratio = (self.sigmas_ratio_max - self.sigmas_ratio_min)*torch.sigmoid(W_shape_var) + self.sigmas_ratio_min    
        scale = torch.cat([torch.ones(batch_size, 1), torch.ones(batch_size, 1) * sigmas_ratio], dim=1)
        rotation = self.rotation_min + (self.rotation_max - self.rotation_min)*torch.sigmoid(W_rotation).view(batch_size)
        offsets = self.coord_min + (self.coord_max - self.coord_min)*torch.sigmoid(W_offsets)

        design_vars = {"scale": base_scale.detach().cpu().numpy().tolist(),
                       "sigmas_ratio": sigmas_ratio.detach().cpu().numpy().tolist(),
                       "rotation": rotation.detach().cpu().numpy().tolist(),
                       "offsets": offsets.detach().cpu().numpy().tolist()}

        return design_vars
    
    def get_geometry(self):

        W_scale = self.W_scale[self.persistent_mask]
        W_shape_var = self.W_shape_var[self.persistent_mask] 
        W_rotation = self.W_rotation[self.persistent_mask]
        W_offsets = self.W_offsets[self.persistent_mask]
        batch_size = W_scale.shape[0]

        base_scale = (self.scale_max - self.scale_min)*torch.sigmoid(W_scale) + self.scale_min
        base_scale = base_scale.squeeze()
        sigmas_ratio = (self.sigmas_ratio_max - self.sigmas_ratio_min)*torch.sigmoid(W_shape_var) + self.sigmas_ratio_min    
        sigmas = torch.cat([torch.ones(batch_size, 1), torch.ones(batch_size, 1) * sigmas_ratio], dim=1)
        rotation = self.rotation_min + (self.rotation_max - self.rotation_min)*torch.sigmoid(W_rotation).view(batch_size)
        offsets = self.coord_min + (self.coord_max - self.coord_min)*torch.sigmoid(W_offsets)

        base_scale = base_scale.detach().cpu().numpy()
        offsets = offsets.detach().cpu().numpy()
        rotation = rotation.detach().cpu().numpy()
        sigmas = sigmas.flatten().detach().cpu().numpy()
        sigmas_ratio = sigmas_ratio.flatten().detach().cpu().numpy()

        geometry_features = []

        for i in range(batch_size):

            a = base_scale[i]
            b = base_scale[i]*sigmas_ratio[i]
            geometry_features.append(["ellipse", a, b, offsets[i], -rotation[i]])

        return geometry_features
from shape_bridge import ShapeBridge

class CombinedMappingDecoderSDF(torch.nn.Module):
    def __init__(self, coords, volumes, args):
        super().__init__()

        self.saved_model_name = args["args"]["saved_model_name"]
        self.config_name = args["args"]["config_name"]
        self.config_dir = args["args"]["config_dir"]
        # num_samples = args["args"]["N_g"]
        num_init = args["args"]["N_g"]
        num_capacity = args["args"].get("N_cap", num_init)

        self.num_init = num_init
        self.num_capacity = num_capacity

        self.ks_smooth_factor = args["args"].get("ks_smooth_factor", 40)
        print(self.ks_smooth_factor)
        self.H_min = 0.5
        self.H_max = 0.5

        self.input_dim = args["args"].get("input_dim", 17)
        self.dataset_type = args["args"].get("dataset_type", "tripple")
        self.sigmoid_limits = args["args"].get("sigmoid_limits", False)

        self.coords = coords.to(torch.float32)   
        self.volumes = volumes.to(torch.float32)
        self.volumes_sum = self.volumes.sum()

        # load stats for latent space
        z_limits = np.load(f"../z_limits/{self.config_name}_full_stats.npz")
        latent_mins = torch.tensor(z_limits['latent_mins'], dtype=torch.float32) * 1.2
        latent_maxs = torch.tensor(z_limits['latent_maxs'], dtype=torch.float32) * 1.2

        latent_dim = latent_mins.shape[0]

        saved_model_path = f'../model_weights/{self.saved_model_name}_full.pt'

        # Load configuration from YAML file
        with open(f'{self.config_dir}/{self.config_name}.yaml', 'r') as file:
            config = yaml.safe_load(file)

        # Initialize VAE model
        model_params = config['model']['params']
        model_params['input_dim'] = self.input_dim
        self.model = models[config['model']['type']](**model_params)
        
        # Load pre-trained weights for the model
        print(saved_model_path)
        state_dict = torch.load(saved_model_path)
        # new_state_dict = self.model.state_dict()

        # # Update the new_state_dict with the loaded state_dict, ignoring size mismatches
        # for key in state_dict:
        #     if key in new_state_dict and state_dict[key].size() == new_state_dict[key].size():
        #         new_state_dict[key] = state_dict[key]

        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.reopen_use_shape_bridge = args["args"].get("reopen_use_shape_bridge", False)
        self.shape_bridge = ShapeBridge(self.model)
        # create variable constraints #######################################################################
        init_scale = args["args"]["init_scale"]
        if len(init_scale) != num_init:
            raise ValueError(f"len(init_scale)={len(init_scale)} != N_g={num_init}")
        self.scale_sigma = torch.tensor(init_scale, dtype=torch.float32)
        self.scale_max = torch.tensor(args["args"].get("scale_max", 1.5))
        self.scale_min = torch.tensor(args["args"].get("scale_min", 0.05))

        self.rotation_min = -torch.pi/2
        self.rotation_max = torch.pi/2
        init_rotation = args["args"].get("init_rotation", 0)
        if not np.isscalar(init_rotation):
            if len(init_rotation) != num_init:
                raise ValueError(f"len(init_rotation)={len(init_rotation)} != N_g={num_init}")
        self.init_rotation = torch.tensor(init_rotation, dtype=torch.float32) * torch.pi/180

        self.shape_var_mins = latent_mins
        self.shape_var_maxs = latent_maxs

        x_min = coords[:, 0].min()
        x_max = coords[:, 0].max()
        y_min = coords[:, 1].min()
        y_max = coords[:, 1].max()

        self.coord_max = torch.tensor([x_max, y_max]).to(torch.float32)
        self.coord_min = torch.tensor([x_min, y_min]).to(torch.float32)
        
        self.coord_min -= self.coord_max * args["args"]["axis_offset"]
        self.coord_max += self.coord_max * args["args"]["axis_offset"]

        # sigmas ratio for shape variables (while merging)
        self.sigmas_ratio_max = 3.5
        self.sigmas_ratio_min = 0.5

        # create initial parameters ##############################################################################

        
        
        try:
            x_centers = args["args"]["init_centers_x"]
            y_centers = args["args"]["init_centers_y"]
            if len(x_centers) != num_init:
                raise ValueError(f"len(init_centers_x)={len(x_centers)} != N_g={num_init}")
            if len(y_centers) != num_init:
                raise ValueError(f"len(init_centers_y)={len(y_centers)} != N_g={num_init}")
            grid_x = torch.tensor(x_centers, dtype=torch.float32)[:, None]
            grid_y = torch.tensor(y_centers, dtype=torch.float32)[:, None]
        except:
            # initialize the feature centers according to the grid
            center_x = (x_max - x_min)/2
            center_y = (y_max - y_min)/2
            x_grid_offset = (x_max - x_min)*0.05
            y_grid_offset = (y_max - y_min)*0.05
            x = torch.linspace(x_min + x_grid_offset, x_max - x_grid_offset, args["args"]["N_g_x"])
            y = torch.linspace(y_min + y_grid_offset, y_max - y_grid_offset, args["args"]["N_g_y"])
            x = x - (x.mean() - center_x)
            y = y - (y.mean() - center_y)
            grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
            grid_x = grid_x.reshape(-1, 1)
            grid_y = grid_y.reshape(-1, 1)

        init_num_samples = num_init
        dist_means_init = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)[:init_num_samples]
        # init_num_samples = dist_means_init.shape[0]

        ## Chi min and max           c  b    x_3  y_3  R_1    R_2   R_3   x_4   y_4   x_5   y_5   R_1   R_2   R_3   R_4
        self.chi_min = torch.tensor([0, 0, -0.8,  0,   0.01, 0.01, 0.01, -0.8, -0.2, -0.8, -0.2, 0.01, 0.01, 0.01, 0.01])
        self.chi_max = torch.tensor([1, 1,  0.8,  0.9, 0.1,   0.1,  0.1,  0.8,  0.8,  0.8,  0.8,  0.1,  0.1,  0.1,  0.1])
        # Create input vector for encoder
        encoder_input = torch.zeros(init_num_samples, self.input_dim)
        # encoder_input[:, 3] = 1/1.5  # Start from circles

        ### start from triangles
        # encoder_input[:, 2] = 0.5
        # encoder_input[:, 4] = 0    # x_3
        # encoder_input[:, 5] = 0.25/3 # y_3
        # encoder_input[:, 6] = 0.25/3 # R_1
        # encoder_input[:, 7] = 0.25/3 # R_2
        # encoder_input[:, 8] = 0.25/3 # R_3

        ### start from squares
        R = 0.45/3
        if "init_chis" in args["args"]:
            print("🚀 检测到 YAML 注入的自定义先验基因，跳过标准正方形初始化！")
            init_chis = args["args"]["init_chis"]

            if len(init_chis) != num_init:
                raise ValueError(f"len(init_chis)={len(init_chis)} != N_g={num_init}")

            for i, chi in enumerate(init_chis):
                if len(chi) != 8:
                    raise ValueError(f"init_chis[{i}] len={len(chi)} != 8")

            custom_chis = torch.tensor(init_chis, dtype=torch.float32)
            encoder_input[:, 2:10] = custom_chis

        elif self.dataset_type == "tripple":
            encoder_input[:, 2] = 1
            chi_offset = 9
            encoder_input[:, chi_offset] = 0.5
            encoder_input[:, chi_offset+1] = 0.5
            encoder_input[:, chi_offset+2] = -0.5
            encoder_input[:, chi_offset+3] = 0.5
            encoder_input[:, chi_offset+4] = R
            encoder_input[:, chi_offset+5] = R
            encoder_input[:, chi_offset+6] = R
            encoder_input[:, chi_offset+7] = R

        elif self.dataset_type == "quadrangle":
            chi_offset = 2
            encoder_input[:, chi_offset] = 0.5
            encoder_input[:, chi_offset+1] = 0.5
            encoder_input[:, chi_offset+2] = -0.5
            encoder_input[:, chi_offset+3] = 0.5
            encoder_input[:, chi_offset+4] = R
            encoder_input[:, chi_offset+5] = R
            encoder_input[:, chi_offset+6] = R
            encoder_input[:, chi_offset+7] = R

        elif self.dataset_type == "triangle":
            chi_offset = 2
            R = 0.5 * torch.tanh(torch.tensor(30 * torch.pi/180))/3 * 0.9
            # R = 0.1
            print("Triangle R: ", R)
            encoder_input[:, chi_offset] = 0
            encoder_input[:, chi_offset+1] = torch.sqrt(torch.tensor(0.75)) - 0.5
            encoder_input[:, chi_offset+2] = R
            encoder_input[:, chi_offset+3] = R
            encoder_input[:, chi_offset+4] = R

        # get initial latent vectors
        output = self.model(encoder_input)
        z = output['z']
        z = torch.clamp(z, self.shape_var_mins, self.shape_var_maxs)

        shape_variables = torch.zeros((num_capacity, z.shape[1])).to(torch.float32)
        offsets = torch.zeros((num_capacity, 2)).to(torch.float32)

        if z.shape[0] != num_init:
            raise ValueError(
                f"latent z count {z.shape[0]} != N_g={num_init}"
            )
        init_num_samples = num_init

        offsets[:init_num_samples] = dist_means_init
        shape_variables[:init_num_samples] = z

        if self.sigmoid_limits:
            offsets_values = torch.logit(
                (offsets - self.coord_min) / (self.coord_max - self.coord_min)
            ).type(torch.float32)

            scale_values_raw = torch.full((num_capacity,), float(self.scale_min), dtype=torch.float32)
            scale_values_raw[:num_init] = self.scale_sigma

            scale_values = torch.logit(
                (scale_values_raw - self.scale_min) / (self.scale_max - self.scale_min)
            ).type(torch.float32)

            rotation_values_raw = torch.zeros((num_capacity,), dtype=torch.float32)
            if self.init_rotation.ndim == 0:
                rotation_values_raw[:num_init] = self.init_rotation
            else:
                rotation_values_raw[:num_init] = self.init_rotation

            rotation_values = torch.logit(
                (rotation_values_raw - self.rotation_min) /
                (self.rotation_max - self.rotation_min)
            ).type(torch.float32).view(-1, 1)

            shape_var_values = torch.logit(
                (shape_variables - self.shape_var_mins) / (self.shape_var_maxs - self.shape_var_mins)
            ).type(torch.float32)
        else:
            print("No sigmoid limits")
            offsets_values = offsets
            scale_values = torch.full((num_capacity,), float(self.scale_min), dtype=torch.float32)
            scale_values[:num_init] = self.scale_sigma
            rotation_values = torch.zeros((num_capacity,), dtype=torch.float32)
            if self.init_rotation.ndim == 0:
                rotation_values[:num_init] = self.init_rotation
            else:
                rotation_values[:num_init] = self.init_rotation
            shape_var_values = shape_variables
        

        self.W_scale = torch.nn.Parameter(scale_values)
        self.W_shape_var = torch.nn.Parameter(shape_var_values)
        self.W_rotation = torch.nn.Parameter(rotation_values)
        self.W_offsets = torch.nn.Parameter(offsets_values)

        left_over_size = num_capacity - init_num_samples
        self.current_marker = init_num_samples

        self.persistent_mask = torch.zeros(num_capacity, dtype=torch.bool)
        self.persistent_mask[:init_num_samples] = True
        self.new_gaussians_mask = torch.zeros_like(self.persistent_mask, dtype=bool)

        # ===== split prior memory =====
        self.split_confidence = torch.zeros(num_capacity, dtype=torch.float32)
        self.split_distance_prior = torch.zeros(num_capacity, dtype=torch.float32)
        self.split_direction = torch.zeros((num_capacity, 2), dtype=torch.float32)

        init_split_conf = args["args"].get("init_split_confidence", [0.0] * num_init)
        init_split_dir_x = args["args"].get("init_split_direction_x", [1.0] * num_init)
        init_split_dir_y = args["args"].get("init_split_direction_y", [0.0] * num_init)
        init_split_dist = args["args"].get("init_split_distance_prior", [0.0] * num_init)

        if len(init_split_conf) != num_init:
            raise ValueError(f"len(init_split_confidence)={len(init_split_conf)} != N_g={num_init}")
        if len(init_split_dir_x) != num_init or len(init_split_dir_y) != num_init:
            raise ValueError("init_split_direction_x/y length mismatch with N_g")
        if len(init_split_dist) != num_init:
            raise ValueError(f"len(init_split_distance_prior)={len(init_split_dist)} != N_g={num_init}")

        self.split_confidence[:num_init] = torch.tensor(init_split_conf, dtype=torch.float32)
        self.split_distance_prior[:num_init] = torch.tensor(init_split_dist, dtype=torch.float32)
        self.split_direction[:num_init, 0] = torch.tensor(init_split_dir_x, dtype=torch.float32)
        self.split_direction[:num_init, 1] = torch.tensor(init_split_dir_y, dtype=torch.float32)

        # 归一化方向，避免零向量
        dir_norm = torch.norm(self.split_direction[:num_init], dim=1, keepdim=True).clamp_min(1e-8)
        self.split_direction[:num_init] = self.split_direction[:num_init] / dir_norm

        # split prior如何参与reopen打分
        self.reopen_prior_weight = args["args"].get("reopen_prior_weight", 0.8)
        self.reopen_grad_weight = args["args"].get("reopen_grad_weight", 1.0)
        self.reopen_prior_decay = args["args"].get("reopen_prior_decay", 0.995)
        self.reopen_min_split_confidence = args["args"].get("reopen_min_split_confidence", 0.10)

        self.feature_birth_iter = torch.full(
            (num_capacity,), -1, dtype=torch.long
        )
        self.feature_birth_iter[:init_num_samples] = 0

        n_expected_intersections = 2
        self.ks_min = torch.log(torch.tensor(n_expected_intersections))/self.ks_smooth_factor
        self.ks_max = torch.log(n_expected_intersections*torch.exp(torch.tensor(self.ks_smooth_factor)))/self.ks_smooth_factor
        self.ks_range = self.ks_max - self.ks_min

        self.kernel_max = 0.5
        self.kernel_min = 0.5

        # additional parameters #########################################################################################

        #### parameters for MMA ####

        self.Emin = args["Emin"]
        self.Emax = args["Emax"]
        self.penal = args["penal"]
        self.volfrac = args["args"]["volfrac"]
        self.volfrac_increment = args["args"]["volfrac_increment"]
        self.volfrac_increment_marker = args["args"]["volfrac_increment_marker"]
        self.volfrac_increment_duration = args["args"]["volfrac_increment_duration"]

        self.compliance_w = args["args"]["compliance_w"]
        self.volfrac_w = args["args"]["volfrac_w"]
        self.gaussian_overlap_w = args["args"]["gaussian_overlap_w"]
        self.gaussian_overlap_scale = args["args"]["gaussian_overlap_scale"]
        self.rc_loss_w = args["args"]["rc_loss_w"]
        self.smooth_k = args["args"]["smooth_k"]
        # self.rs_loss = args["args"]["rs_loss"]
        # self.rs_loss_start_iter = args["args"]["rs_loss_start_iter"]

        self.max_iter = args["args"]["max_iter"]

        try:
            self.offset_grad_deceleration = args["args"]["offset_grad_deceleration"]
        except:
            self.offset_grad_deceleration = 0.5

        self.merging_markers = args["args"]["merging_markers"]
        self.merging_adaptation = args["args"]["merging_adaptation"]

        self.shape_optimization_ranges = args["args"]["shape_optimization_ranges"]
        self.refactoring_markers = args["args"]["refactoring_markers"]
        
        # ===== reopening params =====
        self.reopen_enable = args["args"].get("reopen_enable", True)
        self.reopen_start_iter = args["args"].get("reopen_start_iter", 20)
        self.reopen_patience = args["args"].get("reopen_patience", 5)
        self.reopen_improve_thresh = args["args"].get("reopen_improve_thresh", 1e-3)
        self.reopen_cooldown = args["args"].get("reopen_cooldown", 8)
        self.reopen_adaptation = args["args"].get("reopen_adaptation", 5)

        self.reopen_offset_step = args["args"].get("reopen_offset_step", 0.2)
        self.reopen_scale_shrink = args["args"].get("reopen_scale_shrink", 0.85)
        
        self.reopen_use_canonical_shape_without_prior = args["args"].get(
            "reopen_use_canonical_shape_without_prior", True
        )
        self.reopen_force_zero_rotation_without_prior = args["args"].get(
            "reopen_force_zero_rotation_without_prior", True
        )
        self.reopen_direction_mode_without_prior = args["args"].get(
            "reopen_direction_mode_without_prior", "8way"
        )
        self.reopen_direction_probe_scale = args["args"].get(
            "reopen_direction_probe_scale", 1.0
        )
        self.last_reopen_iter = -10**9
        self.loss_history = []
        self.reopen_parent_idx = None
        self.reopen_canonicalize_old_feature_shapes = args["args"].get(
            "reopen_canonicalize_old_feature_shapes", False
        )
        self.reopen_old_feature_canonical_blend = args["args"].get(
            "reopen_old_feature_canonical_blend", 1
        )
        self.reopen_old_feature_offset_canonical_blend = args["args"].get(
            "reopen_old_feature_offset_canonical_blend", 1
        )
        self.reopen_exploration_steps = args["args"].get("reopen_exploration_steps", 8)
        self.reopen_transition_steps = args["args"].get("reopen_transition_steps", 15)
        self.reopen_explore_compliance_weight = args["args"].get("reopen_compliance_weight", 1)
        self.reopen_explore_volume_weight = args["args"].get("reopen_volume_weight", 0.2)
        self.reopen_constraint_slack = args["args"].get("reopen_constraint_slack", 0.0)
        
        # ===== callback / rollback state =====
        self.callback_enable = args["args"].get("callback_enable", True)
        self.callback_accept_rel_improve = args["args"].get("callback_accept_rel_improve", 5e-4)
        self.callback_reject_vol_slack = args["args"].get("callback_reject_vol_slack", 0.02)
        self.callback_observation_steps = args["args"].get("callback_observation_steps", 50)
        self.reopen_disabled_by_callback = False
        self.pending_event = None
        self.event_counter = 0
        self.reopen_post_vol_constraint_baseline = None

        self.reopen_shrink_old_features = args["args"].get("reopen_shrink_old_features", False)
        self.reopen_old_feature_shrink_ratio = args["args"].get("reopen_old_feature_shrink_ratio", 0.7)
        self.reopen_push_old_features = args["args"].get("reopen_push_old_features", False)
       
        
        self.reopen_use_distribution_prior_without_simp = args["args"].get(
            "reopen_use_distribution_prior_without_simp", True
        )
        self.reopen_offset_shrink_without_prior = args["args"].get(
            "reopen_offset_shrink_without_prior", 1.5
        )
        self.points_inside_elements = {}
        self.point_ids_inside_elements = {}
      
    def export_bounds(self):
        """
        Export per-parameter lower and upper bounds for the MMA optimizer.
        Returns:
            lb_list (list of Tensor): lower bounds in order
                [W_scale, W_shape_var, W_rotation, W_offsets]
            ub_list (list of Tensor): upper bounds in the same order
        """
        # Scale bounds (scalar parameters broadcast to W_scale shape)
        lb_scale = torch.full_like(self.W_scale, self.scale_min)
        ub_scale = torch.full_like(self.W_scale, self.scale_max)

        # Shape‐variable bounds (vector per gaussian, broadcast from shape_var_mins/maxs)
        lb_shape = self.shape_var_mins.unsqueeze(0).expand_as(self.W_shape_var)
        ub_shape = self.shape_var_maxs.unsqueeze(0).expand_as(self.W_shape_var)

        # Rotation bounds (scalar per gaussian)
        lb_rot = torch.full_like(self.W_rotation, self.rotation_min)
        ub_rot = torch.full_like(self.W_rotation, self.rotation_max)

        # Offset bounds (vector per gaussian, broadcast from coord_min/max)
        lb_off = self.coord_min.unsqueeze(0).expand_as(self.W_offsets)
        ub_off = self.coord_max.unsqueeze(0).expand_as(self.W_offsets)

        return [lb_scale, lb_shape, lb_rot, lb_off], [ub_scale, ub_shape, ub_rot, ub_off]

    def refactoring(self):
        print("Refactoring started ...")
        W_shape_var = self.W_shape_var[self.persistent_mask] 
        shape_var = (self.shape_var_maxs - self.shape_var_mins)*torch.sigmoid(W_shape_var) + self.shape_var_mins    

        chis = self.model.decoder_input(shape_var)

        # chis = torch.clamp(chis, self.chi_min, self.chi_max)
        encoder_input = torch.zeros(chis.shape[0], self.input_dim)
        encoder_input[:, 2:] = chis

        # TODO: add radiuses clipping

        # get initial latent vectors
        output = self.model(encoder_input)
        z = output['z']
        z = torch.clamp(z, self.shape_var_mins, self.shape_var_maxs)

        shape_var_values = torch.logit(
            (z - self.shape_var_mins) / (self.shape_var_maxs - self.shape_var_mins)
        ).type(torch.float32)

        self.W_shape_var.data[self.persistent_mask] = shape_var_values
        
    def get_geometry(self):

        W_scale = self.W_scale[self.persistent_mask]
        W_shape_var = self.W_shape_var[self.persistent_mask] 
        W_rotation = self.W_rotation[self.persistent_mask]
        W_offsets = self.W_offsets[self.persistent_mask]
        batch_size = W_scale.shape[0]

        if self.sigmoid_limits: 
            base_scale = (self.scale_max - self.scale_min)*torch.sigmoid(W_scale) + self.scale_min  
            shape_var = (self.shape_var_maxs - self.shape_var_mins)*torch.sigmoid(W_shape_var) + self.shape_var_mins    
            rotation = self.rotation_min + (self.rotation_max - self.rotation_min)*torch.sigmoid(W_rotation).view(batch_size)
            offsets = self.coord_min + (self.coord_max - self.coord_min)*torch.sigmoid(W_offsets)
        else:
            base_scale = W_scale
            shape_var = W_shape_var
            rotation = W_rotation
            offsets = W_offsets

        cos_rot = torch.cos(rotation)
        sin_rot = torch.sin(rotation)

        # rotation matrix
        R = torch.stack([
            torch.stack([cos_rot, -sin_rot], dim=-1),
            torch.stack([sin_rot, cos_rot], dim=-1)
        ], dim=-2)

        chis = self.model.decoder_input(shape_var).clone()
        base_scale = base_scale.detach().cpu().numpy()
        offsets = offsets.detach().cpu().numpy()
        rotation = rotation.detach().cpu().numpy()
        R = R.detach().cpu().numpy()

        geometry_features = []

        for i in range(batch_size):
            geometry_type, geometry_params = extract_geometry(chis[i].detach().cpu().numpy(), self.dataset_type)

            if geometry_type == "ellipse":
                a = geometry_params[1]*base_scale[i]
                b = geometry_params[2]*base_scale[i]
                geometry_features.append(["ellipse", a, b, offsets[i], -rotation[i]])

            elif geometry_type == "polygon":
                vertices = geometry_params[0] @ R[i]
                vertices = vertices*base_scale[i] + offsets[i]
                radiuses = geometry_params[1]*base_scale[i]
                line_segments, arc_segments, arcs_intersection = get_rounded_polygon(vertices, radiuses)
                geometry_features.append(["polygon", vertices, radiuses, line_segments, arc_segments])

        return geometry_features

    def if_refactoring(self, global_i):
        if global_i in self.refactoring_markers:
            return True

    def if_merging_preparation(self, global_i):
        if global_i + 1 in self.merging_markers:
            return True

    def if_merging(self, global_i):
        if global_i in self.merging_markers:
            return True
        
    def if_merging_adaptation(self, global_i):
        in_range = False
        for marker in self.merging_markers:
            if global_i > marker and global_i < marker + self.merging_adaptation:
                in_range = True
        
        return in_range
    
    def compute_merging_pairs(self, kernel_T):
        # Compute pairwise sums of kernel vectors
        kernel = kernel_T.T
        n = kernel.shape[0]  # number of vectors
        pairwise_sums = torch.zeros((n, n), device=kernel.device)

        close_gauss_pairs = []
        close_pair_kernel_sums = []
        self.points_inside_elements = {}

        for i in range(n):
            for j in range(n):
                if j > i:
                    sum_ij = kernel[i] + kernel[j]
                    pairwise_sums[i,j] = sum_ij.max()
                    if sum_ij.max() > 1.5:
                        close_gauss_pairs.append((i, j))
                        close_pair_kernel_sums.append(sum_ij.max())

                        if i not in self.points_inside_elements.keys():
                            # self.points_inside_elements[i] = self.coords[kernel[i] > 0.5]
                            self.point_ids_inside_elements[i] = kernel[i] > 0.5

                        if j not in self.points_inside_elements.keys():
                            # self.points_inside_elements[j] = self.coords[kernel[j] > 0.5]
                            self.point_ids_inside_elements[j] = kernel[j] > 0.5
        

        # Convert pairs and sums to tensors and sort by sum values
        if close_gauss_pairs:  # Only process if there are pairs
            pairs_tensor = torch.tensor(close_gauss_pairs, device=pairwise_sums.device)
            sums_tensor = torch.tensor(close_pair_kernel_sums, device=pairwise_sums.device)
            
            # Sort by kernel sums in descending order
            sorted_indices = torch.argsort(sums_tensor, descending=True)
            pairs_tensor = pairs_tensor[sorted_indices]
            sums_tensor = sums_tensor[sorted_indices]
            
            close_gauss_pairs = pairs_tensor.tolist()
            close_pair_kernel_sums = sums_tensor.tolist()

        self.merging_pairs = close_gauss_pairs
        self.merging_kernel_sums = close_pair_kernel_sums

    def merging(self):
        # TODO: rotation is inncorrectly used (need to take into account sin). 
        # also check the result areas
        # check determinant before splitting
        # W_scale = self.W_scale[self.persistent_mask]
        # W_shape_var = self.W_shape_var[self.persistent_mask]
        # batch_size = W_scale.shape[0]

        # base_scale = (self.scale_max - self.scale_min)*torch.sigmoid(W_scale) + self.scale_min
        # sigmas_ratio = (self.sigmas_ratio_max - self.sigmas_ratio_min)*torch.sigmoid(W_shape_var) + self.sigmas_ratio_min
        # scale = torch.cat([base_scale, base_scale * sigmas_ratio], dim=1)
        
        # covariance_diag = scale.pow(2)
        # det_covariance = torch.prod(covariance_diag, dim=1)

        ###############################################################################################################
        persistent_mask_indices = torch.arange(self.persistent_mask.shape[0])[self.persistent_mask]
        self.new_gaussians_mask = torch.zeros_like(self.persistent_mask, dtype=bool)

        # areas = np.pi * np.sqrt(det_covariance.detach().cpu().numpy())

        while len(self.merging_pairs) > 0:
            process_pair = self.merging_pairs.pop(0)
           
            current_index = self.current_marker + 1

            p_ids_inside = torch.logical_or(self.point_ids_inside_elements[process_pair[0]], self.point_ids_inside_elements[process_pair[1]])
            x = self.coords[p_ids_inside]
            x = x.unique(dim=0)
            x_mean = x.mean(dim=0).clone()
            x = x - x_mean
            x = x.detach().cpu().numpy()

            target_area = self.volumes[p_ids_inside].sum()

            center, R, Sigma = fit_ellipsoid(x, target_area)
            center = x_mean - torch.tensor(center)
            R = torch.tensor(R)

            alpha = torch.arccos(R[0, 0])
            if alpha > torch.pi/2:
                alpha = alpha - torch.pi
            elif alpha < -torch.pi/2:
                alpha = alpha + torch.pi

            scale = torch.tensor(Sigma[0])
            ratio = torch.tensor(Sigma[1]/Sigma[0])

            scale = min(
                self.scale_max - 0.1 * (self.scale_max - self.scale_min),
                max(self.scale_min + 0.1 * (self.scale_max - self.scale_min), scale)
            )
            ratio = min(
                self.sigmas_ratio_max - 0.1 * (self.sigmas_ratio_max - self.sigmas_ratio_min),
                max(self.sigmas_ratio_min + 0.1 * (self.sigmas_ratio_max - self.sigmas_ratio_min), ratio)
            )

            ##############################################################################
            scale = torch.tensor(scale)
            ratio = torch.tensor(ratio)

            # Create input vector for encoder
            encoder_input = torch.zeros(1, 17)
            encoder_input[:, 3] = ratio  # Start from circles

            # get initial latent vectors
            _, _, z = self.model(encoder_input)
            z = torch.clamp(z, self.shape_var_mins, self.shape_var_maxs)

            self.W_rotation.data[current_index] = torch.logit((alpha - self.rotation_min)/(self.rotation_max - self.rotation_min))
            self.W_scale.data[current_index] = torch.logit((scale - self.scale_min)/(self.scale_max - self.scale_min))
            self.W_shape_var.data[current_index] = torch.logit((z - self.shape_var_mins)/(self.shape_var_maxs - self.shape_var_mins))
            self.W_offsets.data[current_index] = torch.logit((center - self.coord_min)/(self.coord_max - self.coord_min))

            self.persistent_mask[current_index] = True
            self.new_gaussians_mask[current_index] = True
            self.persistent_mask[persistent_mask_indices[process_pair[0]]] = False
            self.persistent_mask[persistent_mask_indices[process_pair[1]]] = False

            # Remove any remaining pairs that contain the merged indices
            self.merging_pairs = [pair for pair in self.merging_pairs if process_pair[0] not in pair and process_pair[1] not in pair]

            self.current_marker += 1
    
    def update_reopen_history(self, compliance_value: float):
        self.loss_history.append(float(compliance_value))
        if len(self.loss_history) > self.reopen_patience + 1:
            self.loss_history = self.loss_history[-(self.reopen_patience + 1):]
    
    def update_split_prior_memory(self):
        # 每轮轻微衰减，让早期SIMP先验不会永久主导
        self.split_confidence[self.persistent_mask] *= self.reopen_prior_decay
        self.split_confidence[~self.persistent_mask] = 0.0
        self.split_distance_prior[~self.persistent_mask] = 0.0
        self.split_direction[~self.persistent_mask] = 0.0
     
    def has_free_slots(self, needed=2):
        free_num = int((~self.persistent_mask).sum().item())
        return free_num >= needed
    
    def compute_reopen_scores(self):
        active_ids = torch.arange(self.persistent_mask.shape[0], device=self.W_scale.device)[self.persistent_mask]
        if len(active_ids) == 0:
            return active_ids, None

        scale_g = self.W_scale.grad.detach().abs().reshape(-1)
        off_g = self.W_offsets.grad.detach().norm(dim=1)
        grad_score = scale_g + off_g

        prior_score = self.split_confidence.to(self.W_scale.device)
        total_score = prior_score
        # total_score = self.reopen_grad_weight * grad_score + self.reopen_prior_weight * prior_score
        return active_ids, total_score
    
    def get_reopen_target_scale(self, active_num_after_split: int) -> float:
        """
        根据分裂后 feature 数量给一个更中性的 child 目标尺度。
        """
        if active_num_after_split <= 3:
            return 0.27
        elif active_num_after_split == 4:
            return 0.2
        elif active_num_after_split == 5:
            return 0.16
        else:
            return 0.12
        
    def if_reopening(self, global_i):
        if (not self.reopen_enable) or self.reopen_disabled_by_callback:
            return False
        if global_i < self.reopen_start_iter:
            return False
        if global_i - self.last_reopen_iter < self.reopen_cooldown:
            return False
        if not self.has_free_slots(needed=2):
            return False
        if len(self.loss_history) < self.reopen_patience + 1:
            return False

        old = self.loss_history[0]
        new = self.loss_history[-1]
        rel_improve = (old - new) / (abs(old) + 1e-8)
        
        print(
            f"[ReopenCheck] iter={global_i} | "
            f"old={old:.6f} new={new:.6f} | "
            f"rel_improve={rel_improve:.6f} | "
            f"thresh={self.reopen_improve_thresh:.6f} | "
            f"free_slots={int((~self.persistent_mask).sum().item())} | "
            f"cooldown_ok={(global_i - self.last_reopen_iter >= self.reopen_cooldown)}"
        )
        print("num_capacity =", self.num_capacity)
        print("active =", int(self.persistent_mask.sum().item()))
        print("free =", int((~self.persistent_mask).sum().item()))
        return rel_improve < self.reopen_improve_thresh
    
    def if_reopening_adaptation(self, global_i):
        return (global_i > self.last_reopen_iter) and (
            global_i < self.last_reopen_iter + self.reopen_adaptation
        )
    
    def if_reopen_exploration(self, global_i):
        return (
            global_i >= self.last_reopen_iter + 1
            and global_i <= self.last_reopen_iter + self.reopen_exploration_steps
        )

    def if_reopen_transition(self, global_i):
        start = self.last_reopen_iter + self.reopen_exploration_steps + 1
        end = self.last_reopen_iter + self.reopen_exploration_steps + self.reopen_transition_steps
        return start <= global_i <= end
    
    def select_best_split_direction_8way(self, parent_idx, parent_off, split_step):
        """
        无先验时，从 8 个方向中选两个最适合 split 的方向。
        评分原则（单方向评分）：
        - child center 与其他 active feature center 的最小距离越大越好
        - 离边界太近要扣分
        - 第二个方向不能和第一个方向太接近（避免相邻）
        """
        device = self.W_offsets.device
        dtype = self.W_offsets.dtype

        dirs = torch.tensor([
            [ 1.0,  0.0],      # right
            [-1.0,  0.0],      # left
            [ 0.0,  1.0],      # up
            [ 0.0, -1.0],      # down
            [ 0.707,  0.707],  # up-right
            [ 0.707, -0.707],  # down-right
            [-0.707,  0.707],  # up-left
            [-0.707, -0.707],  # down-left
        ], dtype=dtype, device=device)
        dirs = dirs / torch.norm(dirs, dim=1, keepdim=True)

        active_ids = torch.arange(self.persistent_mask.shape[0], device=device)[self.persistent_mask]
        other_ids = active_ids[active_ids != parent_idx]

        if self.sigmoid_limits:
            offsets_phys = self.coord_min + (self.coord_max - self.coord_min) * torch.sigmoid(self.W_offsets.data)
        else:
            offsets_phys = self.W_offsets.data

        scores = []

        for d in dirs:
            # 单方向候选 child center
            c = parent_off + split_step * d

            # ---------- 1) 与其他 active feature 的距离 ----------
            if len(other_ids) > 0:
                other_centers = offsets_phys[other_ids]
                dist_score = torch.norm(other_centers - c.unsqueeze(0), dim=1).min()
            else:
                dist_score = torch.tensor(1.0, dtype=dtype, device=device)

            # ---------- 2) 边界惩罚 ----------
            margin = torch.min(torch.stack([
                c[0] - self.coord_min[0],
                self.coord_max[0] - c[0],
                c[1] - self.coord_min[1],
                self.coord_max[1] - c[1],
            ]))

            # 出界直接重罚
            outside_penalty = torch.tensor(0.0, dtype=dtype, device=device)
            if margin < 0:
                outside_penalty = torch.tensor(1e6, dtype=dtype, device=device)

            score = dist_score + 0.2 * margin - outside_penalty
            scores.append(score)

        scores = torch.stack(scores)  # [8]
        sorted_idx = torch.argsort(scores, descending=True)

        # 第一方向：最高分
        d1 = dirs[sorted_idx[0]].clone()

        # 第二方向：不和 d1 太接近
        d2 = None
        for idx in sorted_idx[1:]:
            cand = dirs[idx]
            # dot > 0.5 认为太接近（夹角小于约60度）
            if torch.dot(cand, d1) < 0.5:
                d2 = cand.clone()
                break

        # fallback：极少发生
        if d2 is None:
            d2 = dirs[sorted_idx[1]].clone()

        print(
            f"[ReopenDir] top1_dir={d1.detach().cpu().numpy()}, "
            f"top2_dir={d2.detach().cpu().numpy()}, "
            f"top1_score={scores[sorted_idx[0]].item():.6f}"
        )

        return d1, d2
    def find_nearest_active_old_feature(self, parent_idx):
        device = self.W_offsets.device

        all_ids = torch.arange(self.persistent_mask.shape[0], device=device)
        active_ids = all_ids[self.persistent_mask]
        other_ids = active_ids[active_ids != parent_idx]

        if len(other_ids) == 0:
            return None

        if self.sigmoid_limits:
            offsets_phys = self.coord_min + (self.coord_max - self.coord_min) * torch.sigmoid(self.W_offsets.data)
        else:
            offsets_phys = self.W_offsets.data

        parent_off = offsets_phys[parent_idx]
        other_offs = offsets_phys[other_ids]

        dists = torch.norm(other_offs - parent_off.unsqueeze(0), dim=1)
        nn_idx = torch.argmin(dists)

        return int(other_ids[nn_idx].item())
    def get_distribution_prior_direction(self, parent_idx, neighbor_idx):
        device = self.W_offsets.device
        dtype = self.W_offsets.dtype

        if self.sigmoid_limits:
            offsets_phys = self.coord_min + (self.coord_max - self.coord_min) * torch.sigmoid(self.W_offsets.data)
        else:
            offsets_phys = self.W_offsets.data

        p = offsets_phys[parent_idx]
        q = offsets_phys[neighbor_idx]

        v = q - p
        v_norm = torch.norm(v)

        if v_norm.item() < 1e-8:
            # fallback: 用 x 方向
            d = torch.tensor([1.0, 0.0], dtype=dtype, device=device)
        else:
            d = v / v_norm

        return d
    
    def compute_even_split_child_offsets(self, parent_global_idx):
        """
        按当前 active feature 的 x 坐标排序，找到 parent 的序号 rank。
        假设 parent 被 split 后，总孔数 +1。
        然后沿主轴均匀放置 split 后的第 rank 和第 rank+1 个孔的位置。
        y 固定为 0。
        """
        device = self.W_offsets.device
        dtype = self.W_offsets.dtype

        all_ids = torch.arange(self.persistent_mask.shape[0], device=device)
        active_ids = all_ids[self.persistent_mask]

        if self.sigmoid_limits:
            offsets_phys = self.coord_min + (self.coord_max - self.coord_min) * torch.sigmoid(self.W_offsets.data)
        else:
            offsets_phys = self.W_offsets.data

        active_offsets = offsets_phys[active_ids]

        # 按 x 坐标从小到大排序
        sort_idx = torch.argsort(active_offsets[:, 0])
        sorted_active_ids = active_ids[sort_idx]

        # 找 parent 在排序后的 rank（从 0 开始）
        rank = (sorted_active_ids == parent_global_idx).nonzero(as_tuple=False).item()

        # split 后总孔数 = 当前 active 数 + 1
        n_after = len(active_ids) + 1

        # 沿 x 轴均匀取 n_after 个点
        # 这里留一点边界 margin，避免太贴边
        x_min = float(self.coord_min[0].item())
        x_max = float(self.coord_max[0].item())
        margin = 0.08 * (x_max - x_min)

        x_positions = torch.linspace(
            x_min + margin,
            x_max - margin,
            n_after,
            device=device,
            dtype=dtype,
        )

        # child 占据 split 后的第 rank 和 rank+1 个位置
        child1_x = x_positions[rank]
        child2_x = x_positions[rank + 1]

        # y 固定为 0
        child_y = 0.5 * (self.coord_min[1] + self.coord_max[1])

        child1_off_phys = torch.stack([child1_x, child_y])
        child2_off_phys = torch.stack([child2_x, child_y])

        print(
            f"[EvenSplit] parent={parent_global_idx} | rank={rank+1}/{len(active_ids)} | "
            f"n_after={n_after} | child1_x={child1_x.item():.6f} | child2_x={child2_x.item():.6f}"
        )

        return child1_off_phys, child2_off_phys
    
    def compute_post_split_canonical_layout(self, parent_global_idx, active_ids_before, old_active_ids):
        """
        按 split 后总孔数统一编号，生成 canonical offsets。

        规则：
        1. 先取 split 前当前 active features，按 x 坐标排序
        2. 找到 parent 在排序后的 rank
       3. split 后总孔数 = 原 active 数 + 1
        4. parent 分裂出来的 child1 / child2 占据 rank 和 rank+1 位置
        5. 其余 old features 保持顺序，但若在 parent 之后则编号顺延 1
        6. 所有 canonical offsets 沿主轴均匀分布，y 固定为 0

        返回：
            child1_off_phys
            child2_off_phys
            canonical_old_dict: {old_feature_id: canonical_offset}
        """
        device = self.W_offsets.device
        dtype = self.W_offsets.dtype

        

        if self.sigmoid_limits:
            offsets_phys = self.coord_min + (self.coord_max - self.coord_min) * torch.sigmoid(self.W_offsets.data)
        else:
            offsets_phys = self.W_offsets.data

        active_offsets_before = offsets_phys[active_ids_before]

        # split 前按 x 排序
        sort_idx = torch.argsort(active_offsets_before[:, 0])
        sorted_active_ids = active_ids_before[sort_idx]

        # parent 在 split 前排序中的位置（0-based）
        parent_rank = (sorted_active_ids == parent_global_idx).nonzero(as_tuple=False).item()

        # split 后总孔数 = 当前 active 数 + 1
        n_after = len(active_ids_before) + 1

        x_min = float(self.coord_min[0].item())
        x_max = float(self.coord_max[0].item())
        margin = 0.08 * (x_max - x_min)

        x_positions = torch.linspace(
            x_min + margin,
            x_max - margin,
            n_after,
            device=device,
            dtype=dtype,
        )

        # 如果你的中轴不是 y=0，这里改成 0.5*(coord_min[1]+coord_max[1])
        y_canonical = 0.5 * (self.coord_min[1] + self.coord_max[1])
    
        canonical_positions = [
            torch.stack([x_positions[i], y_canonical]) for i in range(n_after)
        ]

        # child 占据 parent_rank 和 parent_rank+1
        child1_off_phys = canonical_positions[parent_rank]
        child2_off_phys = canonical_positions[parent_rank + 1]

        # old feature 的 canonical 位置
        canonical_old_dict = {}

        for old_id in old_active_ids:
            old_rank_before = (sorted_active_ids == old_id).nonzero(as_tuple=False).item()

            if old_rank_before < parent_rank:
                old_rank_after = old_rank_before
            else:
                old_rank_after = old_rank_before + 1

            canonical_old_dict[int(old_id.item())] = canonical_positions[old_rank_after]

        print(
            f"[CanonicalLayout] parent={parent_global_idx} | "
            f"parent_rank={parent_rank+1}/{len(active_ids_before)} | "
            f"n_after={n_after}"
        )

        return child1_off_phys, child2_off_phys, canonical_old_dict
    # def reopening(self, global_i):
    #     import math
    #     if not self.has_free_slots(needed=2):
    #         print("[Reopen] No free slots left.")
    #         return False

    #     # 选 active feature 中梯度最大的那个作为 parent
    #     active_ids, total_score = self.compute_reopen_scores()
    #     if active_ids is None or len(active_ids) == 0:
    #         return False

    #     active_score = total_score[active_ids]

    #     # 低于最小split置信度的孔，不要被先验强推；但梯度特别大仍可选
    #     active_prior = self.split_confidence[active_ids].to(active_score.device)
    #     active_score = torch.where(
    #         active_prior >= self.reopen_min_split_confidence,
    #         active_score,
    #         self.reopen_grad_weight * active_score
    #     )

    #     parent_global_idx = active_ids[torch.argmax(active_score)].item()
    #     self.reopen_parent_idx = parent_global_idx

    #     # 找两个空槽位
    #     free_ids = torch.arange(self.persistent_mask.shape[0], device=self.W_scale.device)[~self.persistent_mask]
    #     child1 = free_ids[0].item()
    #     child2 = free_ids[1].item()

    #     # 取父 feature 当前参数
    #     parent_scale = self.W_scale.data[parent_global_idx].clone()
    #     parent_shape = self.W_shape_var.data[parent_global_idx].clone()
    #     parent_rot = self.W_rotation.data[parent_global_idx].clone()
    #     parent_off = self.W_offsets.data[parent_global_idx].clone()

    #     #分裂1
    #     # prior_dir = self.split_direction[parent_global_idx].to(
    #     #     device=self.W_offsets.device,
    #     #     dtype=self.W_offsets.dtype,
    #     # )
    #     # prior_norm = torch.norm(prior_dir)

    #     # if prior_norm.item() > 1e-8:
    #     #     direction = prior_dir / prior_norm
    #     # else:
    #     #     angle = parent_rot.item()
    #     #     direction = torch.tensor(
    #     #         [np.cos(angle), np.sin(angle)],
    #     #         dtype=self.W_offsets.dtype,
    #     #         device=self.W_offsets.device,
    #     #     )

    #     # # ===== split distance =====
    #     # # density2feature给出的建议距离，与最小reopen距离取较大值
    #     # dist_prior = float(self.split_distance_prior[parent_global_idx].item())
    #     # split_step = max(float(self.reopen_offset_step), dist_prior)

    #     # # 如果当前参数不是物理空间而是logit空间，这里不能直接拿 parent_off ± delta
    #     # # 你当前 sigmoid_limits=False 时直接这样做是对的；
    #     # # 若以后开启 sigmoid_limits，需要先转到物理空间再转回logit
    #     # delta = split_step * direction

    #     # # ===== child shape initialization =====
    #     # if self.reopen_use_shape_bridge:
    #     #     child1_shape, child2_shape, parent_chi, child1_chi, child2_chi = \
    #     #         self.shape_bridge.split_latent(parent_shape, use_mu=True)

    #     #     self.W_shape_var.data[child1] = child1_shape.clone()
    #     #     self.W_shape_var.data[child2] = child2_shape.clone()

    #     #     print("[Reopen-Shape] parent chi =", parent_chi.detach().cpu().numpy())
    #     #     print("[Reopen-Shape] child1 chi =", child1_chi.detach().cpu().numpy())
    #     #     print("[Reopen-Shape] child2 chi =", child2_chi.detach().cpu().numpy())
    #     # else:
    #     #     self.W_shape_var.data[child1] = parent_shape.clone()
    #     #     self.W_shape_var.data[child2] = parent_shape.clone()

    #     # # ===== child scale / rotation =====
    #     # self.W_scale.data[child1] = parent_scale * self.reopen_scale_shrink
    #     # self.W_scale.data[child2] = parent_scale * self.reopen_scale_shrink

    #     # self.W_rotation.data[child1] = parent_rot.clone()
    #     # self.W_rotation.data[child2] = parent_rot.clone()

    #     # # ===== USE PRIOR DIRECTION + DISTANCE HERE =====
    #     # self.W_offsets.data[child1] = parent_off - delta
    #     # self.W_offsets.data[child2] = parent_off + delta

    #     # print(
    #     #     f"[ReopenInit] parent={parent_global_idx} | "
    #     #     f"direction={direction.detach().cpu().numpy()} | "
    #     #     f"dist_prior={dist_prior:.6f} | split_step={split_step:.6f}"
    #     # )
    #     # print(
    #     #     f"[ReopenInit] parent_off={parent_off.detach().cpu().numpy()} | "
    #     #     f"child1_off={self.W_offsets.data[child1].detach().cpu().numpy()} | "
    #     #     f"child2_off={self.W_offsets.data[child2].detach().cpu().numpy()}"
    #     # )


    #     # 分裂2
    #     # prior_dir = self.split_direction[parent_global_idx].to(self.W_offsets.device)
    #     # prior_norm = torch.norm(prior_dir).item()

    #     # if prior_norm > 1e-8:
    #     #     direction = prior_dir / torch.norm(prior_dir)
    #     # else:
    #     #     angle = parent_rot.item()
    #     #     direction = torch.tensor(
    #     #         [np.cos(angle), np.sin(angle)],
    #     #         dtype=self.W_offsets.dtype,
    #     #         device=self.W_offsets.device,
    #     #     )

    #     # # 分裂距离：优先使用density2feature传来的先验
    #     # dist_prior = float(self.split_distance_prior[parent_global_idx].item())
    #     # split_step = max(self.reopen_offset_step, dist_prior)
    #     # delta = split_step * direction

    #     # if self.reopen_use_shape_bridge:
    #     #     child1_shape, child2_shape, parent_chi, child1_chi, child2_chi = \
    #     #         self.shape_bridge.split_latent(parent_shape, use_mu=True)

    #     #     self.W_shape_var.data[child1] = child1_shape.clone()
    #     #     self.W_shape_var.data[child2] = child2_shape.clone()

    #     #     print("[Reopen-Shape] parent chi =", parent_chi.detach().cpu().numpy())
    #     #     print("[Reopen-Shape] child1 chi =", child1_chi.detach().cpu().numpy())
    #     #     print("[Reopen-Shape] child2 chi =", child2_chi.detach().cpu().numpy())
    #     # else:
    #     #     self.W_shape_var.data[child1] = parent_shape.clone()
    #     #     self.W_shape_var.data[child2] = parent_shape.clone()
    #     # dx = 0.25 * math.sqrt(2.0) * parent_scale * torch.sin(parent_rot - math.pi / 4)
    #     # dy = 0.25 * math.sqrt(2.0) * parent_scale * torch.cos(parent_rot - math.pi / 4)
    #     # #         ===== split direction from prior / fallback to parent rotation =====
        
    #     # # child1
    #     # self.W_scale.data[child1] = parent_scale * self.reopen_scale_shrink
        
    #     # self.W_rotation.data[child1] = parent_rot.clone()
    #     # self.W_offsets.data[child1, 0] = parent_off[0] + dx
    #     # self.W_offsets.data[child1, 1] = parent_off[1] - dy 
    #     # # child2
    #     # self.W_scale.data[child2] = parent_scale * self.reopen_scale_shrink
        
    #     # self.W_rotation.data[child2] = parent_rot.clone()
    #     # self.W_offsets.data[child2, 0] = self.W_offsets.data[child1, 0] + (0.5*parent_scale+0.06)*torch.cos(parent_rot) 
    #     # self.W_offsets.data[child2, 1] = self.W_offsets.data[child1, 1] + (0.5*parent_scale+0.06)*torch.sin(parent_rot) 
    #             # =========================================================
    #     # 分裂3
    #     # =========================================================
    #     prior_dir = self.split_direction[parent_global_idx].to(
    #         device=self.W_offsets.device,
    #         dtype=self.W_offsets.dtype,
    #     )
    #     prior_norm = torch.norm(prior_dir).item()
    #     parent_conf = float(self.split_confidence[parent_global_idx].item())

    #     has_reliable_prior = (
    #         (prior_norm > 1e-8) and
    #         (parent_conf >= self.reopen_min_split_confidence)
    #     )

    #     # split distance
    #     dist_prior = float(self.split_distance_prior[parent_global_idx].item())
    #     split_step = max(float(self.reopen_offset_step), dist_prior)

    #     if self.sigmoid_limits:
    #         parent_off_phys = self.coord_min + (self.coord_max - self.coord_min) * torch.sigmoid(parent_off)
    #     else:
    #         parent_off_phys = parent_off.clone()

    #     if has_reliable_prior:
    #         direction = prior_dir / torch.norm(prior_dir)
    #         d1 = -direction
    #         d2 = direction
    #     else:
    #         d1, d2 = self.select_best_split_direction_8way(
    #            parent_global_idx, parent_off_phys, split_step
    #         )

    #     child1_off_phys = parent_off_phys + split_step * d1
    #     child2_off_phys = parent_off_phys + split_step * d2

    #     # =========================================================
    #     # 2) child shape initialization
    #     # =========================================================
        
    #     if self.reopen_use_canonical_shape_without_prior:
    #         child1_shape, child2_shape, parent_chi, child1_chi, child2_chi = \
    #             self.shape_bridge.split_latent_canonical(use_mu=True)
    #         self.W_shape_var.data[child1] = child1_shape.clone()
    #         self.W_shape_var.data[child2] = child2_shape.clone()

    #         print("[Reopen-Shape-Canonical] child chi =", child1_chi.detach().cpu().numpy())
    #     else:
    #         self.W_shape_var.data[child1] = parent_shape.clone()
    #         self.W_shape_var.data[child2] = parent_shape.clone()

    #     # =========================================================
    #     # 3) child scale
    #     # =========================================================
    #     active_num_after_split = int(self.persistent_mask.sum().item()) - 1 + 2
    #     target_scale = self.get_reopen_target_scale(active_num_after_split)

    #     child_scale_value = min(
    #         float(parent_scale.item()) * float(self.reopen_scale_shrink),
    #         float(target_scale)
    #     )

    #     self.W_scale.data[child1] = torch.tensor(
    #         child_scale_value, dtype=self.W_scale.dtype, device=self.W_scale.device
    #     )
    #     self.W_scale.data[child2] = torch.tensor(
    #         child_scale_value, dtype=self.W_scale.dtype, device=self.W_scale.device
    #     )

    #     # =========================================================
    #     # 4) child rotation
    #     # =========================================================
        
    #     if self.reopen_force_zero_rotation_without_prior:
    #         self.W_rotation.data[child1] = torch.tensor(
    #             0.0, dtype=self.W_rotation.dtype, device=self.W_rotation.device
    #         )
    #         self.W_rotation.data[child2] = torch.tensor(
    #             0.0, dtype=self.W_rotation.dtype, device=self.W_rotation.device
    #         )
    #     else:
    #         self.W_rotation.data[child1] = parent_rot.clone()
    #         self.W_rotation.data[child2] = parent_rot.clone()

    #     # =========================================================
    #     # 5) child offsets：真正使用 direction + delta
    #     # =========================================================
        

    #     if self.sigmoid_limits:
    #         eps = 1e-6
    #         child1_off_norm = (child1_off_phys - self.coord_min) / (self.coord_max - self.coord_min)
    #         child2_off_norm = (child2_off_phys - self.coord_min) / (self.coord_max - self.coord_min)

    #         child1_off_norm = torch.clamp(child1_off_norm, eps, 1 - eps)
    #         child2_off_norm = torch.clamp(child2_off_norm, eps, 1 - eps)

    #         self.W_offsets.data[child1] = torch.logit(child1_off_norm)
    #         self.W_offsets.data[child2] = torch.logit(child2_off_norm)
    #     else:
    #         self.W_offsets.data[child1] = child1_off_phys
    #         self.W_offsets.data[child2] = child2_off_phys

    #     print(
    #         f"[ReopenInit] parent={parent_global_idx} | "
    #         f"has_prior={has_reliable_prior} | "
    #         f"d1={d1.detach().cpu().numpy()} | "
    #         f"d2={d2.detach().cpu().numpy()} | "
    #         f"split_step={split_step:.6f}"
    #     )
    #     print(
    #         f"[ReopenInit] parent_off={parent_off_phys.detach().cpu().numpy()} | "
    #         f"child1_off={child1_off_phys.detach().cpu().numpy()} | "
    #         f"child2_off={child2_off_phys.detach().cpu().numpy()} | "
    #         f"child_scale={child_scale_value:.6f}"
    #     )
    #     # mask 更新：关父开子
    #     self.persistent_mask[parent_global_idx] = False
    #     self.persistent_mask[child1] = True
    #     self.persistent_mask[child2] = True

    #     self.new_gaussians_mask = torch.zeros_like(self.persistent_mask, dtype=bool)
    #     self.new_gaussians_mask[child1] = True
    #     self.new_gaussians_mask[child2] = True

    #             # =========================================================
    #             # =========================================================
    #     # 6) identify remaining active old features
    #     #    现在还 active，但不是新 child
    #     # =========================================================
    #     all_ids = torch.arange(self.persistent_mask.shape[0], device=self.W_scale.device)

    #     old_active_mask = self.persistent_mask.clone()
    #     old_active_mask[child1] = False
    #     old_active_mask[child2] = False

    #     old_active_ids = all_ids[old_active_mask]

    #     # =========================================================
    #     # 7) optional: shrink remaining active old features a bit
    #     #    给 child 留一些空间，增强探索
    #     # =========================================================
    #     if self.reopen_shrink_old_features and len(old_active_ids) > 0:
    #         shrink_ratio = float(self.reopen_old_feature_shrink_ratio)

    #         if self.sigmoid_limits:
    #             # 物理尺度下缩小，再映射回 logit
    #             old_scale_phys = (
    #                 (self.scale_max - self.scale_min) * torch.sigmoid(self.W_scale.data[old_active_ids])
    #                 + self.scale_min
    #             )
    #             old_scale_phys = old_scale_phys * shrink_ratio
    #             old_scale_phys = torch.clamp(
    #                 old_scale_phys,
    #                 self.scale_min + 1e-6,
    #                 self.scale_max - 1e-6
    #             )
    #             old_scale_norm = (old_scale_phys - self.scale_min) / (self.scale_max - self.scale_min)
    #             old_scale_norm = torch.clamp(old_scale_norm, 1e-6, 1 - 1e-6)
    #             self.W_scale.data[old_active_ids] = torch.logit(old_scale_norm)
    #         else:
    #             self.W_scale.data[old_active_ids] *= shrink_ratio
    #             self.W_scale.data[old_active_ids] = torch.clamp(
    #                 self.W_scale.data[old_active_ids],
    #                 min=float(self.scale_min),
    #                 max=float(self.scale_max)
    #             )

    #         print(
    #             f"[Reopen] Shrunk {len(old_active_ids)} remaining old features "
    #             f"by ratio={shrink_ratio:.4f}"
    #         )

    #     # =========================================================
    #     # 8) optional: push remaining old features slightly away from parent
    #     #    给 child 再留一点空间
    #     # =========================================================
    #     if self.reopen_push_old_features and len(old_active_ids) > 0:
    #         push_ratio = float(self.reopen_old_feature_push_ratio)
    #         push_step = push_ratio * float(split_step)

    #         if self.sigmoid_limits:
    #             offsets_phys_all = self.coord_min + (self.coord_max - self.coord_min) * torch.sigmoid(self.W_offsets.data)
    #         else:
    #             offsets_phys_all = self.W_offsets.data.clone()

    #         old_centers = offsets_phys_all[old_active_ids]

    #         # ---------- 可选：只推离 parent 比较近的 old features ----------
    #         dist_to_parent = torch.norm(old_centers - parent_off_phys.unsqueeze(0), dim=1)
    #         near_mask = dist_to_parent < (200 * split_step)

    #         if near_mask.any():
    #             near_old_ids = old_active_ids[near_mask]
    #             near_old_centers = old_centers[near_mask]

    #             vec = near_old_centers - parent_off_phys.unsqueeze(0)
    #             vec_norm = torch.norm(vec, dim=1, keepdim=True).clamp_min(1e-8)
    #             push_dir = vec / vec_norm

    #             new_old_centers = near_old_centers + push_step * push_dir

    #             # 防止出界
    #             new_old_centers[:, 0] = torch.clamp(
    #                 new_old_centers[:, 0],
    #                 self.coord_min[0] + 1e-6,
    #                 self.coord_max[0] - 1e-6
    #             )
    #             new_old_centers[:, 1] = torch.clamp(
    #                 new_old_centers[:, 1],
    #                 self.coord_min[1] + 1e-6,
    #                 self.coord_max[1] - 1e-6
    #             )

    #             if self.sigmoid_limits:
    #                 old_norm = (new_old_centers - self.coord_min) / (self.coord_max - self.coord_min)
    #                 old_norm = torch.clamp(old_norm, 1e-6, 1 - 1e-6)
    #                 self.W_offsets.data[near_old_ids] = torch.logit(old_norm)
    #             else:
    #                 self.W_offsets.data[near_old_ids] = new_old_centers

    #             print(
    #                 f"[Reopen] Pushed {len(near_old_ids)} nearby old features "
    #                 f"away from parent by step={push_step:.6f}"
    #             )
    #             # ===== split prior inheritance =====
    #     parent_conf = float(self.split_confidence[parent_global_idx].item())
    #     parent_dir = self.split_direction[parent_global_idx].clone()
    #     parent_dist = float(self.split_distance_prior[parent_global_idx].item())

    #     # 分裂后两个child仍继承“可继续编辑”的倾向，但略微衰减
    #     child_conf = max(0.0, 0.85 * parent_conf)

    #     self.split_confidence[child1] = child_conf
    #     self.split_confidence[child2] = child_conf

    #     self.split_direction[child1] = parent_dir
    #     self.split_direction[child2] = parent_dir

    #     # child后续仍可沿当前方向进一步微分裂，但步长略减
    #     self.split_distance_prior[child1] = 0.75 * parent_dist
    #     self.split_distance_prior[child2] = 0.75 * parent_dist

    #     # parent关闭
    #     self.split_confidence[parent_global_idx] = 0.0
    #     self.split_direction[parent_global_idx] = 0.0
    #     self.split_distance_prior[parent_global_idx] = 0.0
    #     self.feature_birth_iter[child1] = global_i
    #     self.feature_birth_iter[child2] = global_i
    #     self.feature_birth_iter[parent_global_idx] = -1
    #     self.current_marker = max(self.current_marker, child2 + 1)
    #     self.last_reopen_children = (child1, child2)
    #     print(f"[Reopen] parent={parent_global_idx} -> children=({child1}, {child2})")
    #     return True
    
    def reopening(self, global_i):
        import math

        if not self.has_free_slots(needed=2):
            print("[Reopen] No free slots left.")
            return False

        # 选 active feature 中梯度最大的那个作为 parent
        active_ids, total_score = self.compute_reopen_scores()
        if active_ids is None or len(active_ids) == 0:
            return False

        active_score = total_score[active_ids]

        # 低于最小 split 置信度的孔，不要被先验强推；但梯度特别大仍可选
        active_prior = self.split_confidence[active_ids].to(active_score.device)
        active_score = torch.where(
            active_prior >= self.reopen_min_split_confidence,
            active_score,
            self.reopen_grad_weight * active_score
        )

        parent_global_idx = active_ids[torch.argmax(active_score)].item()
        active_ids_before_split = torch.arange(
            self.persistent_mask.shape[0], device=self.W_scale.device
        )[self.persistent_mask].clone()
        self.reopen_parent_idx = parent_global_idx

        # 找两个空槽位
        free_ids = torch.arange(self.persistent_mask.shape[0], device=self.W_scale.device)[~self.persistent_mask]
        child1 = free_ids[0].item()
        child2 = free_ids[1].item()

        # 取父 feature 当前参数
        parent_scale = self.W_scale.data[parent_global_idx].clone()
        parent_shape = self.W_shape_var.data[parent_global_idx].clone()
        parent_rot = self.W_rotation.data[parent_global_idx].clone()
        parent_off = self.W_offsets.data[parent_global_idx].clone()

        # =========================================================
        # 分裂4：有先验走物理先验，无先验走分布先验
        # =========================================================
        prior_dir = self.split_direction[parent_global_idx].to(
            device=self.W_offsets.device,
            dtype=self.W_offsets.dtype,
        )
        prior_norm = torch.norm(prior_dir).item()
        parent_conf = float(self.split_confidence[parent_global_idx].item())

        has_reliable_prior = (
            (prior_norm > 1e-8) and
            (parent_conf >= self.reopen_min_split_confidence)
        )

        dist_prior = float(self.split_distance_prior[parent_global_idx].item())
        split_step = max(float(self.reopen_offset_step), dist_prior)

        if self.sigmoid_limits:
            parent_off_phys = self.coord_min + (self.coord_max - self.coord_min) * torch.sigmoid(parent_off)
        else:
            parent_off_phys = parent_off.clone()

        # =========================================================
        # 1) 方向与形状先验
        # =========================================================
        # =========================================================
        # 1) 方向与形状先验
        # =========================================================
        if has_reliable_prior:
            # ---------- 有 SIMP/物理先验 ----------
            direction = prior_dir / torch.norm(prior_dir)
            d1 = -direction
            d2 = direction
            used_split_step = split_step

            # 🔥 修改点：不再进行 latent 分裂，直接重置为规范形状 (与无先验分支一致)
            child1_shape, child2_shape, parent_chi, child1_chi, child2_chi = \
                self.shape_bridge.split_latent_ver(parent_shape, use_mu=True)
            self.W_shape_var.data[child1] = child1_shape.clone()
            self.W_shape_var.data[child2] = child2_shape.clone()
            # dx = 0.25 * math.sqrt(2.0) * parent_scale * torch.sin(parent_rot - math.pi / 4)
            # dy = 0.25 * math.sqrt(2.0) * parent_scale * torch.cos(parent_rot - math.pi / 4)
            #     # child1
            # self.W_scale.data[child1] = parent_scale * self.reopen_scale_shrink
        
            # self.W_rotation.data[child1] = parent_rot.clone()
            # self.W_offsets.data[child1, 0] = parent_off[0] + dx
            # self.W_offsets.data[child1, 1] = parent_off[1] - dy 
            # # child2
            # self.W_scale.data[child2] = parent_scale * self.reopen_scale_shrink
        
            # self.W_rotation.data[child2] = parent_rot.clone()
            # self.W_offsets.data[child2, 0] = self.W_offsets.data[child1, 0] + (0.5*parent_scale+0.08)*torch.cos(parent_rot) 
            # self.W_offsets.data[child2, 1] = self.W_offsets.data[child1, 1] + (0.5*parent_scale+0.08)*torch.sin(parent_rot) 
            dx = 0.25 * math.sqrt(1) * parent_scale * torch.sin(parent_rot)
            dy = 0.25 * math.sqrt(1) * parent_scale * torch.cos(parent_rot)
                # child1
            self.W_scale.data[child1] = parent_scale * self.reopen_scale_shrink
        
            self.W_rotation.data[child1] = parent_rot.clone()
            self.W_offsets.data[child1, 0] = parent_off[0] 
            self.W_offsets.data[child1, 1] = parent_off[1] 
            # child2
            self.W_scale.data[child2] = parent_scale * self.reopen_scale_shrink
        
            self.W_rotation.data[child2] = parent_rot.clone()
            self.W_offsets.data[child2, 0] = self.W_offsets.data[child1, 0] + (0.06)*torch.sin(parent_rot) + dx
            self.W_offsets.data[child2, 1] = self.W_offsets.data[child1, 1] - (0.06)*torch.cos(parent_rot) - dy
            
    
        else:
            # ---------- 无 SIMP 先验：使用当前 feature 分布与邻域形状先验 ----------
            nn_idx = self.find_nearest_active_old_feature(parent_global_idx)

            if (nn_idx is not None) and self.reopen_use_distribution_prior_without_simp:
                direction = self.get_distribution_prior_direction(parent_global_idx, nn_idx)
                d1 = -direction
                d2 = direction
                child1_shift_ratio = 0.7
                child2_shift_ratio = 0.3
                print(
                    f"[Reopen-DistributionPrior] parent={parent_global_idx}, "
                    f"neighbor={nn_idx}, direction={direction.detach().cpu().numpy()}"
                )

                # # child1 继承 parent 形状
                # self.W_shape_var.data[child1] = parent_shape.clone()

                # # child2 继承邻近孔洞形状
                # self.W_shape_var.data[child2] = self.W_shape_var.data[nn_idx].clone()

                # # child1 / child2 rotation 分别继承 parent / neighbor
                # child1_rot_value = parent_rot.clone()
                # child2_rot_value = self.W_rotation.data[nn_idx].clone()
                canonical_child_z, canonical_child_chi = self.shape_bridge.get_canonical_latent(use_mu=True)

                # child1 / child2 都改成 canonical latent
                self.W_shape_var.data[child1] = canonical_child_z.clone()
                self.W_shape_var.data[child2] = canonical_child_z.clone()

                print(
                    "[Reopen-Shape-Canonical-DistributionPrior] child chi =",
                    canonical_child_chi.detach().cpu().numpy()
                )

                # rotation 先保持你现在的分布先验逻辑不变
                child1_rot_value = parent_rot.clone()
                child2_rot_value = self.W_rotation.data[nn_idx].clone()
            else:
                # fallback：如果没有邻居，就退回你当前的 8 方向 + canonical
                d1, d2 = self.select_best_split_direction_8way(
                    parent_global_idx, parent_off_phys, split_step
                )
                child1_shift_ratio = 0.7
                child2_shift_ratio = 0.3
                if self.reopen_use_canonical_shape_without_prior:
                    child1_shape, child2_shape, parent_chi, child1_chi, child2_chi = \
                        self.shape_bridge.split_latent_canonical(use_mu=True)
                    self.W_shape_var.data[child1] = child1_shape.clone()
                    self.W_shape_var.data[child2] = child2_shape.clone()

                    print("[Reopen-Shape-Canonical] child chi =", child1_chi.detach().cpu().numpy())
                else:
                    self.W_shape_var.data[child1] = parent_shape.clone()
                    self.W_shape_var.data[child2] = parent_shape.clone()

                child1_rot_value = parent_rot.clone()
                if self.reopen_force_zero_rotation_without_prior:
                    child2_rot_value = torch.tensor(
                        0.0, dtype=self.W_rotation.dtype, device=self.W_rotation.device
                    )
                else:
                    child2_rot_value = parent_rot.clone()

            # 无先验时 offset 更保守一些
            used_split_step = float(self.reopen_offset_shrink_without_prior) * float(split_step)

        # =========================================================
        # 2) child scale
        # =========================================================
        active_num_after_split = int(self.persistent_mask.sum().item()) - 1 + 2
        target_scale = self.get_reopen_target_scale(active_num_after_split)
        if has_reliable_prior:
             child_scale_value = float(parent_scale.item())*0.5
        else:
             child_scale_value = float(0.2)
            # float(parent_scale.item())*0.6
            # float(target_scale)
        
        

        self.W_scale.data[child1] = torch.tensor(
            child_scale_value, dtype=self.W_scale.dtype, device=self.W_scale.device
        )
        self.W_scale.data[child2] = torch.tensor(
            child_scale_value, dtype=self.W_scale.dtype, device=self.W_scale.device
        )

        # =========================================================
        # 3) child rotation
        # =========================================================
        if has_reliable_prior:
            pass
        else:
            self.W_rotation.data[child1] = torch.tensor(
                0.0, dtype=self.W_rotation.dtype, device=self.W_rotation.device
            )
            self.W_rotation.data[child2] = torch.tensor(
                0.0, dtype=self.W_rotation.dtype, device=self.W_rotation.device
            )
        

        # =========================================================
        # 4) child offsets
        # =========================================================
        # child1_off_phys = parent_off_phys - child1_shift_ratio * used_split_step * direction
        # child2_off_phys = parent_off_phys + child2_shift_ratio * used_split_step * direction
        # child1_off_phys, child2_off_phys = self.compute_even_split_child_offsets(parent_global_idx)
        # if self.sigmoid_limits:
        #     eps = 1e-6
        #     child1_off_norm = (child1_off_phys - self.coord_min) / (self.coord_max - self.coord_min)
        #     child2_off_norm = (child2_off_phys - self.coord_min) / (self.coord_max - self.coord_min)

        #     child1_off_norm = torch.clamp(child1_off_norm, eps, 1 - eps)
        #     child2_off_norm = torch.clamp(child2_off_norm, eps, 1 - eps)

        #     self.W_offsets.data[child1] = torch.logit(child1_off_norm)
        #     self.W_offsets.data[child2] = torch.logit(child2_off_norm)
        # else:
        #     self.W_offsets.data[child1] = child1_off_phys
        #     self.W_offsets.data[child2] = child2_off_phys

        # print(
        #     f"[ReopenInit-v4] parent={parent_global_idx} | "
        #     f"has_prior={has_reliable_prior} | "
        #     f"d1={d1.detach().cpu().numpy()} | "
        #     f"d2={d2.detach().cpu().numpy()} | "
        #     f"used_split_step={used_split_step:.6f}"
        # )
        # print(
        #     f"[ReopenInit-v4] parent_off={parent_off_phys.detach().cpu().numpy()} | "
        #     f"child1_off={child1_off_phys.detach().cpu().numpy()} | "
        #     f"child2_off={child2_off_phys.detach().cpu().numpy()} | "
        #     f"child_scale={child_scale_value:.6f}"
        # )

        # =========================================================
        # 5) mask 更新：关父开子
        # =========================================================
        self.persistent_mask[parent_global_idx] = False
        self.persistent_mask[child1] = True
        self.persistent_mask[child2] = True

        self.new_gaussians_mask = torch.zeros_like(self.persistent_mask, dtype=bool)
        self.new_gaussians_mask[child1] = True
        self.new_gaussians_mask[child2] = True

        # =========================================================
        # 6) identify remaining active old features
        # =========================================================
        all_ids = torch.arange(self.persistent_mask.shape[0], device=self.W_scale.device)

        old_active_mask = self.persistent_mask.clone()
        old_active_mask[child1] = False
        old_active_mask[child2] = False

        old_active_ids = all_ids[old_active_mask]
        # =========================================================
        # 6.5) child offsets: use post-split canonical layout
        # =========================================================
        canonical_old_dict = None  # 🔥 修复点 1：在这里提前初始化为 None
        child1_off_phys_canon, child2_off_phys_canon, canonical_old_dict = \
            self.compute_post_split_canonical_layout(parent_global_idx, active_ids_before_split, old_active_ids)
        if has_reliable_prior:
            # 如果有可靠先验，直接跳过全局一维布局，使用刚才计算好的物理坐标
            pass 
        else:
            # 只有无先验时，才跑那个强迫症般的“一维规范排布”
            child1_off_phys, child2_off_phys, canonical_old_dict = \
                self.compute_post_split_canonical_layout(parent_global_idx, active_ids_before_split, old_active_ids)
            self.W_offsets.data[child1] = child1_off_phys_canon
            self.W_offsets.data[child2] = child2_off_phys_canon
        # if self.sigmoid_limits:
        #     eps = 1e-6
        #     child1_off_norm = (child1_off_phys - self.coord_min) / (self.coord_max - self.coord_min)
        #     child2_off_norm = (child2_off_phys - self.coord_min) / (self.coord_max - self.coord_min)

        #     child1_off_norm = torch.clamp(child1_off_norm, eps, 1 - eps)
        #     child2_off_norm = torch.clamp(child2_off_norm, eps, 1 - eps)

        #     self.W_offsets.data[child1] = torch.logit(child1_off_norm)
        #     self.W_offsets.data[child2] = torch.logit(child2_off_norm)
        # else:
        #     self.W_offsets.data[child1] = child1_off_phys
        #     self.W_offsets.data[child2] = child2_off_phys
        # =========================================================
        # 6.5) optional: canonicalize remaining old feature shapes
        #      只改 shape latent，保留它们当前的 offset / scale / rotation
        # =========================================================
        if getattr(self, "reopen_canonicalize_old_feature_shapes", False) and len(old_active_ids) > 0:
            canonical_z, canonical_chi = self.shape_bridge.get_canonical_latent(use_mu=True)
            alpha = float(getattr(self, "reopen_old_feature_canonical_blend", 1.0))
            alpha = max(0.0, min(1.0, alpha))

            old_z = self.W_shape_var.data[old_active_ids].clone()
            canonical_batch = canonical_z.unsqueeze(0).expand_as(old_z)
            blended_z = (1.0 - alpha) * old_z + alpha * canonical_batch

            self.W_shape_var.data[old_active_ids] = blended_z

            print(
                f"[Reopen] Blended {len(old_active_ids)} old feature shapes with canonical "
                f"(alpha={alpha:.3f}) | chi={canonical_chi.detach().cpu().numpy()}"
            )
        # =========================================================
        # 7) optional: shrink remaining active old features
        # =========================================================
        if self.reopen_shrink_old_features and len(old_active_ids) > 0:
            shrink_ratio = float(self.reopen_old_feature_shrink_ratio)

            if self.sigmoid_limits:
                old_scale_phys = (
                    (self.scale_max - self.scale_min) * torch.sigmoid(self.W_scale.data[old_active_ids])
                    + self.scale_min
                )
                old_scale_phys = old_scale_phys * shrink_ratio
                old_scale_phys = torch.clamp(
                    old_scale_phys,
                    self.scale_min + 1e-6,
                    self.scale_max - 1e-6
                )
                old_scale_norm = (old_scale_phys - self.scale_min) / (self.scale_max - self.scale_min)
                old_scale_norm = torch.clamp(old_scale_norm, 1e-6, 1 - 1e-6)
                self.W_scale.data[old_active_ids] = torch.logit(old_scale_norm)
            else:
                # self.W_scale.data[old_active_ids] *= shrink_ratio
                # self.W_scale.data[old_active_ids] = torch.clamp(
                #     self.W_scale.data[old_active_ids],
                #     min=float(self.scale_min),
                #     max=float(self.scale_max)
                # )
                self.W_scale.data[old_active_ids] = torch.full_like(
                    self.W_scale.data[old_active_ids],
                    0.25
                )
                self.W_scale.data[old_active_ids] = torch.clamp(
                    self.W_scale.data[old_active_ids],
                    min=float(self.scale_min),
                    max=float(self.scale_max)
                )
            print(
                f"[Reopen] Shrunk {len(old_active_ids)} remaining old features "
                f"by ratio={shrink_ratio:.4f}"
            )

        # =========================================================
        # 8) optional: blend remaining old feature offsets with canonical offsets
        #    canonical offsets 采用 split 后总孔数统一编号
        # =========================================================
        if self.reopen_push_old_features and len(old_active_ids) > 0 and canonical_old_dict is not None:
            alpha = float(getattr(self, "reopen_old_feature_offset_canonical_blend", 0.5))
            alpha = max(0.0, min(1.0, alpha))

            if self.sigmoid_limits:
               offsets_phys_all = self.coord_min + (self.coord_max - self.coord_min) * torch.sigmoid(self.W_offsets.data)
            else:
                offsets_phys_all = self.W_offsets.data.clone()

            old_centers = offsets_phys_all[old_active_ids]

            canonical_centers_aligned = torch.stack(
                [canonical_old_dict[int(oid.item())] for oid in old_active_ids],
                dim=0
            )

            new_old_centers = (1.0 - alpha) * old_centers + alpha * canonical_centers_aligned

            new_old_centers[:, 0] = torch.clamp(
                new_old_centers[:, 0],
                self.coord_min[0] + 1e-6,
                self.coord_max[0] - 1e-6
            )
            new_old_centers[:, 1] = torch.clamp(
                new_old_centers[:, 1],
                self.coord_min[1] + 1e-6,
                self.coord_max[1] - 1e-6
            )

            if self.sigmoid_limits:
                old_norm = (new_old_centers - self.coord_min) / (self.coord_max - self.coord_min)
                old_norm = torch.clamp(old_norm, 1e-6, 1 - 1e-6)
                self.W_offsets.data[old_active_ids] = torch.logit(old_norm)
            else:
                self.W_offsets.data[old_active_ids] = new_old_centers

            print(
                f"[Reopen] Blended {len(old_active_ids)} old feature offsets with post-split canonical offsets "
                f"(alpha={alpha:.3f})"
            )

        # =========================================================
        # 9) split prior inheritance
        # =========================================================
        parent_conf = float(self.split_confidence[parent_global_idx].item())
        parent_dir = self.split_direction[parent_global_idx].clone()
        parent_dist = float(self.split_distance_prior[parent_global_idx].item())

        child_conf = max(0.0, 0.85 * parent_conf)

        self.split_confidence[child1] = child_conf
        self.split_confidence[child2] = child_conf
    
        self.split_direction[child1] = parent_dir
        self.split_direction[child2] = parent_dir

        self.split_distance_prior[child1] = 0.75 * parent_dist
        self.split_distance_prior[child2] = 0.75 * parent_dist

        # parent关闭
        self.split_confidence[parent_global_idx] = 0.0
        self.split_direction[parent_global_idx] = 0.0
        self.split_distance_prior[parent_global_idx] = 0.0

        self.feature_birth_iter[child1] = global_i
        self.feature_birth_iter[child2] = global_i
        self.feature_birth_iter[parent_global_idx] = -1

        self.current_marker = max(self.current_marker, child2 + 1)
                # ===== debug: check volume state immediately after reopen initialization =====
        _ = self.get_x(global_i)

        current_volfrac_after_reopen = (self.H.T @ self.volumes / self.volumes.sum()).item()

        volfrac_goal = self.volfrac - self.volfrac_increment * max(
            0, min(1, (global_i - self.volfrac_increment_marker) / self.volfrac_increment_duration)
        )
        volume_goal = self.volumes_sum * volfrac_goal
        current_vol_constraint_after_reopen = (self.H.T @ self.volumes / volume_goal - 1).item()

        print(
            f"[ReopenInit-Debug] iter={global_i} | "
            f"volfrac_after_reopen={current_volfrac_after_reopen:.6f} | "
            f"vol_constraint_after_reopen={current_vol_constraint_after_reopen:.6f}"
        )
        self.last_reopen_children = (child1, child2)

        print(f"[Reopen-v4] parent={parent_global_idx} -> children=({child1}, {child2})")
        return True
    
    def get_x(self, global_i):

        W_scale = self.W_scale[self.persistent_mask]
        W_shape_var = self.W_shape_var[self.persistent_mask] 
        W_rotation = self.W_rotation[self.persistent_mask]
        W_offsets = self.W_offsets[self.persistent_mask]
        batch_size = W_scale.shape[0]

        if self.sigmoid_limits:
            base_scale = (self.scale_max - self.scale_min)*torch.sigmoid(W_scale) + self.scale_min
            shape_var = (self.shape_var_maxs - self.shape_var_mins)*torch.sigmoid(W_shape_var) + self.shape_var_mins   
            rotation = self.rotation_min + (self.rotation_max - self.rotation_min)*torch.sigmoid(W_rotation).view(batch_size)
            offsets = self.coord_min + (self.coord_max - self.coord_min)*torch.sigmoid(W_offsets)
        else:
            base_scale = W_scale
            shape_var = W_shape_var
            rotation = W_rotation
            offsets = W_offsets

        cos_rot = torch.cos(rotation)
        sin_rot = torch.sin(rotation)

        # rotation matrix
        R = torch.stack([
            torch.stack([cos_rot, -sin_rot], dim=-1),
            torch.stack([sin_rot, cos_rot], dim=-1)
        ], dim=-2)

        # Compute inverse covariance
        coords = self.coords.unsqueeze(0).expand(batch_size, -1, -1, -1)
        offsets = offsets[:, None, None, :]
        xy = coords - offsets

        scaled_xy = xy /((base_scale[:, None, None, None])+1e-8)
        scaled_xy_shifted = xy/(self.gaussian_overlap_scale*base_scale[:, None, None, None]+1e-8)

        # Apply rotation to coordinates
        rotated_scaled_xy = torch.einsum('bij,bxyj->bxyi', R, scaled_xy)
        rotated_scaled_xy_shifted = torch.einsum('bij,bxyj->bxyi', R, scaled_xy_shifted)
        
        rotated_scaled_xy = rotated_scaled_xy.view(batch_size, -1, 2)
        rotated_scaled_xy_shifted = rotated_scaled_xy_shifted.view(batch_size, -1, 2)

        # Calculate z using pre_inv_covariance (no rotation)
        # rot_scaled_xy = torch.einsum('bxyi,bij,bxyj->bxy', rotated_xy, -0.5 * pre_inv_covariance, rotated_xy)
        kernel = torch.zeros(self.coords.shape[0], batch_size)#.double()
        kernel_shifted = torch.zeros(self.coords.shape[0], batch_size)#.double()

        for i in range(batch_size):
            mask = (rotated_scaled_xy[i].abs().max(dim=-1)[0] < 1)
            mask_shifted = (rotated_scaled_xy_shifted[i].abs().max(dim=-1)[0] < 1)
            heaviside = self.model.sdf(shape_var[i], rotated_scaled_xy[i, mask, :])
            heaviside_shifted = self.model.sdf(shape_var[i], rotated_scaled_xy_shifted[i, mask_shifted, :])
            kernel[mask, i] = heaviside.squeeze()#.double()
            kernel_shifted[mask_shifted, i] = heaviside_shifted.squeeze()#.double()

        if self.if_merging_preparation(global_i):
            self.compute_merging_pairs(kernel)

        # Kreisselmeier-Steinhause
        sm_coeff = self.ks_smooth_factor
        kernel_sum = kernel.sum(dim=1)
        kernel = torch.sigmoid(self.smooth_k*(kernel - 0.5))
        
        # self.kernel_max = max(self.kernel_max, kernel.max())
        # self.kernel_min = min(self.kernel_min, kernel.min())
        # print("kernel min and max: ", kernel.min(), kernel.max())
        exp_kernel_sum = torch.exp(sm_coeff*kernel).sum(dim=1)+1e-8
        # print("exp_kernel_sum min and max: ", exp_kernel_sum.min(), exp_kernel_sum.max())
        # exp_kernel_sum_shifted = torch.exp(kernel_shifted).sum(dim=1)+1e-8
        # self.H = (1 - self.Emin)*torch.sigmoid(-self.smooth_k*(kernel_sum - 0.5)) + self.Emin
        self.H = (1 - self.Emin)*(self.ks_max - torch.log(exp_kernel_sum)/sm_coeff)/self.ks_range + self.Emin

        self.H_min = min(self.H_min, self.H.min())
        self.H_max = max(self.H_max, self.H.max())

        # print("H min and max: ", self.H.min(), self.H.max())

        # self.H_splitted = torch.sigmoid(0.1*self.smooth_k*(kernel_shifted - 0.5))
        self.H_splitted = kernel_shifted
        self.H_splitted_sum = self.H_splitted.sum(dim=1)
        # self.H_inverted = kernel_sum*0.5

        self.H_splitted_sum_clipped = torch.nn.functional.relu(
            self.H_splitted_sum - 1
        )

        return self.H
    def snapshot_state(self):
        return {
            "W_scale": self.W_scale.data.clone(),
            "W_shape_var": self.W_shape_var.data.clone(),
            "W_rotation": self.W_rotation.data.clone(),
            "W_offsets": self.W_offsets.data.clone(),
            "persistent_mask": self.persistent_mask.clone(),
            "new_gaussians_mask": self.new_gaussians_mask.clone(),
            "split_confidence": self.split_confidence.clone(),
            "split_distance_prior": self.split_distance_prior.clone(),
            "split_direction": self.split_direction.clone(),
            "feature_birth_iter": self.feature_birth_iter.clone(),
            "current_marker": int(self.current_marker),
            "last_reopen_iter": int(self.last_reopen_iter),
            "reopen_parent_idx": -1 if self.reopen_parent_idx is None else int(self.reopen_parent_idx),
            "last_reopen_children": None if not hasattr(self, "last_reopen_children") else self.last_reopen_children,
        }
    
    def restore_state(self, state):
        self.W_scale.data.copy_(state["W_scale"])
        self.W_shape_var.data.copy_(state["W_shape_var"])
        self.W_rotation.data.copy_(state["W_rotation"])
        self.W_offsets.data.copy_(state["W_offsets"])
        self.persistent_mask.copy_(state["persistent_mask"])
        self.new_gaussians_mask.copy_(state["new_gaussians_mask"])
        self.split_confidence.copy_(state["split_confidence"])
        self.split_distance_prior.copy_(state["split_distance_prior"])
        self.split_direction.copy_(state["split_direction"])
        self.feature_birth_iter.copy_(state["feature_birth_iter"])
        self.current_marker = int(state["current_marker"])
        self.last_reopen_iter = int(state["last_reopen_iter"])

        rp = int(state["reopen_parent_idx"])
        self.reopen_parent_idx = None if rp < 0 else rp
        self.last_reopen_children = state["last_reopen_children"]
    
    def start_reopen_callback(self, global_i, splitted_loss):
        if not self.callback_enable:
            return

        compliance_now = float(splitted_loss["compliance"].item())
        vol_cons_now = float(splitted_loss["volfrac_constraint"].item())
        child_pair = getattr(self, "last_reopen_children", None)
        child_dist0 = None
        if child_pair is not None:
            c1, c2 = child_pair
            off1 = self.W_offsets.data[c1]
            off2 = self.W_offsets.data[c2]
            child_dist0 = float(torch.norm(off1 - off2).item())
        self.event_counter += 1
        self.pending_event = {
            "event_id": int(self.event_counter),
            "type": "reopen",
            "start_iter": int(global_i),
            "end_iter": int(global_i + self.callback_observation_steps),
            "baseline_compliance": compliance_now,
            "baseline_vol_constraint": vol_cons_now,
            "best_compliance": compliance_now,
            "best_vol_constraint": vol_cons_now,
            "best_iter": int(global_i),
            "snapshot_before": self.snapshot_state(),
            "best_state": self.snapshot_state(),
            "status": "running",
            # ===== child collapse tracking =====
            "child_pair": child_pair,
            "child_dist_start": child_dist0,
            "child_dist_last": child_dist0,
            "child_dist_history": [] if child_dist0 is None else [child_dist0],
            "child_closer_count": 0,
            "child_total_updates": 0,
        }

        print(
            f"[Callback] Start reopen event #{self.pending_event['event_id']} "
            f"at iter={global_i}, baseline compliance={compliance_now:.6f}, "
            f"child_dist0={child_dist0}"
        )
    
    def update_callback(self, global_i, splitted_loss):
        if (not self.callback_enable) or (self.pending_event is None):
            return None

        event = self.pending_event
        if event["status"] != "running":
            return None

        comp = float(splitted_loss["compliance"].item())
        vol_cons = float(splitted_loss["volfrac_constraint"].item())

        # =========================================================
        # 1) 更新 best compliance / best state
        # =========================================================
        if comp < event["best_compliance"]:
            event["best_compliance"] = comp
            event["best_vol_constraint"] = vol_cons
            event["best_iter"] = int(global_i)
            event["best_state"] = self.snapshot_state()

        # =========================================================
        # 2) 更新 child collapse trend
        # =========================================================
        child_pair = event.get("child_pair", None)
        if child_pair is not None:
            c1, c2 = child_pair

            # 只有两个 child 仍然 active 时，这个趋势才有意义
            if self.persistent_mask[c1] and self.persistent_mask[c2]:
                off1 = self.W_offsets.data[c1]
                off2 = self.W_offsets.data[c2]
                dist_now = float(torch.norm(off1 - off2).item())

                last_dist = event.get("child_dist_last", None)
                if last_dist is not None and dist_now < last_dist:
                    event["child_closer_count"] += 1

                event["child_total_updates"] += 1
                event["child_dist_last"] = dist_now
                event["child_dist_history"].append(dist_now)

        # =========================================================
        # 3) 还没到窗口末尾，继续
        # =========================================================
        if global_i < event["end_iter"]:
            return {
                "action": "continue",
                "event_id": event["event_id"],
            }

        # =========================================================
        # 4) 到窗口末尾，综合裁决
        # =========================================================
        baseline = event["baseline_compliance"]
        best = event["best_compliance"]
        rel_improve = (baseline - best) / (abs(baseline) + 1e-8)

        vol_ok = event["best_vol_constraint"] <= (
            event["baseline_vol_constraint"] + self.callback_reject_vol_slack
        )

        # ----- child collapse indicators -----
        collapse_ratio = None
        closer_ratio = None
        collapse_flag = False

        d0 = event.get("child_dist_start", None)
        d_last = event.get("child_dist_last", None)
        total_updates = max(1, int(event.get("child_total_updates", 0)))
        closer_count = int(event.get("child_closer_count", 0))

        if (d0 is not None) and (d_last is not None):
            collapse_ratio = d_last / (d0 + 1e-8)
            closer_ratio = closer_count / total_updates

            # 两个条件都满足时，认为 child 正在重新塌缩/靠拢
            collapse_flag = (
                collapse_ratio < 0.50 and
                closer_ratio > 0.80
            )

        # =========================================================
        # 5) accept / rollback
        # =========================================================
        # 规则：
        # - 柔度必须有足够改善
        # - 体积约束不能明显恶化
        # - child 不应明显重新塌缩
        accept = (
            (rel_improve >= self.callback_accept_rel_improve)
            and (not collapse_flag)
        )

        if accept:
            
            event["status"] = "accepted"

            print(
                f"[Callback] Accept event #{event['event_id']} | "
                f"best_iter={event['best_iter']} | "
                f"rel_improve={rel_improve:.6f} | "
                f"collapse_ratio={collapse_ratio} | "
                f"closer_ratio={closer_ratio}"
            )

            self.pending_event = None
            return {
                "action": "accept",
                "event_id": event["event_id"],
                "rel_improve": rel_improve,
                "collapse_ratio": collapse_ratio,
                "closer_ratio": closer_ratio,
            }

        else:
            self.restore_state(event["snapshot_before"])
            event["status"] = "rejected"

            print(
                f"[Callback] Rollback event #{event['event_id']} | "
                f"rel_improve={rel_improve:.6f} | "
                f"vol_ok={vol_ok} | "
                f"collapse_ratio={collapse_ratio} | "
                f"closer_ratio={closer_ratio} | "
                f"collapse_flag={collapse_flag}"
            )

            self.pending_event = None
            return {
                "action": "rollback",
                "event_id": event["event_id"],
                "rel_improve": rel_improve,
                "collapse_ratio": collapse_ratio,
                "closer_ratio": closer_ratio,
            }
        
    def get_young_feature_mask(self, global_i):
        age = global_i - self.feature_birth_iter.to(self.W_scale.device)
        young_mask = (
            self.persistent_mask
            & (self.feature_birth_iter >= 0)
            & (age >= 0)
            & (age < self.reopen_adaptation)
        )
        return young_mask
    
    def prepare_grads(self, global_i):

        evolving_mask = self.persistent_mask


        self.W_scale.grad.data[~evolving_mask] = 0.0
        self.W_rotation.grad.data[~evolving_mask] = 0.0
        self.W_offsets.grad.data[~evolving_mask] = 0.0
        self.W_shape_var.grad.data[~evolving_mask] = 0.0

        # in_shape_optimization_range = any(start <= global_i <= end for start, end in self.shape_optimization_ranges)

        shape_opt_range = False
        for start, end in self.shape_optimization_ranges:
            if start <= global_i <= end:
                print(f"Global iteration {global_i} is within one of the shape optimization ranges.")
                # Calculate the proportion of the current iteration within the shape optimization range
                proportion = max(0, min(1, (global_i - start) / (end - start)))
                # Decrease self.offset_grad_deceleration from its initial value to 0
                self.W_offsets.grad.data[evolving_mask] *= (1 - proportion) * self.offset_grad_deceleration
                shape_opt_range = True

        if not shape_opt_range:
            self.W_shape_var.grad.data[evolving_mask] = 0.0
            self.W_rotation.grad.data[evolving_mask] = self.W_rotation.grad.data[evolving_mask] * 0.6
        young_mask = self.get_young_feature_mask(global_i)
        old_mask = self.persistent_mask & (~young_mask)

        if young_mask.any():
            young_age = global_i - self.feature_birth_iter[young_mask]
            young_age = young_age.to(torch.float32)

            # 0 -> 1 的归一化进度
            t = young_age / max(1, self.reopen_adaptation - 1)
            t = torch.clamp(t, 0.0, 1.0)

            # 注意：下面是逐 feature 缩放，所以需要 unsqueeze 对齐维度
            offset_factor = 1.0 - 0.3 * t          # 1.0 -> 0.7
            scale_factor = 0.8 - 0.4 * t           # 0.8 -> 0.4
            rotation_factor = 0.2 + 0.4 * t        # 0.2 -> 0.6
            shape_factor = 0.1 + 0.5 * t           # 0.1 -> 0.6

            self.W_offsets.grad.data[young_mask] *= offset_factor.unsqueeze(-1)
            self.W_scale.grad.data[young_mask] *= scale_factor
            self.W_rotation.grad.data[young_mask] *= rotation_factor
            self.W_shape_var.grad.data[young_mask] *= shape_factor.unsqueeze(-1)

            # 老 feature 稍微稳一点，但别压太狠
            self.W_offsets.grad.data[old_mask] *= 0.8
    
    def suppress_old_features_after_reopen(self, global_i):
        if not self.if_reopen_exploration(global_i):
            return

        all_ids = torch.arange(self.persistent_mask.shape[0], device=self.W_scale.device)
        old_active_mask = self.persistent_mask.clone()
        old_active_mask[self.new_gaussians_mask] = False
        old_active_ids = all_ids[old_active_mask]

        if len(old_active_ids) == 0:
            return

        # 1) exploration 阶段冻结 old feature scale
        if self.W_scale.grad is not None:
            self.W_scale.grad[old_active_ids] = 0.0

        # 2) old feature 的 offset 只允许小幅动
        if self.W_offsets.grad is not None:
            self.W_offsets.grad[old_active_ids] *= 0.2

        # 3) new children 可稍微放大一点探索能力
        new_ids = all_ids[self.new_gaussians_mask]
        if len(new_ids) > 0:
            if self.W_offsets.grad is not None:
                self.W_offsets.grad[new_ids] *= 1.2

    def compute_rc_loss(self, global_i): # form factor loss
        
        if global_i > 100:
            W_shape_var = self.W_shape_var[self.persistent_mask]
            shape_var = (self.shape_var_maxs - self.shape_var_mins)*torch.sigmoid(W_shape_var) + self.shape_var_mins             
            radius_sum = torch.nn.functional.relu(self.model.tau(shape_var)-0.1)
            return radius_sum.sum()

        #######################################################################

        else:
            return torch.tensor(0.0)
        
    def update_shape_combination(self, global_i):

        if self.if_refactoring(global_i):
            self.refactoring()
    
    def forward(self, ce, global_i):
        self.get_x(global_i)
        rc_loss = self.compute_rc_loss(global_i)

        # Add numerical stability to compliance calculation
        H_vec = torch.clamp(self.H, min=self.Emin, max=self.Emax)**self.penal
        # compliance = torch.dot(H_vec, ce.float())
        # 【兼容性修改开始】=========================================
        if isinstance(ce, dict):
            # --- 走双工况 (CASR) 逻辑 ---
            ce_p = ce["ce_p"].float().to(H_vec.device)
            ce_c = ce["ce_c"].float().to(H_vec.device)
            
            compliance_p = torch.dot(H_vec, ce_p)
            compliance_c = torch.dot(H_vec, ce_c)
            
            # 权重设置：w_c 越大，越强调交叉轴抑制
            w_p = 0.03
            w_c = 1.0 
            # 2. 计算每个单元的真实局部应变能（密度惩罚后）
            # H_vec 是当前材料密度场，ce_p 是单元初始应变能
            local_ce_p = H_vec * ce_p
        
            # 3. 计算 P-范数 (P-norm)，提取全场应力/应变能最高的最薄弱点
            # p 值越大，优化器越会死盯着“最痛”的那个点去修复 (通常设为 8 或 10)
            p_value = 8
            max_local_energy = torch.norm(local_ce_p, p=p_value)
            # 4. 设定应力集中惩罚权重
           # 这个值需要调参：如果依然出现细杆，就把这个值加大；如果结构变得太死板，就减小。
            w_stress = 5.0 
        
            # 5. 新的总目标函数：最小化交叉轴变形 - 最大化主轴变形 + 狠狠惩罚最危险的应力集中点
            compliance = w_c * compliance_c - w_p * compliance_p + w_stress * max_local_energy
            print(f"[调参观察] 总 Loss: {compliance.item():.2f} | "
                  f"①交叉轴惩罚: {(w_c * compliance_c).item():.2f} | "
                  f"②主轴奖励: {-(w_p * compliance_p).item():.2f} | "
                  f"③应力惩罚: {(w_stress * max_local_energy).item():.2f}")
        else:
            # --- 走原始单工况逻辑 ---
            ce_tensor = ce.float().to(H_vec.device)
            compliance = torch.dot(H_vec, ce_tensor)
        # 【兼容性修改结束】=========================================
        # volfrac_goal = self.volfrac - 0.1*max(0, min(1, (global_i-20)/20))
        volfrac_goal = self.volfrac - self.volfrac_increment * max(
            0, min(1, (global_i - self.volfrac_increment_marker) / self.volfrac_increment_duration)
        )
        volume_goal = self.volumes_sum * volfrac_goal
        # Safeguard the loss calculations
        # TODO: change to vectorproduct with volume vectors
        volfrac_loss_pre = torch.nn.functional.relu(
            self.H.T @ self.volumes/volume_goal - 1
        )
        # # 【修改A】：计算体积误差
        # vol_ratio_err = self.H.T @ self.volumes / volume_goal - 1.0
        
        # # 【修改B】：神经网络端的体积惩罚改为平方误差（绝对值）
        # volfrac_loss_pre = vol_ratio_err ** 2
        gaussian_overlap = self.H_splitted_sum_clipped.mean()

        print(f"Iteration: {global_i} | current volfrac: {(self.H.T @ self.volumes/self.volumes.sum()).item():.6f} | compliance: {compliance.item():.6f}")
      
        # check the sign
        # Start of Selection
        obj_ce = (
            -compliance * self.compliance_w
            + rc_loss * self.rc_loss_w
            + volfrac_loss_pre * self.volfrac_w
            + gaussian_overlap * self.gaussian_overlap_w
        )
        obj_real = (
            compliance * self.compliance_w
            + rc_loss * self.rc_loss_w
            + volfrac_loss_pre * self.volfrac_w
            + gaussian_overlap * self.gaussian_overlap_w
        )


        splitted_loss = {
            "gaussian_overlap": gaussian_overlap,
            "volfrac_loss_pre": volfrac_loss_pre,
            "volfrac_constraint": self.H.T @ self.volumes/volume_goal - 1,
            # "volfrac_constraint": vol_ratio_err ** 2 - 0.01,
            "compliance": compliance,
            "rc_loss": rc_loss,
            "obj_ce": obj_ce,
            "obj_real": obj_real,
        }

        return obj_ce, splitted_loss

# Add MMA-based FeatureMappingTopOptMMA
class FeatureMappingTopOptMMA(FeatureMappingTopOpt):
    """Feature mapping top opt using MMA optimizer instead of Adam."""
    def __init__(self, args) -> None:
        super().__init__(args)
        # Override Adam with MMA optimizer
        from mmapy2.mma_pytorch import MMA
        params = [
            self.gaussian_core.W_scale,
            self.gaussian_core.W_shape_var,
            self.gaussian_core.W_rotation,
            self.gaussian_core.W_offsets,
        ]
        self.reset_mma_after_reopen = False
        self.last_reopen_phase = None
        # Flatten total design variable count
        n = sum(p.numel() for p in params)
        # Set wide bounds for MMA
        # lower_bound = torch.full((n, 1), -1e3,
        #                          dtype=params[0].dtype,
        #                          device=params[0].device)
        # upper_bound = torch.full((n, 1), 1e3,
        #                          dtype=params[0].dtype,
        #                          device=params[0].device)

        # Export per‐parameter bounds from gaussian_core
        lb_list, ub_list = self.gaussian_core.export_bounds()
        # Flatten and concatenate into (n,1) tensors for MMA
        lower_bound = torch.cat([b.reshape(-1, 1) for b in lb_list], dim=0)
        upper_bound = torch.cat([b.reshape(-1, 1) for b in ub_list], dim=0)

        
        num_cons = 1
        # Instantiate MMA optimizer
        self.optim = MMA(params, num_cons, lower_bound, upper_bound)

    def get_reopen_phase(self):
        if self.gaussian_core.pending_event is None:
            return "normal"
        if self.gaussian_core.if_reopen_exploration(self.global_i):
            return "explore"
        if self.gaussian_core.if_reopen_transition(self.global_i):
            return "transition"
        return "normal"
    
    def rebuild_mma_with_custom_bounds(self, lb_list, ub_list):
        from mmapy2.mma_pytorch import MMA

        params = [
            self.gaussian_core.W_scale,
            self.gaussian_core.W_shape_var,
            self.gaussian_core.W_rotation,
            self.gaussian_core.W_offsets,
        ]

        lower_bound = torch.cat([b.reshape(-1, 1) for b in lb_list], dim=0)
        upper_bound = torch.cat([b.reshape(-1, 1) for b in ub_list], dim=0)

        self.optim = MMA(params, 1, lower_bound, upper_bound)
    def rebuild_mma_with_old_scale_frozen(self):
        lb_list, ub_list = self.gaussian_core.export_bounds()

        all_ids = torch.arange(
            self.gaussian_core.persistent_mask.shape[0],
            device=self.gaussian_core.W_scale.device
        )

        old_active_mask = self.gaussian_core.persistent_mask.clone()
        old_active_mask[self.gaussian_core.new_gaussians_mask] = False
        old_active_ids = all_ids[old_active_mask]

        print("[Freeze-Debug] persistent_mask =", self.gaussian_core.persistent_mask.detach().cpu().numpy())
        print("[Freeze-Debug] new_gaussians_mask =", self.gaussian_core.new_gaussians_mask.detach().cpu().numpy())
        print("[Freeze-Debug] old_active_ids =", old_active_ids.detach().cpu().numpy())

        if len(old_active_ids) > 0:
            current_scale = self.gaussian_core.W_scale.data.detach().clone()

            # 不要让 lb == ub，会导致 MMA 内部除零
            freeze_eps = 1e-4

            lb_scale = lb_list[0].clone()
            ub_scale = ub_list[0].clone()

            lb_scale[old_active_ids] = torch.clamp(
                current_scale[old_active_ids] - freeze_eps,
                min=self.gaussian_core.scale_min,
                max=self.gaussian_core.scale_max
            )
            ub_scale[old_active_ids] = torch.clamp(
                current_scale[old_active_ids] + freeze_eps,
                min=self.gaussian_core.scale_min,
                max=self.gaussian_core.scale_max
            )

            lb_list[0] = lb_scale
            ub_list[0] = ub_scale

            print(
                f"[Reopen-Freeze] Nearly frozen scale of {len(old_active_ids)} old active features "
                f"with eps={freeze_eps}"
            )
            print(
                "[Freeze-Debug] frozen scale values =",
                current_scale[old_active_ids].detach().cpu().numpy()
            )

            

        self.rebuild_mma_with_custom_bounds(lb_list, ub_list)

    def parameter_opt_step(self, ce):
        from mmapy2.mma_pytorch import MMA
        print(f"[DEBUG] parameter_opt_step iter={self.global_i}")
        # 【新增：兼容字典和单数组的 Tensor 转换】
        if isinstance(ce, dict):
            ce_tensor_input = {
                "ce_p": torch.tensor(ce["ce_p"], device=self.gaussian_core.W_scale.device),
                "ce_c": torch.tensor(ce["ce_c"], device=self.gaussian_core.W_scale.device)
            }
        else:
            ce_tensor_input = torch.tensor(ce, device=self.gaussian_core.W_scale.device)
        # Compute loss and component metrics
        for p in self.gaussian_core.parameters():
            if p.grad is not None:
                p.grad.zero_()

        if self.reset_mma_after_reopen:
            print("[Reopen] Resetting MMA optimizer state after structural event.")
            self.last_reopen_phase = None
            self.reset_mma_after_reopen = False
        
        current_phase = self.get_reopen_phase()

        if current_phase != self.last_reopen_phase:
            print(f"[Reopen] MMA phase switch: {self.last_reopen_phase} -> {current_phase}")
            print(f"[Phase-Debug] iter={self.global_i} | current_phase={current_phase} | last_phase={self.last_reopen_phase} | reset_flag={self.reset_mma_after_reopen}")
            if current_phase in ["explore", "transition"]:
                self.rebuild_mma_with_old_scale_frozen()
            else:
                lb_list, ub_list = self.gaussian_core.export_bounds()
                self.rebuild_mma_with_custom_bounds(lb_list, ub_list)

            self.last_reopen_phase = current_phase
        # 关键修改：先用当前参数重算一次 H
        _ = self.gaussian_core.get_x(self.global_i)
        _, splitted_loss = self.gaussian_core(ce_tensor_input, self.global_i)

        # =========================================================
        # callback 更新：
        # 如果当前有 pending event，就每轮更新一次；
        # 只有在窗口结束时才真正 accept / rollback
        # =========================================================
        callback_result = None
        if self.gaussian_core.pending_event is not None:
            callback_result = self.gaussian_core.update_callback(self.global_i, splitted_loss)

            if callback_result is not None and callback_result["action"] in ["accept", "rollback"]:
                print(f"[Callback] structural event finished with action={callback_result['action']}")

                # 结构状态变了，下一轮重置 MMA 后再继续
                self.reset_mma_after_reopen = True

                # 如果 callback reject / rollback，则彻底关闭后续 reopen
                if callback_result["action"] == "rollback":
                    self.gaussian_core.reopen_enable = False
                    self.gaussian_core.reopen_disabled_by_callback = True
                    print("[Callback] reopen has been permanently disabled for the rest of this run.")

                combined_metrics = {**splitted_loss}

                action_code = 1.0 if callback_result["action"] == "accept" else -1.0
                combined_metrics["callback_action_code"] = torch.tensor(
                    action_code,
                    device=splitted_loss["compliance"].device
                )
                combined_metrics["callback_event_id"] = torch.tensor(
                    float(callback_result["event_id"]),
                    device=splitted_loss["compliance"].device
                )

                if "rel_improve" in callback_result:
                    combined_metrics["callback_rel_improve"] = torch.tensor(
                        float(callback_result["rel_improve"]),
                        device=splitted_loss["compliance"].device
                    )

                if "collapse_ratio" in callback_result and callback_result["collapse_ratio"] is not None:
                    combined_metrics["callback_collapse_ratio"] = torch.tensor(
                        float(callback_result["collapse_ratio"]),
                        device=splitted_loss["compliance"].device
                    )

                if "closer_ratio" in callback_result and callback_result["closer_ratio"] is not None:
                    combined_metrics["callback_closer_ratio"] = torch.tensor(
                        float(callback_result["closer_ratio"]),
                        device=splitted_loss["compliance"].device
                    )
                self.gaussian_core.reopen_post_vol_constraint_baseline = None
                # 本轮不再继续 step，下一轮用 accept/rollback 后的新结构继续
                return splitted_loss["compliance"] * (-1), combined_metrics

        base_compliance = splitted_loss["compliance"] * (-1)
        base_cons = splitted_loss["volfrac_constraint"]

        # 默认：正常阶段
        loss = base_compliance
        cons = base_cons

        # ===== reopen-aware objective schedule: exploration -> transition -> normal =====
        if self.gaussian_core.if_reopen_exploration(self.global_i):
            print("[Reopen] exploration phase")

            comp_w = self.gaussian_core.reopen_explore_compliance_weight
            vol_w = self.gaussian_core.reopen_explore_volume_weight

            loss = comp_w * base_compliance

            if self.gaussian_core.reopen_post_vol_constraint_baseline is not None:
                baseline_vol = float(self.gaussian_core.reopen_post_vol_constraint_baseline)
                if (base_cons - baseline_vol)>0:cons = base_cons
                else: cons = base_cons
            else:
                cons = base_cons

        elif self.gaussian_core.if_reopen_transition(self.global_i):
            print("[Reopen] transition phase")

            t_num = self.global_i - (
                self.gaussian_core.last_reopen_iter + self.gaussian_core.reopen_exploration_steps + 1
            )
            t_den = max(1, self.gaussian_core.reopen_transition_steps - 1)
            t = t_num / t_den

            comp_w = self.gaussian_core.reopen_explore_compliance_weight

            # transition 里逐渐把体积项加回 objective
            vol_w = 0.0 * (1 - t) + 1.0 * t

            loss = comp_w * base_compliance

            if self.gaussian_core.pending_event is not None:
                cons = base_cons
            else:
                cons = base_cons

        else:
           cons = base_cons
        
                # ===== debug: what exactly is sent to MMA =====
        current_volfrac = (self.gaussian_core.H.T @ self.gaussian_core.volumes / self.gaussian_core.volumes.sum()).item()
        current_vol_constraint = splitted_loss["volfrac_constraint"].item()
        current_compliance = splitted_loss["compliance"].item()

        if self.gaussian_core.pending_event is not None:
            baseline_vol = self.gaussian_core.pending_event["baseline_vol_constraint"]
        else:
            baseline_vol = None

        print(
            f"[MMA-Debug] iter={self.global_i} | "
            f"phase={'explore' if self.gaussian_core.if_reopen_exploration(self.global_i) else 'transition' if self.gaussian_core.if_reopen_transition(self.global_i) else 'normal'} | "
            f"compliance={current_compliance:.6f} | "
            f"volfrac={current_volfrac:.6f} | "
            f"vol_constraint={current_vol_constraint:.6f} | "
            f"baseline_vol_constraint={baseline_vol if baseline_vol is not None else 'None'} | "
            f"loss={loss.item():.6f} | cons={cons.item():.6f}"
        )
        if torch.isnan(loss):
            print("Warning: NaN detected in loss")

        # Backward pass
        loss.backward(retain_graph=True)

        # Clean NaN gradients
        for name, param in self.gaussian_core.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN gradient detected in {name}")
                param.grad = torch.where(
                    torch.isnan(param.grad),
                    torch.zeros_like(param.grad),
                    param.grad
                )

        # Prepare gradients
        self.gaussian_core.prepare_grads(self.global_i)
        self.gaussian_core.suppress_old_features_after_reopen(self.global_i)
        self.gaussian_core.update_split_prior_memory()

        # ===== update reopen history =====
        print("[DEBUG] about to check reopening")
        self.gaussian_core.update_reopen_history(float(splitted_loss["compliance"].item()))
        print("[DEBUG] reopen history updated")
        print("[DEBUG] calling if_reopening now")
        print("free slots =", int((~self.gaussian_core.persistent_mask).sum().item()))
        print("history len =", len(self.gaussian_core.loss_history))

        # ===== event-triggered reopening =====
        # 只有当前没有 pending callback event，才允许触发新的 reopen
        if self.gaussian_core.pending_event is None:
            if self.gaussian_core.if_reopening(self.global_i):
                # reopen 前保存事件前快照
                snapshot_before = self.gaussian_core.snapshot_state()

                did_reopen = self.gaussian_core.reopening(self.global_i)
                if did_reopen:
                    self.gaussian_core.last_reopen_iter = self.global_i
                    self.reset_mma_after_reopen = True

                    # ===== 单独记录 reopen 后的体积 baseline（不给 callback 用）=====
                    _ = self.gaussian_core.get_x(self.global_i)
                    _, splitted_loss_after_reopen = self.gaussian_core(ce_tensor_input, self.global_i)

                    self.gaussian_core.reopen_post_vol_constraint_baseline = float(
                        splitted_loss_after_reopen["volfrac_constraint"].item()
                    )
                    self.gaussian_core.reopen_post_volfrac_baseline = float(
                        (self.gaussian_core.H.T @ self.gaussian_core.volumes / self.gaussian_core.volumes.sum()).item()
                    )

                    print(
                        f"[Reopen-PostVolBaseline] iter={self.global_i} | "
                        f"volfrac={self.gaussian_core.reopen_post_volfrac_baseline:.6f} | "
                        f"vol_constraint={self.gaussian_core.reopen_post_vol_constraint_baseline:.6f}"
                    )
                    # 启动 callback 观察窗口
                    self.gaussian_core.start_reopen_callback(self.global_i, splitted_loss)

                    # 用真正的 split 前快照覆盖 callback 默认快照
                    if self.gaussian_core.pending_event is not None:
                        self.gaussian_core.pending_event["snapshot_before"] = snapshot_before
                        self.gaussian_core.pending_event["best_state"] = self.gaussian_core.snapshot_state()

                    # reopen 后本轮先不 step，下一轮用新 active set 重新计算
                    combined_metrics = {**splitted_loss}
                    combined_metrics["reopened"] = torch.tensor(1.0, device=loss.device)
                    return loss, combined_metrics

        # Collect gradient norms
        grad_norms = {
            name: param.grad.norm()
            for name, param in self.gaussian_core.named_parameters()
            if param.grad is not None
        }

        combined_metrics = {**splitted_loss, **grad_norms}

        # Run MMA step
        scale_before = self.gaussian_core.W_scale.data.detach().clone()
        off_before = self.gaussian_core.W_offsets.data.detach().clone()
        rot_before = self.gaussian_core.W_rotation.data.detach().clone()
        shape_before = self.gaussian_core.W_shape_var.data.detach().clone()

        self.optim.step(loss, cons)

        print("[Param-Debug] max |Δscale| =", (self.gaussian_core.W_scale.data - scale_before).abs().max().item())
        print("[Param-Debug] max |Δoffset| =", (self.gaussian_core.W_offsets.data - off_before).abs().max().item())
        print("[Param-Debug] max |Δrot| =", (self.gaussian_core.W_rotation.data - rot_before).abs().max().item())
        print("[Param-Debug] max |Δshape| =", (self.gaussian_core.W_shape_var.data - shape_before).abs().max().item())

        # Shape combination if needed
        if self.shape_combination:
            self.gaussian_core.update_shape_combination(self.global_i)

        return loss, combined_metrics

    # Disable unused methods
    def plot_H_gradients(self, *args, **kwargs):
        raise NotImplementedError("plot_H_gradients is not supported in MMA version")

    def update_problem_config(self, *args, **kwargs):
        raise NotImplementedError("update_problem_config is not supported in MMA version")
