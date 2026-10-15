import torch
import numpy as np


class ShapeBridge:
    """
    用于在 latent shape code (W_shape_var) 和显式几何参数 chi 之间来回转换，
    并支持在几何层做编辑。
    """

    def __init__(self, vae_model, device=None):
        self.vae_model = vae_model
        if device is None:
            device = next(vae_model.parameters()).device
        self.device = device
        self.canonical_chi_quadrangle = torch.tensor(
            [0.5, 0.5, -0.5, 0.5, 0.15, 0.15, 0.15, 0.15],
            dtype=torch.float32,
            device=self.device,
        )
    
    @torch.no_grad()
    def get_canonical_latent(self, use_mu: bool = True):
        """
        返回单个 canonical quadrangle 的 latent。
        """
        chi = self.canonical_chi_quadrangle.clone()
        z = self.chi_to_latent(chi, use_mu=use_mu)
        return z.clone(), chi.clone()
    
    @torch.no_grad()
    def latent_to_chi(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [latent_dim] or [B, latent_dim]
        return:
            chi: [8] or [B, 8]   (quadrangle case)
        """
        if z.ndim == 1:
            z = z.unsqueeze(0)
        z = z.to(self.device).float()

        chi = self.vae_model.decoder_input(z)
        return chi.squeeze(0)

    @torch.no_grad()
    def chi_to_latent(self, chi: torch.Tensor, use_mu: bool = True) -> torch.Tensor:
        """
        chi: [8] or [B, 8]
        encoder 在 forward 里是 encode(x[:, 2:])，所以这里直接喂几何参数即可。
        return:
            z: [latent_dim] or [B, latent_dim]
        """
        if chi.ndim == 1:
            chi = chi.unsqueeze(0)
        chi = chi.to(self.device).float()

        mu, log_var = self.vae_model.encode(chi)
        z = mu if use_mu else self.vae_model.reparameterize(mu, log_var)
        return z.squeeze(0)

    @torch.no_grad()
    def latent_roundtrip(self, z: torch.Tensor) -> torch.Tensor:
        """
        z -> chi -> z_recon
        用于检查桥接误差。
        """
        chi = self.latent_to_chi(z)
        z_recon = self.chi_to_latent(chi, use_mu=True)
        return z_recon
    
    @torch.no_grad()
    def split_latent_canonical(self, use_mu: bool = True):
        """
        直接返回 canonical child latent。
        适合无先验 reopen，用中性形状初始化两个 child。
        """
        chi = self.canonical_chi_quadrangle.clone()
        z = self.chi_to_latent(chi, use_mu=use_mu)
        return z.clone(), z.clone(), chi.clone(), chi.clone(), chi.clone()
    
    @torch.no_grad()
    def split_chi_along_axis(
        self,
        chi: torch.Tensor,
        
        delta: float = 0.08,
        scale_shrink: float = 0.9,
    ):
        """
        最小可运行的几何层 split：
        - 先不改 chi 的内部形状语义太多
        - 主要通过平移 + 轻微缩放创造两个 child

        注意：
        这里 chi 是 shape 参数，不含 offset/scale/rotation。
        所以第一版 split_chi 只做很小扰动，真正的大位移还是放在外部 offset 上做。
        """
        if chi.ndim != 1:
            raise ValueError(f"chi should be 1D, got shape={chi.shape}")
        chi0 = chi.clone()
        chi1 = chi.clone()
        chi2 = chi.clone()

        # quadrangle 的 8 维参数目前语义近似视为 4 个坐标自由度 + 4 个圆角
        # 第一版只对子形状做保守扰动，避免 decoder/encoder 回环不稳定
        
        chi1[0] = chi0[0]+chi0[2]+0.5
        chi1[1] = chi0[1]+chi0[3]+0.5
        chi1[2] = 2*chi0[2]+0.5
        chi1[3] = 2*chi0[3]+0.5

        chi2[0] = 2*chi0[0]-0.5
        
        chi2[1] = 2*chi0[1]+0.5
        chi2[2] = chi0[0]+chi0[2]-0.5
        
        chi2[3] = chi0[1]+chi0[3]+0.5
        

        return chi1, chi2
    
    @torch.no_grad()
    def split_chi_ver_axis(
        self,
        chi: torch.Tensor,
        
        delta: float = 0.08,
        scale_shrink: float = 0.9,
    ):
        """
        最小可运行的几何层 split：
        - 先不改 chi 的内部形状语义太多
        - 主要通过平移 + 轻微缩放创造两个 child

        注意：
        这里 chi 是 shape 参数，不含 offset/scale/rotation。
        所以第一版 split_chi 只做很小扰动，真正的大位移还是放在外部 offset 上做。
        """
        if chi.ndim != 1:
            raise ValueError(f"chi should be 1D, got shape={chi.shape}")
        chi0 = chi.clone()
        chi1 = chi.clone()
        chi2 = chi.clone()

        # quadrangle 的 8 维参数目前语义近似视为 4 个坐标自由度 + 4 个圆角
        # 第一版只对子形状做保守扰动，避免 decoder/encoder 回环不稳定
        
        
        
        chi2[0] = 0.5
        chi2[1] = 0
        chi2[2] = -0.5
        chi2[3] = 0
        

        return chi1, chi2
    
    @torch.no_grad()
    def split_latent(
        self,
        z: torch.Tensor,
        delta: float = 0.08,
        scale_shrink: float = 0.9,
        use_mu: bool = True,
    ):
        """
        z -> chi -> split chi -> z1, z2
        """
        chi = self.latent_to_chi(z)
        chi1, chi2 = self.split_chi_along_axis(
            chi,
            delta=delta,
            scale_shrink=scale_shrink,
        )
        z1 = self.chi_to_latent(chi1, use_mu=use_mu)
        z2 = self.chi_to_latent(chi2, use_mu=use_mu)
        return z1, z2, chi, chi1, chi2
    
    @torch.no_grad()
    def split_latent_ver(
        self,
        z: torch.Tensor,
        delta: float = 0.08,
        scale_shrink: float = 0.9,
        use_mu: bool = True,
    ):
        """
        z -> chi -> split chi -> z1, z2
        """
        chi = self.latent_to_chi(z)
        chi1, chi2 = self.split_chi_ver_axis(
            chi,
            delta=delta,
            scale_shrink=scale_shrink,
        )
        z1 = self.chi_to_latent(chi1, use_mu=use_mu)
        z2 = self.chi_to_latent(chi2, use_mu=use_mu)
        return z1, z2, chi, chi1, chi2