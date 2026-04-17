"""
Verified PDE Loss Functions for Training

已验证通过的数据集 (MSE < 1e-5):
    1. Active Matter - 不可压缩 + 对流
    2. Gray-Scott - 反应扩散系统
    3. Viscoelastic - 仅连续性方程

所有方法:
    - 纯数值计算 (FFT/有限差分)
    - 支持反向传播
    - 不使用 torch.autograd.grad()

Author: Claude
Date: February 2025
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional


# =============================================================================
# 1. Active Matter PDE Loss (2D 不可压缩 + 对流)
# =============================================================================

class ActiveMatterPDELoss(nn.Module):
    """
    Active Matter PDE Loss - 双周期边界

    方程:
        div(u) = 0                (不可压缩, MSE ~ 1e-13)
        dc/dt + u·∇c = 0          (对流方程, MSE ~ 8e-7)

    数值方法:
        - 空间: FFT谱导数
        - 时间: 一阶前向差分
    """

    def __init__(
        self,
        nx: int = 256,
        ny: int = 256,
        dx: float = 10.0 / 256,
        dy: float = 10.0 / 256,
        dt: float = 1.0,
    ):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dt = dt

        # 预计算FFT波数
        kx = torch.fft.fftfreq(nx, d=dx) * 2 * np.pi
        ky = torch.fft.fftfreq(ny, d=dy) * 2 * np.pi
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')

        self.register_buffer('KX', KX)
        self.register_buffer('KY', KY)

    def fft_dx(self, f: torch.Tensor) -> torch.Tensor:
        """FFT x方向导数"""
        KX = self.KX.to(f.device)
        while KX.dim() < f.dim():
            KX = KX.unsqueeze(0)
        f_hat = torch.fft.fft2(f)
        return torch.fft.ifft2(1j * KX * f_hat).real

    def fft_dy(self, f: torch.Tensor) -> torch.Tensor:
        """FFT y方向导数"""
        KY = self.KY.to(f.device)
        while KY.dim() < f.dim():
            KY = KY.unsqueeze(0)
        f_hat = torch.fft.fft2(f)
        return torch.fft.ifft2(1j * KY * f_hat).real

    def forward(
        self,
        c: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        reduction: str = 'mean',
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算PDE残差loss

        Args:
            c: concentration [B, T, X, Y] or [T, X, Y]
            u: x-velocity, same shape
            v: y-velocity, same shape
            reduction: 'mean', 'sum', or 'none'

        Returns:
            total_loss, losses_dict
        """
        # 确保batch维度
        had_batch = c.dim() == 4
        if not had_batch:
            c = c.unsqueeze(0)
            u = u.unsqueeze(0)
            v = v.unsqueeze(0)

        # 时间导数
        dc_dt = (c[:, 1:] - c[:, :-1]) / self.dt

        # 空间导数 (在n时刻)
        c_n = c[:, :-1]
        u_n = u[:, :-1]
        v_n = v[:, :-1]

        # div(u) = 0
        du_dx = self.fft_dx(u_n)
        dv_dy = self.fft_dy(v_n)
        R_cont = du_dx + dv_dy

        # dc/dt + u·∇c = 0
        dc_dx = self.fft_dx(c_n)
        dc_dy = self.fft_dy(c_n)
        R_adv = dc_dt + u_n * dc_dx + v_n * dc_dy

        # Loss计算
        if reduction == 'mean':
            loss_cont = torch.mean(R_cont ** 2)
            loss_adv = torch.mean(R_adv ** 2)
        elif reduction == 'sum':
            loss_cont = torch.sum(R_cont ** 2)
            loss_adv = torch.sum(R_adv ** 2)
        else:
            loss_cont = R_cont ** 2
            loss_adv = R_adv ** 2

        total_loss = loss_cont + loss_adv

        return total_loss, {
            'continuity': loss_cont,
            'advection': loss_adv,
            'total': total_loss,
        }


# =============================================================================
# 2. Gray-Scott PDE Loss (2D 反应扩散)
# =============================================================================

class GrayScottPDELoss(nn.Module):
    """
    Gray-Scott Reaction-Diffusion PDE Loss - 双周期边界

    方程:
        dA/dt = D_A*∇²A - A*B² + F*(1-A)    (MSE ~ 1.4e-6)
        dB/dt = D_B*∇²B + A*B² - (F+k)*B    (MSE ~ 1.5e-6)

    数值方法:
        - 空间: FFT Laplacian
        - 时间: 二阶中心差分 (关键!)
    """

    DEFAULT_EQ_SCALES = {
        'A': 1.0,
        'B': 1.0,
    }

    def __init__(
        self,
        nx: int = 128,
        ny: int = 128,
        dx: float = 2.0 / 128,
        dy: float = 2.0 / 128,
        dt: float = 10.0,
        F: float = 0.098,
        k: float = 0.057,
        D_A: float = 1.81e-5,
        D_B: float = 1.39e-5,
        eq_scales: Optional[Dict[str, float]] = None,
        eq_weights: Optional[Dict[str, float]] = None,
        eq_scales_per_t: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.F = F
        self.k = k
        self.D_A = D_A
        self.D_B = D_B
        self.eq_scales = {**self.DEFAULT_EQ_SCALES, **(eq_scales or {})}
        self.eq_weights = {'A': 1.0, 'B': 1.0, **(eq_weights or {})}

        # Per-timestep scales (time-aware normalization)
        self.use_per_t_scales = eq_scales_per_t is not None
        if eq_scales_per_t is not None:
            for name, scales in eq_scales_per_t.items():
                self.register_buffer(f'scales_t_{name}', scales.float())

        # 预计算FFT波数
        kx = torch.fft.fftfreq(nx, d=dx) * 2 * np.pi
        ky = torch.fft.fftfreq(ny, d=dy) * 2 * np.pi
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')
        K2 = KX**2 + KY**2

        self.register_buffer('K2', K2)

    def fft_laplacian(self, f: torch.Tensor) -> torch.Tensor:
        """FFT Laplacian"""
        K2 = self.K2.to(f.device)
        while K2.dim() < f.dim():
            K2 = K2.unsqueeze(0)
        f_hat = torch.fft.fft2(f)
        return torch.fft.ifft2(-K2 * f_hat).real

    def forward(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        reduction: str = 'mean',
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算PDE残差loss

        Args:
            A: concentration A [B, T, X, Y] or [T, X, Y]
            B: concentration B, same shape
            reduction: 'mean', 'sum', or 'none'

        Returns:
            total_loss, losses_dict
        """
        had_batch = A.dim() == 4
        if not had_batch:
            A = A.unsqueeze(0)
            B = B.unsqueeze(0)

        # 二阶中心时间差分 (关键改进!)
        # (f[n+1] - f[n-1]) / (2*dt)
        dA_dt = (A[:, 2:] - A[:, :-2]) / (2 * self.dt)
        dB_dt = (B[:, 2:] - B[:, :-2]) / (2 * self.dt)

        # 空间项在中间时刻 n
        A_n = A[:, 1:-1]
        B_n = B[:, 1:-1]

        # Laplacian
        lap_A = self.fft_laplacian(A_n)
        lap_B = self.fft_laplacian(B_n)

        # 反应项
        AB2 = A_n * B_n**2

        # 残差
        R_A = dA_dt - self.D_A * lap_A + AB2 - self.F * (1 - A_n)
        R_B = dB_dt - self.D_B * lap_B - AB2 + (self.F + self.k) * B_n

        # Loss计算 (per-equation normalized by eq_scales or per-timestep scales)
        w_A = self.eq_weights['A']
        w_B = self.eq_weights['B']

        if self.use_per_t_scales and reduction == 'mean':
            # Per-timestep normalization: R / s_t -> mean(normalized_R^2)
            T_res = R_A.shape[1]

            if hasattr(self, 'scales_t_A_equation'):
                s_t_A = self.scales_t_A_equation[:T_res].clamp(min=1e-8)
                R_A_norm = R_A / s_t_A[None, :, None, None]
                loss_A = w_A * torch.mean(R_A_norm ** 2)
            else:
                s_A = self.eq_scales['A']
                loss_A = w_A * torch.mean(R_A ** 2) / (s_A ** 2)

            if hasattr(self, 'scales_t_B_equation'):
                s_t_B = self.scales_t_B_equation[:T_res].clamp(min=1e-8)
                R_B_norm = R_B / s_t_B[None, :, None, None]
                loss_B = w_B * torch.mean(R_B_norm ** 2)
            else:
                s_B = self.eq_scales['B']
                loss_B = w_B * torch.mean(R_B ** 2) / (s_B ** 2)
        else:
            # Fallback: fixed per-equation scales
            s_A = self.eq_scales['A']
            s_B = self.eq_scales['B']

            if reduction == 'mean':
                loss_A = w_A * torch.mean(R_A ** 2) / (s_A ** 2)
                loss_B = w_B * torch.mean(R_B ** 2) / (s_B ** 2)
            elif reduction == 'sum':
                loss_A = w_A * torch.sum(R_A ** 2) / (s_A ** 2)
                loss_B = w_B * torch.sum(R_B ** 2) / (s_B ** 2)
            else:
                loss_A = w_A * R_A ** 2 / (s_A ** 2)
                loss_B = w_B * R_B ** 2 / (s_B ** 2)

        total_loss = loss_A + loss_B

        return total_loss, {
            'A_equation': loss_A,
            'B_equation': loss_B,
            'total': total_loss,
        }


# =============================================================================
# 3. Viscoelastic Continuity Loss (2D 仅不可压缩)
# =============================================================================

class ViscoelasticContinuityLoss(nn.Module):
    """
    Viscoelastic Instability - 仅连续性方程

    方程:
        div(u) = ∂u/∂x + ∂v/∂y = 0    (MSE ~ 3e-7)

    边界:
        - x方向: 周期 (FFT)
        - y方向: 墙面 (中心差分)

    注意: 动量方程和本构方程未验证通过
    """

    def __init__(
        self,
        nx: int = 512,
        ny: int = 512,
        dx: float = 2 * np.pi / 512,
        dy: float = 2.0 / 512,
    ):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy

        # x方向FFT波数
        kx = torch.fft.fftfreq(nx, d=dx) * 2 * np.pi
        self.register_buffer('kx', kx.view(nx, 1))

    def fft_dx(self, f: torch.Tensor) -> torch.Tensor:
        """FFT x方向导数 (周期)"""
        # f: [..., X, Y]
        kx = self.kx.to(f.device)
        f_hat = torch.fft.fft(f, dim=-2)
        return torch.fft.ifft(1j * kx * f_hat, dim=-2).real

    def fd_dy(self, f: torch.Tensor) -> torch.Tensor:
        """中心差分 y方向导数 (墙面)"""
        # f: [..., X, Y]
        return (f[..., 2:] - f[..., :-2]) / (2 * self.dy)

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        reduction: str = 'mean',
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算连续性方程残差

        Args:
            u: x-velocity [B, T, X, Y] or [T, X, Y]
            v: y-velocity, same shape
            reduction: 'mean', 'sum', or 'none'

        Returns:
            total_loss, losses_dict
        """
        # du/dx (FFT, 周期)
        du_dx = self.fft_dx(u)

        # dv/dy (中心差分, 墙面)
        dv_dy = self.fd_dy(v)

        # 匹配形状: 去掉y边界点
        R_cont = du_dx[..., 1:-1] + dv_dy

        # Loss计算
        if reduction == 'mean':
            loss_cont = torch.mean(R_cont ** 2)
        elif reduction == 'sum':
            loss_cont = torch.sum(R_cont ** 2)
        else:
            loss_cont = R_cont ** 2

        return loss_cont, {
            'continuity': loss_cont,
            'total': loss_cont,
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_active_matter_loss(data_path: Optional[str] = None) -> ActiveMatterPDELoss:
    """创建Active Matter PDE loss，可从数据文件读取参数"""
    if data_path is not None:
        import h5py
        with h5py.File(data_path, 'r') as f:
            L = float(f.attrs['L'])
            x = f['dimensions/x'][:]
            y = f['dimensions/y'][:]
            t = f['dimensions/time'][:]
            nx, ny = len(x), len(y)
            dx = L / nx
            dy = L / ny
            dt = float(t[1] - t[0])
        return ActiveMatterPDELoss(nx=nx, ny=ny, dx=dx, dy=dy, dt=dt)
    return ActiveMatterPDELoss()


def create_gray_scott_loss(data_path: Optional[str] = None) -> GrayScottPDELoss:
    """创建Gray-Scott PDE loss，可从数据文件读取参数"""
    if data_path is not None:
        import h5py
        with h5py.File(data_path, 'r') as f:
            F = float(f.attrs['F'])
            k = float(f.attrs['k'])
            x = f['dimensions/x'][:]
            y = f['dimensions/y'][:]
            t = f['dimensions/time'][:]
            nx, ny = len(x), len(y)
            Lx = x[-1] - x[0] + (x[1] - x[0])
            Ly = y[-1] - y[0] + (y[1] - y[0])
            dx = Lx / nx
            dy = Ly / ny
            dt = float(t[1] - t[0])
        return GrayScottPDELoss(
            nx=nx, ny=ny, dx=dx, dy=dy, dt=dt,
            F=F, k=k, D_A=1.81e-5, D_B=1.39e-5
        )
    return GrayScottPDELoss()


def create_viscoelastic_loss(data_path: Optional[str] = None) -> ViscoelasticContinuityLoss:
    """创建Viscoelastic continuity loss，可从数据文件读取参数"""
    if data_path is not None:
        import h5py
        with h5py.File(data_path, 'r') as f:
            x = f['dimensions/x'][:]
            y = f['dimensions/y'][:]
            nx, ny = len(x), len(y)
            Lx = 2 * np.pi  # x in [0, 2π]
            Ly = 2.0        # y in [-1, 1]
            dx = Lx / nx
            dy = Ly / ny
        return ViscoelasticContinuityLoss(nx=nx, ny=ny, dx=dx, dy=dy)
    return ViscoelasticContinuityLoss()


# =============================================================================
# Testing
# =============================================================================

def test_all_losses():
    """测试所有PDE loss函数"""
    import h5py

    print("=" * 70)
    print("Testing All Verified PDE Losses")
    print("=" * 70)

    # Test Active Matter
    print("\n--- Active Matter ---")
    am_path = "data/test/active_matter_L_10.0_zeta_1.0_alpha_-1.0.hdf5"
    try:
        with h5py.File(am_path, 'r') as f:
            c = torch.tensor(f['t0_fields/concentration'][0, :20], dtype=torch.float64)
            vel = torch.tensor(f['t1_fields/velocity'][0, :20], dtype=torch.float64)
            u, v = vel[..., 0], vel[..., 1]

        loss_fn = create_active_matter_loss(am_path)
        with torch.no_grad():
            total, losses = loss_fn(c, u, v)
        for name, val in losses.items():
            print(f"  {name}: {val.item():.6e}")
    except FileNotFoundError:
        print("  [SKIP] Data file not found")

    # Test Gray-Scott
    print("\n--- Gray-Scott ---")
    gs_path = "data/test/gray_scott_reaction_diffusion_bubbles_F_0.098_k_0.057.hdf5"
    try:
        with h5py.File(gs_path, 'r') as f:
            A = torch.tensor(f['t0_fields/A'][0, :50], dtype=torch.float64)
            B = torch.tensor(f['t0_fields/B'][0, :50], dtype=torch.float64)

        loss_fn = create_gray_scott_loss(gs_path)
        with torch.no_grad():
            total, losses = loss_fn(A, B)
        for name, val in losses.items():
            print(f"  {name}: {val.item():.6e}")
    except FileNotFoundError:
        print("  [SKIP] Data file not found")

    # Test Viscoelastic
    print("\n--- Viscoelastic (Continuity only) ---")
    ve_path = "data/test/viscoelastic_instability_AH.hdf5"
    try:
        with h5py.File(ve_path, 'r') as f:
            vel = torch.tensor(f['t1_fields/velocity'][0, :20], dtype=torch.float64)
            u, v = vel[..., 0], vel[..., 1]

        loss_fn = create_viscoelastic_loss(ve_path)
        with torch.no_grad():
            total, losses = loss_fn(u, v)
        for name, val in losses.items():
            print(f"  {name}: {val.item():.6e}")
    except FileNotFoundError:
        print("  [SKIP] Data file not found")

    print("\n" + "=" * 70)
    print("Testing complete!")


if __name__ == "__main__":
    test_all_losses()


# ============================================================
# Active Matter n-PINN PDE Loss (conservative upwind)
# ============================================================

class ActiveMatterNPINNPDELoss(nn.Module):
    """
    Active Matter PDE Loss (n-PINN conservative upwind).

    Equations:
        1. Continuity: du/dx + dv/dy = 0
        2. Concentration: dc/dt + div(c*u) = d_T * Laplacian(c)   (+ div correction)
        3. D_xx advection: dD_xx/dt + div(D_xx*u) = d_T * Laplacian(D_xx)  (simplified, + div correction)

    All periodic BC, n-PINN conservative upwind scheme with 2nd-order upwind
    face-value interpolation.

    Args:
        nx, ny: grid points (256, 256)
        Lx, Ly: domain size (10.0, 10.0)
        dt: time step (0.25 for data output interval)
        d_T: translational diffusion coefficient (0.05)
        use_div_correction: whether to apply -f*div correction (default True)
        eq_scales: per-equation normalization dict
        eq_weights: per-equation loss weights dict
    """

    DEFAULT_EQ_SCALES = {
        'continuity': 1.0,
        'concentration': 1.0,
        'Dxx': 1.0,
    }

    def __init__(
        self,
        nx: int = 256,
        ny: int = 256,
        Lx: float = 10.0,
        Ly: float = 10.0,
        dt: float = 0.25,
        d_T: float = 0.05,
        use_div_correction: bool = True,
        eq_scales: Optional[Dict[str, float]] = None,
        eq_weights: Optional[Dict[str, float]] = None,
        eq_scales_per_t: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.dt = dt
        self.d_T = d_T
        self.use_div_correction = use_div_correction

        # Per-equation normalization scales
        self.eq_scales = dict(self.DEFAULT_EQ_SCALES)
        if eq_scales is not None:
            self.eq_scales.update(eq_scales)

        # Per-equation weights
        self.eq_weights = {'continuity': 1.0, 'concentration': 1.0, 'Dxx': 1.0}
        if eq_weights is not None:
            self.eq_weights.update(eq_weights)

        # Per-timestep scales (time-aware normalization)
        self.use_per_t_scales = eq_scales_per_t is not None
        if eq_scales_per_t is not None:
            for name, scales in eq_scales_per_t.items():
                self.register_buffer(f'scales_t_{name}', scales.float())

    def _face_velocities(
        self, u: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Face-averaged velocities for conservative fluxes (periodic BC via roll)."""
        # East/West faces (x-direction)
        uc_e = 0.5 * (u + torch.roll(u, -1, dims=-1))  # east face
        uc_w = 0.5 * (u + torch.roll(u, 1, dims=-1))   # west face
        # North/South faces (y-direction)
        vc_n = 0.5 * (v + torch.roll(v, -1, dims=-2))   # north face
        vc_s = 0.5 * (v + torch.roll(v, 1, dims=-2))    # south face
        return uc_e, uc_w, vc_n, vc_s

    def _upwind_convection(
        self,
        f: torch.Tensor,
        uc_e: torch.Tensor, uc_w: torch.Tensor,
        vc_n: torch.Tensor, vc_s: torch.Tensor,
    ) -> torch.Tensor:
        """Conservative upwind convection: div(f*u) with 2nd-order upwind face values.

        Periodic BC via torch.roll.
        """
        # Neighbors for 2nd-order upwind stencil
        f_ip1 = torch.roll(f, -1, dims=-1)  # f[i+1]
        f_im1 = torch.roll(f, 1, dims=-1)   # f[i-1]
        f_ip2 = torch.roll(f, -2, dims=-1)  # f[i+2]
        f_im2 = torch.roll(f, 2, dims=-1)   # f[i-2]

        f_jp1 = torch.roll(f, -1, dims=-2)  # f[j+1]
        f_jm1 = torch.roll(f, 1, dims=-2)   # f[j-1]
        f_jp2 = torch.roll(f, -2, dims=-2)  # f[j+2]
        f_jm2 = torch.roll(f, 2, dims=-2)   # f[j-2]

        # East face: 2nd-order upwind
        Fe_pos = 1.5 * f - 0.5 * f_im1       # uc_e >= 0: upstream = i, i-1
        Fe_neg = 1.5 * f_ip1 - 0.5 * f_ip2   # uc_e < 0: upstream = i+1, i+2
        Fe = torch.where(uc_e >= 0, Fe_pos, Fe_neg)

        # West face
        Fw_pos = 1.5 * f_im1 - 0.5 * f_im2   # uc_w >= 0
        Fw_neg = 1.5 * f - 0.5 * f_ip1         # uc_w < 0
        Fw = torch.where(uc_w >= 0, Fw_pos, Fw_neg)

        # North face
        Fn_pos = 1.5 * f - 0.5 * f_jm1
        Fn_neg = 1.5 * f_jp1 - 0.5 * f_jp2
        Fn = torch.where(vc_n >= 0, Fn_pos, Fn_neg)

        # South face
        Fs_pos = 1.5 * f_jm1 - 0.5 * f_jm2
        Fs_neg = 1.5 * f - 0.5 * f_jp1
        Fs = torch.where(vc_s >= 0, Fs_pos, Fs_neg)

        # Conservative flux divergence
        conv = (uc_e * Fe - uc_w * Fw) / self.dx + (vc_n * Fn - vc_s * Fs) / self.dy
        return conv

    def _laplacian_2d(self, f: torch.Tensor) -> torch.Tensor:
        """2nd-order central Laplacian with periodic BC."""
        lap_x = (torch.roll(f, -1, dims=-1) - 2 * f + torch.roll(f, 1, dims=-1)) / (self.dx ** 2)
        lap_y = (torch.roll(f, -1, dims=-2) - 2 * f + torch.roll(f, 1, dims=-2)) / (self.dy ** 2)
        return lap_x + lap_y

    def _divergence(
        self,
        uc_e: torch.Tensor, uc_w: torch.Tensor,
        vc_n: torch.Tensor, vc_s: torch.Tensor,
    ) -> torch.Tensor:
        """Face-velocity divergence."""
        return (uc_e - uc_w) / self.dx + (vc_n - vc_s) / self.dy

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        c: torch.Tensor,
        Dxx: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute PDE residual losses.

        Args:
            u, v, c, Dxx: [B, T, H, W] where T includes prepended t0 frame.
                           Time derivative uses t=1..T-2 (2nd-order central).

        Returns:
            total_loss, {eq_name: loss_value}
        """
        losses = {}

        # ---- Eq 1: Continuity (all timesteps except boundaries) ----
        # Use interior timesteps t=1..T-2 for consistency with transport eqs
        u_int = u[:, 1:-1]
        v_int = v[:, 1:-1]

        uc_e, uc_w, vc_n, vc_s = self._face_velocities(u_int, v_int)
        div = self._divergence(uc_e, uc_w, vc_n, vc_s)

        w_cont = self.eq_weights['continuity']
        T_res = div.shape[1]

        if self.use_per_t_scales and hasattr(self, 'scales_t_continuity'):
            s_t_cont = self.scales_t_continuity[:T_res].clamp(min=1e-8)
            div_norm = div / s_t_cont[None, :, None, None]
            losses['continuity'] = w_cont * torch.mean(div_norm ** 2)
        else:
            scale_cont = self.eq_scales['continuity']
            losses['continuity'] = w_cont * torch.mean(div ** 2) / (scale_cont ** 2)

        # ---- Eq 2: Concentration advection-diffusion ----
        w_conc = self.eq_weights['concentration']
        if w_conc > 0:
            c_int = c[:, 1:-1]
            dc_dt = (c[:, 2:] - c[:, :-2]) / (2 * self.dt)

            conv_c = self._upwind_convection(c_int, uc_e, uc_w, vc_n, vc_s)
            lap_c = self._laplacian_2d(c_int)

            R_c = dc_dt + conv_c - self.d_T * lap_c
            if self.use_div_correction:
                R_c = R_c - c_int * div

            if self.use_per_t_scales and hasattr(self, 'scales_t_concentration'):
                s_t_conc = self.scales_t_concentration[:T_res].clamp(min=1e-8)
                R_c_norm = R_c / s_t_conc[None, :, None, None]
                losses['concentration'] = w_conc * torch.mean(R_c_norm ** 2)
            else:
                scale_conc = self.eq_scales['concentration']
                losses['concentration'] = w_conc * torch.mean(R_c ** 2) / (scale_conc ** 2)
        else:
            losses['concentration'] = torch.tensor(0.0, device=u.device)

        # ---- Eq 3: D_xx advection-diffusion (simplified) ----
        w_dxx = self.eq_weights['Dxx']
        if w_dxx > 0:
            Dxx_int = Dxx[:, 1:-1]
            dDxx_dt = (Dxx[:, 2:] - Dxx[:, :-2]) / (2 * self.dt)

            conv_Dxx = self._upwind_convection(Dxx_int, uc_e, uc_w, vc_n, vc_s)
            lap_Dxx = self._laplacian_2d(Dxx_int)

            R_Dxx = dDxx_dt + conv_Dxx - self.d_T * lap_Dxx
            if self.use_div_correction:
                R_Dxx = R_Dxx - Dxx_int * div

            scale_dxx = self.eq_scales['Dxx']
            losses['Dxx'] = w_dxx * torch.mean(R_Dxx ** 2) / (scale_dxx ** 2)
        else:
            losses['Dxx'] = torch.tensor(0.0, device=u.device)

        total_loss = sum(losses.values())
        return total_loss, losses


# =============================================================================
# 4. Rayleigh-Bénard Full PDE Loss (2D Incompressible Boussinesq)
# =============================================================================

class RayleighBenardFullPDELoss(nn.Module):
    """
    Rayleigh-Bénard PDE Loss — Incompressible Boussinesq (2D).

    Equations:
        1. Continuity: ∂u/∂x + ∂v/∂y = 0
        2. Buoyancy transport: ∂b/∂t + u·∂b/∂x + v·∂b/∂y = κΔb

    BCs:
        x: periodic (4th-order central with torch.roll)
        y: Dirichlet (2nd-order central interior + ghost cell extrapolation)

    Time: 2nd-order central (f[t+1] - f[t-1]) / (2dt)
    skip_bl: skip boundary layer rows from both y edges for loss computation.

    Data layout: [B, T, Nx, Ny] where Nx = periodic dim, Ny = Dirichlet dim.
    """

    DEFAULT_EQ_SCALES = {
        'continuity': 1.0,
        'buoyancy': 1.0,
    }

    def __init__(
        self,
        nx: int = 512,
        ny: int = 128,
        Lx: float = 4.0,
        Ly: float = 1.0,
        dt: float = 0.25,
        kappa: float = 1e-5,
        skip_bl: int = 15,
        eq_scales: Optional[Dict[str, float]] = None,
        eq_weights: Optional[Dict[str, float]] = None,
        eq_scales_per_t: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.dx = Lx / nx        # periodic: Lx / nx
        self.dy = Ly / (ny - 1)  # Dirichlet: Ly / (ny - 1)
        self.dt = dt
        self.kappa = kappa
        self.skip_bl = skip_bl

        self.eq_scales = dict(self.DEFAULT_EQ_SCALES)
        if eq_scales is not None:
            self.eq_scales.update(eq_scales)

        self.eq_weights = {'continuity': 1.0, 'buoyancy': 1.0}
        if eq_weights is not None:
            self.eq_weights.update(eq_weights)

        # Per-timestep normalization scales (optional)
        self.has_per_t_scales = eq_scales_per_t is not None
        if eq_scales_per_t is not None:
            for name, scales in eq_scales_per_t.items():
                self.register_buffer(f'scales_t_{name}', scales.float())

    # ---- x derivatives: 4th-order central, periodic via roll ----

    def _dx_4th(self, f: torch.Tensor) -> torch.Tensor:
        """4th-order central x-derivative (periodic). f: [..., Nx, Ny]"""
        return (-torch.roll(f, -2, dims=-2) + 8 * torch.roll(f, -1, dims=-2)
                - 8 * torch.roll(f, 1, dims=-2) + torch.roll(f, 2, dims=-2)) / (12 * self.dx)

    def _d2x_4th(self, f: torch.Tensor) -> torch.Tensor:
        """4th-order central 2nd x-derivative (periodic). f: [..., Nx, Ny]"""
        return (-torch.roll(f, -2, dims=-2) + 16 * torch.roll(f, -1, dims=-2)
                - 30 * f + 16 * torch.roll(f, 1, dims=-2)
                - torch.roll(f, 2, dims=-2)) / (12 * self.dx ** 2)

    # ---- y derivatives: 2nd-order central with ghost cell ----

    def _dy_ghost(self, f: torch.Tensor) -> torch.Tensor:
        """
        1st y-derivative with ghost cell extrapolation at boundaries.
        f: [..., Nx, Ny]. Returns [..., Nx, Ny] (same shape).
        Interior: (f[j+1] - f[j-1]) / (2*dy)
        j=0:  ghost f[-1] = 2*f[0] - f[1] → df/dy = (f[1] - f[0]) / dy
        j=Ny-1: ghost f[Ny] = 2*f[Ny-1] - f[Ny-2] → df/dy = (f[Ny-1] - f[Ny-2]) / dy
        """
        dy = self.dy
        # Interior: j=1..Ny-2
        interior = (f[..., 2:] - f[..., :-2]) / (2 * dy)
        # Boundary j=0
        left = (f[..., 1:2] - f[..., 0:1]) / dy
        # Boundary j=Ny-1
        right = (f[..., -1:] - f[..., -2:-1]) / dy
        return torch.cat([left, interior, right], dim=-1)

    def _d2y_ghost(self, f: torch.Tensor) -> torch.Tensor:
        """
        2nd y-derivative with ghost cell extrapolation at boundaries.
        f: [..., Nx, Ny]. Returns [..., Nx, Ny] (same shape).
        Interior: (f[j+1] - 2*f[j] + f[j-1]) / dy²
        j=0:  ghost → (f[1] - 2*f[0] + (2*f[0]-f[1])) / dy² = 0
        j=Ny-1: ghost → 0
        """
        dy2 = self.dy ** 2
        interior = (f[..., 2:] - 2 * f[..., 1:-1] + f[..., :-2]) / dy2
        # Boundaries: ghost cell gives 0
        zeros = torch.zeros_like(f[..., 0:1])
        return torch.cat([zeros, interior, zeros], dim=-1)

    def _get_per_t_scale(
        self, eq_name: str, T_res: int, t_offset: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Get per-timestep scale for normalization.

        Args:
            eq_name: equation name (e.g. 'continuity', 'buoyancy')
            T_res: number of interior timesteps in current residual
            t_offset: global timestep index of the first frame in the clip [B]
                      Interior timesteps start at t_offset+1 in global coords.

        Returns:
            scale tensor [1, T_res, 1, 1] or None if not available.
        """
        buf_name = f'scales_t_{eq_name}'
        if not hasattr(self, buf_name):
            return None
        scales_full = getattr(self, buf_name)  # [T_total_interior]

        if t_offset is not None:
            # Each sample has different offset; take per-sample scales and average
            # Interior timesteps of the clip: global indices t_offset+1 .. t_offset+T_res
            # In the GT scales buffer, index 0 = global interior timestep 1
            # So scale index = t_offset (since GT interior starts at global t=1, offset by 1)
            B = t_offset.shape[0]
            T_full = scales_full.shape[0]
            scales_batch = []
            for i in range(B):
                off = int(t_offset[i].item())
                # Clip interior timestep indices in GT buffer
                idx_start = off  # GT buffer idx 0 = global t=1, clip interior starts at clip t=1 = global t=off+1
                idx_end = idx_start + T_res
                if idx_end <= T_full and idx_start >= 0:
                    scales_batch.append(scales_full[idx_start:idx_end])
                else:
                    # Fallback: clamp to valid range
                    idx_start = max(0, min(idx_start, T_full - T_res))
                    idx_end = idx_start + T_res
                    scales_batch.append(scales_full[idx_start:idx_end])
            # Average across batch (different offsets) → [T_res]
            scale_t = torch.stack(scales_batch, dim=0).mean(dim=0)  # [T_res]
        else:
            # No offset info: use global average (fallback)
            scale_t = scales_full[:T_res] if T_res <= scales_full.shape[0] else scales_full.mean().expand(T_res)

        # Clamp to avoid division by near-zero
        scale_t = scale_t.clamp(min=1e-8)
        return scale_t.view(1, T_res, 1, 1)

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        b: torch.Tensor,
        kappa: Optional[torch.Tensor] = None,
        t_offset: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute PDE residual losses.

        Args:
            u: x-velocity [B, T, Nx, Ny]
            v: y-velocity [B, T, Nx, Ny]
            b: buoyancy   [B, T, Nx, Ny]
            kappa: optional per-sample κ [B] overriding self.kappa
            t_offset: optional global timestep offset [B] for per-timestep scaling

        Returns:
            total_loss, {eq_name: loss_value}
        """
        losses: Dict[str, torch.Tensor] = {}
        skip = self.skip_bl

        # Interior timesteps for 2nd-order central time derivative
        u_int = u[:, 1:-1]
        v_int = v[:, 1:-1]
        b_int = b[:, 1:-1]
        T_int = u_int.shape[1]

        # ---- Eq 1: Continuity ----
        du_dx = self._dx_4th(u_int)
        dv_dy = self._dy_ghost(v_int)

        R_cont = du_dx + dv_dy
        # Skip boundary layer
        if skip > 0:
            R_cont = R_cont[..., skip:-skip]

        w_cont = self.eq_weights['continuity']
        if self.has_per_t_scales:
            scale_cont_t = self._get_per_t_scale('continuity', T_int, t_offset)
            if scale_cont_t is not None:
                # Per-timestep normalized: R / scale_t, then MSE
                R_cont_norm = R_cont / scale_cont_t
                losses['continuity'] = w_cont * torch.mean(R_cont_norm ** 2)
            else:
                scale_cont = self.eq_scales['continuity']
                losses['continuity'] = w_cont * torch.mean(R_cont ** 2) / (scale_cont ** 2)
        else:
            scale_cont = self.eq_scales['continuity']
            losses['continuity'] = w_cont * torch.mean(R_cont ** 2) / (scale_cont ** 2)

        # ---- Eq 2: Buoyancy transport ----
        w_buoy = self.eq_weights['buoyancy']
        if w_buoy > 0:
            # Time derivative: 2nd-order central
            db_dt = (b[:, 2:] - b[:, :-2]) / (2 * self.dt)

            # Spatial derivatives at interior timestep
            db_dx = self._dx_4th(b_int)
            db_dy = self._dy_ghost(b_int)

            # Laplacian
            d2b_dx2 = self._d2x_4th(b_int)
            d2b_dy2 = self._d2y_ghost(b_int)
            lap_b = d2b_dx2 + d2b_dy2

            # Advection: u·∂b/∂x + v·∂b/∂y
            adv = u_int * db_dx + v_int * db_dy

            if kappa is not None:
                # Per-sample kappa: [B] → [B, 1, 1, 1]
                kappa_val = kappa.view(-1, 1, 1, 1)
            else:
                kappa_val = self.kappa

            # Residual: ∂b/∂t + u·∇b - κΔb = 0
            R_buoy = db_dt + adv - kappa_val * lap_b

            if skip > 0:
                R_buoy = R_buoy[..., skip:-skip]

            if self.has_per_t_scales:
                scale_buoy_t = self._get_per_t_scale('buoyancy', T_int, t_offset)
                if scale_buoy_t is not None:
                    R_buoy_norm = R_buoy / scale_buoy_t
                    losses['buoyancy'] = w_buoy * torch.mean(R_buoy_norm ** 2)
                else:
                    scale_buoy = self.eq_scales['buoyancy']
                    losses['buoyancy'] = w_buoy * torch.mean(R_buoy ** 2) / (scale_buoy ** 2)
            else:
                scale_buoy = self.eq_scales['buoyancy']
                losses['buoyancy'] = w_buoy * torch.mean(R_buoy ** 2) / (scale_buoy ** 2)
        else:
            losses['buoyancy'] = torch.tensor(0.0, device=u.device)

        total_loss = sum(losses.values())
        return total_loss, losses


# =============================================================================
# 5. Turbulent Radiative PDE Loss (2D Compressible Euler)
# =============================================================================

class TurbulentRadiativePDELoss(nn.Module):
    """
    Turbulent Radiative Layer PDE Loss — Compressible Euler (no energy equation).

    Equations:
        1. Mass: ∂ρ/∂t + ∂(ρvx)/∂x + ∂(ρvy)/∂y = 0
        2. x-Mom: ∂(ρvx)/∂t + ∂(ρvx²+P)/∂x + ∂(ρvx·vy)/∂y = 0
        3. y-Mom: ∂(ρvy)/∂t + ∂(ρvx·vy)/∂x + ∂(ρvy²+P)/∂y = 0

    BCs:
        x: periodic (4th-order central with torch.roll)
        y: open/Neumann (2nd-order central interior only → Ny-2 points)

    Time: 2nd-order central (f[t+1] - f[t-1]) / (2dt)
    skip_bl: skip boundary rows from y edges for loss computation.
    eq_scales: per-equation normalization (GT RMS), eq_weights: per-equation loss weights.

    Data layout: [B, T, Nx, Ny] where Nx = periodic (dim -2), Ny = open (dim -1).
    """

    DEFAULT_EQ_SCALES = {
        'mass': 1.0,
        'x_momentum': 1.0,
        'y_momentum': 1.0,
    }

    def __init__(
        self,
        nx: int = 128,
        ny: int = 384,
        Lx: float = 1.0,
        Ly: float = 3.0,
        dt: float = 1.597033,
        skip_bl: int = 10,
        eq_scales: Optional[Dict[str, float]] = None,
        eq_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.dx = Lx / nx         # periodic
        self.dy = Ly / (ny - 1)   # open: Ly / (ny - 1)
        self.dt = dt
        self.skip_bl = skip_bl

        self.eq_scales = dict(self.DEFAULT_EQ_SCALES)
        if eq_scales is not None:
            self.eq_scales.update(eq_scales)

        self.eq_weights = {k: 1.0 for k in self.DEFAULT_EQ_SCALES}
        if eq_weights is not None:
            self.eq_weights.update(eq_weights)

    def _dx_4th(self, f: torch.Tensor) -> torch.Tensor:
        """4th-order central x-derivative (periodic). f: [..., Nx, Ny]"""
        return (-torch.roll(f, -2, dims=-2) + 8 * torch.roll(f, -1, dims=-2)
                - 8 * torch.roll(f, 1, dims=-2) + torch.roll(f, 2, dims=-2)) / (12 * self.dx)

    def _dy_central(self, f: torch.Tensor) -> torch.Tensor:
        """2nd-order central y-derivative (interior only, Ny → Ny-2). f: [..., Nx, Ny]"""
        return (f[..., 2:] - f[..., :-2]) / (2 * self.dy)

    def forward(
        self,
        vx: torch.Tensor,
        vy: torch.Tensor,
        rho: torch.Tensor,
        P: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute PDE residual losses.

        Args:
            vx:  x-velocity [B, T, Nx, Ny]
            vy:  y-velocity [B, T, Nx, Ny]
            rho: density    [B, T, Nx, Ny]
            P:   pressure   [B, T, Nx, Ny]

        Returns:
            total_loss, {eq_name: loss_value}
        """
        losses: Dict[str, torch.Tensor] = {}
        skip = self.skip_bl

        # Interior timesteps (for 2nd-order central time derivative)
        rho_n = rho[:, 1:-1]
        vx_n = vx[:, 1:-1]
        vy_n = vy[:, 1:-1]
        P_n = P[:, 1:-1]

        # ---- Eq 1: Mass ∂ρ/∂t + ∂(ρvx)/∂x + ∂(ρvy)/∂y = 0 ----
        drho_dt = (rho[:, 2:] - rho[:, :-2]) / (2 * self.dt)

        drhovx_dx = self._dx_4th(rho_n * vx_n)         # [..., Nx, Ny]
        drhovy_dy = self._dy_central(rho_n * vy_n)      # [..., Nx, Ny-2]

        # Trim x-derivative to match y interior
        R_mass = drho_dt[..., 1:-1] + drhovx_dx[..., 1:-1] + drhovy_dy
        if skip > 0:
            R_mass = R_mass[..., skip:-skip]
        raw_mass = torch.mean(R_mass ** 2)
        s, w = self.eq_scales, self.eq_weights
        losses['mass'] = w['mass'] * raw_mass / (s['mass'] ** 2)

        # ---- Eq 2: x-Momentum ∂(ρvx)/∂t + ∂(ρvx²+P)/∂x + ∂(ρvx·vy)/∂y = 0 ----
        rhovx = rho * vx
        drhovx_dt = (rhovx[:, 2:] - rhovx[:, :-2]) / (2 * self.dt)

        dFx_x = self._dx_4th(rho_n * vx_n ** 2 + P_n)
        dGy_x = self._dy_central(rho_n * vx_n * vy_n)

        R_xmom = drhovx_dt[..., 1:-1] + dFx_x[..., 1:-1] + dGy_x
        if skip > 0:
            R_xmom = R_xmom[..., skip:-skip]
        raw_xmom = torch.mean(R_xmom ** 2)
        losses['x_momentum'] = w['x_momentum'] * raw_xmom / (s['x_momentum'] ** 2)

        # ---- Eq 3: y-Momentum ∂(ρvy)/∂t + ∂(ρvx·vy)/∂x + ∂(ρvy²+P)/∂y = 0 ----
        rhovy = rho * vy
        drhovy_dt = (rhovy[:, 2:] - rhovy[:, :-2]) / (2 * self.dt)

        dFx_y = self._dx_4th(rho_n * vx_n * vy_n)
        dGy_y = self._dy_central(rho_n * vy_n ** 2 + P_n)

        R_ymom = drhovy_dt[..., 1:-1] + dFx_y[..., 1:-1] + dGy_y
        if skip > 0:
            R_ymom = R_ymom[..., skip:-skip]
        raw_ymom = torch.mean(R_ymom ** 2)
        losses['y_momentum'] = w['y_momentum'] * raw_ymom / (s['y_momentum'] ** 2)

        total_loss = sum(losses.values())
        return total_loss, losses


# =============================================================================
# 6. Shear Flow PDE Loss — n-PINN Conservative Upwind
# =============================================================================

class ShearFlowPDELossNPINN(nn.Module):
    """
    Shear Flow PDE Loss (n-PINN conservative upwind) — Incompressible NS + Passive Tracer.

    Equations:
        1. Continuity: face-velocity divergence = 0
        2. x-Momentum: du/dt + div(u·u) + dp/dx - ν·Δu = 0
        3. y-Momentum: dv/dt + div(v·u) + dp/dy - ν·Δv = 0
        4. Tracer:      ds/dt + div(s·u) - D·Δs = 0

    n-PINN scheme: face velocities, 2nd-order upwind, div correction.
    Plus face-averaged pressure gradient.

    All periodic BCs (x and y) via torch.roll.

    Data layout: [B, T, H, W] where H=Nx (x-periodic), W=Ny (y-periodic).
    """

    DEFAULT_EQ_SCALES = {
        'continuity': 1.0,
        'x_momentum': 1.0,
        'y_momentum': 1.0,
        'tracer': 1.0,
    }

    def __init__(
        self,
        nx: int = 256,
        ny: int = 512,
        Lx: float = 1.0,
        Ly: float = 2.0,
        dt: float = 0.1,
        nu: float = 1e-4,
        D: float = 1e-3,
        use_div_correction: bool = True,
        eq_scales: Optional[Dict[str, float]] = None,
        eq_weights: Optional[Dict[str, float]] = None,
        eq_scales_per_t: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.dt = dt
        self.nu = nu
        self.D = D
        self.use_div_correction = use_div_correction

        self.eq_scales = dict(self.DEFAULT_EQ_SCALES)
        if eq_scales is not None:
            self.eq_scales.update(eq_scales)

        self.eq_weights = {'continuity': 1.0, 'x_momentum': 1.0,
                           'y_momentum': 1.0, 'tracer': 1.0}
        if eq_weights is not None:
            self.eq_weights.update(eq_weights)

        # Per-timestep normalization scales (optional)
        self.has_per_t_scales = eq_scales_per_t is not None
        if eq_scales_per_t is not None:
            for name, scales in eq_scales_per_t.items():
                self.register_buffer(f'scales_t_{name}', scales.float())

    # ---- Face velocities ----

    def _face_velocities(
        self, u: torch.Tensor, v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Face-averaged velocities (periodic via roll).
        Data layout: [B, T, H, W] where H=Nx (x, dim=-2), W=Ny (y, dim=-1).
        East/West = x-direction (dim=-2), North/South = y-direction (dim=-1).
        """
        uc_e = 0.5 * (u + torch.roll(u, -1, dims=-2))  # east (x+)
        uc_w = 0.5 * (u + torch.roll(u, 1, dims=-2))   # west (x-)
        vc_n = 0.5 * (v + torch.roll(v, -1, dims=-1))   # north (y+)
        vc_s = 0.5 * (v + torch.roll(v, 1, dims=-1))    # south (y-)
        return uc_e, uc_w, vc_n, vc_s

    # ---- Face-velocity divergence ----

    def _divergence(
        self,
        uc_e: torch.Tensor, uc_w: torch.Tensor,
        vc_n: torch.Tensor, vc_s: torch.Tensor,
    ) -> torch.Tensor:
        return (uc_e - uc_w) / self.dx + (vc_n - vc_s) / self.dy

    # ---- 2nd-order upwind convection (periodic) ----

    def _upwind_convection(
        self,
        f: torch.Tensor,
        uc_e: torch.Tensor, uc_w: torch.Tensor,
        vc_n: torch.Tensor, vc_s: torch.Tensor,
    ) -> torch.Tensor:
        """Conservative upwind convection: div(f*u) with 2nd-order upwind face values.
        x-direction = dim=-2 (H=Nx), y-direction = dim=-1 (W=Ny).
        """
        # x-direction neighbors (dim=-2, H=Nx)
        f_ip1 = torch.roll(f, -1, dims=-2)
        f_im1 = torch.roll(f, 1, dims=-2)
        f_ip2 = torch.roll(f, -2, dims=-2)
        f_im2 = torch.roll(f, 2, dims=-2)

        # y-direction neighbors (dim=-1, W=Ny)
        f_jp1 = torch.roll(f, -1, dims=-1)
        f_jm1 = torch.roll(f, 1, dims=-1)
        f_jp2 = torch.roll(f, -2, dims=-1)
        f_jm2 = torch.roll(f, 2, dims=-1)

        # East face (x+)
        Fe_pos = 1.5 * f - 0.5 * f_im1
        Fe_neg = 1.5 * f_ip1 - 0.5 * f_ip2
        Fe = torch.where(uc_e >= 0, Fe_pos, Fe_neg)

        # West face (x-)
        Fw_pos = 1.5 * f_im1 - 0.5 * f_im2
        Fw_neg = 1.5 * f - 0.5 * f_ip1
        Fw = torch.where(uc_w >= 0, Fw_pos, Fw_neg)

        # North face (y+)
        Fn_pos = 1.5 * f - 0.5 * f_jm1
        Fn_neg = 1.5 * f_jp1 - 0.5 * f_jp2
        Fn = torch.where(vc_n >= 0, Fn_pos, Fn_neg)

        # South face (y-)
        Fs_pos = 1.5 * f_jm1 - 0.5 * f_jm2
        Fs_neg = 1.5 * f - 0.5 * f_jp1
        Fs = torch.where(vc_s >= 0, Fs_pos, Fs_neg)

        conv = (uc_e * Fe - uc_w * Fw) / self.dx + (vc_n * Fn - vc_s * Fs) / self.dy
        return conv

    # ---- Laplacian (periodic) ----

    def _laplacian_2d(self, f: torch.Tensor) -> torch.Tensor:
        """2nd-order central Laplacian with periodic BC.
        x = dim=-2 (dx), y = dim=-1 (dy).
        """
        lap_x = (torch.roll(f, -1, dims=-2) - 2 * f + torch.roll(f, 1, dims=-2)) / (self.dx ** 2)
        lap_y = (torch.roll(f, -1, dims=-1) - 2 * f + torch.roll(f, 1, dims=-1)) / (self.dy ** 2)
        return lap_x + lap_y

    # ---- Face-averaged pressure gradient ----

    def _pressure_grad_x(self, p: torch.Tensor) -> torch.Tensor:
        """dp/dx via face average. x = dim=-2."""
        p_E = torch.roll(p, -1, dims=-2)
        p_W = torch.roll(p, 1, dims=-2)
        return (p_E - p_W) / (2 * self.dx)

    def _pressure_grad_y(self, p: torch.Tensor) -> torch.Tensor:
        """dp/dy via face average. y = dim=-1."""
        p_N = torch.roll(p, -1, dims=-1)
        p_S = torch.roll(p, 1, dims=-1)
        return (p_N - p_S) / (2 * self.dy)

    def _get_per_t_scale(
        self, eq_name: str, T_res: int, t_offset: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Get per-timestep scale for normalization.

        Args:
            eq_name: equation name
            T_res: number of interior timesteps in current residual
            t_offset: global timestep index of the first frame in the clip [B]

        Returns:
            scale tensor [1, T_res, 1, 1] or None if not available.
        """
        buf_name = f'scales_t_{eq_name}'
        if not hasattr(self, buf_name):
            return None
        scales_full = getattr(self, buf_name)  # [T_total_interior]

        if t_offset is not None:
            B = t_offset.shape[0]
            T_full = scales_full.shape[0]
            scales_batch = []
            for i in range(B):
                off = int(t_offset[i].item())
                idx_start = off
                idx_end = idx_start + T_res
                if idx_end <= T_full and idx_start >= 0:
                    scales_batch.append(scales_full[idx_start:idx_end])
                else:
                    idx_start = max(0, min(idx_start, T_full - T_res))
                    idx_end = idx_start + T_res
                    scales_batch.append(scales_full[idx_start:idx_end])
            scale_t = torch.stack(scales_batch, dim=0).mean(dim=0)
        else:
            scale_t = scales_full[:T_res] if T_res <= scales_full.shape[0] else scales_full.mean().expand(T_res)

        scale_t = scale_t.clamp(min=1e-8)
        return scale_t.view(1, T_res, 1, 1)

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        s: torch.Tensor,
        reduction: str = 'mean',
        t_offset: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute PDE residual losses.

        Args:
            u, v, p, s: [B, T, H, W]
            reduction: 'mean' (default)
            t_offset: optional global timestep offset [B] for per-timestep scaling

        Returns:
            total_loss, {eq_name: loss_value}
        """
        losses: Dict[str, torch.Tensor] = {}

        # Interior timesteps (2nd-order central time derivative)
        u_int = u[:, 1:-1]
        v_int = v[:, 1:-1]
        p_int = p[:, 1:-1]
        s_int = s[:, 1:-1]
        T_int = u_int.shape[1]

        # Face velocities
        uc_e, uc_w, vc_n, vc_s = self._face_velocities(u_int, v_int)

        # Divergence
        div = self._divergence(uc_e, uc_w, vc_n, vc_s)

        # ---- Eq 1: Continuity ----
        w_cont = self.eq_weights['continuity']
        if self.has_per_t_scales:
            scale_cont_t = self._get_per_t_scale('continuity', T_int, t_offset)
            if scale_cont_t is not None:
                div_norm = div / scale_cont_t
                losses['continuity'] = w_cont * torch.mean(div_norm ** 2)
            else:
                scale_cont = self.eq_scales['continuity']
                losses['continuity'] = w_cont * torch.mean(div ** 2) / (scale_cont ** 2)
        else:
            scale_cont = self.eq_scales['continuity']
            losses['continuity'] = w_cont * torch.mean(div ** 2) / (scale_cont ** 2)

        # ---- Eq 2: x-Momentum ----
        w_umom = self.eq_weights['x_momentum']
        if w_umom > 0:
            du_dt = (u[:, 2:] - u[:, :-2]) / (2 * self.dt)
            conv_u = self._upwind_convection(u_int, uc_e, uc_w, vc_n, vc_s)
            dpx = self._pressure_grad_x(p_int)
            lap_u = self._laplacian_2d(u_int)

            R_u = du_dt + conv_u + dpx - self.nu * lap_u
            if self.use_div_correction:
                R_u = R_u - u_int * div

            if self.has_per_t_scales:
                scale_u_t = self._get_per_t_scale('x_momentum', T_int, t_offset)
                if scale_u_t is not None:
                    R_u_norm = R_u / scale_u_t
                    losses['x_momentum'] = w_umom * torch.mean(R_u_norm ** 2)
                else:
                    scale_u = self.eq_scales['x_momentum']
                    losses['x_momentum'] = w_umom * torch.mean(R_u ** 2) / (scale_u ** 2)
            else:
                scale_u = self.eq_scales['x_momentum']
                losses['x_momentum'] = w_umom * torch.mean(R_u ** 2) / (scale_u ** 2)
        else:
            losses['x_momentum'] = torch.tensor(0.0, device=u.device)

        # ---- Eq 3: y-Momentum ----
        w_vmom = self.eq_weights['y_momentum']
        if w_vmom > 0:
            dv_dt = (v[:, 2:] - v[:, :-2]) / (2 * self.dt)
            conv_v = self._upwind_convection(v_int, uc_e, uc_w, vc_n, vc_s)
            dpy = self._pressure_grad_y(p_int)
            lap_v = self._laplacian_2d(v_int)

            R_v = dv_dt + conv_v + dpy - self.nu * lap_v
            if self.use_div_correction:
                R_v = R_v - v_int * div

            if self.has_per_t_scales:
                scale_v_t = self._get_per_t_scale('y_momentum', T_int, t_offset)
                if scale_v_t is not None:
                    R_v_norm = R_v / scale_v_t
                    losses['y_momentum'] = w_vmom * torch.mean(R_v_norm ** 2)
                else:
                    scale_v = self.eq_scales['y_momentum']
                    losses['y_momentum'] = w_vmom * torch.mean(R_v ** 2) / (scale_v ** 2)
            else:
                scale_v = self.eq_scales['y_momentum']
                losses['y_momentum'] = w_vmom * torch.mean(R_v ** 2) / (scale_v ** 2)
        else:
            losses['y_momentum'] = torch.tensor(0.0, device=u.device)

        # ---- Eq 4: Tracer ----
        w_tracer = self.eq_weights['tracer']
        if w_tracer > 0:
            ds_dt = (s[:, 2:] - s[:, :-2]) / (2 * self.dt)
            conv_s = self._upwind_convection(s_int, uc_e, uc_w, vc_n, vc_s)
            lap_s = self._laplacian_2d(s_int)

            R_s = ds_dt + conv_s - self.D * lap_s
            if self.use_div_correction:
                R_s = R_s - s_int * div

            if self.has_per_t_scales:
                scale_s_t = self._get_per_t_scale('tracer', T_int, t_offset)
                if scale_s_t is not None:
                    R_s_norm = R_s / scale_s_t
                    losses['tracer'] = w_tracer * torch.mean(R_s_norm ** 2)
                else:
                    scale_s = self.eq_scales['tracer']
                    losses['tracer'] = w_tracer * torch.mean(R_s ** 2) / (scale_s ** 2)
            else:
                scale_s = self.eq_scales['tracer']
                losses['tracer'] = w_tracer * torch.mean(R_s ** 2) / (scale_s ** 2)
        else:
            losses['tracer'] = torch.tensor(0.0, device=u.device)

        total_loss = sum(losses.values())
        return total_loss, losses


# =============================================================================
# Wave-Gauss Acoustic Wave Equation PDE Loss
# =============================================================================

class WaveGaussPDELoss(nn.Module):
    """
    Acoustic Wave Equation PDE Loss: u_tt = c(x)^2 * nabla^2 u

    改写为 u_tt/c^2 - nabla^2 u = 0，消除 c^2 量级放大。

    边界条件:
        - 非周期 (absorbing BC) → 2阶中心差分, 只在内部点计算
        - skip_boundary: 进一步跳过靠近边界的像素

    数值方法:
        - 时间: 2阶中心差分 u_tt = (u[t+1] - 2*u[t] + u[t-1]) / dt^2
        - 空间: 2阶中心差分 Laplacian (非周期)
            d2u/dx2 = (u[i+1] - 2u[i] + u[i-1]) / dx^2
            d2u/dy2 = (u[j+1] - 2u[j] + u[j-1]) / dy^2
        - 只在内部点 [1:-1, 1:-1] 有效, 再跳 skip_boundary 行/列

    GT RMS (eq_scales): ~90.6 (from test_gt_pde_wave_gauss.py)
    """

    DEFAULT_EQ_SCALES = {
        'wave': 1.0,
    }

    def __init__(
        self,
        nx: int = 128,
        ny: int = 128,
        Lx: float = 1.0,
        Ly: float = 1.0,
        dt: float = 1.0 / 14,
        skip_boundary: int = 2,
        eq_scales: Optional[Dict[str, float]] = None,
        eq_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.dt = dt
        self.skip_boundary = skip_boundary

        self.eq_scales = {**self.DEFAULT_EQ_SCALES, **(eq_scales or {})}
        self.eq_weights = {'wave': 1.0, **(eq_weights or {})}

    def forward(
        self,
        u: torch.Tensor,
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算 acoustic wave PDE 残差 loss

        Args:
            u: displacement [B, T, H, W]
            c: wave speed [B, H, W] or [B, 1, H, W] (time-independent)

        Returns:
            total_loss, losses_dict
        """
        dx, dy, dt = self.dx, self.dy, self.dt
        sb = self.skip_boundary

        if c.dim() == 3:
            c = c.unsqueeze(1)

        # ---- Time derivative: u_tt (2nd-order central) ----
        u_tt = (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / (dt ** 2)
        u_mid = u[:, 1:-1]

        # ---- Spatial Laplacian (2nd-order central, non-periodic) ----
        # Only valid at interior points [1:-1, 1:-1]
        d2u_dx2 = (u_mid[..., 2:, 1:-1] - 2 * u_mid[..., 1:-1, 1:-1] + u_mid[..., :-2, 1:-1]) / (dx ** 2)
        d2u_dy2 = (u_mid[..., 1:-1, 2:] - 2 * u_mid[..., 1:-1, 1:-1] + u_mid[..., 1:-1, :-2]) / (dy ** 2)
        lap_u = d2u_dx2 + d2u_dy2

        u_tt_interior = u_tt[..., 1:-1, 1:-1]
        c_interior = c[..., 1:-1, 1:-1]

        # ---- Residual: R = u_tt/c^2 - lap_u ----
        R_wave = u_tt_interior / (c_interior ** 2) - lap_u

        if sb > 0:
            R_wave = R_wave[..., sb:-sb, sb:-sb]

        # ---- Per-equation normalized loss ----
        s_wave = self.eq_scales['wave']
        w_wave = self.eq_weights['wave']

        loss_wave = w_wave * torch.mean(R_wave ** 2) / (s_wave ** 2)

        return loss_wave, {'wave': loss_wave, 'total': loss_wave}


# =============================================================================
# NS-PwC / NS-SVS PDE Loss (Incompressible NS + Passive Tracer, 2nd-order central)
# =============================================================================

class NSPwCPDELoss(nn.Module):
    """
    NS-PwC / NS-SVS PDE Loss — Incompressible NS + Passive Tracer.

    Uses vorticity formulation to avoid pressure:
        1. Divergence: ∂u/∂x + ∂v/∂y = 0
        2. Vorticity transport: ∂ω/∂t + u·∂ω/∂x + v·∂ω/∂y = ν·Δω
           where ω = ∂v/∂x - ∂u/∂y
        3. Tracer transport: ∂s/∂t + u·∂s/∂x + v·∂s/∂y = κ·Δs

    All periodic BCs, 2nd-order central FD (best GT residual on 128x128 grid).
    Time: 2nd-order central (f[t+1] - f[t-1]) / (2dt).
    Per-equation normalization via eq_scales / eq_weights.

    Note: 2nd-order central beats 4th-order and n-PINN on this dataset
    (128x128 resolution, spectral solver data).
    """

    DEFAULT_EQ_SCALES = {
        'divergence': 1.0,
        'vorticity': 1.0,
        'tracer': 1.0,
    }

    def __init__(
        self,
        nx: int = 128,
        ny: int = 128,
        Lx: float = 1.0,
        Ly: float = 1.0,
        dt: float = 0.05,
        nu: float = 4e-4,
        kappa: float = 4e-4,
        eq_scales: Optional[Dict[str, float]] = None,
        eq_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.dt = dt
        self.nu = nu
        self.kappa = kappa

        self.eq_scales = dict(self.DEFAULT_EQ_SCALES)
        if eq_scales is not None:
            self.eq_scales.update(eq_scales)

        self.eq_weights = {'divergence': 1.0, 'vorticity': 1.0, 'tracer': 1.0}
        if eq_weights is not None:
            self.eq_weights.update(eq_weights)

    def _dx(self, f: torch.Tensor) -> torch.Tensor:
        """2nd-order central x-derivative (periodic)."""
        return (torch.roll(f, -1, dims=-1) - torch.roll(f, 1, dims=-1)) / (2 * self.dx)

    def _dy(self, f: torch.Tensor) -> torch.Tensor:
        """2nd-order central y-derivative (periodic)."""
        return (torch.roll(f, -1, dims=-2) - torch.roll(f, 1, dims=-2)) / (2 * self.dy)

    def _laplacian(self, f: torch.Tensor) -> torch.Tensor:
        """2nd-order central Laplacian (periodic)."""
        lx = (torch.roll(f, -1, dims=-1) - 2*f + torch.roll(f, 1, dims=-1)) / self.dx**2
        ly = (torch.roll(f, -1, dims=-2) - 2*f + torch.roll(f, 1, dims=-2)) / self.dy**2
        return lx + ly

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        s: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute PDE residual losses.

        Args:
            u: x-velocity [B, T, H, W] (T includes prepended t0 frame)
            v: y-velocity [B, T, H, W]
            s: tracer     [B, T, H, W]

        Returns:
            total_loss, {eq_name: loss_value}
        """
        losses: Dict[str, torch.Tensor] = {}

        # Interior timesteps
        u_int = u[:, 1:-1]
        v_int = v[:, 1:-1]
        s_int = s[:, 1:-1]

        # ---- Eq 1: Divergence ----
        div = self._dx(u_int) + self._dy(v_int)
        scale_div = self.eq_scales['divergence']
        w_div = self.eq_weights['divergence']
        losses['divergence'] = w_div * torch.mean(div ** 2) / (scale_div ** 2)

        # ---- Eq 2: Vorticity transport ----
        w_vort = self.eq_weights['vorticity']
        if w_vort > 0:
            # ω = ∂v/∂x - ∂u/∂y at interior timesteps
            omega_int = self._dx(v_int) - self._dy(u_int)

            # ∂ω/∂t via 2nd-order central on the full ω field
            omega_next = self._dx(v[:, 2:]) - self._dy(u[:, 2:])
            omega_prev = self._dx(v[:, :-2]) - self._dy(u[:, :-2])
            domega_dt = (omega_next - omega_prev) / (2 * self.dt)

            # Advection: u·∂ω/∂x + v·∂ω/∂y
            adv_omega = u_int * self._dx(omega_int) + v_int * self._dy(omega_int)

            # Diffusion: ν·Δω
            lap_omega = self._laplacian(omega_int)

            R_vort = domega_dt + adv_omega - self.nu * lap_omega
            scale_vort = self.eq_scales['vorticity']
            losses['vorticity'] = w_vort * torch.mean(R_vort ** 2) / (scale_vort ** 2)
        else:
            losses['vorticity'] = torch.tensor(0.0, device=u.device)

        # ---- Eq 3: Tracer transport ----
        w_tracer = self.eq_weights['tracer']
        if w_tracer > 0:
            ds_dt = (s[:, 2:] - s[:, :-2]) / (2 * self.dt)
            adv_s = u_int * self._dx(s_int) + v_int * self._dy(s_int)
            lap_s = self._laplacian(s_int)

            R_tracer = ds_dt + adv_s - self.kappa * lap_s
            scale_tracer = self.eq_scales['tracer']
            losses['tracer'] = w_tracer * torch.mean(R_tracer ** 2) / (scale_tracer ** 2)
        else:
            losses['tracer'] = torch.tensor(0.0, device=u.device)

        total_loss = sum(losses.values())
        return total_loss, losses


# =============================================================================
# 10. Taylor-Green Vortex 2D PDE Loss (Incompressible Navier-Stokes)
# =============================================================================

class TaylorGreen2DPDELoss(nn.Module):
    """
    Taylor-Green Vortex 2D PDE Loss — Incompressible Navier-Stokes.

    Equations:
        1. Continuity: du/dx + dv/dy = 0
        2. x-momentum: du/dt + u*du/dx + v*du/dy + dp/dx - nu*lap(u) = 0
        3. y-momentum: dv/dt + u*dv/dx + v*dv/dy + dp/dy - nu*lap(v) = 0

    BCs: Periodic in both x and y (4th-order central with torch.roll)
    Time: 2nd-order central (f[t+1] - f[t-1]) / (2*dt)
    Data layout: [B, T, H, W] where H=W=256, domain [0, 2pi]^2
    """

    DEFAULT_EQ_SCALES = {
        'continuity': 1.0,
        'x_momentum': 1.0,
        'y_momentum': 1.0,
    }

    def __init__(
        self,
        nx: int = 256,
        ny: int = 256,
        Lx: float = 6.283185307179586,  # 2*pi
        Ly: float = 6.283185307179586,  # 2*pi
        dt: float = 0.01,
        eq_scales: Optional[Dict[str, float]] = None,
        eq_weights: Optional[Dict[str, float]] = None,
        eq_scales_per_t: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.dx = Lx / nx  # boundary-exclusive periodic: dx = L / N
        self.dy = Ly / ny
        self.dt = dt

        self.eq_scales = dict(self.DEFAULT_EQ_SCALES)
        if eq_scales is not None:
            self.eq_scales.update(eq_scales)

        self.eq_weights = {'continuity': 1.0, 'x_momentum': 1.0, 'y_momentum': 1.0}
        if eq_weights is not None:
            self.eq_weights.update(eq_weights)

        # Per-timestep normalization scales (optional)
        self.has_per_t_scales = eq_scales_per_t is not None
        if eq_scales_per_t is not None:
            for name, scales in eq_scales_per_t.items():
                self.register_buffer(f'scales_t_{name}', scales.float())

    # ---- Spatial derivatives: 4th-order central, periodic via roll ----

    def _dx_4th(self, f: torch.Tensor) -> torch.Tensor:
        """4th-order central x-derivative (periodic). f: [..., H, W], x along W."""
        return (-torch.roll(f, -2, dims=-1) + 8 * torch.roll(f, -1, dims=-1)
                - 8 * torch.roll(f, 1, dims=-1) + torch.roll(f, 2, dims=-1)) / (12 * self.dx)

    def _dy_4th(self, f: torch.Tensor) -> torch.Tensor:
        """4th-order central y-derivative (periodic). f: [..., H, W], y along H."""
        return (-torch.roll(f, -2, dims=-2) + 8 * torch.roll(f, -1, dims=-2)
                - 8 * torch.roll(f, 1, dims=-2) + torch.roll(f, 2, dims=-2)) / (12 * self.dy)

    def _d2x_4th(self, f: torch.Tensor) -> torch.Tensor:
        """4th-order central 2nd x-derivative (periodic)."""
        return (-torch.roll(f, -2, dims=-1) + 16 * torch.roll(f, -1, dims=-1)
                - 30 * f + 16 * torch.roll(f, 1, dims=-1)
                - torch.roll(f, 2, dims=-1)) / (12 * self.dx ** 2)

    def _d2y_4th(self, f: torch.Tensor) -> torch.Tensor:
        """4th-order central 2nd y-derivative (periodic)."""
        return (-torch.roll(f, -2, dims=-2) + 16 * torch.roll(f, -1, dims=-2)
                - 30 * f + 16 * torch.roll(f, 1, dims=-2)
                - torch.roll(f, 2, dims=-2)) / (12 * self.dy ** 2)

    def _get_per_t_scale(
        self, eq_name: str, T_res: int, t_offset: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Get per-timestep scale for normalization."""
        buf_name = f'scales_t_{eq_name}'
        if not hasattr(self, buf_name):
            return None
        scales_full = getattr(self, buf_name)  # [T_total_interior]

        if t_offset is not None:
            B = t_offset.shape[0]
            T_full = scales_full.shape[0]
            scales_batch = []
            for i in range(B):
                off = int(t_offset[i].item())
                idx_start = off
                idx_end = idx_start + T_res
                if idx_end <= T_full and idx_start >= 0:
                    scales_batch.append(scales_full[idx_start:idx_end])
                else:
                    idx_start = max(0, min(idx_start, T_full - T_res))
                    idx_end = idx_start + T_res
                    scales_batch.append(scales_full[idx_start:idx_end])
            scale_t = torch.stack(scales_batch, dim=0).mean(dim=0)
        else:
            scale_t = scales_full[:T_res] if T_res <= scales_full.shape[0] else scales_full.mean().expand(T_res)

        scale_t = scale_t.clamp(min=1e-8)
        return scale_t.view(1, T_res, 1, 1)

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        nu: Optional[torch.Tensor] = None,
        t_offset: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute PDE residual losses.

        Args:
            u: x-velocity [B, T, H, W]
            v: y-velocity [B, T, H, W]
            p: pressure   [B, T, H, W]
            nu: per-sample viscosity [B]
            t_offset: optional global timestep offset [B]

        Returns:
            total_loss, {eq_name: loss_value}
        """
        losses: Dict[str, torch.Tensor] = {}

        # Interior timesteps for 2nd-order central time derivative
        u_int = u[:, 1:-1]
        v_int = v[:, 1:-1]
        p_int = p[:, 1:-1]
        T_int = u_int.shape[1]

        # Per-sample nu: [B] -> [B, 1, 1, 1]
        if nu is not None:
            nu_4d = nu.view(-1, 1, 1, 1)
        else:
            nu_4d = 0.01  # fallback default

        # ---- Eq 1: Continuity ----
        du_dx = self._dx_4th(u_int)
        dv_dy = self._dy_4th(v_int)
        R_cont = du_dx + dv_dy

        w_cont = self.eq_weights['continuity']
        if self.has_per_t_scales:
            scale_cont_t = self._get_per_t_scale('continuity', T_int, t_offset)
            if scale_cont_t is not None:
                losses['continuity'] = w_cont * torch.mean((R_cont / scale_cont_t) ** 2)
            else:
                scale_cont = self.eq_scales['continuity']
                losses['continuity'] = w_cont * torch.mean(R_cont ** 2) / (scale_cont ** 2)
        else:
            scale_cont = self.eq_scales['continuity']
            losses['continuity'] = w_cont * torch.mean(R_cont ** 2) / (scale_cont ** 2)

        # ---- Eq 2: x-momentum ----
        w_xmom = self.eq_weights['x_momentum']
        if w_xmom > 0:
            du_dt = (u[:, 2:] - u[:, :-2]) / (2 * self.dt)
            du_dy = self._dy_4th(u_int)
            dp_dx = self._dx_4th(p_int)
            lap_u = self._d2x_4th(u_int) + self._d2y_4th(u_int)

            R_xmom = du_dt + u_int * du_dx + v_int * du_dy + dp_dx - nu_4d * lap_u

            if self.has_per_t_scales:
                scale_xmom_t = self._get_per_t_scale('x_momentum', T_int, t_offset)
                if scale_xmom_t is not None:
                    losses['x_momentum'] = w_xmom * torch.mean((R_xmom / scale_xmom_t) ** 2)
                else:
                    scale_xmom = self.eq_scales['x_momentum']
                    losses['x_momentum'] = w_xmom * torch.mean(R_xmom ** 2) / (scale_xmom ** 2)
            else:
                scale_xmom = self.eq_scales['x_momentum']
                losses['x_momentum'] = w_xmom * torch.mean(R_xmom ** 2) / (scale_xmom ** 2)
        else:
            losses['x_momentum'] = torch.tensor(0.0, device=u.device)

        # ---- Eq 3: y-momentum ----
        w_ymom = self.eq_weights['y_momentum']
        if w_ymom > 0:
            dv_dt = (v[:, 2:] - v[:, :-2]) / (2 * self.dt)
            dv_dx = self._dx_4th(v_int)
            dv_dy_mom = self._dy_4th(v_int)
            dp_dy = self._dy_4th(p_int)
            lap_v = self._d2x_4th(v_int) + self._d2y_4th(v_int)

            R_ymom = dv_dt + u_int * dv_dx + v_int * dv_dy_mom + dp_dy - nu_4d * lap_v

            if self.has_per_t_scales:
                scale_ymom_t = self._get_per_t_scale('y_momentum', T_int, t_offset)
                if scale_ymom_t is not None:
                    losses['y_momentum'] = w_ymom * torch.mean((R_ymom / scale_ymom_t) ** 2)
                else:
                    scale_ymom = self.eq_scales['y_momentum']
                    losses['y_momentum'] = w_ymom * torch.mean(R_ymom ** 2) / (scale_ymom ** 2)
            else:
                scale_ymom = self.eq_scales['y_momentum']
                losses['y_momentum'] = w_ymom * torch.mean(R_ymom ** 2) / (scale_ymom ** 2)
        else:
            losses['y_momentum'] = torch.tensor(0.0, device=u.device)

        total_loss = sum(losses.values())
        return total_loss, losses


# =============================================================================
# 11. KP-II (Kadomtsev-Petviashvili) PDE Loss
# =============================================================================

class KP2DPDELoss(nn.Module):
    """
    KP-II PDE Loss — 2D periodic domain.

    PDE (expanded form):
        u_xt + 6*(u_x)^2 + 6*u*u_xx + u_xxxx + 3*u_yy = 0

    Derived from the original form:
        d/dx(u_t + 6u*u_x + u_xxx) + 3*u_yy = 0

    Numerical methods:
        - Time: 2nd-order central difference
        - Space: 4th-order central FD (periodic via torch.roll)
        - u_xxxx: 2nd-order 5-point stencil (direct)

    NOTE: The line soliton is not truly periodic on [0,L]^2. A skip_boundary
          margin is used to exclude boundary artifacts from the loss.

    Args:
        nx, ny: grid points (256, 256)
        dx, dy: grid spacing
        dt: time step
        skip_boundary: pixels to skip from each spatial edge (default 20)
        eq_scales: per-equation normalization dict
        eq_weights: per-equation loss weights dict
        eq_scales_per_t: per-timestep normalization dict
    """

    DEFAULT_EQ_SCALES = {
        'kp2_equation': 1.0,
    }

    def __init__(
        self,
        nx: int = 256,
        ny: int = 256,
        dx: float = 20.0 / 256,
        dy: float = 20.0 / 256,
        dt: float = 0.01,
        skip_boundary: int = 20,
        eq_scales: Optional[Dict[str, float]] = None,
        eq_weights: Optional[Dict[str, float]] = None,
        eq_scales_per_t: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.skip_boundary = skip_boundary

        self.eq_scales = dict(self.DEFAULT_EQ_SCALES)
        if eq_scales is not None:
            self.eq_scales.update(eq_scales)

        self.eq_weights = {'kp2_equation': 1.0}
        if eq_weights is not None:
            self.eq_weights.update(eq_weights)

        # Per-timestep scales
        self.use_per_t_scales = eq_scales_per_t is not None
        if eq_scales_per_t is not None:
            for name, scales in eq_scales_per_t.items():
                self.register_buffer(f'scales_t_{name}', scales.float())

    def _dx_4th(self, f: torch.Tensor) -> torch.Tensor:
        """4th-order central 1st x-derivative (periodic). f: [..., H, W], x=dim -2."""
        return (-torch.roll(f, -2, dims=-2) + 8 * torch.roll(f, -1, dims=-2)
                - 8 * torch.roll(f, 1, dims=-2) + torch.roll(f, 2, dims=-2)) / (12 * self.dx)

    def _d2x_4th(self, f: torch.Tensor) -> torch.Tensor:
        """4th-order central 2nd x-derivative (periodic). f: [..., H, W], x=dim -2."""
        return (-torch.roll(f, -2, dims=-2) + 16 * torch.roll(f, -1, dims=-2)
                - 30 * f + 16 * torch.roll(f, 1, dims=-2)
                - torch.roll(f, 2, dims=-2)) / (12 * self.dx ** 2)

    def _d2y_4th(self, f: torch.Tensor) -> torch.Tensor:
        """4th-order central 2nd y-derivative (periodic). f: [..., H, W], y=dim -1."""
        return (-torch.roll(f, -2, dims=-1) + 16 * torch.roll(f, -1, dims=-1)
                - 30 * f + 16 * torch.roll(f, 1, dims=-1)
                - torch.roll(f, 2, dims=-1)) / (12 * self.dy ** 2)

    def _d4x(self, f: torch.Tensor) -> torch.Tensor:
        """2nd-order 5-point stencil for 4th x-derivative (periodic). f: [..., H, W]."""
        fm2 = torch.roll(f, 2, dims=-2)
        fm1 = torch.roll(f, 1, dims=-2)
        fp1 = torch.roll(f, -1, dims=-2)
        fp2 = torch.roll(f, -2, dims=-2)
        return (fm2 - 4 * fm1 + 6 * f - 4 * fp1 + fp2) / (self.dx ** 4)

    def forward(
        self,
        u: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute KP-II PDE residual loss.

        Args:
            u: scalar field [B, T, H, W] where T includes prepended t0 frame.
               Time derivative uses t=1..T-2 (2nd-order central).

        Returns:
            total_loss, {eq_name: loss_value}
        """
        losses: Dict[str, torch.Tensor] = {}
        sb = self.skip_boundary

        # Spatial derivatives at all timesteps
        u_x = self._dx_4th(u)  # [B, T, H, W]

        # Time derivative of u_x at interior timesteps t=1..T-2
        u_xt = (u_x[:, 2:] - u_x[:, :-2]) / (2 * self.dt)  # [B, T-2, H, W]

        # Spatial terms at interior timesteps
        u_mid = u[:, 1:-1]       # [B, T-2, H, W]
        u_x_mid = u_x[:, 1:-1]

        u_xx = self._d2x_4th(u_mid)
        u_xxxx = self._d4x(u_mid)
        u_yy = self._d2y_4th(u_mid)

        # KP-II residual: u_xt + 6*(u_x)^2 + 6*u*u_xx + u_xxxx + 3*u_yy = 0
        R = u_xt + 6 * u_x_mid**2 + 6 * u_mid * u_xx + u_xxxx + 3 * u_yy

        # Crop boundary region
        if sb > 0:
            R = R[:, :, sb:-sb, sb:-sb]

        # Compute loss
        w_kp2 = self.eq_weights['kp2_equation']
        T_res = R.shape[1]

        if self.use_per_t_scales and hasattr(self, 'scales_t_kp2_equation'):
            s_t = self.scales_t_kp2_equation[:T_res].clamp(min=1e-8)
            R_norm = R / s_t[None, :, None, None]
            losses['kp2_equation'] = w_kp2 * torch.mean(R_norm ** 2)
        else:
            scale = self.eq_scales['kp2_equation']
            losses['kp2_equation'] = w_kp2 * torch.mean(R ** 2) / (scale ** 2)

        total_loss = sum(losses.values())
        return total_loss, losses


# =============================================================================
# 12. Burgers 2D Cole-Hopf PDE Loss (n-PINN Conservative Upwind)
# =============================================================================

class Burgers2DCHPDELoss(nn.Module):
    """
    2D Burgers (Cole-Hopf, periodic) PDE Loss — n-PINN conservative upwind.

    Equations:
        1. u-momentum: du/dt + div(u*u_vec) - u*div(u_vec) - nu*lap(u) = 0
        2. v-momentum: dv/dt + div(v*u_vec) - v*div(u_vec) - nu*lap(v) = 0
        3. irrotational: dv/dx - du/dy = 0

    No pressure (unlike NS). Per-sample nu from dataset.

    n-PINN scheme: face velocities, 2nd-order upwind, div correction.
    All periodic BCs (x and y) via torch.roll.

    Data layout: [B, T, H, W] where H=W=256, domain [0, 2pi]^2.
    """

    DEFAULT_EQ_SCALES = {
        'u_momentum': 1.0,
        'v_momentum': 1.0,
        'irrotational': 1.0,
    }

    def __init__(
        self,
        nx: int = 256,
        ny: int = 256,
        Lx: float = 6.283185307179586,  # 2*pi
        Ly: float = 6.283185307179586,  # 2*pi
        dt: float = 0.01,
        eq_scales: Optional[Dict[str, float]] = None,
        eq_weights: Optional[Dict[str, float]] = None,
        eq_scales_per_t: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.dx = Lx / nx  # boundary-exclusive periodic: dx = L / N
        self.dy = Ly / ny
        self.dt = dt

        self.eq_scales = dict(self.DEFAULT_EQ_SCALES)
        if eq_scales is not None:
            self.eq_scales.update(eq_scales)

        self.eq_weights = {
            'u_momentum': 1.0,
            'v_momentum': 1.0,
            'irrotational': 1.0,
        }
        if eq_weights is not None:
            self.eq_weights.update(eq_weights)

        # Per-timestep normalization scales (optional)
        self.has_per_t_scales = eq_scales_per_t is not None
        if eq_scales_per_t is not None:
            for name, scales in eq_scales_per_t.items():
                self.register_buffer(f'scales_t_{name}', scales.float())

    # ---- Face velocities ----

    def _face_velocities(
        self, u: torch.Tensor, v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Face-averaged velocities (periodic via roll)."""
        uc_e = 0.5 * (u + torch.roll(u, -1, dims=-1))
        uc_w = 0.5 * (u + torch.roll(u, 1, dims=-1))
        vc_n = 0.5 * (v + torch.roll(v, -1, dims=-2))
        vc_s = 0.5 * (v + torch.roll(v, 1, dims=-2))
        return uc_e, uc_w, vc_n, vc_s

    # ---- Face-velocity divergence ----

    def _divergence(
        self,
        uc_e: torch.Tensor, uc_w: torch.Tensor,
        vc_n: torch.Tensor, vc_s: torch.Tensor,
    ) -> torch.Tensor:
        return (uc_e - uc_w) / self.dx + (vc_n - vc_s) / self.dy

    # ---- 2nd-order upwind convection (periodic) ----

    def _upwind_convection(
        self,
        f: torch.Tensor,
        uc_e: torch.Tensor, uc_w: torch.Tensor,
        vc_n: torch.Tensor, vc_s: torch.Tensor,
    ) -> torch.Tensor:
        """Conservative upwind convection: div(f*u) with 2nd-order upwind face values."""
        f_ip1 = torch.roll(f, -1, dims=-1)
        f_im1 = torch.roll(f, 1, dims=-1)
        f_ip2 = torch.roll(f, -2, dims=-1)
        f_im2 = torch.roll(f, 2, dims=-1)

        f_jp1 = torch.roll(f, -1, dims=-2)
        f_jm1 = torch.roll(f, 1, dims=-2)
        f_jp2 = torch.roll(f, -2, dims=-2)
        f_jm2 = torch.roll(f, 2, dims=-2)

        Fe_pos = 1.5 * f - 0.5 * f_im1
        Fe_neg = 1.5 * f_ip1 - 0.5 * f_ip2
        Fe = torch.where(uc_e >= 0, Fe_pos, Fe_neg)

        Fw_pos = 1.5 * f_im1 - 0.5 * f_im2
        Fw_neg = 1.5 * f - 0.5 * f_ip1
        Fw = torch.where(uc_w >= 0, Fw_pos, Fw_neg)

        Fn_pos = 1.5 * f - 0.5 * f_jm1
        Fn_neg = 1.5 * f_jp1 - 0.5 * f_jp2
        Fn = torch.where(vc_n >= 0, Fn_pos, Fn_neg)

        Fs_pos = 1.5 * f_jm1 - 0.5 * f_jm2
        Fs_neg = 1.5 * f - 0.5 * f_jp1
        Fs = torch.where(vc_s >= 0, Fs_pos, Fs_neg)

        conv = (uc_e * Fe - uc_w * Fw) / self.dx + (vc_n * Fn - vc_s * Fs) / self.dy
        return conv

    # ---- Laplacian (periodic) ----

    def _laplacian_2d(self, f: torch.Tensor) -> torch.Tensor:
        """2nd-order central Laplacian with periodic BC."""
        lap_x = (torch.roll(f, -1, dims=-1) - 2 * f + torch.roll(f, 1, dims=-1)) / (self.dx ** 2)
        lap_y = (torch.roll(f, -1, dims=-2) - 2 * f + torch.roll(f, 1, dims=-2)) / (self.dy ** 2)
        return lap_x + lap_y

    # ---- 4th-order derivatives for irrotational constraint ----

    def _dx_4th(self, f: torch.Tensor) -> torch.Tensor:
        """4th-order central x-derivative (periodic)."""
        return (-torch.roll(f, -2, dims=-1) + 8 * torch.roll(f, -1, dims=-1)
                - 8 * torch.roll(f, 1, dims=-1) + torch.roll(f, 2, dims=-1)) / (12 * self.dx)

    def _dy_4th(self, f: torch.Tensor) -> torch.Tensor:
        """4th-order central y-derivative (periodic)."""
        return (-torch.roll(f, -2, dims=-2) + 8 * torch.roll(f, -1, dims=-2)
                - 8 * torch.roll(f, 1, dims=-2) + torch.roll(f, 2, dims=-2)) / (12 * self.dy)

    # ---- Per-timestep scale helper ----

    def _get_per_t_scale(
        self, eq_name: str, T_res: int, t_offset: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Get per-timestep scale for normalization."""
        buf_name = f'scales_t_{eq_name}'
        if not hasattr(self, buf_name):
            return None
        scales_full = getattr(self, buf_name)

        if t_offset is not None:
            B = t_offset.shape[0]
            T_full = scales_full.shape[0]
            scales_batch = []
            for i in range(B):
                off = int(t_offset[i].item())
                idx_start = off
                idx_end = idx_start + T_res
                if idx_end <= T_full and idx_start >= 0:
                    scales_batch.append(scales_full[idx_start:idx_end])
                else:
                    idx_start = max(0, min(idx_start, T_full - T_res))
                    idx_end = idx_start + T_res
                    scales_batch.append(scales_full[idx_start:idx_end])
            scale_t = torch.stack(scales_batch, dim=0).mean(dim=0)
        else:
            scale_t = (scales_full[:T_res] if T_res <= scales_full.shape[0]
                       else scales_full.mean().expand(T_res))

        scale_t = scale_t.clamp(min=1e-8)
        return scale_t.view(1, T_res, 1, 1)

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        nu: torch.Tensor,
        reduction: str = 'mean',
        t_offset: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute PDE residual losses for 2D Burgers (Cole-Hopf).

        Args:
            u: x-velocity [B, T, H, W] (T includes prepended t0 frame)
            v: y-velocity [B, T, H, W]
            nu: per-sample viscosity [B] or scalar
            reduction: 'mean' (default)
            t_offset: optional global timestep offset [B]

        Returns:
            total_loss, {eq_name: loss_value}
        """
        losses: Dict[str, torch.Tensor] = {}

        # Broadcast nu to [B, 1, 1, 1]
        if nu.dim() == 0:
            nu_b = nu.view(1, 1, 1, 1)
        else:
            nu_b = nu.view(-1, 1, 1, 1)

        # Interior timesteps (2nd-order central time derivative)
        u_int = u[:, 1:-1]
        v_int = v[:, 1:-1]
        T_int = u_int.shape[1]

        # Face velocities
        uc_e, uc_w, vc_n, vc_s = self._face_velocities(u_int, v_int)

        # Divergence (for div correction)
        div_uv = self._divergence(uc_e, uc_w, vc_n, vc_s)

        # ---- Eq 1: u-momentum ----
        w_umom = self.eq_weights['u_momentum']
        if w_umom > 0:
            du_dt = (u[:, 2:] - u[:, :-2]) / (2 * self.dt)
            conv_u = self._upwind_convection(u_int, uc_e, uc_w, vc_n, vc_s)
            lap_u = self._laplacian_2d(u_int)

            R_u = du_dt + conv_u - u_int * div_uv - nu_b * lap_u

            if self.has_per_t_scales:
                scale_u_t = self._get_per_t_scale('u_momentum', T_int, t_offset)
                if scale_u_t is not None:
                    R_u_norm = R_u / scale_u_t
                    losses['u_momentum'] = w_umom * torch.mean(R_u_norm ** 2)
                else:
                    scale_u = self.eq_scales['u_momentum']
                    losses['u_momentum'] = w_umom * torch.mean(R_u ** 2) / (scale_u ** 2)
            else:
                scale_u = self.eq_scales['u_momentum']
                losses['u_momentum'] = w_umom * torch.mean(R_u ** 2) / (scale_u ** 2)
        else:
            losses['u_momentum'] = torch.tensor(0.0, device=u.device)

        # ---- Eq 2: v-momentum ----
        w_vmom = self.eq_weights['v_momentum']
        if w_vmom > 0:
            dv_dt = (v[:, 2:] - v[:, :-2]) / (2 * self.dt)
            conv_v = self._upwind_convection(v_int, uc_e, uc_w, vc_n, vc_s)
            lap_v = self._laplacian_2d(v_int)

            R_v = dv_dt + conv_v - v_int * div_uv - nu_b * lap_v

            if self.has_per_t_scales:
                scale_v_t = self._get_per_t_scale('v_momentum', T_int, t_offset)
                if scale_v_t is not None:
                    R_v_norm = R_v / scale_v_t
                    losses['v_momentum'] = w_vmom * torch.mean(R_v_norm ** 2)
                else:
                    scale_v = self.eq_scales['v_momentum']
                    losses['v_momentum'] = w_vmom * torch.mean(R_v ** 2) / (scale_v ** 2)
            else:
                scale_v = self.eq_scales['v_momentum']
                losses['v_momentum'] = w_vmom * torch.mean(R_v ** 2) / (scale_v ** 2)
        else:
            losses['v_momentum'] = torch.tensor(0.0, device=u.device)

        # ---- Eq 3: irrotational constraint ----
        w_irrot = self.eq_weights['irrotational']
        if w_irrot > 0:
            vx = self._dx_4th(v_int)
            uy = self._dy_4th(u_int)
            R_irrot = vx - uy

            if self.has_per_t_scales:
                scale_irrot_t = self._get_per_t_scale('irrotational', T_int, t_offset)
                if scale_irrot_t is not None:
                    R_irrot_norm = R_irrot / scale_irrot_t
                    losses['irrotational'] = w_irrot * torch.mean(R_irrot_norm ** 2)
                else:
                    scale_irrot = self.eq_scales['irrotational']
                    if scale_irrot < 1e-10:
                        scale_irrot = 1.0
                    losses['irrotational'] = w_irrot * torch.mean(R_irrot ** 2) / (scale_irrot ** 2)
            else:
                scale_irrot = self.eq_scales['irrotational']
                if scale_irrot < 1e-10:
                    scale_irrot = 1.0  # avoid div by zero for exact solutions
                losses['irrotational'] = w_irrot * torch.mean(R_irrot ** 2) / (scale_irrot ** 2)
        else:
            losses['irrotational'] = torch.tensor(0.0, device=u.device)

        total_loss = sum(losses.values())
        return total_loss, losses


# =============================================================================
# Advection-Diffusion 2D PDE Loss (periodic, 4th-order central FD)
# =============================================================================

class AdvDiff2DPDELoss(nn.Module):
    """
    2D Advection-Diffusion PDE Loss — periodic BC.

    Equation:
        u_t + a*u_x + b*u_y = nu * (u_xx + u_yy)

    Numerical method:
        - Spatial: 4th-order central difference (periodic via torch.roll)
        - Temporal: 2nd-order central difference

    Args:
        nx, ny: grid points (256, 256)
        dx, dy: grid spacing (2*pi/256)
        dt: time step (0.01)
        a, b: advection velocity components (fixed)
        nu: diffusion coefficient (fixed, used if per-sample nu not provided)
        eq_scales: per-equation normalization dict
        eq_weights: per-equation loss weights dict
        eq_scales_per_t: per-timestep normalization dict
    """

    DEFAULT_EQ_SCALES = {
        'advdiff': 1.0,
    }

    def __init__(
        self,
        nx: int = 256,
        ny: int = 256,
        dx: float = 2 * np.pi / 256,
        dy: float = 2 * np.pi / 256,
        dt: float = 0.01,
        a: float = 1.0,
        b: float = 0.5,
        nu: float = 0.05,
        eq_scales: Optional[Dict[str, float]] = None,
        eq_weights: Optional[Dict[str, float]] = None,
        eq_scales_per_t: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.a = a
        self.b = b
        self.nu = nu

        self.eq_scales = {**self.DEFAULT_EQ_SCALES, **(eq_scales or {})}
        self.eq_weights = {'advdiff': 1.0, **(eq_weights or {})}

        # Per-timestep scales
        self.use_per_t_scales = eq_scales_per_t is not None
        if eq_scales_per_t is not None:
            for name, scales in eq_scales_per_t.items():
                self.register_buffer(f'scales_t_{name}', scales.float())

    def _dx_4th(self, f: torch.Tensor) -> torch.Tensor:
        """4th-order central x-derivative (periodic, dims=-1)."""
        return (
            -torch.roll(f, -2, dims=-1) + 8 * torch.roll(f, -1, dims=-1)
            - 8 * torch.roll(f, 1, dims=-1) + torch.roll(f, 2, dims=-1)
        ) / (12 * self.dx)

    def _dy_4th(self, f: torch.Tensor) -> torch.Tensor:
        """4th-order central y-derivative (periodic, dims=-2)."""
        return (
            -torch.roll(f, -2, dims=-2) + 8 * torch.roll(f, -1, dims=-2)
            - 8 * torch.roll(f, 1, dims=-2) + torch.roll(f, 2, dims=-2)
        ) / (12 * self.dy)

    def _laplacian_4th(self, f: torch.Tensor) -> torch.Tensor:
        """4th-order central Laplacian (periodic)."""
        d2f_dx2 = (
            -torch.roll(f, -2, dims=-1) + 16 * torch.roll(f, -1, dims=-1)
            - 30 * f
            + 16 * torch.roll(f, 1, dims=-1) - torch.roll(f, 2, dims=-1)
        ) / (12 * self.dx ** 2)
        d2f_dy2 = (
            -torch.roll(f, -2, dims=-2) + 16 * torch.roll(f, -1, dims=-2)
            - 30 * f
            + 16 * torch.roll(f, 1, dims=-2) - torch.roll(f, 2, dims=-2)
        ) / (12 * self.dy ** 2)
        return d2f_dx2 + d2f_dy2

    def forward(
        self,
        u: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute advection-diffusion PDE residual loss.

        Args:
            u: scalar field [B, T, H, W] where T includes prepended t0 frame.
               Time derivative uses 2nd-order central at t=1..T-2.

        Returns:
            total_loss, {eq_name: loss_value}
        """
        # 2nd-order central time derivative
        du_dt = (u[:, 2:] - u[:, :-2]) / (2 * self.dt)

        # Spatial terms at interior timesteps
        u_int = u[:, 1:-1]

        du_dx = self._dx_4th(u_int)
        du_dy = self._dy_4th(u_int)
        lap_u = self._laplacian_4th(u_int)

        # Residual: u_t + a*u_x + b*u_y - nu*(u_xx + u_yy) = 0
        R = du_dt + self.a * du_dx + self.b * du_dy - self.nu * lap_u

        # Per-equation normalized loss
        w_advdiff = self.eq_weights['advdiff']
        T_res = R.shape[1]

        if self.use_per_t_scales and hasattr(self, 'scales_t_advdiff'):
            s_t = self.scales_t_advdiff[:T_res].clamp(min=1e-8)
            R_norm = R / s_t[None, :, None, None]
            loss_advdiff = w_advdiff * torch.mean(R_norm ** 2)
        else:
            s = self.eq_scales['advdiff']
            loss_advdiff = w_advdiff * torch.mean(R ** 2) / (s ** 2)

        return loss_advdiff, {'advdiff': loss_advdiff, 'total': loss_advdiff}


# =============================================================================
# Anisotropic Diffusion 2D PDE Loss (periodic, full tensor diffusion)
# =============================================================================

class AnisoDiff2DPDELoss(nn.Module):
    """
    2D Anisotropic Diffusion PDE Loss — periodic BC.

    Equation:
        u_t = div(A * grad(u))
        u_t = A_xx * u_xx + (A_xy + A_yx) * u_xy + A_yy * u_yy

    Default APEBench coefficients:
        A = [[0.001, 0.0005],
             [0.0005, 0.002]]

    Domain: [0, L)^2 periodic, L=1.0, 160x160, dt=0.1

    Numerical method:
        - Spatial: 4th-order central difference (periodic via torch.roll)
        - Temporal: 2nd-order central difference
        - Cross derivative: chain 4th-order d/dx then d/dy

    Args:
        nx, ny: grid points (160, 160)
        dx, dy: grid spacing (1/160)
        dt: time step (0.1)
        A_xx, A_xy, A_yx, A_yy: diffusion tensor components
        eq_scales: per-equation normalization dict
        eq_weights: per-equation loss weights dict
        eq_scales_per_t: per-timestep normalization dict
    """

    DEFAULT_EQ_SCALES = {
        'aniso_diff': 1.0,
    }

    def __init__(
        self,
        nx: int = 160,
        ny: int = 160,
        dx: float = 1.0 / 160,
        dy: float = 1.0 / 160,
        dt: float = 0.1,
        A_xx: float = 0.001,
        A_xy: float = 0.0005,
        A_yx: float = 0.0005,
        A_yy: float = 0.002,
        eq_scales: Optional[Dict[str, float]] = None,
        eq_weights: Optional[Dict[str, float]] = None,
        eq_scales_per_t: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.A_xx = A_xx
        self.A_xy = A_xy
        self.A_yx = A_yx
        self.A_yy = A_yy

        self.eq_scales = {**self.DEFAULT_EQ_SCALES, **(eq_scales or {})}
        self.eq_weights = {'aniso_diff': 1.0, **(eq_weights or {})}

        # Per-timestep scales
        self.use_per_t_scales = eq_scales_per_t is not None
        if eq_scales_per_t is not None:
            for name, scales in eq_scales_per_t.items():
                self.register_buffer(f'scales_t_{name}', scales.float())

    def _dx_4th(self, f: torch.Tensor) -> torch.Tensor:
        """4th-order central x-derivative (periodic, dims=-1)."""
        return (
            -torch.roll(f, -2, dims=-1) + 8 * torch.roll(f, -1, dims=-1)
            - 8 * torch.roll(f, 1, dims=-1) + torch.roll(f, 2, dims=-1)
        ) / (12 * self.dx)

    def _dy_4th(self, f: torch.Tensor) -> torch.Tensor:
        """4th-order central y-derivative (periodic, dims=-2)."""
        return (
            -torch.roll(f, -2, dims=-2) + 8 * torch.roll(f, -1, dims=-2)
            - 8 * torch.roll(f, 1, dims=-2) + torch.roll(f, 2, dims=-2)
        ) / (12 * self.dy)

    def _d2x_4th(self, f: torch.Tensor) -> torch.Tensor:
        """4th-order central d^2/dx^2 (periodic, dims=-1)."""
        return (
            -torch.roll(f, -2, dims=-1) + 16 * torch.roll(f, -1, dims=-1)
            - 30 * f
            + 16 * torch.roll(f, 1, dims=-1) - torch.roll(f, 2, dims=-1)
        ) / (12 * self.dx ** 2)

    def _d2y_4th(self, f: torch.Tensor) -> torch.Tensor:
        """4th-order central d^2/dy^2 (periodic, dims=-2)."""
        return (
            -torch.roll(f, -2, dims=-2) + 16 * torch.roll(f, -1, dims=-2)
            - 30 * f
            + 16 * torch.roll(f, 1, dims=-2) - torch.roll(f, 2, dims=-2)
        ) / (12 * self.dy ** 2)

    def _dxy_4th(self, f: torch.Tensor) -> torch.Tensor:
        """4th-order central cross derivative d^2/(dx dy) via chain rule."""
        return self._dy_4th(self._dx_4th(f))

    def forward(
        self,
        u: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute anisotropic diffusion PDE residual loss.

        Args:
            u: scalar field [B, T, H, W] where T includes prepended t0 frame.
               Time derivative uses 2nd-order central at t=1..T-2.

        Returns:
            total_loss, {eq_name: loss_value}
        """
        # 2nd-order central time derivative
        du_dt = (u[:, 2:] - u[:, :-2]) / (2 * self.dt)

        # Spatial terms at interior timesteps
        u_int = u[:, 1:-1]

        u_xx = self._d2x_4th(u_int)
        u_yy = self._d2y_4th(u_int)
        u_xy = self._dxy_4th(u_int)

        # Residual: u_t - A_xx*u_xx - (A_xy + A_yx)*u_xy - A_yy*u_yy = 0
        R = du_dt - self.A_xx * u_xx - (self.A_xy + self.A_yx) * u_xy - self.A_yy * u_yy

        # Per-equation normalized loss
        w_aniso = self.eq_weights['aniso_diff']
        T_res = R.shape[1]

        if self.use_per_t_scales and hasattr(self, 'scales_t_aniso_diff'):
            s_t = self.scales_t_aniso_diff[:T_res].clamp(min=1e-8)
            R_norm = R / s_t[None, :, None, None]
            loss_aniso = w_aniso * torch.mean(R_norm ** 2)
        else:
            s = self.eq_scales['aniso_diff']
            loss_aniso = w_aniso * torch.mean(R ** 2) / (s ** 2)

        return loss_aniso, {'aniso_diff': loss_aniso, 'total': loss_aniso}


# =============================================================================
# Wave 2D PDE Loss (periodic, first-order system)
# =============================================================================

class Wave2DPDELoss(nn.Module):
    """
    2D Wave Equation PDE Loss — periodic BC.

    First-order system:
        Eq 1: u_t - w = 0
        Eq 2: w_t - c^2(u_xx + u_yy) = 0

    Numerical method:
        - Spatial: 4th-order central FD Laplacian (periodic, torch.roll)
        - Temporal: 2nd-order central difference

    The wave speed c varies per sample and is passed as a [B] tensor to forward().

    Grid: boundary-exclusive [0, 2pi), dx = 2pi/256.

    Args:
        dx, dy: grid spacing (default 2pi/256)
        dt: time step (default 0.01)
        eq_scales: per-equation normalization scales
        eq_weights: per-equation loss weights
        eq_scales_per_t: per-timestep normalization dict
    """

    DEFAULT_EQ_SCALES = {
        'u_equation': 1.0,
        'w_equation': 1.0,
    }

    def __init__(
        self,
        dx: float = 2 * np.pi / 256,
        dy: float = 2 * np.pi / 256,
        dt: float = 0.01,
        eq_scales: Optional[Dict[str, float]] = None,
        eq_weights: Optional[Dict[str, float]] = None,
        eq_scales_per_t: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.eq_scales = {**self.DEFAULT_EQ_SCALES, **(eq_scales or {})}
        self.eq_weights = {'u_equation': 1.0, 'w_equation': 1.0, **(eq_weights or {})}

        # Per-timestep scales
        self.use_per_t_scales = eq_scales_per_t is not None
        if eq_scales_per_t is not None:
            for name, scales in eq_scales_per_t.items():
                self.register_buffer(f'scales_t_{name}', scales.float())

    def _laplacian_4th(self, f: torch.Tensor) -> torch.Tensor:
        """4th-order central FD Laplacian (periodic BC, torch.roll).

        f shape: [B, T, H, W]
        """
        d2f_dx2 = (
            -torch.roll(f, -2, dims=-2) + 16 * torch.roll(f, -1, dims=-2)
            - 30 * f
            + 16 * torch.roll(f, 1, dims=-2) - torch.roll(f, 2, dims=-2)
        ) / (12 * self.dx**2)
        d2f_dy2 = (
            -torch.roll(f, -2, dims=-1) + 16 * torch.roll(f, -1, dims=-1)
            - 30 * f
            + 16 * torch.roll(f, 1, dims=-1) - torch.roll(f, 2, dims=-1)
        ) / (12 * self.dy**2)
        return d2f_dx2 + d2f_dy2

    def forward(
        self,
        u: torch.Tensor,
        w: torch.Tensor,
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute Wave Equation PDE residual loss.

        Args:
            u: displacement field [B, T, H, W]
            w: velocity field (u_t) [B, T, H, W]
            c: wave speed [B] (per-sample)

        Returns:
            total_loss, losses_dict
        """
        # 2nd-order central time difference
        du_dt = (u[:, 2:] - u[:, :-2]) / (2 * self.dt)  # [B, T-2, H, W]
        dw_dt = (w[:, 2:] - w[:, :-2]) / (2 * self.dt)

        # Spatial terms at mid-frame
        u_mid = u[:, 1:-1]  # [B, T-2, H, W]
        w_mid = w[:, 1:-1]

        # Laplacian of u
        lap_u = self._laplacian_4th(u_mid)

        # c^2 per sample: [B] -> [B, 1, 1, 1]
        c2 = (c ** 2).view(-1, 1, 1, 1)

        # Eq 1: u_t - w = 0
        R_u = du_dt - w_mid

        # Eq 2: w_t - c^2 * lap(u) = 0
        R_w = dw_dt - c2 * lap_u

        # Loss computation
        w_u = self.eq_weights['u_equation']
        w_w = self.eq_weights['w_equation']

        if self.use_per_t_scales:
            T_res = R_u.shape[1]

            if hasattr(self, 'scales_t_u_equation'):
                s_t_u = self.scales_t_u_equation[:T_res].clamp(min=1e-8)
                R_u_norm = R_u / s_t_u[None, :, None, None]
                loss_u = w_u * torch.mean(R_u_norm ** 2)
            else:
                s_u = self.eq_scales['u_equation']
                loss_u = w_u * torch.mean(R_u ** 2) / (s_u ** 2)

            if hasattr(self, 'scales_t_w_equation'):
                s_t_w = self.scales_t_w_equation[:T_res].clamp(min=1e-8)
                R_w_norm = R_w / s_t_w[None, :, None, None]
                loss_w = w_w * torch.mean(R_w_norm ** 2)
            else:
                s_w = self.eq_scales['w_equation']
                loss_w = w_w * torch.mean(R_w ** 2) / (s_w ** 2)
        else:
            s_u = self.eq_scales['u_equation']
            s_w = self.eq_scales['w_equation']
            loss_u = w_u * torch.mean(R_u ** 2) / (s_u ** 2)
            loss_w = w_w * torch.mean(R_w ** 2) / (s_w ** 2)

        total_loss = loss_u + loss_w

        return total_loss, {
            'u_equation': loss_u,
            'w_equation': loss_w,
            'total': total_loss,
        }


# =============================================================================
# 3D Advection PDE Loss (periodic)
# =============================================================================

class Advection3DPDELoss(nn.Module):
    """
    3D Advection PDE Loss — periodic BC.

    Equation:
        u_t + a*u_x + b*u_y + c*u_z = 0

    Numerical method:
        - Spatial: 4th-order central difference (periodic via torch.roll)
        - Temporal: 2nd-order central difference

    Data layout: [B, T, X, Y, Z] where x=dim=-3, y=dim=-2, z=dim=-1.

    Args:
        dx, dy, dz: grid spacing (2*pi/64)
        dt: time step (0.05)
        eq_scales: per-equation normalization dict
        eq_weights: per-equation loss weights dict
        eq_scales_per_t: per-timestep normalization dict
    """

    DEFAULT_EQ_SCALES = {
        'advection': 1.0,
    }

    def __init__(
        self,
        dx: float = 2 * np.pi / 64,
        dy: float = 2 * np.pi / 64,
        dz: float = 2 * np.pi / 64,
        dt: float = 0.05,
        eq_scales: Optional[Dict[str, float]] = None,
        eq_weights: Optional[Dict[str, float]] = None,
        eq_scales_per_t: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dt = dt

        self.eq_scales = {**self.DEFAULT_EQ_SCALES, **(eq_scales or {})}
        self.eq_weights = {'advection': 1.0, **(eq_weights or {})}

        # Per-timestep scales
        self.use_per_t_scales = eq_scales_per_t is not None
        if eq_scales_per_t is not None:
            for name, scales in eq_scales_per_t.items():
                self.register_buffer(f'scales_t_{name}', scales.float())

    def _dx_4th(self, f: torch.Tensor) -> torch.Tensor:
        """4th-order central x-derivative (periodic, dims=-3)."""
        return (
            -torch.roll(f, -2, dims=-3) + 8 * torch.roll(f, -1, dims=-3)
            - 8 * torch.roll(f, 1, dims=-3) + torch.roll(f, 2, dims=-3)
        ) / (12 * self.dx)

    def _dy_4th(self, f: torch.Tensor) -> torch.Tensor:
        """4th-order central y-derivative (periodic, dims=-2)."""
        return (
            -torch.roll(f, -2, dims=-2) + 8 * torch.roll(f, -1, dims=-2)
            - 8 * torch.roll(f, 1, dims=-2) + torch.roll(f, 2, dims=-2)
        ) / (12 * self.dy)

    def _dz_4th(self, f: torch.Tensor) -> torch.Tensor:
        """4th-order central z-derivative (periodic, dims=-1)."""
        return (
            -torch.roll(f, -2, dims=-1) + 8 * torch.roll(f, -1, dims=-1)
            - 8 * torch.roll(f, 1, dims=-1) + torch.roll(f, 2, dims=-1)
        ) / (12 * self.dz)

    def forward(
        self,
        u: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute 3D advection PDE residual loss.

        Args:
            u: scalar field [B, T, X, Y, Z] where T includes prepended t0 frame.
               Time derivative uses 2nd-order central at t=1..T-2.
            a: advection velocity x-component [B] (per-sample)
            b: advection velocity y-component [B] (per-sample)
            c: advection velocity z-component [B] (per-sample)

        Returns:
            total_loss, {eq_name: loss_value}
        """
        # 2nd-order central time derivative
        du_dt = (u[:, 2:] - u[:, :-2]) / (2 * self.dt)  # [B, T-2, X, Y, Z]

        # Spatial terms at interior timesteps
        u_int = u[:, 1:-1]  # [B, T-2, X, Y, Z]

        du_dx = self._dx_4th(u_int)
        du_dy = self._dy_4th(u_int)
        du_dz = self._dz_4th(u_int)

        # Per-sample velocities: [B] -> [B, 1, 1, 1, 1]
        a_v = a.view(-1, 1, 1, 1, 1)
        b_v = b.view(-1, 1, 1, 1, 1)
        c_v = c.view(-1, 1, 1, 1, 1)

        # Residual: u_t + a*u_x + b*u_y + c*u_z = 0
        R = du_dt + a_v * du_dx + b_v * du_dy + c_v * du_dz

        # Per-equation normalized loss
        w_adv = self.eq_weights['advection']
        T_res = R.shape[1]

        if self.use_per_t_scales and hasattr(self, 'scales_t_advection'):
            s_t = self.scales_t_advection[:T_res].clamp(min=1e-8)
            R_norm = R / s_t[None, :, None, None, None]
            loss_adv = w_adv * torch.mean(R_norm ** 2)
        else:
            s = self.eq_scales['advection']
            loss_adv = w_adv * torch.mean(R ** 2) / (s ** 2)

        return loss_adv, {'advection': loss_adv, 'total': loss_adv}


# =============================================================================
# 1D Burgers PDE Loss (periodic, per-sample nu)
# =============================================================================

class Burgers1DPDELoss(nn.Module):
    """
    1D Burgers PDE Loss — periodic BC.

    Equation:
        u_t + u * u_x - nu * u_xx = 0

    Numerical method:
        - Spatial: 4th-order central difference (periodic via torch.roll)
        - Temporal: 2nd-order central difference

    Data layout: [B, T, X] where x=dim=-1.

    Args:
        dx: grid spacing
        dt: time step
        eq_scales: per-equation normalization dict
        eq_weights: per-equation loss weights dict
        eq_scales_per_t: per-timestep normalization dict
    """

    DEFAULT_EQ_SCALES = {
        'momentum': 1.0,
    }

    def __init__(
        self,
        dx: float = 2 * np.pi / 256,
        dt: float = 0.01,
        eq_scales: Optional[Dict[str, float]] = None,
        eq_weights: Optional[Dict[str, float]] = None,
        eq_scales_per_t: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.dx = dx
        self.dt = dt

        self.eq_scales = {**self.DEFAULT_EQ_SCALES, **(eq_scales or {})}
        self.eq_weights = {'momentum': 1.0, **(eq_weights or {})}

        self.use_per_t_scales = eq_scales_per_t is not None
        if eq_scales_per_t is not None:
            for name, scales in eq_scales_per_t.items():
                self.register_buffer(f'scales_t_{name}', scales.float())

    def _dx_4th(self, f: torch.Tensor) -> torch.Tensor:
        """4th-order central x-derivative (periodic, dim=-1)."""
        return (
            -torch.roll(f, -2, dims=-1) + 8 * torch.roll(f, -1, dims=-1)
            - 8 * torch.roll(f, 1, dims=-1) + torch.roll(f, 2, dims=-1)
        ) / (12 * self.dx)

    def _d2x_4th(self, f: torch.Tensor) -> torch.Tensor:
        """4th-order central 2nd x-derivative (periodic, dim=-1)."""
        return (
            -torch.roll(f, -2, dims=-1) + 16 * torch.roll(f, -1, dims=-1)
            - 30 * f
            + 16 * torch.roll(f, 1, dims=-1) - torch.roll(f, 2, dims=-1)
        ) / (12 * self.dx ** 2)

    def forward(
        self,
        u: torch.Tensor,
        nu: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute 1D Burgers PDE residual loss.

        Args:
            u: velocity field [B, T, X] where T includes prepended t0 frame.
               Time derivative uses 2nd-order central at t=1..T-2.
            nu: viscosity [B] (per-sample)

        Returns:
            total_loss, {eq_name: loss_value}
        """
        # 2nd-order central time derivative
        du_dt = (u[:, 2:] - u[:, :-2]) / (2 * self.dt)  # [B, T-2, X]

        # Spatial terms at interior timesteps
        u_int = u[:, 1:-1]  # [B, T-2, X]

        du_dx = self._dx_4th(u_int)
        d2u_dx2 = self._d2x_4th(u_int)

        # Per-sample nu: [B] -> [B, 1, 1]
        nu_v = nu.view(-1, 1, 1)

        # Residual: u_t + u * u_x - nu * u_xx = 0
        R = du_dt + u_int * du_dx - nu_v * d2u_dx2

        # Per-equation normalized loss
        w_mom = self.eq_weights['momentum']
        T_res = R.shape[1]

        if self.use_per_t_scales and hasattr(self, 'scales_t_momentum'):
            s_t = self.scales_t_momentum[:T_res].clamp(min=1e-8)
            R_norm = R / s_t[None, :, None]
            loss_mom = w_mom * torch.mean(R_norm ** 2)
        else:
            s = self.eq_scales['momentum']
            loss_mom = w_mom * torch.mean(R ** 2) / (s ** 2)

        return loss_mom, {'momentum': loss_mom, 'total': loss_mom}


# =============================================================================
# 1D Advection PDE Loss (periodic, per-sample velocity a)
# =============================================================================

class Advection1DPDELoss(nn.Module):
    """
    1D Advection PDE Loss — periodic BC.

    Equation:
        u_t + a * u_x = 0

    Numerical method:
        - Spatial: 4th-order central difference (same as Burgers1DPDELoss)
        - Temporal: 2nd-order central difference

    Data layout: [B, T, X] where x=dim=-1.

    Args:
        dx: grid spacing
        dt: time step
        eq_scales: per-equation normalization dict
        eq_weights: per-equation loss weights dict
        eq_scales_per_t: per-timestep normalization dict
    """

    DEFAULT_EQ_SCALES = {
        'advection': 1.0,
    }

    def __init__(
        self,
        dx: float = 2 * np.pi / 256,
        dt: float = 0.01,
        eq_scales: Optional[Dict[str, float]] = None,
        eq_weights: Optional[Dict[str, float]] = None,
        eq_scales_per_t: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.dx = dx
        self.dt = dt

        self.eq_scales = {**self.DEFAULT_EQ_SCALES, **(eq_scales or {})}
        self.eq_weights = {'advection': 1.0, **(eq_weights or {})}

        self.use_per_t_scales = eq_scales_per_t is not None
        if eq_scales_per_t is not None:
            for name, scales in eq_scales_per_t.items():
                self.register_buffer(f'scales_t_{name}', scales.float())

    def _dx_4th(self, f: torch.Tensor) -> torch.Tensor:
        """4th-order central d/dx, periodic (same as Burgers1DPDELoss)."""
        fE  = torch.roll(f, -1, dims=-1)
        fW  = torch.roll(f,  1, dims=-1)
        fEE = torch.roll(f, -2, dims=-1)
        fWW = torch.roll(f,  2, dims=-1)
        return (-fEE + 8 * fE - 8 * fW + fWW) / (12 * self.dx)

    def forward(
        self,
        u: torch.Tensor,
        a: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute 1D advection PDE residual loss.

        Args:
            u: scalar field [B, T, X] where T includes prepended t0 frame.
               Time derivative uses 2nd-order central at t=1..T-2.
            a: advection velocity [B] (per-sample)

        Returns:
            total_loss, {eq_name: loss_value}
        """
        # 2nd-order central time derivative
        du_dt = (u[:, 2:] - u[:, :-2]) / (2 * self.dt)  # [B, T-2, X]

        # Spatial terms at interior timesteps
        u_int = u[:, 1:-1]  # [B, T-2, X]

        # Per-sample velocity: [B] -> [B, 1, 1]
        a_v = a.view(-1, 1, 1)

        # Residual: u_t + a * u_x = 0
        R = du_dt + a_v * self._dx_4th(u_int)

        # Per-equation normalized loss
        w_adv = self.eq_weights['advection']
        T_res = R.shape[1]

        if self.use_per_t_scales and hasattr(self, 'scales_t_advection'):
            s_t = self.scales_t_advection[:T_res].clamp(min=1e-8)
            R_norm = R / s_t[None, :, None]
            loss_adv = w_adv * torch.mean(R_norm ** 2)
        else:
            s = self.eq_scales['advection']
            loss_adv = w_adv * torch.mean(R ** 2) / (s ** 2)

        return loss_adv, {'advection': loss_adv, 'total': loss_adv}


# =============================================================================
# 1D Heat PDE Loss (periodic, per-sample alpha)
# =============================================================================

class Heat1DPDELoss(nn.Module):
    """
    1D Heat PDE Loss — periodic BC.

    Equation:
        u_t - alpha * u_xx = 0

    Numerical method:
        - Spatial: 4th-order central difference (periodic via torch.roll)
        - Temporal: 2nd-order central difference

    Data layout: [B, T, X] where x=dim=-1.

    Args:
        dx: grid spacing
        dt: time step
        eq_scales: per-equation normalization dict
        eq_weights: per-equation loss weights dict
        eq_scales_per_t: per-timestep normalization dict
    """

    DEFAULT_EQ_SCALES = {
        'heat': 1.0,
    }

    def __init__(
        self,
        dx: float = 2 * np.pi / 256,
        dt: float = 0.01,
        eq_scales: Optional[Dict[str, float]] = None,
        eq_weights: Optional[Dict[str, float]] = None,
        eq_scales_per_t: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.dx = dx
        self.dt = dt

        self.eq_scales = {**self.DEFAULT_EQ_SCALES, **(eq_scales or {})}
        self.eq_weights = {'heat': 1.0, **(eq_weights or {})}

        self.use_per_t_scales = eq_scales_per_t is not None
        if eq_scales_per_t is not None:
            for name, scales in eq_scales_per_t.items():
                self.register_buffer(f'scales_t_{name}', scales.float())

    def _d2x_4th(self, f: torch.Tensor) -> torch.Tensor:
        """4th-order central 2nd x-derivative (periodic, dim=-1)."""
        return (
            -torch.roll(f, -2, dims=-1) + 16 * torch.roll(f, -1, dims=-1)
            - 30 * f
            + 16 * torch.roll(f, 1, dims=-1) - torch.roll(f, 2, dims=-1)
        ) / (12 * self.dx ** 2)

    def forward(
        self,
        u: torch.Tensor,
        alpha: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute 1D heat PDE residual loss.

        Args:
            u: temperature field [B, T, X] where T includes prepended t0 frame.
               Time derivative uses 2nd-order central at t=1..T-2.
            alpha: thermal diffusivity [B] (per-sample)

        Returns:
            total_loss, {eq_name: loss_value}
        """
        # 2nd-order central time derivative
        du_dt = (u[:, 2:] - u[:, :-2]) / (2 * self.dt)  # [B, T-2, X]

        # Spatial terms at interior timesteps
        u_int = u[:, 1:-1]  # [B, T-2, X]

        d2u_dx2 = self._d2x_4th(u_int)

        # Per-sample alpha: [B] -> [B, 1, 1]
        alpha_v = alpha.view(-1, 1, 1)

        # Residual: u_t - alpha * u_xx = 0
        R = du_dt - alpha_v * d2u_dx2

        # Per-equation normalized loss
        w_heat = self.eq_weights['heat']
        T_res = R.shape[1]

        if self.use_per_t_scales and hasattr(self, 'scales_t_heat'):
            s_t = self.scales_t_heat[:T_res].clamp(min=1e-8)
            R_norm = R / s_t[None, :, None]
            loss_heat = w_heat * torch.mean(R_norm ** 2)
        else:
            s = self.eq_scales['heat']
            loss_heat = w_heat * torch.mean(R ** 2) / (s ** 2)

        return loss_heat, {'heat': loss_heat, 'total': loss_heat}


# =============================================================================
# NLS 1D PDE Loss (focusing nonlinear Schrödinger, bright soliton)
# =============================================================================


# =============================================================================
# APEBench 2D Burgers PDE Loss
# =============================================================================

class APEBenchBurgers2DPDELoss(nn.Module):
    """
    APEBench 2D Burgers PDE Loss — normalized coefficients.

    PDE: u_t = alpha_2 * lap(u) + delta_conv * (u*u_x + v*u_y)
         v_t = alpha_2 * lap(v) + delta_conv * (u*v_x + v*v_y)

    All coefficients are time-independent and fixed across samples.
    Uses 4th-order central FD (spatial) + 2nd-order central (temporal).
    Periodic BC via torch.roll.

    Data layout: [B, T, H, W] where H=y (dim=-2), W=x (dim=-1).
    """

    DEFAULT_EQ_SCALES = {
        'u_momentum': 1.0,
        'v_momentum': 1.0,
    }

    def __init__(
        self,
        nx: int = 160,
        ny: int = 160,
        Lx: float = 1.0,
        Ly: float = 1.0,
        dt: float = 1.0,
        alpha_2: float = 1.4648e-5,
        delta_conv: float = -2.3438e-3,
        eq_scales: Optional[Dict[str, float]] = None,
        eq_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.dt = dt
        self.alpha_2 = alpha_2
        self.delta_conv = delta_conv

        self.eq_scales = dict(self.DEFAULT_EQ_SCALES)
        if eq_scales is not None:
            self.eq_scales.update(eq_scales)

        self.eq_weights = {'u_momentum': 1.0, 'v_momentum': 1.0}
        if eq_weights is not None:
            self.eq_weights.update(eq_weights)

    def _grad_4th(self, f: torch.Tensor, dim: int, d: float) -> torch.Tensor:
        """4th-order central difference along dim."""
        return (-torch.roll(f, -2, dims=dim) + 8 * torch.roll(f, -1, dims=dim)
                - 8 * torch.roll(f, 1, dims=dim) + torch.roll(f, 2, dims=dim)) / (12 * d)

    def _laplacian(self, f: torch.Tensor) -> torch.Tensor:
        """2nd-order central Laplacian."""
        lap_x = (torch.roll(f, -1, dims=-1) - 2 * f + torch.roll(f, 1, dims=-1)) / (self.dx ** 2)
        lap_y = (torch.roll(f, -1, dims=-2) - 2 * f + torch.roll(f, 1, dims=-2)) / (self.dy ** 2)
        return lap_x + lap_y

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            u, v: [B, T, H, W] — includes prepended t0 frame.
        Returns:
            total_loss, {eq_name: loss_value}
        """
        # Interior timesteps
        u_int = u[:, 1:-1]
        v_int = v[:, 1:-1]

        # Time derivatives (2nd-order central)
        du_dt = (u[:, 2:] - u[:, :-2]) / (2 * self.dt)
        dv_dt = (v[:, 2:] - v[:, :-2]) / (2 * self.dt)

        # Spatial derivatives (4th-order central)
        du_dx = self._grad_4th(u_int, dim=-1, d=self.dx)
        du_dy = self._grad_4th(u_int, dim=-2, d=self.dy)
        dv_dx = self._grad_4th(v_int, dim=-1, d=self.dx)
        dv_dy = self._grad_4th(v_int, dim=-2, d=self.dy)

        # Laplacians
        lap_u = self._laplacian(u_int)
        lap_v = self._laplacian(v_int)

        # Residuals: u_t = alpha_2*lap(u) + delta_conv*(u*u_x + v*u_y)
        R_u = du_dt - self.alpha_2 * lap_u - self.delta_conv * (u_int * du_dx + v_int * du_dy)
        R_v = dv_dt - self.alpha_2 * lap_v - self.delta_conv * (u_int * dv_dx + v_int * dv_dy)

        losses = {}
        scale_u = self.eq_scales['u_momentum']
        scale_v = self.eq_scales['v_momentum']
        w_u = self.eq_weights['u_momentum']
        w_v = self.eq_weights['v_momentum']

        losses['u_momentum'] = w_u * torch.mean(R_u ** 2) / (scale_u ** 2)
        losses['v_momentum'] = w_v * torch.mean(R_v ** 2) / (scale_v ** 2)

        total_loss = sum(losses.values())
        return total_loss, losses


# =============================================================================
# APEBench 1D Burgers PDE Loss
# =============================================================================

class APEBenchBurgers1DPDELoss(nn.Module):
    """
    APEBench 1D Burgers PDE Loss — normalized coefficients.

    PDE: u_t = alpha_2 * u_xx + delta_conv * u * u_x

    All coefficients are time-independent and fixed across samples.
    Uses 4th-order central FD (spatial) + 2nd-order central (temporal).
    Periodic BC via torch.roll.

    Data layout: [B, T, X] where x = dim=-1.
    """

    DEFAULT_EQ_SCALES = {
        'momentum': 1.0,
    }

    def __init__(
        self,
        nx: int = 160,
        Lx: float = 1.0,
        dt: float = 1.0,
        alpha_2: float = 2.9297e-5,
        delta_conv: float = -4.6875e-3,
        eq_scales: Optional[Dict[str, float]] = None,
        eq_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.dx = Lx / nx
        self.dt = dt
        self.alpha_2 = alpha_2
        self.delta_conv = delta_conv

        self.eq_scales = dict(self.DEFAULT_EQ_SCALES)
        if eq_scales is not None:
            self.eq_scales.update(eq_scales)

        self.eq_weights = {'momentum': 1.0}
        if eq_weights is not None:
            self.eq_weights.update(eq_weights)

    def forward(
        self,
        u: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            u: [B, T, X] — includes prepended t0 frame.
        Returns:
            total_loss, {eq_name: loss_value}
        """
        u_int = u[:, 1:-1]

        # Time derivative
        du_dt = (u[:, 2:] - u[:, :-2]) / (2 * self.dt)

        # 4th-order central u_x
        du_dx = (-torch.roll(u_int, -2, dims=-1) + 8 * torch.roll(u_int, -1, dims=-1)
                 - 8 * torch.roll(u_int, 1, dims=-1) + torch.roll(u_int, 2, dims=-1)) / (12 * self.dx)

        # 2nd-order central u_xx
        u_xx = (torch.roll(u_int, -1, dims=-1) - 2 * u_int + torch.roll(u_int, 1, dims=-1)) / (self.dx ** 2)

        # Residual: u_t = alpha_2*u_xx + delta_conv*u*u_x
        R = du_dt - self.alpha_2 * u_xx - self.delta_conv * u_int * du_dx

        scale = self.eq_scales['momentum']
        w = self.eq_weights['momentum']
        losses = {'momentum': w * torch.mean(R ** 2) / (scale ** 2)}

        total_loss = losses['momentum']
        return total_loss, losses


# =============================================================================
# APEBench 3D Burgers PDE Loss
# =============================================================================

class APEBenchBurgers3DPDELoss(nn.Module):
    """
    APEBench 3D Burgers PDE Loss — normalized coefficients, conservative form.

    PDE: u_t = -beta_1 * 1/2 * div(u tensor u) + alpha_2 * lap(u)
    In components:
        u_t = -beta_1/2 * [d(uu)/dx + d(uv)/dy + d(uw)/dz] + alpha_2 * lap(u)
        v_t = -beta_1/2 * [d(vu)/dx + d(vv)/dy + d(vw)/dz] + alpha_2 * lap(v)
        w_t = -beta_1/2 * [d(wu)/dx + d(wv)/dy + d(ww)/dz] + alpha_2 * lap(w)

    Data layout: [B, T, X, Y, Z] with x=dim=-3, y=dim=-2, z=dim=-1.
    """

    DEFAULT_EQ_SCALES = {
        'u_momentum': 1.0,
        'v_momentum': 1.0,
        'w_momentum': 1.0,
    }

    def __init__(
        self,
        nx: int = 32,
        Lx: float = 1.0,
        dt: float = 1.0,
        alpha_2: float = 2.4414e-4,
        beta_1: float = 1.5625e-2,
        eq_scales: Optional[Dict[str, float]] = None,
        eq_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.dx = Lx / nx
        self.dt = dt
        self.alpha_2 = alpha_2
        self.beta_1 = beta_1

        self.eq_scales = dict(self.DEFAULT_EQ_SCALES)
        if eq_scales is not None:
            self.eq_scales.update(eq_scales)

        self.eq_weights = {'u_momentum': 1.0, 'v_momentum': 1.0, 'w_momentum': 1.0}
        if eq_weights is not None:
            self.eq_weights.update(eq_weights)

    def _grad_4th(self, f: torch.Tensor, dim: int) -> torch.Tensor:
        """4th-order central difference along dim."""
        return (-torch.roll(f, -2, dims=dim) + 8 * torch.roll(f, -1, dims=dim)
                - 8 * torch.roll(f, 1, dims=dim) + torch.roll(f, 2, dims=dim)) / (12 * self.dx)

    def _laplacian_3d(self, f: torch.Tensor) -> torch.Tensor:
        """2nd-order central Laplacian in 3D."""
        lap = torch.zeros_like(f)
        for dim in [-3, -2, -1]:
            lap = lap + (torch.roll(f, -1, dims=dim) - 2 * f + torch.roll(f, 1, dims=dim)) / (self.dx ** 2)
        return lap

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            u, v, w: [B, T, X, Y, Z] — includes prepended t0 frame.
        """
        u_int = u[:, 1:-1]
        v_int = v[:, 1:-1]
        w_int = w[:, 1:-1]

        # Time derivatives
        du_dt = (u[:, 2:] - u[:, :-2]) / (2 * self.dt)
        dv_dt = (v[:, 2:] - v[:, :-2]) / (2 * self.dt)
        dw_dt = (w[:, 2:] - w[:, :-2]) / (2 * self.dt)

        # Conservative convection: 1/2 * div(phi tensor u)
        conv_u = 0.5 * (self._grad_4th(u_int * u_int, -3) +
                        self._grad_4th(u_int * v_int, -2) +
                        self._grad_4th(u_int * w_int, -1))
        conv_v = 0.5 * (self._grad_4th(v_int * u_int, -3) +
                        self._grad_4th(v_int * v_int, -2) +
                        self._grad_4th(v_int * w_int, -1))
        conv_w = 0.5 * (self._grad_4th(w_int * u_int, -3) +
                        self._grad_4th(w_int * v_int, -2) +
                        self._grad_4th(w_int * w_int, -1))

        # Diffusion
        lap_u = self._laplacian_3d(u_int)
        lap_v = self._laplacian_3d(v_int)
        lap_w = self._laplacian_3d(w_int)

        # Residuals: du/dt + beta_1*conv - alpha_2*lap = 0
        R_u = du_dt + self.beta_1 * conv_u - self.alpha_2 * lap_u
        R_v = dv_dt + self.beta_1 * conv_v - self.alpha_2 * lap_v
        R_w = dw_dt + self.beta_1 * conv_w - self.alpha_2 * lap_w

        losses = {}
        for name, R in [('u_momentum', R_u), ('v_momentum', R_v), ('w_momentum', R_w)]:
            scale = self.eq_scales[name]
            w_eq = self.eq_weights[name]
            losses[name] = w_eq * torch.mean(R ** 2) / (scale ** 2)

        total_loss = sum(losses.values())
        return total_loss, losses


# =============================================================================
# APEBench 1D Kuramoto-Sivashinsky PDE Loss
# =============================================================================

class KS1DPDELoss(nn.Module):
    """
    APEBench 1D Kuramoto-Sivashinsky PDE Loss — normalized coefficients.

    PDE (on normalized domain [0,1], dt=1):
        u_t = alpha_2 * u_xx + alpha_4 * u_xxxx + beta_2 * 1/2 * (u_x)^2

    APEBench difficulty params: gammas=(0,0,-1.2,0,-15), deltas=(0,0,-6)
    Normalized (N=160, D=1):
        alpha_2 = -2.34375e-5   (negative diffusion, energy producing)
        alpha_4 = -2.86102e-9   (hyper-diffusion, stabilizing)
        beta_2  = -2.34375e-4   (gradient norm nonlinearity)

    Spatial discretization:
        - u_x:    4th-order central FD (periodic)
        - u_xx:   4th-order central FD (periodic)
        - u_xxxx: direct 5-point stencil (2nd order, periodic)
    Temporal discretization:
        - u_t:    2nd-order central

    Data layout: [B, T, X] where x=dim=-1.
    """

    DEFAULT_EQ_SCALES = {
        'momentum': 1.0,
    }

    def __init__(
        self,
        nx: int = 160,
        Lx: float = 1.0,
        dt: float = 1.0,
        alpha_2: float = -2.34375e-5,
        alpha_4: float = -2.8610229492e-9,
        beta_2: float = -2.34375e-4,
        eq_scales: Optional[Dict[str, float]] = None,
        eq_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.dx = Lx / nx
        self.dt = dt
        self.alpha_2 = alpha_2
        self.alpha_4 = alpha_4
        self.beta_2 = beta_2

        self.eq_scales = dict(self.DEFAULT_EQ_SCALES)
        if eq_scales is not None:
            self.eq_scales.update(eq_scales)

        self.eq_weights = {'momentum': 1.0}
        if eq_weights is not None:
            self.eq_weights.update(eq_weights)

    def _dx_4th(self, f: torch.Tensor) -> torch.Tensor:
        """4th-order central d/dx (periodic, dim=-1)."""
        return (
            -torch.roll(f, -2, dims=-1) + 8 * torch.roll(f, -1, dims=-1)
            - 8 * torch.roll(f, 1, dims=-1) + torch.roll(f, 2, dims=-1)
        ) / (12 * self.dx)

    def _d2x_4th(self, f: torch.Tensor) -> torch.Tensor:
        """4th-order central d^2/dx^2 (periodic, dim=-1)."""
        return (
            -torch.roll(f, -2, dims=-1) + 16 * torch.roll(f, -1, dims=-1)
            - 30 * f
            + 16 * torch.roll(f, 1, dims=-1) - torch.roll(f, 2, dims=-1)
        ) / (12 * self.dx ** 2)

    def _d4x_direct(self, f: torch.Tensor) -> torch.Tensor:
        """Direct 5-point stencil for d^4/dx^4 (2nd order, periodic, dim=-1).

        d^4u/dx^4 = (u[i-2] - 4u[i-1] + 6u[i] - 4u[i+1] + u[i+2]) / dx^4
        """
        return (
            torch.roll(f, -2, dims=-1) - 4 * torch.roll(f, -1, dims=-1)
            + 6 * f
            - 4 * torch.roll(f, 1, dims=-1) + torch.roll(f, 2, dims=-1)
        ) / (self.dx ** 4)

    def forward(
        self,
        u: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute 1D KS PDE residual loss.

        Args:
            u: scalar field [B, T, X] — includes prepended t0 frame.
               Time derivative uses 2nd-order central at t=1..T-2.

        Returns:
            total_loss, {eq_name: loss_value}
        """
        u_int = u[:, 1:-1]

        # Time derivative (2nd-order central)
        du_dt = (u[:, 2:] - u[:, :-2]) / (2 * self.dt)

        # Spatial derivatives
        du_dx = self._dx_4th(u_int)
        d2u_dx2 = self._d2x_4th(u_int)
        d4u_dx4 = self._d4x_direct(u_int)

        # Gradient norm: 1/2 * (u_x)^2
        grad_norm = 0.5 * du_dx ** 2

        # Residual: u_t - alpha_2*u_xx - alpha_4*u_xxxx - beta_2*1/2*(u_x)^2 = 0
        R = du_dt - self.alpha_2 * d2u_dx2 - self.alpha_4 * d4u_dx4 - self.beta_2 * grad_norm

        scale = self.eq_scales['momentum']
        w = self.eq_weights['momentum']
        losses = {'momentum': w * torch.mean(R ** 2) / (scale ** 2)}

        total_loss = losses['momentum']
        return total_loss, losses


# =============================================================================
# APEBench 1D KdV (Korteweg-de Vries) PDE Loss
# =============================================================================

class KdV1DPDELoss(nn.Module):
    """
    APEBench 1D KdV PDE Loss — periodic BC.

    PDE (from APEBench difficulty-based coefficients):
        u_t = beta_1 * u * u_x + alpha_3 * u_xxx + alpha_4 * u_xxxx

    where (gammas=(0,0,0,-14,-9), deltas=(0,-2,0), N=160, D=1, M=1.0):
        alpha_3 = gamma_3 / (N^3 * 2^2 * D) = -8.5449e-7  (dispersion)
        alpha_4 = gamma_4 / (N^4 * 2^3 * D) = -1.7166e-9  (hyper-diffusion)
        beta_1  = delta_1 / (M * N * D)      = -1.2500e-2  (convection)

    Residual form:
        R = u_t - beta_1 * u * u_x - alpha_3 * u_xxx - alpha_4 * u_xxxx

    Numerical method:
        - Spatial: 4th-order central difference (periodic via torch.roll)
        - Temporal: 2nd-order central difference

    Data layout: [B, T, X] where x = dim=-1.
    """

    DEFAULT_EQ_SCALES = {
        'kdv': 1.0,
    }

    def __init__(
        self,
        nx: int = 160,
        Lx: float = 1.0,
        dt: float = 1.0,
        alpha_3: float = -8.5449e-7,
        alpha_4: float = -1.7166e-9,
        beta_1: float = -1.2500e-2,
        eq_scales: Optional[Dict[str, float]] = None,
        eq_weights: Optional[Dict[str, float]] = None,
        eq_scales_per_t: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.dx = Lx / nx
        self.dt = dt
        self.alpha_3 = alpha_3
        self.alpha_4 = alpha_4
        self.beta_1 = beta_1

        self.eq_scales = {**self.DEFAULT_EQ_SCALES, **(eq_scales or {})}
        self.eq_weights = {'kdv': 1.0, **(eq_weights or {})}

        self.use_per_t_scales = eq_scales_per_t is not None
        if eq_scales_per_t is not None:
            for name, scales in eq_scales_per_t.items():
                self.register_buffer(f'scales_t_{name}', scales.float())

    def _dx_4th(self, f: torch.Tensor) -> torch.Tensor:
        """4th-order central d/dx (periodic, dim=-1)."""
        return (
            -torch.roll(f, -2, dims=-1) + 8 * torch.roll(f, -1, dims=-1)
            - 8 * torch.roll(f, 1, dims=-1) + torch.roll(f, 2, dims=-1)
        ) / (12 * self.dx)

    def _d3x_4th(self, f: torch.Tensor) -> torch.Tensor:
        """4th-order central d^3/dx^3 (periodic, dim=-1)."""
        fp1 = torch.roll(f, -1, dims=-1)
        fm1 = torch.roll(f, 1, dims=-1)
        fp2 = torch.roll(f, -2, dims=-1)
        fm2 = torch.roll(f, 2, dims=-1)
        fp3 = torch.roll(f, -3, dims=-1)
        fm3 = torch.roll(f, 3, dims=-1)
        return (-fp3/8 + fp2 - 13*fp1/8 + 13*fm1/8 - fm2 + fm3/8) / (self.dx**3)

    def _d4x_4th(self, f: torch.Tensor) -> torch.Tensor:
        """4th-order central d^4/dx^4 (periodic, dim=-1)."""
        fp1 = torch.roll(f, -1, dims=-1)
        fm1 = torch.roll(f, 1, dims=-1)
        fp2 = torch.roll(f, -2, dims=-1)
        fm2 = torch.roll(f, 2, dims=-1)
        fp3 = torch.roll(f, -3, dims=-1)
        fm3 = torch.roll(f, 3, dims=-1)
        return (-fp3/6 + 2*fp2 - 13*fp1/2 + 28*f/3 - 13*fm1/2 + 2*fm2 - fm3/6) / (self.dx**4)

    def forward(
        self,
        u: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute 1D KdV PDE residual loss.

        Args:
            u: scalar field [B, T, X] where T includes prepended t0 frame.
               Time derivative uses 2nd-order central at t=1..T-2.

        Returns:
            total_loss, {eq_name: loss_value}
        """
        # 2nd-order central time derivative
        du_dt = (u[:, 2:] - u[:, :-2]) / (2 * self.dt)  # [B, T-2, X]

        # Spatial terms at interior timesteps
        u_int = u[:, 1:-1]  # [B, T-2, X]

        du_dx = self._dx_4th(u_int)
        d3u_dx3 = self._d3x_4th(u_int)
        d4u_dx4 = self._d4x_4th(u_int)

        # Residual: u_t - beta_1*u*u_x - alpha_3*u_xxx - alpha_4*u_xxxx = 0
        R = du_dt - self.beta_1 * u_int * du_dx - self.alpha_3 * d3u_dx3 - self.alpha_4 * d4u_dx4

        # Per-equation normalized loss
        w_kdv = self.eq_weights['kdv']
        T_res = R.shape[1]

        if self.use_per_t_scales and hasattr(self, 'scales_t_kdv'):
            s_t = self.scales_t_kdv[:T_res].clamp(min=1e-8)
            R_norm = R / s_t[None, :, None]
            loss_kdv = w_kdv * torch.mean(R_norm ** 2)
        else:
            s = self.eq_scales['kdv']
            loss_kdv = w_kdv * torch.mean(R ** 2) / (s ** 2)

        return loss_kdv, {'kdv': loss_kdv, 'total': loss_kdv}


# =============================================================================
# Swift-Hohenberg 3D PDE Loss
# =============================================================================

class SwiftHohenberg3DPDELoss(nn.Module):
    """
    3D Swift-Hohenberg PDE Loss.

    PDE: u_t = r*u - (k + Δ)²*u + u² - u³
       = (r-k²)*u - 2k*Δu - Δ²u + u² - u³

    Periodic BC, 2nd-order central Laplacian, biharmonic = Laplacian(Laplacian).
    Time: 2nd-order central.

    Args:
        domain_extent: L (default 10π)
        num_points: N per dim (default 32)
        dt: time step between snapshots (default 0.1)
        reactivity: r (default 0.7)
        critical_number: k (default 1.0)
        eq_scales, eq_weights, eq_scales_per_t: normalization
    """

    DEFAULT_EQ_SCALES = {'swift_hohenberg': 1.0}

    def __init__(
        self,
        domain_extent: float = 31.41592653589793,
        num_points: int = 32,
        dt: float = 0.1,
        reactivity: float = 0.7,
        critical_number: float = 1.0,
        eq_scales: Optional[Dict[str, float]] = None,
        eq_weights: Optional[Dict[str, float]] = None,
        eq_scales_per_t: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.dx = domain_extent / num_points
        self.dt = dt
        self.r = reactivity
        self.k = critical_number
        self.linear_coeff = reactivity - critical_number ** 2  # r - k²

        self.eq_scales = dict(self.DEFAULT_EQ_SCALES)
        if eq_scales:
            self.eq_scales.update(eq_scales)

        self.eq_weights = {'swift_hohenberg': 1.0}
        if eq_weights:
            self.eq_weights.update(eq_weights)

        self.use_per_t_scales = eq_scales_per_t is not None
        if eq_scales_per_t is not None:
            for name, scales in eq_scales_per_t.items():
                self.register_buffer(f'scales_t_{name}', scales.float())

    def _laplacian_3d(self, f: torch.Tensor) -> torch.Tensor:
        """2nd-order central Laplacian, periodic 3D."""
        dx2 = self.dx ** 2
        lap = torch.zeros_like(f)
        for dim in [-3, -2, -1]:
            lap += (torch.roll(f, -1, dims=dim) - 2*f + torch.roll(f, 1, dims=dim)) / dx2
        return lap

    def forward(
        self,
        u: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            u: [B, T, X, Y, Z] with T including prepended t0 frame.
        Returns:
            total_loss, losses_dict
        """
        # Time derivative: 2nd-order central
        u_mid = u[:, 1:-1]
        du_dt = (u[:, 2:] - u[:, :-2]) / (2 * self.dt)

        # Spatial operators
        lap_u = self._laplacian_3d(u_mid)
        bilap_u = self._laplacian_3d(lap_u)

        # Residual: du/dt - [(r-k²)*u - 2k*Δu - Δ²u + u² - u³]
        R = du_dt - self.linear_coeff * u_mid + 2*self.k * lap_u + bilap_u - u_mid**2 + u_mid**3

        # Normalization
        w = self.eq_weights['swift_hohenberg']
        T_res = R.shape[1]

        if self.use_per_t_scales and hasattr(self, 'scales_t_swift_hohenberg'):
            s_t = self.scales_t_swift_hohenberg[:T_res].clamp(min=1e-8)
            # s_t shape [T_res] -> [1, T_res, 1, 1, 1]
            R_norm = R / s_t[None, :, None, None, None]
            loss = w * torch.mean(R_norm ** 2)
        else:
            s = self.eq_scales['swift_hohenberg']
            loss = w * torch.mean(R ** 2) / (s ** 2)

        return loss, {'swift_hohenberg': loss, 'total': loss}
