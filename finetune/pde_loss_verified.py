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
        KX = self.KX
        while KX.dim() < f.dim():
            KX = KX.unsqueeze(0)
        f_hat = torch.fft.fft2(f)
        return torch.fft.ifft2(1j * KX * f_hat).real

    def fft_dy(self, f: torch.Tensor) -> torch.Tensor:
        """FFT y方向导数"""
        KY = self.KY
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

        # 预计算FFT波数
        kx = torch.fft.fftfreq(nx, d=dx) * 2 * np.pi
        ky = torch.fft.fftfreq(ny, d=dy) * 2 * np.pi
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')
        K2 = KX**2 + KY**2

        self.register_buffer('K2', K2)

    def fft_laplacian(self, f: torch.Tensor) -> torch.Tensor:
        """FFT Laplacian"""
        K2 = self.K2
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

        # Loss计算
        if reduction == 'mean':
            loss_A = torch.mean(R_A ** 2)
            loss_B = torch.mean(R_B ** 2)
        elif reduction == 'sum':
            loss_A = torch.sum(R_A ** 2)
            loss_B = torch.sum(R_B ** 2)
        else:
            loss_A = R_A ** 2
            loss_B = R_B ** 2

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
        kx = self.kx
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

        scale_cont = self.eq_scales['continuity']
        w_cont = self.eq_weights['continuity']
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

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        b: torch.Tensor,
        kappa: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute PDE residual losses.

        Args:
            u: x-velocity [B, T, Nx, Ny]
            v: y-velocity [B, T, Nx, Ny]
            b: buoyancy   [B, T, Nx, Ny]
            kappa: optional per-sample κ [B] overriding self.kappa

        Returns:
            total_loss, {eq_name: loss_value}
        """
        losses: Dict[str, torch.Tensor] = {}
        skip = self.skip_bl

        # Interior timesteps for 2nd-order central time derivative
        u_int = u[:, 1:-1]
        v_int = v[:, 1:-1]
        b_int = b[:, 1:-1]

        # ---- Eq 1: Continuity ----
        du_dx = self._dx_4th(u_int)
        dv_dy = self._dy_ghost(v_int)

        R_cont = du_dx + dv_dy
        # Skip boundary layer
        if skip > 0:
            R_cont = R_cont[..., skip:-skip]

        scale_cont = self.eq_scales['continuity']
        w_cont = self.eq_weights['continuity']
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

    # ---- Face velocities ----

    def _face_velocities(
        self, u: torch.Tensor, v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Face-averaged velocities (periodic via roll)."""
        uc_e = 0.5 * (u + torch.roll(u, -1, dims=-1))  # east
        uc_w = 0.5 * (u + torch.roll(u, 1, dims=-1))   # west
        vc_n = 0.5 * (v + torch.roll(v, -1, dims=-2))   # north
        vc_s = 0.5 * (v + torch.roll(v, 1, dims=-2))    # south
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
        # x-direction neighbors (dim=-1)
        f_ip1 = torch.roll(f, -1, dims=-1)
        f_im1 = torch.roll(f, 1, dims=-1)
        f_ip2 = torch.roll(f, -2, dims=-1)
        f_im2 = torch.roll(f, 2, dims=-1)

        # y-direction neighbors (dim=-2)
        f_jp1 = torch.roll(f, -1, dims=-2)
        f_jm1 = torch.roll(f, 1, dims=-2)
        f_jp2 = torch.roll(f, -2, dims=-2)
        f_jm2 = torch.roll(f, 2, dims=-2)

        # East face
        Fe_pos = 1.5 * f - 0.5 * f_im1
        Fe_neg = 1.5 * f_ip1 - 0.5 * f_ip2
        Fe = torch.where(uc_e >= 0, Fe_pos, Fe_neg)

        # West face
        Fw_pos = 1.5 * f_im1 - 0.5 * f_im2
        Fw_neg = 1.5 * f - 0.5 * f_ip1
        Fw = torch.where(uc_w >= 0, Fw_pos, Fw_neg)

        # North face
        Fn_pos = 1.5 * f - 0.5 * f_jm1
        Fn_neg = 1.5 * f_jp1 - 0.5 * f_jp2
        Fn = torch.where(vc_n >= 0, Fn_pos, Fn_neg)

        # South face
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

    # ---- Face-averaged pressure gradient ----

    def _pressure_grad_x(self, p: torch.Tensor) -> torch.Tensor:
        """dp/dx via face average: (p_E - p_W) / (2*dx)."""
        p_E = torch.roll(p, -1, dims=-1)
        p_W = torch.roll(p, 1, dims=-1)
        return (p_E - p_W) / (2 * self.dx)

    def _pressure_grad_y(self, p: torch.Tensor) -> torch.Tensor:
        """dp/dy via face average: (p_N - p_S) / (2*dy)."""
        p_N = torch.roll(p, -1, dims=-2)
        p_S = torch.roll(p, 1, dims=-2)
        return (p_N - p_S) / (2 * self.dy)

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        s: torch.Tensor,
        reduction: str = 'mean',
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute PDE residual losses.

        Args:
            u, v, p, s: [B, T, H, W]
            reduction: 'mean' (default)

        Returns:
            total_loss, {eq_name: loss_value}
        """
        losses: Dict[str, torch.Tensor] = {}

        # Interior timesteps (2nd-order central time derivative)
        u_int = u[:, 1:-1]
        v_int = v[:, 1:-1]
        p_int = p[:, 1:-1]
        s_int = s[:, 1:-1]

        # Face velocities
        uc_e, uc_w, vc_n, vc_s = self._face_velocities(u_int, v_int)

        # Divergence
        div = self._divergence(uc_e, uc_w, vc_n, vc_s)

        # ---- Eq 1: Continuity ----
        scale_cont = self.eq_scales['continuity']
        w_cont = self.eq_weights['continuity']
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
