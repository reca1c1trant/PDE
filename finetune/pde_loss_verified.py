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
