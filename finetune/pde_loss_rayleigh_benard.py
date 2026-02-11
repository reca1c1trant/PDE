"""
Rayleigh-Bénard Convection PDE residual loss - 谱方法 (Spectral)

控制方程 (无量纲形式):
    ∂b/∂t + u·∂b/∂x + w·∂b/∂y = κ·∇²b           (浮力方程)
    ∂u/∂t + u·∂u/∂x + w·∂u/∂y = -∂p/∂x + ν·∇²u  (x动量)
    ∂w/∂t + u·∂w/∂x + w·∂w/∂y = -∂p/∂y + ν·∇²w + b  (y动量, 含浮力项)
    ∂u/∂x + ∂w/∂y = 0                           (连续性方程)

物理参数:
    κ = (Ra × Pr)^(-1/2)  热扩散系数
    ν = (Ra / Pr)^(-1/2)  运动粘度

边界条件:
    x方向: 周期性 → FFT 谱导数
    y方向: Dirichlet (y=0: b=1, u=w=0; y=1: b=0, u=w=0)
          → 中心差分，排除边界 [2:-2]

差分格式 (匹配 Dedalus 谱方法生成的数据):
    x方向: FFT 谱导数 (精确匹配周期性 Fourier 基)
        ∂f/∂x = IFFT(i·k·FFT(f))
        ∂²f/∂x² = IFFT(-k²·FFT(f))
    y方向: 二阶中心差分 (Dirichlet 边界，内部点)
    时间导数: 一阶前向差分

Author: Ziye
Date: February 2025
"""

import torch
import numpy as np
from typing import Tuple, Dict


def chebyshev_derivative_matrix(N: int, Ly: float = 1.0) -> torch.Tensor:
    """
    构建 Chebyshev 微分矩阵 (Chebyshev-Gauss-Lobatto 点).

    D[i,j] = dT_j(y_i) / dy，其中 y_i = cos(πi/N) 映射到 [0, Ly]

    Returns: D [N+1, N+1]
    """
    # Chebyshev-Gauss-Lobatto 点 (在 [-1, 1] 上)
    j = np.arange(N + 1)
    y = np.cos(np.pi * j / N)  # y_j = cos(πj/N), j=0,...,N

    # 系数 c_j
    c = np.ones(N + 1)
    c[0] = 2.0
    c[N] = 2.0

    # 构建微分矩阵
    D = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for k in range(N + 1):
            if i != k:
                D[i, k] = (c[i] / c[k]) * ((-1) ** (i + k)) / (y[i] - y[k])
            elif i == 0:
                D[i, i] = (2 * N * N + 1) / 6.0
            elif i == N:
                D[i, i] = -(2 * N * N + 1) / 6.0
            else:
                D[i, i] = -y[i] / (2 * (1 - y[i] ** 2))

    # 缩放: [-1, 1] -> [0, Ly]，导数乘以 2/Ly
    D = D * (2.0 / Ly)

    return torch.tensor(D, dtype=torch.float64)


def spectral_derivative_y_chebyshev(phi: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """
    Y方向 Chebyshev 谱导数。

    phi: [*, X, Y] where Y = N+1 (Chebyshev points)
    D: [Y, Y] Chebyshev differentiation matrix

    Returns: [*, X, Y]
    """
    # phi @ D.T 对最后一维做矩阵乘法
    return torch.einsum('...y,zy->...z', phi, D)


def compute_transport_coefficients(Ra: float, Pr: float) -> Tuple[float, float]:
    """从 Ra, Pr 计算传输系数。"""
    kappa = (Ra * Pr) ** (-0.5)
    nu = (Ra / Pr) ** (-0.5)
    return kappa, nu


def spectral_derivative_x(phi: torch.Tensor, Lx: float) -> torch.Tensor:
    """
    X方向谱导数 (周期性边界, FFT)。

    ∂f/∂x in Fourier space: i * k * f̂(k)

    phi: [*, X, Y]
    Lx: 域长度
    """
    X = phi.shape[-2]
    # 波数
    kx = torch.fft.fftfreq(X, d=Lx/X, device=phi.device, dtype=phi.dtype) * 2 * torch.pi
    kx = kx.view(*([1] * (phi.dim() - 2)), X, 1)

    # FFT -> 乘以 ik -> IFFT
    phi_hat = torch.fft.fft(phi, dim=-2)
    dphi_hat = 1j * kx * phi_hat
    dphi = torch.fft.ifft(dphi_hat, dim=-2).real

    return dphi


def spectral_derivative_xx(phi: torch.Tensor, Lx: float) -> torch.Tensor:
    """
    X方向二阶谱导数 (周期性边界, FFT)。

    ∂²f/∂x² in Fourier space: -k² * f̂(k)
    """
    X = phi.shape[-2]
    kx = torch.fft.fftfreq(X, d=Lx/X, device=phi.device, dtype=phi.dtype) * 2 * torch.pi
    kx = kx.view(*([1] * (phi.dim() - 2)), X, 1)

    phi_hat = torch.fft.fft(phi, dim=-2)
    d2phi_hat = -kx**2 * phi_hat
    d2phi = torch.fft.ifft(d2phi_hat, dim=-2).real

    return d2phi


def upwind_x_periodic(phi: torch.Tensor, vel_x: torch.Tensor, dx: float) -> torch.Tensor:
    """
    X方向二阶迎风 (周期性边界) - 保留作为备用。

    phi, vel_x: [*, X, Y]
    """
    phi_E = torch.roll(phi, -1, dims=-2)   # i+1
    phi_W = torch.roll(phi, 1, dims=-2)    # i-1
    phi_EE = torch.roll(phi, -2, dims=-2)  # i+2
    phi_WW = torch.roll(phi, 2, dims=-2)   # i-2

    # u >= 0: backward difference
    dphi_backward = (3*phi - 4*phi_W + phi_WW) / (2*dx)
    # u < 0: forward difference
    dphi_forward = (-3*phi + 4*phi_E - phi_EE) / (2*dx)

    return torch.where(vel_x >= 0, dphi_backward, dphi_forward)


def upwind_y_interior(phi: torch.Tensor, vel_y: torch.Tensor, dy: float) -> torch.Tensor:
    """
    Y方向二阶迎风 (内部点, 排除边界2层)。

    phi, vel_y: [*, X, Y]
    返回: [*, X, Y-4]
    """
    phi_C = phi[..., 2:-2]
    phi_N = phi[..., 3:-1]   # i+1
    phi_S = phi[..., 1:-3]   # i-1
    phi_NN = phi[..., 4:]    # i+2
    phi_SS = phi[..., :-4]   # i-2

    vel_C = vel_y[..., 2:-2]

    dphi_backward = (3*phi_C - 4*phi_S + phi_SS) / (2*dy)
    dphi_forward = (-3*phi_C + 4*phi_N - phi_NN) / (2*dy)

    return torch.where(vel_C >= 0, dphi_backward, dphi_forward)


def laplacian_mixed_bc(phi: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """
    Laplacian: ∇²φ = ∂²φ/∂x² + ∂²φ/∂y²

    X方向: 周期性 (中心差分)
    Y方向: 内部点 [2:-2]

    phi: [*, X, Y]
    返回: [*, X, Y-4]
    """
    # X方向二阶导数 (周期性)
    phi_xx = (torch.roll(phi, -1, dims=-2) - 2*phi + torch.roll(phi, 1, dims=-2)) / (dx ** 2)

    # Y方向二阶导数 (内部点, 使用2层stencil)
    phi_yy = (phi[..., 4:] - 2*phi[..., 2:-2] + phi[..., :-4]) / (dy ** 2)

    return phi_xx[..., 2:-2] + phi_yy


def pressure_grad_x_periodic(p: torch.Tensor, dx: float) -> torch.Tensor:
    """压力X方向梯度 (中心差分, 周期性)。"""
    return (torch.roll(p, -1, dims=-2) - torch.roll(p, 1, dims=-2)) / (2 * dx)


def pressure_grad_y_interior(p: torch.Tensor, dy: float) -> torch.Tensor:
    """压力Y方向梯度 (中心差分, 内部点)。"""
    return (p[..., 3:-1] - p[..., 1:-3]) / (2 * dy)


def rayleigh_benard_pde_loss(
    b: torch.Tensor,
    u: torch.Tensor,
    w: torch.Tensor,
    p: torch.Tensor,
    Ra: float,
    Pr: float,
    dx: float,
    dy: float,
    dt: float,
    Lx: float = 4.0,
    Ly: float = 1.0,
    use_spectral: bool = True,
    D_cheb: torch.Tensor = None,  # Chebyshev differentiation matrix
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    计算 Rayleigh-Bénard 方程的 PDE 残差 (二阶迎风格式)。

    Parameters:
    -----------
    b : torch.Tensor [*, T, X, Y]
        浮力场
    u : torch.Tensor [*, T, X, Y]
        水平速度
    w : torch.Tensor [*, T, X, Y]
        垂直速度
    p : torch.Tensor [*, T, X, Y]
        压力场
    Ra : float
        Rayleigh number
    Pr : float
        Prandtl number
    dx, dy, dt : float
        网格间距和时间步长

    Returns:
    --------
    total_loss : torch.Tensor
        总 PDE 残差 loss
    losses : dict
        各方程的残差 loss
    """
    # 计算传输系数
    kappa, nu = compute_transport_coefficients(Ra, Pr)

    # 时间导数 (前向差分)
    db_dt = (b[..., 1:, :, :] - b[..., :-1, :, :]) / dt
    du_dt = (u[..., 1:, :, :] - u[..., :-1, :, :]) / dt
    dw_dt = (w[..., 1:, :, :] - w[..., :-1, :, :]) / dt

    # 取 n 时刻做空间导数
    b_n = b[..., :-1, :, :]
    u_n = u[..., :-1, :, :]
    w_n = w[..., :-1, :, :]
    p_n = p[..., :-1, :, :]

    if use_spectral:
        # X方向用谱导数 (FFT)
        db_dx = spectral_derivative_x(b_n, Lx)
        du_dx = spectral_derivative_x(u_n, Lx)
        dw_dx = spectral_derivative_x(w_n, Lx)
        dp_dx = spectral_derivative_x(p_n, Lx)

        # X方向二阶导数 (FFT)
        d2b_dx2 = spectral_derivative_xx(b_n, Lx)
        d2u_dx2 = spectral_derivative_xx(u_n, Lx)
        d2w_dx2 = spectral_derivative_xx(w_n, Lx)
    else:
        # X方向用有限差分
        db_dx = upwind_x_periodic(b_n, u_n, dx)
        du_dx = upwind_x_periodic(u_n, u_n, dx)
        dw_dx = upwind_x_periodic(w_n, u_n, dx)
        dp_dx = pressure_grad_x_periodic(p_n, dx)

        d2b_dx2 = (torch.roll(b_n, -1, dims=-2) - 2*b_n + torch.roll(b_n, 1, dims=-2)) / dx**2
        d2u_dx2 = (torch.roll(u_n, -1, dims=-2) - 2*u_n + torch.roll(u_n, 1, dims=-2)) / dx**2
        d2w_dx2 = (torch.roll(w_n, -1, dims=-2) - 2*w_n + torch.roll(w_n, 1, dims=-2)) / dx**2

    # Y方向导数
    if use_spectral and D_cheb is not None:
        # Chebyshev 谱导数 (精确匹配 Dedalus 的 Chebyshev 基)
        db_dy_full = spectral_derivative_y_chebyshev(b_n, D_cheb)
        du_dy_full = spectral_derivative_y_chebyshev(u_n, D_cheb)
        dw_dy_full = spectral_derivative_y_chebyshev(w_n, D_cheb)
        dp_dy_full = spectral_derivative_y_chebyshev(p_n, D_cheb)

        # 二阶导数: D @ D
        D2_cheb = D_cheb @ D_cheb
        d2b_dy2_full = spectral_derivative_y_chebyshev(b_n, D2_cheb)
        d2u_dy2_full = spectral_derivative_y_chebyshev(u_n, D2_cheb)
        d2w_dy2_full = spectral_derivative_y_chebyshev(w_n, D2_cheb)

        # 取内部点 [2:-2]
        db_dy = db_dy_full[..., 2:-2]
        du_dy = du_dy_full[..., 2:-2]
        dw_dy = dw_dy_full[..., 2:-2]
        dp_dy = dp_dy_full[..., 2:-2]
        d2b_dy2 = d2b_dy2_full[..., 2:-2]
        d2u_dy2 = d2u_dy2_full[..., 2:-2]
        d2w_dy2 = d2w_dy2_full[..., 2:-2]
    else:
        # 中心差分 (Dirichlet 边界，排除边界2层 → 内部点 [2:-2])
        db_dy = (b_n[..., 3:-1] - b_n[..., 1:-3]) / (2 * dy)
        du_dy = (u_n[..., 3:-1] - u_n[..., 1:-3]) / (2 * dy)
        dw_dy = (w_n[..., 3:-1] - w_n[..., 1:-3]) / (2 * dy)
        dp_dy = (p_n[..., 3:-1] - p_n[..., 1:-3]) / (2 * dy)

        d2b_dy2 = (b_n[..., 3:-1] - 2*b_n[..., 2:-2] + b_n[..., 1:-3]) / dy**2
        d2u_dy2 = (u_n[..., 3:-1] - 2*u_n[..., 2:-2] + u_n[..., 1:-3]) / dy**2
        d2w_dy2 = (w_n[..., 3:-1] - 2*w_n[..., 2:-2] + u_n[..., 1:-3]) / dy**2

    # Laplacian = d²/dx² + d²/dy² (取内部点 [2:-2])
    lap_b = d2b_dx2[..., 2:-2] + d2b_dy2
    lap_u = d2u_dx2[..., 2:-2] + d2u_dy2
    lap_w = d2w_dx2[..., 2:-2] + d2w_dy2

    # 所有变量取内部点 [2:-2]
    db_dt_int = db_dt[..., 2:-2]
    du_dt_int = du_dt[..., 2:-2]
    dw_dt_int = dw_dt[..., 2:-2]
    db_dx_int = db_dx[..., 2:-2]
    du_dx_int = du_dx[..., 2:-2]
    dw_dx_int = dw_dx[..., 2:-2]
    dp_dx_int = dp_dx[..., 2:-2]
    b_int = b_n[..., 2:-2]
    u_int = u_n[..., 2:-2]
    w_int = w_n[..., 2:-2]

    # PDE 残差 (所有项 shape: [*, T-1, X, Y-4])
    # 浮力: ∂b/∂t + u·∂b/∂x + w·∂b/∂y - κ·∇²b = 0
    R_b = db_dt_int + u_int * db_dx_int + w_int * db_dy - kappa * lap_b

    # x动量: ∂u/∂t + u·∂u/∂x + w·∂u/∂y + ∂p/∂x - ν·∇²u = 0
    R_u = du_dt_int + u_int * du_dx_int + w_int * du_dy + dp_dx_int - nu * lap_u

    # y动量: ∂w/∂t + u·∂w/∂x + w·∂w/∂y + ∂p/∂y - ν·∇²w - b = 0
    R_w = dw_dt_int + u_int * dw_dx_int + w_int * dw_dy + dp_dy - nu * lap_w - b_int

    # 连续性: ∂u/∂x + ∂w/∂y = 0
    R_cont = du_dx_int + dw_dy

    # Loss 计算
    loss_b = torch.mean(R_b ** 2)
    loss_u = torch.mean(R_u ** 2)
    loss_w = torch.mean(R_w ** 2)
    loss_cont = torch.mean(R_cont ** 2)

    total_loss = loss_b + loss_u + loss_w + loss_cont

    losses = {
        'buoyancy': loss_b,
        'momentum_x': loss_u,
        'momentum_y': loss_w,
        'continuity': loss_cont,
        'total': total_loss,
    }

    return total_loss, losses


def test_with_gt_data(data_path: str):
    """用 GT 数据测试 PDE 残差。"""
    import h5py

    print("=" * 60)
    print("Testing Rayleigh-Bénard PDE Loss (谱方法)")
    print("=" * 60)

    with h5py.File(data_path, 'r') as f:
        # 物理参数
        Ra = float(f.attrs['Rayleigh'])
        Pr = float(f.attrs['Prandtl'])
        print(f"Ra = {Ra:.2e}, Pr = {Pr}")

        kappa, nu = compute_transport_coefficients(Ra, Pr)
        print(f"κ = {kappa:.6e}, ν = {nu:.6e}")

        # 网格信息
        x = f['dimensions/x'][:]
        y = f['dimensions/y'][:]
        t = f['dimensions/time'][:]

        # 域长度 (用于谱导数)
        Lx = float(x[-1] - x[0]) + (x[1] - x[0])  # 周期性边界，加一个dx
        Ly = float(y[-1] - y[0])

        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dt = t[1] - t[0]
        print(f"Lx = {Lx:.6f}, Ly = {Ly:.6f}")
        print(f"dx = {dx:.6f}, dy = {dy:.6f}, dt = {dt:.4f}")

        # 加载数据 (第一个 trajectory, 前20个时间步)
        b = torch.tensor(f['t0_fields/buoyancy'][0, :20], dtype=torch.float64)
        p = torch.tensor(f['t0_fields/pressure'][0, :20], dtype=torch.float64)
        vel = torch.tensor(f['t1_fields/velocity'][0, :20], dtype=torch.float64)
        u = vel[..., 0]
        w = vel[..., 1]

        Ny = b.shape[-1]  # Y方向点数

        print(f"\nData shapes: b={b.shape}, u={u.shape}, w={w.shape}, p={p.shape}")
        print("  Format: [T, X, Y] where X=512 (periodic), Y=128 (Dirichlet)")
        print(f"  Ny = {Ny}")

        # 检查 y 网格是否为 Chebyshev 点
        y_expected_cheb = Ly * (1 - np.cos(np.pi * np.arange(Ny) / (Ny - 1))) / 2
        y_diff = np.abs(y - y_expected_cheb)
        is_chebyshev = np.max(y_diff) < 1e-10
        print(f"  Y grid is Chebyshev: {is_chebyshev} (max diff: {np.max(y_diff):.2e})")

    # 构建 Chebyshev 微分矩阵
    if is_chebyshev:
        print("\n使用 Chebyshev 谱导数 (Y方向)")
        D_cheb = chebyshev_derivative_matrix(Ny - 1, Ly)
    else:
        print("\n使用中心差分 (Y方向非 Chebyshev 网格)")
        D_cheb = None

    # 计算 PDE 残差 (用谱方法)
    total_loss, losses = rayleigh_benard_pde_loss(
        b, u, w, p, Ra, Pr, dx, dy, dt, Lx=Lx, Ly=Ly, use_spectral=True, D_cheb=D_cheb
    )

    print(f"\n{'='*50}")
    print("PDE Residuals (GT data):")
    print(f"{'='*50}")
    for name, loss in losses.items():
        status = "✅" if loss.item() < 1e-6 else ("⚠️" if loss.item() < 1e-3 else "❌")
        print(f"  {name:15s}: {loss.item():.6e}  {status}")

    print(f"\n{'='*50}")
    if total_loss.item() < 1e-6:
        print("✅ PDE loss 验证通过！残差接近机器精度。")
    elif total_loss.item() < 1e-3:
        print("⚠️ 残差较小，可能由于数值精度或时间步长差异。")
    else:
        print("❌ 残差较大，请检查数据格式或差分格式。")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Rayleigh-Bénard PDE loss")
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to GT data file (HDF5)')
    args = parser.parse_args()

    test_with_gt_data(args.data_path)
