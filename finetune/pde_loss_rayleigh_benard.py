"""
Rayleigh-Bénard Convection PDE residual loss

控制方程 (无量纲形式):
    ∂b/∂t + u·∂b/∂x + w·∂b/∂z = κ·∇²b           (浮力方程)
    ∂u/∂t + u·∂u/∂x + w·∂u/∂z = -∂p/∂x + ν·∇²u  (x动量)
    ∂w/∂t + u·∂w/∂x + w·∂w/∂z = -∂p/∂z + ν·∇²w + b  (z动量, 含浮力项)
    ∂u/∂x + ∂w/∂z = 0                           (连续性方程)

物理参数:
    κ = (Ra × Pr)^(-1/2)  热扩散系数
    ν = (Ra / Pr)^(-1/2)  运动粘度

边界条件:
    x方向: 周期性 (用 torch.roll 处理)
    z方向: Dirichlet (z=0: b=1, u=w=0; z=1: b=0, u=w=0)
          -> 边界点不参与 PDE loss 计算

网格:
    Lx=4, Lz=1, 512×128
    dx = 4/512 ≈ 0.0078, dz = 1/128 ≈ 0.0078
    dt = 0.25

Author: Ziye
Date: February 2025
"""

import torch
from typing import Tuple, Dict
import math


def compute_transport_coefficients(Ra: float, Pr: float) -> Tuple[float, float]:
    """
    从 Ra, Pr 计算传输系数。

    κ = (Ra × Pr)^(-1/2)
    ν = (Ra / Pr)^(-1/2)
    """
    kappa = (Ra * Pr) ** (-0.5)
    nu = (Ra / Pr) ** (-0.5)
    return kappa, nu


def central_diff_x_periodic(phi: torch.Tensor, dx: float) -> torch.Tensor:
    """
    x方向一阶中心差分 (周期性边界)。

    phi: [B, T, H, W]
    """
    phi_E = torch.roll(phi, shifts=-1, dims=-1)
    phi_W = torch.roll(phi, shifts=1, dims=-1)
    return (phi_E - phi_W) / (2 * dx)


def central_diff_z(phi: torch.Tensor, dz: float) -> torch.Tensor:
    """
    z方向一阶中心差分 (非周期性，内部点)。

    phi: [B, T, H, W]
    返回: [B, T, H-2, W] (排除上下边界)
    """
    phi_N = phi[:, :, 2:, :]   # i+1
    phi_S = phi[:, :, :-2, :]  # i-1
    return (phi_N - phi_S) / (2 * dz)


def laplacian_mixed_bc(phi: torch.Tensor, dx: float, dz: float) -> torch.Tensor:
    """
    Laplacian: ∇²φ = ∂²φ/∂x² + ∂²φ/∂z²

    x方向: 周期性 (中心差分)
    z方向: 非周期性 (内部点只)

    phi: [B, T, H, W]
    返回: [B, T, H-2, W] (排除上下边界)
    """
    # x方向二阶导数 (周期性)
    phi_E = torch.roll(phi, shifts=-1, dims=-1)
    phi_W = torch.roll(phi, shifts=1, dims=-1)
    phi_xx = (phi_E - 2*phi + phi_W) / (dx ** 2)

    # z方向二阶导数 (内部点)
    phi_zz = (phi[:, :, 2:, :] - 2*phi[:, :, 1:-1, :] + phi[:, :, :-2, :]) / (dz ** 2)

    # phi_xx 也取内部点
    phi_xx_interior = phi_xx[:, :, 1:-1, :]

    return phi_xx_interior + phi_zz


def rayleigh_benard_pde_loss(
    pred: torch.Tensor,
    Ra: float,
    Pr: float,
    dt: float = 0.25,
    Lx: float = 4.0,
    Lz: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    计算 Rayleigh-Bénard 方程的 PDE 残差。

    Parameters:
    -----------
    pred : torch.Tensor [B, T, H, W, C]
        预测场，C=4: (b, u, w, p)
        H=128 (z方向), W=512 (x方向)
    Ra : float
        Rayleigh number
    Pr : float
        Prandtl number
    dt : float
        时间步长 (默认 0.25)
    Lx, Lz : float
        域大小 (默认 4.0, 1.0)

    Returns:
    --------
    total_loss : torch.Tensor
        总 PDE 残差 loss
    losses : dict
        各方程的残差 loss
    """
    B, T, H, W, C = pred.shape
    assert C == 4, f"Expected 4 channels (b, u, w, p), got {C}"
    assert T >= 2, f"Need at least 2 timesteps, got {T}"

    # 计算传输系数
    kappa, nu = compute_transport_coefficients(Ra, Pr)

    # 网格间距
    dx = Lx / W
    dz = Lz / H

    # 提取场变量 [B, T, H, W]
    b = pred[..., 0]  # buoyancy
    u = pred[..., 1]  # horizontal velocity
    w = pred[..., 2]  # vertical velocity
    p = pred[..., 3]  # pressure

    # ========== 时间导数 (前向差分) ==========
    # [B, T-1, H, W]
    db_dt = (b[:, 1:] - b[:, :-1]) / dt
    du_dt = (u[:, 1:] - u[:, :-1]) / dt
    dw_dt = (w[:, 1:] - w[:, :-1]) / dt

    # 取 n 时刻的场用于空间导数计算
    b_n = b[:, :-1]  # [B, T-1, H, W]
    u_n = u[:, :-1]
    w_n = w[:, :-1]
    p_n = p[:, :-1]

    # ========== 空间导数 ==========
    # x方向 (周期性)
    db_dx = central_diff_x_periodic(b_n, dx)  # [B, T-1, H, W]
    du_dx = central_diff_x_periodic(u_n, dx)
    dw_dx = central_diff_x_periodic(w_n, dx)
    dp_dx = central_diff_x_periodic(p_n, dx)

    # z方向 (内部点, 排除边界)
    db_dz = central_diff_z(b_n, dz)  # [B, T-1, H-2, W]
    du_dz = central_diff_z(u_n, dz)
    dw_dz = central_diff_z(w_n, dz)
    dp_dz = central_diff_z(p_n, dz)

    # Laplacian (内部点)
    lap_b = laplacian_mixed_bc(b_n, dx, dz)  # [B, T-1, H-2, W]
    lap_u = laplacian_mixed_bc(u_n, dx, dz)
    lap_w = laplacian_mixed_bc(w_n, dx, dz)

    # 时间导数取内部点
    db_dt_int = db_dt[:, :, 1:-1, :]  # [B, T-1, H-2, W]
    du_dt_int = du_dt[:, :, 1:-1, :]
    dw_dt_int = dw_dt[:, :, 1:-1, :]

    # x导数也取内部点
    db_dx_int = db_dx[:, :, 1:-1, :]
    du_dx_int = du_dx[:, :, 1:-1, :]
    dw_dx_int = dw_dx[:, :, 1:-1, :]
    dp_dx_int = dp_dx[:, :, 1:-1, :]

    # 场变量取内部点
    b_int = b_n[:, :, 1:-1, :]
    u_int = u_n[:, :, 1:-1, :]
    w_int = w_n[:, :, 1:-1, :]

    # ========== PDE 残差 ==========
    # 浮力方程: ∂b/∂t + u·∂b/∂x + w·∂b/∂z - κ·∇²b = 0
    R_b = db_dt_int + u_int * db_dx_int + w_int * db_dz - kappa * lap_b

    # x动量方程: ∂u/∂t + u·∂u/∂x + w·∂u/∂z + ∂p/∂x - ν·∇²u = 0
    R_u = du_dt_int + u_int * du_dx_int + w_int * du_dz + dp_dx_int - nu * lap_u

    # z动量方程: ∂w/∂t + u·∂w/∂x + w·∂w/∂z + ∂p/∂z - ν·∇²w - b = 0
    R_w = dw_dt_int + u_int * dw_dx_int + w_int * dw_dz + dp_dz - nu * lap_w - b_int

    # 连续性方程: ∂u/∂x + ∂w/∂z = 0
    R_cont = du_dx_int + dw_dz

    # ========== Loss 计算 ==========
    loss_b = torch.mean(R_b ** 2)
    loss_u = torch.mean(R_u ** 2)
    loss_w = torch.mean(R_w ** 2)
    loss_cont = torch.mean(R_cont ** 2)

    total_loss = loss_b + loss_u + loss_w + loss_cont

    losses = {
        'buoyancy': loss_b,
        'momentum_x': loss_u,
        'momentum_z': loss_w,
        'continuity': loss_cont,
        'total': total_loss,
    }

    return total_loss, losses


def test_with_gt_data(data_path: str = None):
    """
    用 GT 数据测试 PDE 残差是否接近零。

    如果残差 < 1e-6，说明差分格式正确。
    """
    print("=" * 60)
    print("Testing Rayleigh-Bénard PDE Loss with GT Data")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if data_path is None:
        print("\n[WARNING] No data path provided. Using random data for testing.")
        print("Please download data and run:")
        print("  python pde_loss_rayleigh_benard.py --data_path /path/to/data")

        # 用随机数据测试代码是否能跑
        B, T, H, W, C = 2, 10, 128, 512, 4
        pred = torch.randn(B, T, H, W, C, device=device)
        Ra, Pr = 1e6, 1.0

        total_loss, losses = rayleigh_benard_pde_loss(pred, Ra, Pr)

        print(f"\nRandom data test (just checking code runs):")
        print(f"  Input shape: [B={B}, T={T}, H={H}, W={W}, C={C}]")
        print(f"  Ra={Ra:.1e}, Pr={Pr}")
        for name, loss in losses.items():
            print(f"  {name}: {loss.item():.6e}")

        print("\nCode runs correctly. Now test with real GT data.")
        return

    # ========== 加载真实数据 ==========
    import h5py
    import numpy as np

    print(f"\nLoading data from: {data_path}")

    with h5py.File(data_path, 'r') as f:
        print(f"\nDataset keys: {list(f.keys())}")

        # 检查数据结构
        for key in f.keys():
            if hasattr(f[key], 'shape'):
                print(f"  {key}: shape={f[key].shape}, dtype={f[key].dtype}")

        # 提取数据 (假设格式)
        # 需要根据实际数据格式调整
        if 'buoyancy' in f:
            b = torch.tensor(f['buoyancy'][:], device=device, dtype=torch.float32)
            u = torch.tensor(f['velocity'][:, ..., 0], device=device, dtype=torch.float32)
            w = torch.tensor(f['velocity'][:, ..., 1], device=device, dtype=torch.float32)
            p = torch.tensor(f['pressure'][:], device=device, dtype=torch.float32)

            # 获取物理参数
            if 'Ra' in f.attrs:
                Ra = float(f.attrs['Ra'])
                Pr = float(f.attrs['Pr'])
            else:
                # 默认值
                Ra, Pr = 1e6, 1.0
                print(f"  [WARNING] Ra, Pr not found in attrs, using default: Ra={Ra}, Pr={Pr}")
        else:
            print(f"  [ERROR] Unexpected data format. Please check the file structure.")
            return

    # 组合成 [B, T, H, W, C]
    # 假设数据是 [N, T, H, W] 或 [T, H, W]
    if b.dim() == 3:
        # [T, H, W] -> [1, T, H, W, C]
        pred = torch.stack([b, u, w, p], dim=-1).unsqueeze(0)
    elif b.dim() == 4:
        # [N, T, H, W] -> [N, T, H, W, C]
        pred = torch.stack([b, u, w, p], dim=-1)

    print(f"\nPred shape: {pred.shape}")
    print(f"Ra = {Ra:.2e}, Pr = {Pr}")

    kappa, nu = compute_transport_coefficients(Ra, Pr)
    print(f"κ = {kappa:.6e}, ν = {nu:.6e}")

    # 计算 PDE 残差
    total_loss, losses = rayleigh_benard_pde_loss(pred, Ra, Pr)

    print(f"\n{'='*40}")
    print("PDE Residuals (GT data):")
    print(f"{'='*40}")
    for name, loss in losses.items():
        status = "✅" if loss.item() < 1e-6 else "⚠️"
        print(f"  {name:15s}: {loss.item():.6e}  {status}")

    print(f"\n{'='*40}")
    if total_loss.item() < 1e-6:
        print("✅ PDE loss 验证通过！残差接近机器精度。")
    elif total_loss.item() < 1e-3:
        print("⚠️ 残差较小，可能由于数值精度或时间步长差异。")
    else:
        print("❌ 残差较大，请检查数据格式或差分格式。")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Rayleigh-Bénard PDE loss")
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to GT data file (HDF5)')
    args = parser.parse_args()

    test_with_gt_data(args.data_path)
