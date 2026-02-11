"""
2D Burgers方程PDE residual loss计算 - Conservative Flux Form (n-PINN style)

适用于 boundary-inclusive grid (128点从0到1)。
PDE loss 只在内部区域 [2:126, 2:126] 计算（索引2到125）。

公式 (conservative form):
    u_t + (u*u)_x + (v*u)_y = ν∇²u
    v_t + (u*v)_x + (v*v)_y = ν∇²v

二阶迎风格式:
1. Cell face 平均速度: uc_e = 0.5*(uE + uC)
2. 二阶迎风插值:
   - phi_e = 1.5*phiC - 0.5*phiW  (if uc_e >= 0)
   - phi_e = 1.5*phiE - 0.5*phiEE (if uc_e < 0)
3. Flux 差分: (uc_e*phi_e - uc_w*phi_w) / dx

Author: Ziye
Date: January 2025
"""

import torch
from typing import Tuple


def extract_stencil_interior(field: torch.Tensor, H: int, W: int) -> dict:
    """
    从完整场中提取内部区域的 stencil。

    内部区域: [2:H-2, 2:W-2]，即索引2到H-3（对于H=128就是2到125）

    Parameters:
    -----------
    field : torch.Tensor [B, T, H, W]
        完整场（包含边界）
    H, W : int
        网格尺寸（128）

    Returns:
    --------
    dict: 包含 C, E, W, N, S, EE, WW, NN, SS（都是内部区域大小）
    """
    # 内部区域: [2:H-2, 2:W-2]
    # 对于128x128，就是[2:126, 2:126]，共124x124个点

    C = field[:, :, 2:H-2, 2:W-2]      # Center
    E = field[:, :, 2:H-2, 3:W-1]      # East (x+dx)
    W_ = field[:, :, 2:H-2, 1:W-3]     # West (x-dx)
    N = field[:, :, 3:H-1, 2:W-2]      # North (y+dy)
    S = field[:, :, 1:H-3, 2:W-2]      # South (y-dy)

    EE = field[:, :, 2:H-2, 4:W]       # East-East (x+2dx)
    WW = field[:, :, 2:H-2, 0:W-4]     # West-West (x-2dx)
    NN = field[:, :, 4:H, 2:W-2]       # North-North (y+2dy)
    SS = field[:, :, 0:H-4, 2:W-2]     # South-South (y-2dy)

    return {
        'C': C, 'E': E, 'W': W_, 'N': N, 'S': S,
        'EE': EE, 'WW': WW, 'NN': NN, 'SS': SS
    }


def compute_convection_flux_x(
    u_stencil: dict,
    phi_stencil: dict,
    dx: float,
) -> torch.Tensor:
    """
    计算 x 方向的对流 flux (n-PINN conservative flux splitting)。

    (u*phi)_x ≈ (uc_e * phi_e - uc_w * phi_w) / dx
    """
    uC, uE, uW = u_stencil['C'], u_stencil['E'], u_stencil['W']
    phiC = phi_stencil['C']
    phiE, phiW = phi_stencil['E'], phi_stencil['W']
    phiEE, phiWW = phi_stencil['EE'], phi_stencil['WW']

    # Cell face 平均速度
    uc_e = 0.5 * (uE + uC)  # East face
    uc_w = 0.5 * (uW + uC)  # West face

    # 二阶迎风插值
    # East face: phi_e
    phi_e_minus = 1.5 * phiC - 0.5 * phiW    # 从左边插值 (uc_e >= 0)
    phi_e_plus = 1.5 * phiE - 0.5 * phiEE    # 从右边插值 (uc_e < 0)
    phi_e = torch.where(uc_e >= 0, phi_e_minus, phi_e_plus)

    # West face: phi_w
    phi_w_minus = 1.5 * phiW - 0.5 * phiWW   # 从左边插值 (uc_w >= 0)
    phi_w_plus = 1.5 * phiC - 0.5 * phiE     # 从右边插值 (uc_w < 0)
    phi_w = torch.where(uc_w >= 0, phi_w_minus, phi_w_plus)

    # Flux 差分
    flux_x = (uc_e * phi_e - uc_w * phi_w) / dx
    return flux_x


def compute_convection_flux_y(
    v_stencil: dict,
    phi_stencil: dict,
    dy: float,
) -> torch.Tensor:
    """
    计算 y 方向的对流 flux (n-PINN conservative flux splitting)。

    (v*phi)_y ≈ (vc_n * phi_n - vc_s * phi_s) / dy
    """
    vC, vN, vS = v_stencil['C'], v_stencil['N'], v_stencil['S']
    phiC = phi_stencil['C']
    phiN, phiS = phi_stencil['N'], phi_stencil['S']
    phiNN, phiSS = phi_stencil['NN'], phi_stencil['SS']

    # Cell face 平均速度
    vc_n = 0.5 * (vN + vC)  # North face
    vc_s = 0.5 * (vS + vC)  # South face

    # 二阶迎风插值
    # North face: phi_n
    phi_n_minus = 1.5 * phiC - 0.5 * phiS    # 从下边插值 (vc_n >= 0)
    phi_n_plus = 1.5 * phiN - 0.5 * phiNN    # 从上边插值 (vc_n < 0)
    phi_n = torch.where(vc_n >= 0, phi_n_minus, phi_n_plus)

    # South face: phi_s
    phi_s_minus = 1.5 * phiS - 0.5 * phiSS   # 从下边插值 (vc_s >= 0)
    phi_s_plus = 1.5 * phiC - 0.5 * phiN     # 从上边插值 (vc_s < 0)
    phi_s = torch.where(vc_s >= 0, phi_s_minus, phi_s_plus)

    # Flux 差分
    flux_y = (vc_n * phi_n - vc_s * phi_s) / dy
    return flux_y


def compute_laplacian(phi_stencil: dict, dx: float, dy: float) -> torch.Tensor:
    """
    计算 Laplacian (二阶中心差分)。

    ∇²phi = (phiE - 2*phiC + phiW) / dx² + (phiN - 2*phiC + phiS) / dy²
    """
    phiC = phi_stencil['C']
    phiE, phiW = phi_stencil['E'], phi_stencil['W']
    phiN, phiS = phi_stencil['N'], phi_stencil['S']

    phi_xx = (phiE - 2 * phiC + phiW) / (dx ** 2)
    phi_yy = (phiN - 2 * phiC + phiS) / (dy ** 2)

    return phi_xx + phi_yy


def burgers_pde_loss(
    pred: torch.Tensor,
    nu: float,
    dt: float = 1/999,
    dx: float = 1/127,
    dy: float = 1/127,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算 2D Burgers 方程的 PDE residual loss。

    只在内部区域 [2:H-2, 2:W-2] 计算（对于128x128就是[2:126, 2:126]）。

    Parameters:
    -----------
    pred : torch.Tensor [B, T, H, W, 2]
        预测的速度场，最后维度是 (u, v)
        H, W = 128 (boundary-inclusive grid)
    nu : float
        粘度系数
    dt : float
        时间步长 (default: 1/999)
    dx, dy : float
        空间步长 (default: 1/127)

    Returns:
    --------
    total_loss : torch.Tensor
    loss_u : torch.Tensor
    loss_v : torch.Tensor
    residual_u : torch.Tensor [B, T-1, H-4, W-4]
    residual_v : torch.Tensor [B, T-1, H-4, W-4]
    """
    B, T, H, W, _ = pred.shape
    assert T >= 2, "T must be at least 2"
    assert H >= 5 and W >= 5, "H, W must be at least 5 for 2nd order scheme"

    # 提取 u 和 v
    u = pred[..., 0]  # [B, T, H, W]
    v = pred[..., 1]

    # ========== 时间导数 (后向差分) ==========
    # 使用当前时刻的空间导数
    u_t = (u[:, 1:] - u[:, :-1]) / dt  # [B, T-1, H, W]
    v_t = (v[:, 1:] - v[:, :-1]) / dt

    # 当前时刻的场 (t=1 to T-1)
    u_current = u[:, 1:]  # [B, T-1, H, W]
    v_current = v[:, 1:]

    # 提取内部区域的 stencils
    u_stencil = extract_stencil_interior(u_current, H, W)
    v_stencil = extract_stencil_interior(v_current, H, W)

    # 时间导数也只取内部区域
    u_t_interior = u_t[:, :, 2:H-2, 2:W-2]
    v_t_interior = v_t[:, :, 2:H-2, 2:W-2]

    # ========== 对流项 (n-PINN flux splitting) ==========
    # u 方程: (u*u)_x + (v*u)_y
    UUx = compute_convection_flux_x(u_stencil, u_stencil, dx)
    VUy = compute_convection_flux_y(v_stencil, u_stencil, dy)

    # v 方程: (u*v)_x + (v*v)_y
    UVx = compute_convection_flux_x(u_stencil, v_stencil, dx)
    VVy = compute_convection_flux_y(v_stencil, v_stencil, dy)

    # ========== 扩散项 (二阶中心差分) ==========
    laplacian_u = compute_laplacian(u_stencil, dx, dy)
    laplacian_v = compute_laplacian(v_stencil, dx, dy)

    # ========== PDE Residual ==========
    # Burgers: u_t + (u*u)_x + (v*u)_y = ν∇²u
    residual_u = u_t_interior + UUx + VUy - nu * laplacian_u
    residual_v = v_t_interior + UVx + VVy - nu * laplacian_v

    # MSE Loss
    loss_u = torch.mean(residual_u ** 2)
    loss_v = torch.mean(residual_v ** 2)
    total_loss = loss_u + loss_v

    return total_loss, loss_u, loss_v, residual_u, residual_v


# Alias for backward compatibility
burgers_pde_loss_upwind = burgers_pde_loss


if __name__ == "__main__":
    """Test PDE loss computation."""
    print("=" * 60)
    print("Testing Burgers PDE Loss (n-PINN, boundary-inclusive grid)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    B, T, H, W = 2, 17, 128, 128
    nu = 0.1
    dx = dy = 1/127

    # Create dummy data
    pred = torch.randn(B, T, H, W, 2, device=device)

    total_loss, loss_u, loss_v, res_u, res_v = burgers_pde_loss(
        pred, nu, dt=1/999, dx=dx, dy=dy
    )

    print(f"Input shape: [B={B}, T={T}, H={H}, W={W}, 2]")
    print(f"Residual shape: {res_u.shape}")
    print(f"  Expected: [B={B}, T-1={T-1}, H-4={H-4}, W-4={W-4}]")
    print(f"\nTotal Loss: {total_loss.item():.6f}")
    print(f"Loss U: {loss_u.item():.6f}")
    print(f"Loss V: {loss_v.item():.6f}")

    print("\nTest passed!")
