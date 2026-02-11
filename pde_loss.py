"""
2D Burgers方程PDE residual loss计算 - Non-conservative Form + 二阶迎风

适用于 boundary-inclusive grid (128点从0到1)。
PDE loss 只在内部区域 [2:126, 2:126] 计算（索引2到125）。

公式 (non-conservative form):
    u_t + u*u_x + v*u_y = ν∇²u
    v_t + u*v_x + v*v_y = ν∇²v

注意：这与 conservative form 不同！
    - Conservative: u_t + (u*u)_x + (v*u)_y = ν∇²u
    - Non-conservative: u_t + u*u_x + v*u_y = ν∇²u

两者只有在 div(u,v)=0 时才等价。Burgers方程的解析解不满足连续性，
所以必须用 non-conservative form。

二阶迎风格式:
- u_x: 根据u的符号选择迎风方向
  - u >= 0: u_x = (3*u_C - 4*u_W + u_WW) / (2*dx)  (后向)
  - u < 0:  u_x = (-3*u_C + 4*u_E - u_EE) / (2*dx) (前向)

Author: Ziye
Date: January 2025
"""

import torch
from typing import Tuple


def extract_stencil_interior(field: torch.Tensor, H: int, W: int) -> dict:
    """
    从完整场中提取内部区域的 stencil。

    内部区域: [2:H-2, 2:W-2]，即索引2到H-3（对于H=128就是2到125）
    """
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


def compute_upwind_derivative_x(
    phi_stencil: dict,
    u_stencil: dict,
    dx: float,
) -> torch.Tensor:
    """
    计算 phi_x 使用二阶迎风格式（non-conservative）。

    根据 u 的符号选择迎风方向：
    - u >= 0: phi_x = (3*phi_C - 4*phi_W + phi_WW) / (2*dx)
    - u < 0:  phi_x = (-3*phi_C + 4*phi_E - phi_EE) / (2*dx)
    """
    uC = u_stencil['C']
    phiC = phi_stencil['C']
    phiE, phiW = phi_stencil['E'], phi_stencil['W']
    phiEE, phiWW = phi_stencil['EE'], phi_stencil['WW']

    # 二阶后向差分 (u >= 0)
    phi_x_backward = (3*phiC - 4*phiW + phiWW) / (2*dx)

    # 二阶前向差分 (u < 0)
    phi_x_forward = (-3*phiC + 4*phiE - phiEE) / (2*dx)

    # 迎风选择
    phi_x = torch.where(uC >= 0, phi_x_backward, phi_x_forward)

    return phi_x


def compute_upwind_derivative_y(
    phi_stencil: dict,
    v_stencil: dict,
    dy: float,
) -> torch.Tensor:
    """
    计算 phi_y 使用二阶迎风格式（non-conservative）。

    根据 v 的符号选择迎风方向：
    - v >= 0: phi_y = (3*phi_C - 4*phi_S + phi_SS) / (2*dy)
    - v < 0:  phi_y = (-3*phi_C + 4*phi_N - phi_NN) / (2*dy)
    """
    vC = v_stencil['C']
    phiC = phi_stencil['C']
    phiN, phiS = phi_stencil['N'], phi_stencil['S']
    phiNN, phiSS = phi_stencil['NN'], phi_stencil['SS']

    # 二阶后向差分 (v >= 0)
    phi_y_backward = (3*phiC - 4*phiS + phiSS) / (2*dy)

    # 二阶前向差分 (v < 0)
    phi_y_forward = (-3*phiC + 4*phiN - phiNN) / (2*dy)

    # 迎风选择
    phi_y = torch.where(vC >= 0, phi_y_backward, phi_y_forward)

    return phi_y


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
    计算 2D Burgers 方程的 PDE residual loss（non-conservative form）。

    PDE (non-conservative):
        u_t + u*u_x + v*u_y = ν∇²u
        v_t + u*v_x + v*v_y = ν∇²v

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

    # ========== 空间导数 (二阶迎风, non-conservative) ==========
    # u 方程: u*u_x + v*u_y
    u_x = compute_upwind_derivative_x(u_stencil, u_stencil, dx)
    u_y = compute_upwind_derivative_y(u_stencil, v_stencil, dy)

    # v 方程: u*v_x + v*v_y
    v_x = compute_upwind_derivative_x(v_stencil, u_stencil, dx)
    v_y = compute_upwind_derivative_y(v_stencil, v_stencil, dy)

    # 中心点速度
    uC = u_stencil['C']
    vC = v_stencil['C']

    # ========== 扩散项 (二阶中心差分) ==========
    laplacian_u = compute_laplacian(u_stencil, dx, dy)
    laplacian_v = compute_laplacian(v_stencil, dx, dy)

    # ========== PDE Residual (non-conservative form) ==========
    # u_t + u*u_x + v*u_y = ν∇²u
    residual_u = u_t_interior + uC * u_x + vC * u_y - nu * laplacian_u
    residual_v = v_t_interior + uC * v_x + vC * v_y - nu * laplacian_v

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
    print("Testing Burgers PDE Loss (non-conservative, 2nd order upwind)")
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
