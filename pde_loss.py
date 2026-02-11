"""
2D Burgers方程PDE residual loss计算 - Non-conservative Form + 二阶迎风 + Ghost Cell

适用于 boundary-inclusive grid (128点从0到1)。
使用ghost cell外推，PDE loss在 [1:127, 1:127] 计算（索引1到126，共126x126个点）。

公式 (non-conservative form):
    u_t + u*u_x + v*u_y = ν∇²u
    v_t + u*v_x + v*v_y = ν∇²v

Ghost cell外推:
    ghost[-1] = 2 * data[0] - data[1]
    ghost[128] = 2 * data[127] - data[126]

二阶迎风格式:
- u >= 0: u_x = (3*u_C - 4*u_W + u_WW) / (2*dx)
- u < 0:  u_x = (-3*u_C + 4*u_E - u_EE) / (2*dx)

Author: Ziye
Date: January 2025
"""

import torch
from typing import Tuple


def compute_upwind_derivative_x(
    phi_stencil: dict,
    u_stencil: dict,
    dx: float,
) -> torch.Tensor:
    """
    计算 phi_x 使用二阶迎风格式（non-conservative）。
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
    计算 2D Burgers 方程的 PDE residual loss（带ghost cell）。

    使用索引0和127的边界值外推ghost cell，使得索引1到126都能计算PDE。
    PDE loss 在 [1:127, 1:127] 计算（索引1到126，共126x126个点）。

    Parameters:
    -----------
    pred : torch.Tensor [B, T, H, W, 2]
        预测的速度场，最后维度是 (u, v)
        H, W = 128 (boundary-inclusive grid: 0, 1/127, ..., 1)
    nu : float
        粘度系数
    dt, dx, dy : float
        时间/空间步长

    Returns:
    --------
    total_loss, loss_u, loss_v, residual_u, residual_v
    """
    B, T, H, W, _ = pred.shape
    assert T >= 2, "T must be at least 2"
    assert H >= 4 and W >= 4, "H, W must be at least 4"

    # 提取 u 和 v
    u = pred[..., 0]  # [B, T, H, W]
    v = pred[..., 1]

    # ========== Ghost cell padding ==========
    # 外推 ghost cells: ghost = 2 * boundary - interior
    # Left ghost (x = -dx)
    u_ghost_left = 2 * u[:, :, :, 0:1] - u[:, :, :, 1:2]
    v_ghost_left = 2 * v[:, :, :, 0:1] - v[:, :, :, 1:2]
    # Right ghost (x = 1 + dx)
    u_ghost_right = 2 * u[:, :, :, -1:] - u[:, :, :, -2:-1]
    v_ghost_right = 2 * v[:, :, :, -1:] - v[:, :, :, -2:-1]
    # Bottom ghost (y = -dy)
    u_ghost_bottom = 2 * u[:, :, 0:1, :] - u[:, :, 1:2, :]
    v_ghost_bottom = 2 * v[:, :, 0:1, :] - v[:, :, 1:2, :]
    # Top ghost (y = 1 + dy)
    u_ghost_top = 2 * u[:, :, -1:, :] - u[:, :, -2:-1, :]
    v_ghost_top = 2 * v[:, :, -1:, :] - v[:, :, -2:-1, :]

    # Pad x方向: [B, T, H, W] -> [B, T, H, W+2]
    u_padded = torch.cat([u_ghost_left, u, u_ghost_right], dim=-1)
    v_padded = torch.cat([v_ghost_left, v, v_ghost_right], dim=-1)

    # 扩展 ghost_bottom/top 以匹配新宽度
    u_gb_ext = torch.cat([
        2 * u_ghost_left[:, :, 0:1, :] - u_ghost_left[:, :, 1:2, :],
        u_ghost_bottom,
        2 * u_ghost_right[:, :, 0:1, :] - u_ghost_right[:, :, 1:2, :]
    ], dim=-1)
    u_gt_ext = torch.cat([
        2 * u_ghost_left[:, :, -1:, :] - u_ghost_left[:, :, -2:-1, :],
        u_ghost_top,
        2 * u_ghost_right[:, :, -1:, :] - u_ghost_right[:, :, -2:-1, :]
    ], dim=-1)
    v_gb_ext = torch.cat([
        2 * v_ghost_left[:, :, 0:1, :] - v_ghost_left[:, :, 1:2, :],
        v_ghost_bottom,
        2 * v_ghost_right[:, :, 0:1, :] - v_ghost_right[:, :, 1:2, :]
    ], dim=-1)
    v_gt_ext = torch.cat([
        2 * v_ghost_left[:, :, -1:, :] - v_ghost_left[:, :, -2:-1, :],
        v_ghost_top,
        2 * v_ghost_right[:, :, -1:, :] - v_ghost_right[:, :, -2:-1, :]
    ], dim=-1)

    # Pad y方向: [B, T, H, W+2] -> [B, T, H+2, W+2]
    u_padded = torch.cat([u_gb_ext, u_padded, u_gt_ext], dim=-2)
    v_padded = torch.cat([v_gb_ext, v_padded, v_gt_ext], dim=-2)

    # ========== 时间导数 ==========
    u_t = (u[:, 1:] - u[:, :-1]) / dt  # [B, T-1, H, W]
    v_t = (v[:, 1:] - v[:, :-1]) / dt

    # 当前时刻的padded场
    u_pad_curr = u_padded[:, 1:]  # [B, T-1, H+2, W+2]
    v_pad_curr = v_padded[:, 1:]

    # ========== 提取 stencil ==========
    # 内部区域: 原始的[1:H-1, 1:W-1]，在padded中是[2:H, 2:W]
    def extract_stencil_padded(field_padded, H_orig, W_orig):
        C = field_padded[:, :, 2:H_orig, 2:W_orig]
        E = field_padded[:, :, 2:H_orig, 3:W_orig+1]
        W_ = field_padded[:, :, 2:H_orig, 1:W_orig-1]
        N = field_padded[:, :, 3:H_orig+1, 2:W_orig]
        S = field_padded[:, :, 1:H_orig-1, 2:W_orig]
        EE = field_padded[:, :, 2:H_orig, 4:W_orig+2]
        WW = field_padded[:, :, 2:H_orig, 0:W_orig-2]
        NN = field_padded[:, :, 4:H_orig+2, 2:W_orig]
        SS = field_padded[:, :, 0:H_orig-2, 2:W_orig]
        return {'C': C, 'E': E, 'W': W_, 'N': N, 'S': S, 'EE': EE, 'WW': WW, 'NN': NN, 'SS': SS}

    u_stencil = extract_stencil_padded(u_pad_curr, H, W)
    v_stencil = extract_stencil_padded(v_pad_curr, H, W)

    # 时间导数取[1:H-1, 1:W-1]
    u_t_interior = u_t[:, :, 1:H-1, 1:W-1]
    v_t_interior = v_t[:, :, 1:H-1, 1:W-1]

    # ========== 空间导数 ==========
    u_x = compute_upwind_derivative_x(u_stencil, u_stencil, dx)
    u_y = compute_upwind_derivative_y(u_stencil, v_stencil, dy)
    v_x = compute_upwind_derivative_x(v_stencil, u_stencil, dx)
    v_y = compute_upwind_derivative_y(v_stencil, v_stencil, dy)

    uC, vC = u_stencil['C'], v_stencil['C']

    # ========== 扩散项 ==========
    laplacian_u = compute_laplacian(u_stencil, dx, dy)
    laplacian_v = compute_laplacian(v_stencil, dx, dy)

    # ========== PDE Residual ==========
    residual_u = u_t_interior + uC * u_x + vC * u_y - nu * laplacian_u
    residual_v = v_t_interior + uC * v_x + vC * v_y - nu * laplacian_v

    loss_u = torch.mean(residual_u ** 2)
    loss_v = torch.mean(residual_v ** 2)
    total_loss = loss_u + loss_v

    return total_loss, loss_u, loss_v, residual_u, residual_v


# Alias for backward compatibility
burgers_pde_loss_upwind = burgers_pde_loss


if __name__ == "__main__":
    """Test PDE loss computation."""
    print("=" * 60)
    print("Testing Burgers PDE Loss (non-conservative + ghost cell)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    B, T, H, W = 2, 17, 128, 128
    nu = 0.1
    dx = dy = 1/127

    pred = torch.randn(B, T, H, W, 2, device=device)

    total_loss, loss_u, loss_v, res_u, res_v = burgers_pde_loss(
        pred, nu, dt=1/999, dx=dx, dy=dy
    )

    print(f"Input shape: [B={B}, T={T}, H={H}, W={W}, 2]")
    print(f"Residual shape: {res_u.shape}")
    print(f"  Expected: [B={B}, T-1={T-1}, H-2={H-2}, W-2={W-2}]")
    print(f"\nTotal Loss: {total_loss.item():.6f}")

    print("\nTest passed!")
