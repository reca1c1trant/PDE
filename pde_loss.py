"""
2D Burgers方程PDE residual loss计算 - Conservative Flux Form (n-PINN style)

采用 n-PINN 论文中的 **conservative flux splitting** 形式计算对流项。

公式:
    u_t + (u*u)_x + (v*u)_y = ν∇²u
    v_t + (u*v)_x + (v*v)_y = ν∇²v

关键实现:
1. Cell face 平均速度: uc_e = 0.5*(uE + uC), uc_w = 0.5*(uW + uC)
2. 二阶迎风插值:
   - phi_e = 1.5*phiC - 0.5*phiW  (if uc_e >= 0)
   - phi_e = 1.5*phiE - 0.5*phiEE (if uc_e < 0)
3. Flux 差分: (uc_e*phi_e - uc_w*phi_w) / dx

Author: Ziye
Date: January 2025
"""

import torch
from typing import Tuple


def pad_with_boundaries_2x(
    interior: torch.Tensor,
    boundary_left: torch.Tensor,
    boundary_right: torch.Tensor,
    boundary_bottom: torch.Tensor,
    boundary_top: torch.Tensor,
) -> torch.Tensor:
    """
    使用边界值对内部网格进行2层padding，用于二阶迎风格式。

    Ghost cell 外推方式 (线性外推):
    - ghost1 = 2*boundary - interior[0]  (1x距离)
    - ghost2 = 3*boundary - 2*interior[0]  (2x距离)

    Parameters:
    -----------
    interior : torch.Tensor [B, T, H, W]
    boundary_* : torch.Tensor

    Returns:
    --------
    padded : torch.Tensor [B, T, H+4, W+4]
    """
    B, T, H, W = interior.shape

    bl = boundary_left.squeeze(-1)      # [B, T, H]
    br = boundary_right.squeeze(-1)
    bb = boundary_bottom.squeeze(-2)    # [B, T, W]
    bt = boundary_top.squeeze(-2)

    # 第一层 ghost cells (1x距离)
    ghost_left_1 = 2 * bl - interior[..., 0]
    ghost_right_1 = 2 * br - interior[..., -1]
    ghost_bottom_1 = 2 * bb - interior[..., 0, :]
    ghost_top_1 = 2 * bt - interior[..., -1, :]

    # 第二层 ghost cells (2x距离)
    ghost_left_2 = 2 * ghost_left_1 - bl
    ghost_right_2 = 2 * ghost_right_1 - br
    ghost_bottom_2 = 2 * ghost_bottom_1 - bb
    ghost_top_2 = 2 * ghost_top_1 - bt

    # 构建 padded tensor
    gl2 = ghost_left_2.unsqueeze(-1)
    gl1 = ghost_left_1.unsqueeze(-1)
    gr1 = ghost_right_1.unsqueeze(-1)
    gr2 = ghost_right_2.unsqueeze(-1)
    middle_rows = torch.cat([gl2, gl1, interior, gr1, gr2], dim=-1)  # [B, T, H, W+4]

    def make_corner(ghost_x, ghost_y):
        return (ghost_x + ghost_y) / 2

    # Bottom rows
    cb2_ll = make_corner(ghost_left_2[:, :, 0], ghost_bottom_2[:, :, 0])
    cb2_l = make_corner(ghost_left_1[:, :, 0], ghost_bottom_2[:, :, 0])
    cb2_r = make_corner(ghost_right_1[:, :, 0], ghost_bottom_2[:, :, -1])
    cb2_rr = make_corner(ghost_right_2[:, :, 0], ghost_bottom_2[:, :, -1])
    bottom_row_2 = torch.cat([
        cb2_ll.unsqueeze(-1).unsqueeze(-1),
        cb2_l.unsqueeze(-1).unsqueeze(-1),
        ghost_bottom_2.unsqueeze(-2),
        cb2_r.unsqueeze(-1).unsqueeze(-1),
        cb2_rr.unsqueeze(-1).unsqueeze(-1)
    ], dim=-1)

    cb1_ll = make_corner(ghost_left_2[:, :, 0], ghost_bottom_1[:, :, 0])
    cb1_l = make_corner(ghost_left_1[:, :, 0], ghost_bottom_1[:, :, 0])
    cb1_r = make_corner(ghost_right_1[:, :, 0], ghost_bottom_1[:, :, -1])
    cb1_rr = make_corner(ghost_right_2[:, :, 0], ghost_bottom_1[:, :, -1])
    bottom_row_1 = torch.cat([
        cb1_ll.unsqueeze(-1).unsqueeze(-1),
        cb1_l.unsqueeze(-1).unsqueeze(-1),
        ghost_bottom_1.unsqueeze(-2),
        cb1_r.unsqueeze(-1).unsqueeze(-1),
        cb1_rr.unsqueeze(-1).unsqueeze(-1)
    ], dim=-1)

    # Top rows
    ct1_ll = make_corner(ghost_left_2[:, :, -1], ghost_top_1[:, :, 0])
    ct1_l = make_corner(ghost_left_1[:, :, -1], ghost_top_1[:, :, 0])
    ct1_r = make_corner(ghost_right_1[:, :, -1], ghost_top_1[:, :, -1])
    ct1_rr = make_corner(ghost_right_2[:, :, -1], ghost_top_1[:, :, -1])
    top_row_1 = torch.cat([
        ct1_ll.unsqueeze(-1).unsqueeze(-1),
        ct1_l.unsqueeze(-1).unsqueeze(-1),
        ghost_top_1.unsqueeze(-2),
        ct1_r.unsqueeze(-1).unsqueeze(-1),
        ct1_rr.unsqueeze(-1).unsqueeze(-1)
    ], dim=-1)

    ct2_ll = make_corner(ghost_left_2[:, :, -1], ghost_top_2[:, :, 0])
    ct2_l = make_corner(ghost_left_1[:, :, -1], ghost_top_2[:, :, 0])
    ct2_r = make_corner(ghost_right_1[:, :, -1], ghost_top_2[:, :, -1])
    ct2_rr = make_corner(ghost_right_2[:, :, -1], ghost_top_2[:, :, -1])
    top_row_2 = torch.cat([
        ct2_ll.unsqueeze(-1).unsqueeze(-1),
        ct2_l.unsqueeze(-1).unsqueeze(-1),
        ghost_top_2.unsqueeze(-2),
        ct2_r.unsqueeze(-1).unsqueeze(-1),
        ct2_rr.unsqueeze(-1).unsqueeze(-1)
    ], dim=-1)

    padded = torch.cat([bottom_row_2, bottom_row_1, middle_rows, top_row_1, top_row_2], dim=-2)
    return padded


def extract_stencil(u_padded: torch.Tensor, H: int, W: int) -> dict:
    """
    从 padded tensor 中提取 5-point stencil + 二阶邻居。

    Parameters:
    -----------
    u_padded : torch.Tensor [B, T, H+4, W+4]
    H, W : int

    Returns:
    --------
    dict: 包含 uC, uE, uW, uN, uS, uEE, uWW, uNN, uSS
    """
    # 内部区域从索引 2 开始
    uC = u_padded[:, :, 2:H+2, 2:W+2]    # Center
    uE = u_padded[:, :, 2:H+2, 3:W+3]    # East (x+dx)
    uW = u_padded[:, :, 2:H+2, 1:W+1]    # West (x-dx)
    uN = u_padded[:, :, 3:H+3, 2:W+2]    # North (y+dy)
    uS = u_padded[:, :, 1:H+1, 2:W+2]    # South (y-dy)

    uEE = u_padded[:, :, 2:H+2, 4:W+4]   # East-East (x+2dx)
    uWW = u_padded[:, :, 2:H+2, 0:W]     # West-West (x-2dx)
    uNN = u_padded[:, :, 4:H+4, 2:W+2]   # North-North (y+2dy)
    uSS = u_padded[:, :, 0:H, 2:W+2]     # South-South (y-2dy)

    return {
        'C': uC, 'E': uE, 'W': uW, 'N': uN, 'S': uS,
        'EE': uEE, 'WW': uWW, 'NN': uNN, 'SS': uSS
    }


def compute_convection_flux_x(
    u_stencil: dict,
    phi_stencil: dict,
    dx: float,
) -> torch.Tensor:
    """
    计算 x 方向的对流 flux (与 n-PINN 完全一致)。

    使用 conservative flux splitting 形式:
    (u*phi)_x ≈ (uc_e * phi_e - uc_w * phi_w) / dx

    其中:
    - uc_e = 0.5*(uE + uC) : East face 的速度
    - uc_w = 0.5*(uW + uC) : West face 的速度
    - phi_e, phi_w : 二阶迎风插值的 phi 值

    Parameters:
    -----------
    u_stencil : dict
        速度场的 stencil (决定迎风方向)
    phi_stencil : dict
        被输运量的 stencil
    dx : float

    Returns:
    --------
    flux_x : torch.Tensor [B, T, H, W]
    """
    uC, uE, uW = u_stencil['C'], u_stencil['E'], u_stencil['W']
    phiC = phi_stencil['C']
    phiE, phiW = phi_stencil['E'], phi_stencil['W']
    phiEE, phiWW = phi_stencil['EE'], phi_stencil['WW']

    # Cell face 平均速度 (中心差分)
    uc_e = 0.5 * (uE + uC)  # East face
    uc_w = 0.5 * (uW + uC)  # West face

    # 二阶迎风插值 (与 n-PINN 一致)
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
    计算 y 方向的对流 flux (与 n-PINN 完全一致)。

    使用 conservative flux splitting 形式:
    (v*phi)_y ≈ (vc_n * phi_n - vc_s * phi_s) / dy

    Parameters:
    -----------
    v_stencil : dict
        速度场的 stencil (决定迎风方向)
    phi_stencil : dict
        被输运量的 stencil
    dy : float

    Returns:
    --------
    flux_y : torch.Tensor [B, T, H, W]
    """
    vC, vN, vS = v_stencil['C'], v_stencil['N'], v_stencil['S']
    phiC = phi_stencil['C']
    phiN, phiS = phi_stencil['N'], phi_stencil['S']
    phiNN, phiSS = phi_stencil['NN'], phi_stencil['SS']

    # Cell face 平均速度 (中心差分)
    vc_n = 0.5 * (vN + vC)  # North face
    vc_s = 0.5 * (vS + vC)  # South face

    # 二阶迎风插值 (与 n-PINN 一致)
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

    Parameters:
    -----------
    phi_stencil : dict
    dx, dy : float

    Returns:
    --------
    laplacian : torch.Tensor [B, T, H, W]
    """
    phiC = phi_stencil['C']
    phiE, phiW = phi_stencil['E'], phi_stencil['W']
    phiN, phiS = phi_stencil['N'], phi_stencil['S']

    phi_xx = (phiE - 2 * phiC + phiW) / (dx ** 2)
    phi_yy = (phiN - 2 * phiC + phiS) / (dy ** 2)

    return phi_xx + phi_yy


def burgers_pde_loss(
    pred: torch.Tensor,
    boundary_left: torch.Tensor,
    boundary_right: torch.Tensor,
    boundary_bottom: torch.Tensor,
    boundary_top: torch.Tensor,
    nu: float,
    dt: float = 1/999,
    Lx: float = 1.0,
    Ly: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算 2D Burgers 方程的 PDE residual loss (与 n-PINN 完全一致)。

    使用 conservative flux splitting 形式:
    - u_t + (u*u)_x + (v*u)_y = ν∇²u
    - v_t + (u*v)_x + (v*v)_y = ν∇²v

    Parameters:
    -----------
    pred : torch.Tensor [B, T, H, W, 2]
        预测的速度场，最后维度是 (u, v)
    boundary_* : torch.Tensor
        边界值
    nu : float
        粘度系数
    dt : float
        时间步长
    Lx, Ly : float
        计算域尺寸

    Returns:
    --------
    total_loss : torch.Tensor
    loss_u : torch.Tensor
    loss_v : torch.Tensor
    residual_u : torch.Tensor [B, T-1, H, W]
    residual_v : torch.Tensor [B, T-1, H, W]
    """
    B, T, H, W, _ = pred.shape
    assert T >= 2, "T must be at least 2"

    dx = Lx / H
    dy = Ly / W

    # 提取 u 和 v
    u = pred[..., 0]  # [B, T, H, W]
    v = pred[..., 1]

    # 提取边界的 u 和 v
    u_left = boundary_left[..., 0]
    v_left = boundary_left[..., 1]
    u_right = boundary_right[..., 0]
    v_right = boundary_right[..., 1]
    u_bottom = boundary_bottom[..., 0]
    v_bottom = boundary_bottom[..., 1]
    u_top = boundary_top[..., 0]
    v_top = boundary_top[..., 1]

    # Padding (2层 ghost cells)
    u_padded = pad_with_boundaries_2x(u, u_left, u_right, u_bottom, u_top)
    v_padded = pad_with_boundaries_2x(v, v_left, v_right, v_bottom, v_top)

    # ========== 时间导数 (后向差分) ==========
    u_t = (u[:, 1:] - u[:, :-1]) / dt  # [B, T-1, H, W]
    v_t = (v[:, 1:] - v[:, :-1]) / dt

    # 当前时刻的 padded 场 [B, T-1, H+4, W+4]
    u_padded_current = u_padded[:, 1:]
    v_padded_current = v_padded[:, 1:]

    # 提取 stencils
    u_stencil = extract_stencil(u_padded_current, H, W)
    v_stencil = extract_stencil(v_padded_current, H, W)

    # ========== 对流项 (n-PINN flux splitting) ==========
    # u 方程: (u*u)_x + (v*u)_y
    UUx = compute_convection_flux_x(u_stencil, u_stencil, dx)  # (u*u)_x
    VUy = compute_convection_flux_y(v_stencil, u_stencil, dy)  # (v*u)_y

    # v 方程: (u*v)_x + (v*v)_y
    UVx = compute_convection_flux_x(u_stencil, v_stencil, dx)  # (u*v)_x
    VVy = compute_convection_flux_y(v_stencil, v_stencil, dy)  # (v*v)_y

    # ========== 扩散项 (二阶中心差分) ==========
    laplacian_u = compute_laplacian(u_stencil, dx, dy)
    laplacian_v = compute_laplacian(v_stencil, dx, dy)

    # ========== PDE Residual ==========
    # Burgers: u_t + (u*u)_x + (v*u)_y = ν∇²u
    residual_u = u_t + UUx + VUy - nu * laplacian_u
    residual_v = v_t + UVx + VVy - nu * laplacian_v

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
    print("Testing Burgers PDE Loss (n-PINN style)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    B, T, H, W = 2, 17, 64, 64
    nu = 0.01

    # Create dummy data
    pred = torch.randn(B, T, H, W, 2, device=device)
    bnd_left = torch.randn(B, T, H, 1, 2, device=device)
    bnd_right = torch.randn(B, T, H, 1, 2, device=device)
    bnd_bottom = torch.randn(B, T, 1, W, 2, device=device)
    bnd_top = torch.randn(B, T, 1, W, 2, device=device)

    total_loss, loss_u, loss_v, res_u, res_v = burgers_pde_loss(
        pred, bnd_left, bnd_right, bnd_bottom, bnd_top, nu
    )

    print(f"Total Loss: {total_loss.item():.6f}")
    print(f"Loss U: {loss_u.item():.6f}")
    print(f"Loss V: {loss_v.item():.6f}")
    print(f"Residual U shape: {res_u.shape}")
    print(f"Residual V shape: {res_v.shape}")

    print("\nBurgers PDE Loss test passed!")
