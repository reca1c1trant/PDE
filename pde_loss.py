"""
2D Burgers方程PDE residual loss计算 - 二阶迎风格式版本

Loss = ||u_t + uu_x + vu_y - ν(u_xx + u_yy)||_2
     + ||v_t + uv_x + vv_y - ν(v_xx + v_yy)||_2

导数计算策略：
- 时间导数：后向差分（从索引1开始）
- 一阶空间导数：**二阶迎风格式** O(h²)
- 二阶空间导数：中心差分 O(h²)

二阶迎风格式公式（与 n-PINN 一致）：
- 正向流动 (velocity >= 0): (3u - 4u_left + u_left_left) / (2dx)
- 负向流动 (velocity < 0): (-u_right_right + 4u_right - 3u) / (2dx)

需要 2x 距离的 ghost cells 用于边界处理。

Author: Ziye
Date: December 2024
Updated: January 2025 - 升级为二阶迎风
"""

import torch


def pad_with_boundaries_2x(interior, boundary_left, boundary_right, boundary_bottom, boundary_top):
    """
    使用边界值对内部网格进行2层padding，用于二阶迎风格式

    外推方式：线性外推
    ghost1 = 2*boundary - interior[0]  (1x距离)
    ghost2 = 3*boundary - 2*interior[0]  (2x距离，继续线性外推)

    Parameters:
    -----------
    interior : torch.Tensor [B, T, H, W]
        内部网格数据
    boundary_left : torch.Tensor [B, T, H, 1]
        左边界 (x=0)
    boundary_right : torch.Tensor [B, T, H, 1]
        右边界 (x=1)
    boundary_bottom : torch.Tensor [B, T, 1, W]
        下边界 (y=0)
    boundary_top : torch.Tensor [B, T, 1, W]
        上边界 (y=1)

    Returns:
    --------
    padded : torch.Tensor [B, T, H+4, W+4]
        Padding后的网格（包含2层ghost cells）

    Note:
        Uses torch.cat instead of index assignment to preserve gradients.
    """
    B, T, H, W = interior.shape

    # 边界值 [B, T, H] or [B, T, W]
    bl = boundary_left.squeeze(-1)      # [B, T, H]
    br = boundary_right.squeeze(-1)     # [B, T, H]
    bb = boundary_bottom.squeeze(-2)    # [B, T, W]
    bt = boundary_top.squeeze(-2)       # [B, T, W]

    # 第一层 ghost cells (1x距离): ghost1 = 2*boundary - interior[adj]
    ghost_left_1 = 2 * bl - interior[..., 0]       # [B, T, H]
    ghost_right_1 = 2 * br - interior[..., -1]     # [B, T, H]
    ghost_bottom_1 = 2 * bb - interior[..., 0, :]  # [B, T, W]
    ghost_top_1 = 2 * bt - interior[..., -1, :]    # [B, T, W]

    # 第二层 ghost cells (2x距离): ghost2 = 2*ghost1 - boundary = 3*boundary - 2*interior[adj]
    ghost_left_2 = 2 * ghost_left_1 - bl       # = 3*bl - 2*interior[..., 0]
    ghost_right_2 = 2 * ghost_right_1 - br     # = 3*br - 2*interior[..., -1]
    ghost_bottom_2 = 2 * ghost_bottom_1 - bb   # = 3*bb - 2*interior[..., 0, :]
    ghost_top_2 = 2 * ghost_top_1 - bt         # = 3*bt - 2*interior[..., -1, :]

    # 中间行：添加左右各2层 ghost → [B, T, H, W+4]
    gl2 = ghost_left_2.unsqueeze(-1)   # [B, T, H, 1]
    gl1 = ghost_left_1.unsqueeze(-1)   # [B, T, H, 1]
    gr1 = ghost_right_1.unsqueeze(-1)  # [B, T, H, 1]
    gr2 = ghost_right_2.unsqueeze(-1)  # [B, T, H, 1]
    middle_rows = torch.cat([gl2, gl1, interior, gr1, gr2], dim=-1)  # [B, T, H, W+4]

    # 构建 bottom 和 top 行需要包含角点
    # 角点用简单的平均近似
    def make_corner(ghost_x, ghost_y):
        """创建角点值，用两个方向的 ghost 的平均"""
        return (ghost_x + ghost_y) / 2

    # Bottom 两行: [B, T, 2, W+4]
    # 行 0: ghost_bottom_2 + 四个角点
    cb2_ll = make_corner(ghost_left_2[:, :, 0], ghost_bottom_2[:, :, 0])     # [B, T]
    cb2_l = make_corner(ghost_left_1[:, :, 0], ghost_bottom_2[:, :, 0])
    cb2_r = make_corner(ghost_right_1[:, :, 0], ghost_bottom_2[:, :, -1])
    cb2_rr = make_corner(ghost_right_2[:, :, 0], ghost_bottom_2[:, :, -1])
    bottom_row_2 = torch.cat([
        cb2_ll.unsqueeze(-1).unsqueeze(-1),  # [B, T, 1, 1]
        cb2_l.unsqueeze(-1).unsqueeze(-1),
        ghost_bottom_2.unsqueeze(-2),         # [B, T, 1, W]
        cb2_r.unsqueeze(-1).unsqueeze(-1),
        cb2_rr.unsqueeze(-1).unsqueeze(-1)
    ], dim=-1)  # [B, T, 1, W+4]

    # 行 1: ghost_bottom_1 + 四个角点
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
    ], dim=-1)  # [B, T, 1, W+4]

    # Top 两行: [B, T, 2, W+4]
    # 行 H+2: ghost_top_1 + 四个角点
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
    ], dim=-1)  # [B, T, 1, W+4]

    # 行 H+3: ghost_top_2 + 四个角点
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
    ], dim=-1)  # [B, T, 1, W+4]

    # 拼接成完整的 padded grid: [B, T, H+4, W+4]
    padded = torch.cat([bottom_row_2, bottom_row_1, middle_rows, top_row_1, top_row_2], dim=-2)

    return padded


def upwind_derivative_x_2nd(u_padded, velocity, H, W, dx):
    """
    计算x方向的二阶迎风导数

    二阶迎风格式 (与 n-PINN 一致):
    - 正向流动 (velocity >= 0): (3u - 4u_left + u_left_left) / (2dx)
    - 负向流动 (velocity < 0): (-u_right_right + 4u_right - 3u) / (2dx)

    Parameters:
    -----------
    u_padded : torch.Tensor [B, T, H+4, W+4]
        padding后的场 (2层ghost cells)
    velocity : torch.Tensor [B, T, H, W]
        决定迎风方向的速度场
    H, W : int
        内部网格尺寸
    dx : float
        网格间距

    Returns:
    --------
    du_dx : torch.Tensor [B, T, H, W]
        二阶迎风格式计算的x方向导数
    """
    # padded 尺寸: [B, T, H+4, W+4]
    # 内部区域从索引 2 开始: u_center = padded[:, :, 2:H+2, 2:W+2]

    u_center = u_padded[:, :, 2:H+2, 2:W+2]      # u[i,j]
    u_left = u_padded[:, :, 2:H+2, 1:W+1]        # u[i,j-1]
    u_left_left = u_padded[:, :, 2:H+2, 0:W]     # u[i,j-2]
    u_right = u_padded[:, :, 2:H+2, 3:W+3]       # u[i,j+1]
    u_right_right = u_padded[:, :, 2:H+2, 4:W+4] # u[i,j+2]

    # 二阶后向差分 (3u - 4u_left + u_left_left) / (2dx)，用于 velocity >= 0
    backward = (3 * u_center - 4 * u_left + u_left_left) / (2 * dx)

    # 二阶前向差分 (-u_right_right + 4u_right - 3u) / (2dx)，用于 velocity < 0
    forward = (-u_right_right + 4 * u_right - 3 * u_center) / (2 * dx)

    # 迎风选择：velocity >= 0 用后向，velocity < 0 用前向
    du_dx = torch.where(velocity >= 0, backward, forward)

    return du_dx


def upwind_derivative_y_2nd(u_padded, velocity, H, W, dy):
    """
    计算y方向的二阶迎风导数

    二阶迎风格式 (与 n-PINN 一致):
    - 正向流动 (velocity >= 0): (3u - 4u_bottom + u_bottom_bottom) / (2dy)
    - 负向流动 (velocity < 0): (-u_top_top + 4u_top - 3u) / (2dy)

    Parameters:
    -----------
    u_padded : torch.Tensor [B, T, H+4, W+4]
        padding后的场 (2层ghost cells)
    velocity : torch.Tensor [B, T, H, W]
        决定迎风方向的速度场
    H, W : int
        内部网格尺寸
    dy : float
        网格间距

    Returns:
    --------
    du_dy : torch.Tensor [B, T, H, W]
        二阶迎风格式计算的y方向导数
    """
    # padded 尺寸: [B, T, H+4, W+4]
    # 内部区域从索引 2 开始

    u_center = u_padded[:, :, 2:H+2, 2:W+2]          # u[i,j]
    u_bottom = u_padded[:, :, 1:H+1, 2:W+2]          # u[i-1,j]
    u_bottom_bottom = u_padded[:, :, 0:H, 2:W+2]     # u[i-2,j]
    u_top = u_padded[:, :, 3:H+3, 2:W+2]             # u[i+1,j]
    u_top_top = u_padded[:, :, 4:H+4, 2:W+2]         # u[i+2,j]

    # 二阶后向差分 (3u - 4u_bottom + u_bottom_bottom) / (2dy)，用于 velocity >= 0
    backward = (3 * u_center - 4 * u_bottom + u_bottom_bottom) / (2 * dy)

    # 二阶前向差分 (-u_top_top + 4u_top - 3u) / (2dy)，用于 velocity < 0
    forward = (-u_top_top + 4 * u_top - 3 * u_center) / (2 * dy)

    # 迎风选择
    du_dy = torch.where(velocity >= 0, backward, forward)

    return du_dy


def burgers_pde_loss_upwind(pred, boundary_left, boundary_right, boundary_bottom, boundary_top,
                            nu, dt=1/999, verbose=False):
    """
    计算2D Burgers方程的PDE residual loss (二阶迎风格式)

    Parameters:
    -----------
    pred : torch.Tensor [B, T, H, W, 2]
        预测的速度场，最后维度是(u, v)
    boundary_left : torch.Tensor [B, T, H, 1, 2]
        左边界(x=0)的值
    boundary_right : torch.Tensor [B, T, H, 1, 2]
        右边界(x=1.0)的值
    boundary_bottom : torch.Tensor [B, T, 1, W, 2]
        下边界(y=0)的值
    boundary_top : torch.Tensor [B, T, 1, W, 2]
        上边界(y=1.0)的值
    nu : float or torch.Tensor
        粘度系数
    dt : float
        时间步长，默认1/999
    verbose : bool
        是否打印debug信息，默认False

    Returns:
    --------
    total_loss : torch.Tensor
        总PDE residual loss (loss_u + loss_v)
    loss_u : torch.Tensor
        u方程的loss
    loss_v : torch.Tensor
        v方程的loss
    residual_u : torch.Tensor [B, T-1, H, W]
        u方程的残差
    residual_v : torch.Tensor [B, T-1, H, W]
        v方程的残差
    """
    B, T, H, W, _ = pred.shape
    assert T >= 2, "T must be at least 2"
    dx = 1.0 / H
    dy = dx

    # 提取u和v
    u = pred[..., 0]  # [B, T, H, W]
    v = pred[..., 1]

    # 提取边界的u和v
    u_left = boundary_left[..., 0]
    v_left = boundary_left[..., 1]
    u_right = boundary_right[..., 0]
    v_right = boundary_right[..., 1]
    u_bottom = boundary_bottom[..., 0]
    v_bottom = boundary_bottom[..., 1]
    u_top = boundary_top[..., 0]
    v_top = boundary_top[..., 1]

    # 使用2层ghost cells进行padding (用于二阶迎风)
    u_padded = pad_with_boundaries_2x(u, u_left, u_right, u_bottom, u_top)  # [B, T, H+4, W+4]
    v_padded = pad_with_boundaries_2x(v, v_left, v_right, v_bottom, v_top)

    # ========== 时间导数（后向差分，从索引1开始）==========
    u_t = (u[:, 1:] - u[:, :-1]) / dt  # [B, T-1, H, W]
    v_t = (v[:, 1:] - v[:, :-1]) / dt

    # 当前时刻的场（索引1到T-1）
    u_current = u[:, 1:]
    v_current = v[:, 1:]

    # 当前时刻的padded场 [B, T-1, H+4, W+4]
    u_padded_current = u_padded[:, 1:]
    v_padded_current = v_padded[:, 1:]

    # ========== 一阶空间导数（二阶迎风格式）==========
    # u方程: u_t + u*u_x + v*u_y = ν(u_xx + u_yy)
    # - u*u_x: 用u作为速度决定u_x的迎风方向
    # - v*u_y: 用v作为速度决定u_y的迎风方向

    u_x = upwind_derivative_x_2nd(u_padded_current, u_current, H, W, dx)
    u_y = upwind_derivative_y_2nd(u_padded_current, v_current, H, W, dy)

    # v方程: v_t + u*v_x + v*v_y = ν(v_xx + v_yy)
    # - u*v_x: 用u作为速度决定v_x的迎风方向
    # - v*v_y: 用v作为速度决定v_y的迎风方向

    v_x = upwind_derivative_x_2nd(v_padded_current, u_current, H, W, dx)
    v_y = upwind_derivative_y_2nd(v_padded_current, v_current, H, W, dy)

    # ========== 二阶空间导数（中心差分，使用2层padding中的内层）==========
    # padded 尺寸: [B, T, H+4, W+4]，内部区域从索引 2 开始
    u_center = u_padded_current[:, :, 2:H+2, 2:W+2]  # [B, T-1, H, W]
    v_center = v_padded_current[:, :, 2:H+2, 2:W+2]

    # 使用 1x 距离的邻居进行中心差分
    u_left_1 = u_padded_current[:, :, 2:H+2, 1:W+1]   # u[i, j-1]
    u_right_1 = u_padded_current[:, :, 2:H+2, 3:W+3]  # u[i, j+1]
    u_bottom_1 = u_padded_current[:, :, 1:H+1, 2:W+2] # u[i-1, j]
    u_top_1 = u_padded_current[:, :, 3:H+3, 2:W+2]    # u[i+1, j]

    v_left_1 = v_padded_current[:, :, 2:H+2, 1:W+1]
    v_right_1 = v_padded_current[:, :, 2:H+2, 3:W+3]
    v_bottom_1 = v_padded_current[:, :, 1:H+1, 2:W+2]
    v_top_1 = v_padded_current[:, :, 3:H+3, 2:W+2]

    u_xx = (u_right_1 - 2 * u_center + u_left_1) / (dx ** 2)
    v_xx = (v_right_1 - 2 * v_center + v_left_1) / (dx ** 2)

    u_yy = (u_top_1 - 2 * u_center + u_bottom_1) / (dy ** 2)
    v_yy = (v_top_1 - 2 * v_center + v_bottom_1) / (dy ** 2)

    # ========== 计算PDE residual ==========
    residual_u = u_t + u_current * u_x + v_current * u_y - nu * (u_xx + u_yy)
    residual_v = v_t + u_current * v_x + v_current * v_y - nu * (v_xx + v_yy)

    # MSE Loss
    loss_u = torch.mean(residual_u ** 2)
    loss_v = torch.mean(residual_v ** 2)
    total_loss = loss_u + loss_v

    # Debug信息
    if verbose:
        with torch.no_grad():
            abs_u = torch.abs(residual_u)
            max_val_u = torch.max(abs_u)
            idx_u = (abs_u == max_val_u).nonzero(as_tuple=False)[0]

            abs_v = torch.abs(residual_v)
            max_val_v = torch.max(abs_v)
            idx_v = (abs_v == max_val_v).nonzero(as_tuple=False)[0]

            print(f"Max Residual U: {max_val_u.item():.5f} at Index (B,T,H,W): {idx_u.tolist()}")
            print(f"Max Residual V: {max_val_v.item():.5f} at Index (B,T,H,W): {idx_v.tolist()}")

    return total_loss, loss_u, loss_v, residual_u, residual_v


# ============== 1x Padding 函数 (用于原始版本对比) ==============
def pad_with_boundaries_1x(interior, boundary_left, boundary_right, boundary_bottom, boundary_top):
    """
    使用边界值对内部网格进行1层padding（原始版本使用）

    Returns:
    --------
    padded : torch.Tensor [B, T, H+2, W+2]
    """
    B, T, H, W = interior.shape

    bl = boundary_left.squeeze(-1)
    br = boundary_right.squeeze(-1)
    bb = boundary_bottom.squeeze(-2)
    bt = boundary_top.squeeze(-2)

    ghost_left = 2 * bl - interior[..., 0]
    ghost_right = 2 * br - interior[..., -1]
    ghost_bottom = 2 * bb - interior[..., 0, :]
    ghost_top = 2 * bt - interior[..., -1, :]

    corner_bl = (ghost_left[:, :, 0:1] + ghost_bottom[:, :, 0:1]) / 2
    corner_br = (ghost_right[:, :, 0:1] + ghost_bottom[:, :, -1:]) / 2
    corner_tl = (ghost_left[:, :, -1:] + ghost_top[:, :, 0:1]) / 2
    corner_tr = (ghost_right[:, :, -1:] + ghost_top[:, :, -1:]) / 2

    middle_rows = torch.cat([ghost_left.unsqueeze(-1), interior, ghost_right.unsqueeze(-1)], dim=-1)
    bottom_row = torch.cat([corner_bl.unsqueeze(-1), ghost_bottom.unsqueeze(-2), corner_br.unsqueeze(-1)], dim=-1)
    top_row = torch.cat([corner_tl.unsqueeze(-1), ghost_top.unsqueeze(-2), corner_tr.unsqueeze(-1)], dim=-1)

    padded = torch.cat([bottom_row, middle_rows, top_row], dim=-2)
    return padded


# ============== 原始版本（保留用于对比）==============
def burgers_pde_loss_original(pred, boundary_left, boundary_right, boundary_bottom, boundary_top,
                              nu, dt=1/999, verbose=False):
    """
    原始版本：统一使用后向差分（用于对比）
    """
    B, T, H, W, _ = pred.shape
    assert T >= 2, "T must be at least 2"

    dx = 1.0 / H
    dy = dx

    u = pred[..., 0]
    v = pred[..., 1]

    u_left = boundary_left[..., 0]
    v_left = boundary_left[..., 1]
    u_right = boundary_right[..., 0]
    v_right = boundary_right[..., 1]
    u_bottom = boundary_bottom[..., 0]
    v_bottom = boundary_bottom[..., 1]
    u_top = boundary_top[..., 0]
    v_top = boundary_top[..., 1]

    u_padded = pad_with_boundaries_1x(u, u_left, u_right, u_bottom, u_top)
    v_padded = pad_with_boundaries_1x(v, v_left, v_right, v_bottom, v_top)
    
    u_t = (u[:, 1:] - u[:, :-1]) / dt
    v_t = (v[:, 1:] - v[:, :-1]) / dt
    
    u_current = u[:, 1:]
    v_current = v[:, 1:]
    
    u_padded_current = u_padded[:, 1:]
    v_padded_current = v_padded[:, 1:]
    
    # 统一后向差分
    u_x = (u_padded_current[:, :, 1:H+1, 1:W+1] - u_padded_current[:, :, 1:H+1, 0:W]) / dx
    v_x = (v_padded_current[:, :, 1:H+1, 1:W+1] - v_padded_current[:, :, 1:H+1, 0:W]) / dx
    u_y = (u_padded_current[:, :, 1:H+1, 1:W+1] - u_padded_current[:, :, 0:H, 1:W+1]) / dy
    v_y = (v_padded_current[:, :, 1:H+1, 1:W+1] - v_padded_current[:, :, 0:H, 1:W+1]) / dy
    
    u_center = u_padded_current[:, :, 1:H+1, 1:W+1]
    v_center = v_padded_current[:, :, 1:H+1, 1:W+1]
    
    u_xx = (u_padded_current[:, :, 1:H+1, 2:W+2] - 2*u_center + u_padded_current[:, :, 1:H+1, 0:W]) / (dx**2)
    v_xx = (v_padded_current[:, :, 1:H+1, 2:W+2] - 2*v_center + v_padded_current[:, :, 1:H+1, 0:W]) / (dx**2)
    u_yy = (u_padded_current[:, :, 2:H+2, 1:W+1] - 2*u_center + u_padded_current[:, :, 0:H, 1:W+1]) / (dy**2)
    v_yy = (v_padded_current[:, :, 2:H+2, 1:W+1] - 2*v_center + v_padded_current[:, :, 0:H, 1:W+1]) / (dy**2)
    
    residual_u = u_t + u_current * u_x + v_current * u_y - nu * (u_xx + u_yy)
    residual_v = v_t + u_current * v_x + v_current * v_y - nu * (v_xx + v_yy)
    
    loss_u = torch.mean(residual_u ** 2)
    loss_v = torch.mean(residual_v ** 2)
    total_loss = loss_u + loss_v
    
    if verbose:
        with torch.no_grad():
            abs_u = torch.abs(residual_u)
            max_val_u = torch.max(abs_u)
            idx_u = (abs_u == max_val_u).nonzero(as_tuple=False)[0]

            abs_v = torch.abs(residual_v)
            max_val_v = torch.max(abs_v)
            idx_v = (abs_v == max_val_v).nonzero(as_tuple=False)[0]

            print(f"[DEBUG-Original] Max Residual U: {max_val_u.item():.5f} at Index (B,T,H,W): {idx_u.tolist()}")
            print(f"[DEBUG-Original] Max Residual V: {max_val_v.item():.5f} at Index (B,T,H,W): {idx_v.tolist()}")

    return total_loss, loss_u, loss_v, residual_u, residual_v


if __name__ == "__main__":
    # 简单测试
    print("Testing 2nd order upwind scheme implementation...")

    B, T, H, W = 1, 10, 32, 32
    pred = torch.randn(B, T, H, W, 2)
    boundary_left = torch.randn(B, T, H, 1, 2)
    boundary_right = torch.randn(B, T, H, 1, 2)
    boundary_bottom = torch.randn(B, T, 1, W, 2)
    boundary_top = torch.randn(B, T, 1, W, 2)
    nu = 0.1

    print("\n--- Original (1st order backward diff) ---")
    loss_orig, _, _, _, _ = burgers_pde_loss_original(
        pred, boundary_left, boundary_right, boundary_bottom, boundary_top, nu, verbose=True
    )
    print(f"Total loss: {loss_orig.item():.6f}")

    print("\n--- 2nd order upwind scheme (n-PINN style) ---")
    loss_upwind, _, _, _, _ = burgers_pde_loss_upwind(
        pred, boundary_left, boundary_right, boundary_bottom, boundary_top, nu, verbose=True
    )
    print(f"Total loss: {loss_upwind.item():.6f}")

    # 测试 2x padding 的形状
    print("\n--- Testing pad_with_boundaries_2x ---")
    u = pred[..., 0]  # [B, T, H, W]
    u_padded = pad_with_boundaries_2x(
        u, boundary_left[..., 0], boundary_right[..., 0],
        boundary_bottom[..., 0], boundary_top[..., 0]
    )
    print(f"Input shape: {u.shape}")
    print(f"Padded shape: {u_padded.shape} (expected: [1, 10, 36, 36])")