"""
2D Burgers方程PDE residual loss计算 - 迎风格式版本

Loss = ||u_t + uu_x + vu_y - ν(u_xx + u_yy)||_2
     + ||v_t + uv_x + vv_y - ν(v_xx + v_yy)||_2

导数计算策略：
- 时间导数：后向差分（从索引1开始）
- 一阶空间导数：迎风格式（根据流速方向选择前向/后向差分）
- 二阶空间导数：中心差分（需要padding）

迎风格式原理：
- 对于 u*u_x：u>0 用后向差分，u<0 用前向差分
- 对于 v*u_y：v>0 用后向差分，v<0 用前向差分
- 同理适用于 v 方程

Author: Ziye
Date: December 2024
"""

import torch


def pad_with_boundaries(interior, boundary_left, boundary_right, boundary_bottom, boundary_top):
    """
    使用边界值对内部网格进行padding，外推ghost cells

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
    padded : torch.Tensor [B, T, H+2, W+2]
        Padding后的网格（包含ghost cells）

    Note:
        Uses torch.cat instead of index assignment to preserve gradients.
    """
    B, T, H, W = interior.shape

    # 外推ghost cells [B, T, H] or [B, T, W]
    ghost_left = 2 * boundary_left.squeeze(-1) - interior[..., 0]      # [B, T, H]
    ghost_right = 2 * boundary_right.squeeze(-1) - interior[..., -1]   # [B, T, H]
    ghost_bottom = 2 * boundary_bottom.squeeze(-2) - interior[..., 0, :]  # [B, T, W]
    ghost_top = 2 * boundary_top.squeeze(-2) - interior[..., -1, :]       # [B, T, W]

    # 四个角点
    corner_bl = (ghost_left[:, :, 0:1] + ghost_bottom[:, :, 0:1]) / 2   # [B, T, 1]
    corner_br = (ghost_right[:, :, 0:1] + ghost_bottom[:, :, -1:]) / 2  # [B, T, 1]
    corner_tl = (ghost_left[:, :, -1:] + ghost_top[:, :, 0:1]) / 2      # [B, T, 1]
    corner_tr = (ghost_right[:, :, -1:] + ghost_top[:, :, -1:]) / 2     # [B, T, 1]

    # 构建中间行（包含左右ghost）: [B, T, H, W+2]
    ghost_left_col = ghost_left.unsqueeze(-1)    # [B, T, H, 1]
    ghost_right_col = ghost_right.unsqueeze(-1)  # [B, T, H, 1]
    middle_rows = torch.cat([ghost_left_col, interior, ghost_right_col], dim=-1)  # [B, T, H, W+2]

    # 构建底部行: [B, T, 1, W+2]
    bottom_row = torch.cat([corner_bl.unsqueeze(-1), ghost_bottom.unsqueeze(-2), corner_br.unsqueeze(-1)], dim=-1)

    # 构建顶部行: [B, T, 1, W+2]
    top_row = torch.cat([corner_tl.unsqueeze(-1), ghost_top.unsqueeze(-2), corner_tr.unsqueeze(-1)], dim=-1)

    # 拼接成完整的padded grid: [B, T, H+2, W+2]
    padded = torch.cat([bottom_row, middle_rows, top_row], dim=-2)

    return padded


def upwind_derivative_x(u_padded, velocity, H, W, dx):
    """
    计算x方向的迎风导数
    
    Parameters:
    -----------
    u_padded : torch.Tensor [B, T, H+2, W+2]
        padding后的场
    velocity : torch.Tensor [B, T, H, W]
        决定迎风方向的速度场
    H, W : int
        内部网格尺寸
    dx : float
        网格间距
    
    Returns:
    --------
    du_dx : torch.Tensor [B, T, H, W]
        迎风格式计算的x方向导数
    """
    # 提取内部及其左右邻居
    u_center = u_padded[:, :, 1:H+1, 1:W+1]  # u[i,j]
    u_left = u_padded[:, :, 1:H+1, 0:W]       # u[i,j-1]
    u_right = u_padded[:, :, 1:H+1, 2:W+2]   # u[i,j+1]
    
    # 后向差分 (u[i,j] - u[i,j-1]) / dx，用于 velocity > 0
    backward = (u_center - u_left) / dx
    
    # 前向差分 (u[i,j+1] - u[i,j]) / dx，用于 velocity < 0
    forward = (u_right - u_center) / dx
    
    # 迎风选择：velocity > 0 用后向，velocity < 0 用前向
    du_dx = torch.where(velocity >= 0, backward, forward)
    
    return du_dx


def upwind_derivative_y(u_padded, velocity, H, W, dy):
    """
    计算y方向的迎风导数
    
    Parameters:
    -----------
    u_padded : torch.Tensor [B, T, H+2, W+2]
        padding后的场
    velocity : torch.Tensor [B, T, H, W]
        决定迎风方向的速度场
    H, W : int
        内部网格尺寸
    dy : float
        网格间距
    
    Returns:
    --------
    du_dy : torch.Tensor [B, T, H, W]
        迎风格式计算的y方向导数
    """
    # 提取内部及其上下邻居
    u_center = u_padded[:, :, 1:H+1, 1:W+1]  # u[i,j]
    u_bottom = u_padded[:, :, 0:H, 1:W+1]     # u[i-1,j]
    u_top = u_padded[:, :, 2:H+2, 1:W+1]     # u[i+1,j]
    
    # 后向差分 (u[i,j] - u[i-1,j]) / dy，用于 velocity > 0
    backward = (u_center - u_bottom) / dy
    
    # 前向差分 (u[i+1,j] - u[i,j]) / dy，用于 velocity < 0
    forward = (u_top - u_center) / dy
    
    # 迎风选择
    du_dy = torch.where(velocity >= 0, backward, forward)
    
    return du_dy


def burgers_pde_loss_upwind(pred, boundary_left, boundary_right, boundary_bottom, boundary_top,
                            nu, dt=1/999, verbose=False):
    """
    计算2D Burgers方程的PDE residual loss

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
    
    # Padding内部网格
    u_padded = pad_with_boundaries(u, u_left, u_right, u_bottom, u_top)
    v_padded = pad_with_boundaries(v, v_left, v_right, v_bottom, v_top)
    
    # ========== 时间导数（后向差分，从索引1开始）==========
    u_t = (u[:, 1:] - u[:, :-1]) / dt  # [B, T-1, H, W]
    v_t = (v[:, 1:] - v[:, :-1]) / dt
    
    # 当前时刻的场（索引1到T-1）
    u_current = u[:, 1:]
    v_current = v[:, 1:]
    
    # 当前时刻的padded场
    u_padded_current = u_padded[:, 1:]
    v_padded_current = v_padded[:, 1:]
    
    # ========== 一阶空间导数（迎风格式）==========
    # u方程: u_t + u*u_x + v*u_y = ν(u_xx + u_yy)
    # - u*u_x: 用u作为速度决定u_x的迎风方向
    # - v*u_y: 用v作为速度决定u_y的迎风方向
    
    u_x = upwind_derivative_x(u_padded_current, u_current, H, W, dx)
    u_y = upwind_derivative_y(u_padded_current, v_current, H, W, dy)
    
    # v方程: v_t + u*v_x + v*v_y = ν(v_xx + v_yy)
    # - u*v_x: 用u作为速度决定v_x的迎风方向
    # - v*v_y: 用v作为速度决定v_y的迎风方向
    
    v_x = upwind_derivative_x(v_padded_current, u_current, H, W, dx)
    v_y = upwind_derivative_y(v_padded_current, v_current, H, W, dy)
    
    # ========== 二阶空间导数（中心差分）==========
    u_center = u_padded_current[:, :, 1:H+1, 1:W+1]
    v_center = v_padded_current[:, :, 1:H+1, 1:W+1]
    
    u_xx = (u_padded_current[:, :, 1:H+1, 2:W+2] - 2*u_center + u_padded_current[:, :, 1:H+1, 0:W]) / (dx**2)
    v_xx = (v_padded_current[:, :, 1:H+1, 2:W+2] - 2*v_center + v_padded_current[:, :, 1:H+1, 0:W]) / (dx**2)
    
    u_yy = (u_padded_current[:, :, 2:H+2, 1:W+1] - 2*u_center + u_padded_current[:, :, 0:H, 1:W+1]) / (dy**2)
    v_yy = (v_padded_current[:, :, 2:H+2, 1:W+1] - 2*v_center + v_padded_current[:, :, 0:H, 1:W+1]) / (dy**2)
    
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
    
    u_padded = pad_with_boundaries(u, u_left, u_right, u_bottom, u_top)
    v_padded = pad_with_boundaries(v, v_left, v_right, v_bottom, v_top)
    
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
    print("Testing upwind scheme implementation...")
    
    B, T, H, W = 1, 10, 32, 32
    pred = torch.randn(B, T, H, W, 2)
    boundary_left = torch.randn(B, T, H, 1, 2)
    boundary_right = torch.randn(B, T, H, 1, 2)
    boundary_bottom = torch.randn(B, T, 1, W, 2)
    boundary_top = torch.randn(B, T, 1, W, 2)
    nu = 0.1
    
    print("\n--- Original (backward diff) ---")
    loss_orig, _, _, _, _ = burgers_pde_loss_original(
        pred, boundary_left, boundary_right, boundary_bottom, boundary_top, nu, verbose=True
    )
    print(f"Total loss: {loss_orig.item():.6f}")

    print("\n--- Upwind scheme ---")
    loss_upwind, _, _, _, _ = burgers_pde_loss_upwind(
        pred, boundary_left, boundary_right, boundary_bottom, boundary_top, nu, verbose=True
    )
    print(f"Total loss: {loss_upwind.item():.6f}")