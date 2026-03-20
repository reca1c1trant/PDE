"""
Compute per-timestep PDE residual RMS scales for Gray-Scott and Active Matter.

For Gray-Scott:
  - Uses FFT Laplacian + 2nd-order central time derivative (same as GrayScottPDELoss)
  - Equations: R_A = dA/dt - D_A*Lap(A) + A*B^2 - F*(1-A)
               R_B = dB/dt - D_B*Lap(B) - A*B^2 + (F+k)*B
  - Per-timestep RMS: rms_t = sqrt(mean_{samples,x,y}(R[:,t,:,:]^2))

For Active Matter:
  - Continuity: du/dx + dv/dy = 0 (4th-order central FD, periodic)
  - Concentration: dc/dt + u*dc/dx + v*dc/dy - d_T*Lap(c) = 0 (4th-order central FD)
  - Per-timestep RMS: rms_t = sqrt(mean_{samples,x,y}(R[:,t,:,:]^2))

Output: torch .pt files with dict of per-timestep RMS tensors.

Usage:
    python tools/compute_gt_scales_per_t.py --dataset gray_scott
    python tools/compute_gt_scales_per_t.py --dataset active_matter
    python tools/compute_gt_scales_per_t.py --dataset all
"""

import argparse
import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def fft_laplacian(
    f: torch.Tensor, K2: torch.Tensor
) -> torch.Tensor:
    """FFT Laplacian for periodic domain. f: [B, T, Nx, Ny], K2: [Nx, Ny]."""
    K2_exp = K2.unsqueeze(0).unsqueeze(0)  # [1, 1, Nx, Ny]
    f_hat = torch.fft.fft2(f)
    return torch.fft.ifft2(-K2_exp * f_hat).real


def dx_4th_periodic(f: torch.Tensor, dx: float) -> torch.Tensor:
    """4th-order central x-derivative, periodic via roll. f: [..., Nx, Ny]."""
    return (-torch.roll(f, -2, dims=-2) + 8 * torch.roll(f, -1, dims=-2)
            - 8 * torch.roll(f, 1, dims=-2) + torch.roll(f, 2, dims=-2)) / (12 * dx)


def dy_4th_periodic(f: torch.Tensor, dy: float) -> torch.Tensor:
    """4th-order central y-derivative, periodic via roll. f: [..., Nx, Ny]."""
    return (-torch.roll(f, -2, dims=-1) + 8 * torch.roll(f, -1, dims=-1)
            - 8 * torch.roll(f, 1, dims=-1) + torch.roll(f, 2, dims=-1)) / (12 * dy)


def laplacian_4th_periodic(f: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """4th-order central Laplacian, periodic. f: [..., Nx, Ny]."""
    d2x = (-torch.roll(f, -2, dims=-2) + 16 * torch.roll(f, -1, dims=-2)
            - 30 * f + 16 * torch.roll(f, 1, dims=-2)
            - torch.roll(f, 2, dims=-2)) / (12 * dx ** 2)
    d2y = (-torch.roll(f, -2, dims=-1) + 16 * torch.roll(f, -1, dims=-1)
            - 30 * f + 16 * torch.roll(f, 1, dims=-1)
            - torch.roll(f, 2, dims=-1)) / (12 * dy ** 2)
    return d2x + d2y


def compute_gray_scott_per_t(
    data_path: str,
    output_path: str,
) -> None:
    """Compute per-timestep RMS of Gray-Scott PDE residuals."""
    import h5py

    print(f"Loading data from {data_path} ...")
    with h5py.File(data_path, 'r') as f:
        # scalar shape: (N, T, 128, 128, 2)
        scalar = torch.tensor(f['scalar'][:], dtype=torch.float64)
        # Physics params
        dt = float(f.attrs['dt'])
        dx = float(f.attrs['dx'])
        dy = float(f.attrs['dy'])
        F_val = float(f.attrs['F'])
        k_val = float(f.attrs['k'])
        D_A = float(f.attrs['D_A'])
        D_B = float(f.attrs['D_B'])

    N, T, Nx, Ny, _ = scalar.shape
    print(f"Data shape: N={N}, T={T}, Nx={Nx}, Ny={Ny}")
    print(f"Physics: dt={dt}, dx={dx}, dy={dy}, F={F_val}, k={k_val}, D_A={D_A}, D_B={D_B}")

    # Extract A and B: [N, T, Nx, Ny]
    A = scalar[..., 0]
    B = scalar[..., 1]

    # Precompute FFT wavenumbers for Laplacian
    kx = torch.fft.fftfreq(Nx, d=dx, dtype=torch.float64) * 2 * np.pi
    ky = torch.fft.fftfreq(Ny, d=dy, dtype=torch.float64) * 2 * np.pi
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2  # [Nx, Ny]

    # Process in chunks to avoid memory issues
    chunk_size = 5  # 5 samples at a time
    n_chunks = (N + chunk_size - 1) // chunk_size

    # 2nd-order central time derivative: interior timesteps t=1..T-2
    # Residual at timestep t (1-indexed in the interior): t goes from 1 to T-2
    T_res = T - 2  # number of interior timesteps
    print(f"Number of residual timesteps: {T_res}")

    # Accumulate sum of squared residuals per timestep
    sum_sq_A = torch.zeros(T_res, dtype=torch.float64)
    sum_sq_B = torch.zeros(T_res, dtype=torch.float64)
    total_spatial = N * Nx * Ny  # total spatial points per timestep across all samples

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, N)
        n_in_chunk = end - start

        A_chunk = A[start:end]  # [n, T, Nx, Ny]
        B_chunk = B[start:end]

        # Time derivative: 2nd-order central
        dA_dt = (A_chunk[:, 2:] - A_chunk[:, :-2]) / (2 * dt)
        dB_dt = (B_chunk[:, 2:] - B_chunk[:, :-2]) / (2 * dt)

        # Spatial terms at interior timesteps
        A_n = A_chunk[:, 1:-1]
        B_n = B_chunk[:, 1:-1]

        # FFT Laplacian
        lap_A = fft_laplacian(A_n, K2)
        lap_B = fft_laplacian(B_n, K2)

        # Reaction term
        AB2 = A_n * B_n**2

        # Residuals
        R_A = dA_dt - D_A * lap_A + AB2 - F_val * (1 - A_n)
        R_B = dB_dt - D_B * lap_B - AB2 + (F_val + k_val) * B_n

        # Per-timestep sum of squares: sum over batch and spatial dims
        # R_A shape: [n, T_res, Nx, Ny]
        for t in range(T_res):
            sum_sq_A[t] += torch.sum(R_A[:, t, :, :] ** 2).item()
            sum_sq_B[t] += torch.sum(R_B[:, t, :, :] ** 2).item()

        if (chunk_idx + 1) % 2 == 0 or chunk_idx == n_chunks - 1:
            print(f"  Processed chunk {chunk_idx+1}/{n_chunks} (samples {start}-{end-1})")

    # Compute per-timestep RMS
    rms_A = torch.sqrt(sum_sq_A / total_spatial)
    rms_B = torch.sqrt(sum_sq_B / total_spatial)

    print(f"\nPer-timestep RMS (A): min={rms_A.min():.6e}, max={rms_A.max():.6e}, mean={rms_A.mean():.6e}")
    print(f"Per-timestep RMS (B): min={rms_B.min():.6e}, max={rms_B.max():.6e}, mean={rms_B.mean():.6e}")

    # Show first/last few values
    n_show = min(10, T_res)
    print(f"\nFirst {n_show} timestep RMS (A): {rms_A[:n_show].tolist()}")
    print(f"First {n_show} timestep RMS (B): {rms_B[:n_show].tolist()}")
    print(f"Last {n_show} timestep RMS (A): {rms_A[-n_show:].tolist()}")
    print(f"Last {n_show} timestep RMS (B): {rms_B[-n_show:].tolist()}")

    # Save
    result = {
        'A_equation': rms_A.float(),
        'B_equation': rms_B.float(),
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(result, output_path)
    print(f"\nSaved to {output_path}")
    print(f"Keys: {list(result.keys())}")
    for k, v in result.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")


def compute_active_matter_per_t(
    data_path: str,
    output_path: str,
) -> None:
    """Compute per-timestep RMS of Active Matter PDE residuals.

    Uses simple 4th-order central FD (not n-PINN) for GT verification.
    - Continuity: du/dx + dv/dy = 0
    - Concentration: dc/dt + u*dc/dx + v*dc/dy - d_T*Lap(c) = 0
    """
    import h5py

    # Physics
    nx, ny = 256, 256
    Lx, Ly = 10.0, 10.0
    dx_val = Lx / nx
    dy_val = Ly / ny
    dt = 0.25
    d_T = 0.05

    print(f"Loading data from {data_path} ...")
    with h5py.File(data_path, 'r') as f:
        # vector: (N, T, 256, 256, 3), scalar: (N, T, 256, 256, 1)
        vector = torch.tensor(f['vector'][:], dtype=torch.float64)
        scalar = torch.tensor(f['scalar'][:], dtype=torch.float64)

    N, T, Nx, Ny, _ = vector.shape
    print(f"Data shape: N={N}, T={T}, Nx={Nx}, Ny={Ny}")
    print(f"Physics: dx={dx_val}, dy={dy_val}, dt={dt}, d_T={d_T}")

    # Extract fields: [N, T, Nx, Ny]
    vx = vector[..., 0]
    vy = vector[..., 1]
    conc = scalar[..., 0]

    # 2nd-order central time: interior timesteps t=1..T-2
    T_res = T - 2
    print(f"Number of residual timesteps: {T_res}")

    # Process in chunks
    chunk_size = 4
    n_chunks = (N + chunk_size - 1) // chunk_size

    sum_sq_cont = torch.zeros(T_res, dtype=torch.float64)
    sum_sq_conc = torch.zeros(T_res, dtype=torch.float64)
    total_spatial = N * Nx * Ny

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, N)

        vx_c = vx[start:end]  # [n, T, Nx, Ny]
        vy_c = vy[start:end]
        c_c = conc[start:end]

        # Interior timesteps
        vx_int = vx_c[:, 1:-1]
        vy_int = vy_c[:, 1:-1]
        c_int = c_c[:, 1:-1]

        # Continuity: du/dx + dv/dy = 0
        du_dx = dx_4th_periodic(vx_int, dx_val)
        dv_dy = dy_4th_periodic(vy_int, dy_val)
        R_cont = du_dx + dv_dy

        # Concentration: dc/dt + u*dc/dx + v*dc/dy - d_T*Lap(c) = 0
        dc_dt = (c_c[:, 2:] - c_c[:, :-2]) / (2 * dt)
        dc_dx = dx_4th_periodic(c_int, dx_val)
        dc_dy = dy_4th_periodic(c_int, dy_val)
        lap_c = laplacian_4th_periodic(c_int, dx_val, dy_val)

        R_conc = dc_dt + vx_int * dc_dx + vy_int * dc_dy - d_T * lap_c

        # Per-timestep accumulation
        for t in range(T_res):
            sum_sq_cont[t] += torch.sum(R_cont[:, t, :, :] ** 2).item()
            sum_sq_conc[t] += torch.sum(R_conc[:, t, :, :] ** 2).item()

        if (chunk_idx + 1) % 4 == 0 or chunk_idx == n_chunks - 1:
            print(f"  Processed chunk {chunk_idx+1}/{n_chunks} (samples {start}-{end-1})")

    # Compute per-timestep RMS
    rms_cont = torch.sqrt(sum_sq_cont / total_spatial)
    rms_conc = torch.sqrt(sum_sq_conc / total_spatial)

    print(f"\nPer-timestep RMS (continuity): min={rms_cont.min():.6e}, max={rms_cont.max():.6e}, mean={rms_cont.mean():.6e}")
    print(f"Per-timestep RMS (concentration): min={rms_conc.min():.6e}, max={rms_conc.max():.6e}, mean={rms_conc.mean():.6e}")

    # Show first/last few values
    n_show = min(10, T_res)
    print(f"\nFirst {n_show} timestep RMS (continuity): {rms_cont[:n_show].tolist()}")
    print(f"First {n_show} timestep RMS (concentration): {rms_conc[:n_show].tolist()}")
    print(f"Last {n_show} timestep RMS (continuity): {rms_cont[-n_show:].tolist()}")
    print(f"Last {n_show} timestep RMS (concentration): {rms_conc[-n_show:].tolist()}")

    # Save
    result = {
        'continuity': rms_cont.float(),
        'concentration': rms_conc.float(),
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(result, output_path)
    print(f"\nSaved to {output_path}")
    print(f"Keys: {list(result.keys())}")
    for k, v in result.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-timestep PDE residual RMS scales"
    )
    parser.add_argument(
        '--dataset', type=str, required=True,
        choices=['gray_scott', 'active_matter', 'all'],
        help='Which dataset to compute scales for'
    )
    parser.add_argument(
        '--gray_scott_path', type=str,
        default='/scratch-share/SONG0304/finetune/gray_scott_test.h5',
        help='Path to gray_scott HDF5 data'
    )
    parser.add_argument(
        '--active_matter_path', type=str,
        default='/home/msai/song0304/code/PDE/data/finetune/active_matter_all.hdf5',
        help='Path to active_matter HDF5 data'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default='./data/gt_scales',
        help='Output directory for .pt files'
    )
    args = parser.parse_args()

    if args.dataset in ('gray_scott', 'all'):
        compute_gray_scott_per_t(
            data_path=args.gray_scott_path,
            output_path=os.path.join(args.output_dir, 'gray_scott_per_t.pt'),
        )

    if args.dataset in ('active_matter', 'all'):
        compute_active_matter_per_t(
            data_path=args.active_matter_path,
            output_path=os.path.join(args.output_dir, 'active_matter_per_t.pt'),
        )


if __name__ == '__main__':
    main()
