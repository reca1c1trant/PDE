"""
Full pipeline test: data -> encoder -> transformer -> decoder -> loss -> backward.
Tests 1D, 2D (existing PDEModelV2), and 3D pipelines.
"""
import sys
sys.path.insert(0, '/home/msai/song0304/code/PDE')

import torch
import torch.nn.functional as F


def test_pipeline_1d():
    """Full 1D pipeline: encoder -> NA2D transformer -> decoder -> loss -> backward."""
    from pretrain.encoder_v2 import PatchifyEncoder1D
    from pretrain.model_v2 import NATransformer1D
    from pretrain.decoder_v2 import PatchifyDecoder1D

    device = 'cuda'
    dtype = torch.float32

    # Config
    B, T, X, C = 1, 4, 256, 3
    hidden_dim = 128
    num_layers = 2
    num_heads = 4
    base_kernel = 3

    print(f"[1D Pipeline] Input: [B={B}, T={T}, X={X}, C={C}]")
    print(f"  hidden_dim={hidden_dim}, layers={num_layers}, heads={num_heads}, kernel={base_kernel}")

    # Build components
    encoder = PatchifyEncoder1D(
        in_channels=C, hidden_dim=hidden_dim, patch_size=16,
        stem_hidden=64, stem_out=128,
        intra_patch_layers=1, intra_patch_heads=4,
    ).to(device, dtype)

    transformer = NATransformer1D(
        hidden_dim=hidden_dim, num_layers=num_layers,
        num_heads=num_heads, base_kernel=base_kernel, dropout=0.0,
    ).to(device, dtype)

    decoder = PatchifyDecoder1D(
        out_channels=C, hidden_dim=hidden_dim, patch_size=16,
        stem_channels=128, decoder_hidden=64,
    ).to(device, dtype)

    total_params = (
        sum(p.numel() for p in encoder.parameters()) +
        sum(p.numel() for p in transformer.parameters()) +
        sum(p.numel() for p in decoder.parameters())
    )
    print(f"  Total params: {total_params:,}")

    # Fabricate data
    x = torch.randn(B, T, X, C, device=device, dtype=dtype, requires_grad=True)

    # Forward
    tokens, shape_info = encoder(x)
    print(f"  Encoder: {x.shape} -> {tokens.shape}, shape_info={shape_info}")

    tokens = transformer(tokens, shape_info)
    print(f"  Transformer: {tokens.shape}")

    output = decoder(tokens, shape_info)
    print(f"  Decoder: {tokens.shape} -> {output.shape}")

    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"

    # Loss + backward
    loss = F.mse_loss(output, x.detach())
    loss.backward()
    print(f"  Loss: {loss.item():.4f}")
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print(f"  Backward: OK")


def test_pipeline_2d():
    """Full 2D pipeline using existing PDEModelV2."""
    from pretrain.model_v2 import PDEModelV2

    device = 'cuda'
    dtype = torch.float32

    B, T, H, W, C = 1, 4, 128, 128, 3
    print(f"\n[2D Pipeline] Input: [B={B}, T={T}, H={H}, W={W}, C={C}]")

    config = {
        'model': {
            'in_channels': C,
            'hidden_dim': 128,
            'patch_size': 16,
            'num_layers': 2,
            'num_heads': 4,
            'dropout': 0.0,
            'encoder': {'stem_hidden': 64, 'stem_out': 128},
            'intra_patch': {'num_layers': 1, 'temporal_window': 3, 'num_heads': 4},
            'na': {'base_kernel': 3},
            'decoder': {'stem_channels': 128, 'hidden_channels': 64},
        }
    }

    model = PDEModelV2(config).to(device, dtype)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")

    x = torch.randn(B, T, H, W, C, device=device, dtype=dtype, requires_grad=True)

    output = model(x)
    print(f"  Forward: {x.shape} -> {output.shape}")
    assert output.shape == x.shape

    loss = F.mse_loss(output, x.detach())
    loss.backward()
    print(f"  Loss: {loss.item():.4f}")
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print(f"  Backward: OK")

    # Also test with return_normalized
    x2 = torch.randn(B, T, H, W, C, device=device, dtype=dtype)
    output_norm, mean, std = model(x2, return_normalized=True)
    assert output_norm.shape == x2.shape
    assert mean.shape == (B, 1, 1, 1, C)
    print(f"  Normalized forward: OK")


def test_pipeline_3d():
    """Full 3D pipeline: encoder -> NA4D transformer -> decoder -> loss -> backward."""
    from pretrain.encoder_v2 import PatchifyEncoder3D
    from pretrain.model_v2 import NATransformer3D
    from pretrain.decoder_v2 import PatchifyDecoder3D

    device = 'cuda'
    dtype = torch.float32

    # Need spatial dims >= kernel_size (3) after patchify: 32/8=4 >= 3
    B, T, D_in, H, W, C = 1, 4, 32, 32, 32, 3
    hidden_dim = 128
    num_layers = 2
    num_heads = 4
    base_kernel = 3

    print(f"\n[3D Pipeline] Input: [B={B}, T={T}, D={D_in}, H={H}, W={W}, C={C}]")
    print(f"  hidden_dim={hidden_dim}, layers={num_layers}, heads={num_heads}, kernel={base_kernel}")

    # Build components
    encoder = PatchifyEncoder3D(
        in_channels=C, hidden_dim=hidden_dim, patch_size=8,
        stem_hidden=64, stem_out=128,
        intra_patch_layers=1, intra_patch_heads=4,
    ).to(device, dtype)

    transformer = NATransformer3D(
        hidden_dim=hidden_dim, num_layers=num_layers,
        num_heads=num_heads, base_kernel=base_kernel, dropout=0.0,
    ).to(device, dtype)

    decoder = PatchifyDecoder3D(
        out_channels=C, hidden_dim=hidden_dim, patch_size=8,
        stem_channels=128, decoder_hidden=64,
    ).to(device, dtype)

    total_params = (
        sum(p.numel() for p in encoder.parameters()) +
        sum(p.numel() for p in transformer.parameters()) +
        sum(p.numel() for p in decoder.parameters())
    )
    print(f"  Total params: {total_params:,}")

    # Fabricate data
    x = torch.randn(B, T, D_in, H, W, C, device=device, dtype=dtype, requires_grad=True)

    # Forward
    tokens, shape_info = encoder(x)
    print(f"  Encoder: {x.shape} -> {tokens.shape}, shape_info={shape_info}")

    tokens = transformer(tokens, shape_info)
    print(f"  Transformer: {tokens.shape}")

    output = decoder(tokens, shape_info)
    print(f"  Decoder: {tokens.shape} -> {output.shape}")

    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"

    # Loss + backward
    loss = F.mse_loss(output, x.detach())
    loss.backward()
    print(f"  Loss: {loss.item():.4f}")
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print(f"  Backward: OK")


def test_pipeline_3d_larger():
    """3D pipeline with larger spatial resolution (32^3)."""
    from pretrain.encoder_v2 import PatchifyEncoder3D
    from pretrain.model_v2 import NATransformer3D
    from pretrain.decoder_v2 import PatchifyDecoder3D

    device = 'cuda'
    dtype = torch.float32

    B, T, D_in, H, W, C = 1, 4, 32, 32, 32, 3
    hidden_dim = 128
    print(f"\n[3D Pipeline Larger] Input: [B={B}, T={T}, D={D_in}, H={H}, W={W}, C={C}]")

    encoder = PatchifyEncoder3D(
        in_channels=C, hidden_dim=hidden_dim, patch_size=8,
        stem_hidden=64, stem_out=128,
        intra_patch_layers=1, intra_patch_heads=4,
    ).to(device, dtype)

    transformer = NATransformer3D(
        hidden_dim=hidden_dim, num_layers=2,
        num_heads=4, base_kernel=3, dropout=0.0,
    ).to(device, dtype)

    decoder = PatchifyDecoder3D(
        out_channels=C, hidden_dim=hidden_dim, patch_size=8,
        stem_channels=128, decoder_hidden=64,
    ).to(device, dtype)

    x = torch.randn(B, T, D_in, H, W, C, device=device, dtype=dtype, requires_grad=True)

    tokens, shape_info = encoder(x)
    n_d, n_h, n_w = D_in // 8, H // 8, W // 8
    print(f"  Encoder: {x.shape} -> {tokens.shape}")
    print(f"  Patches: {n_d}x{n_h}x{n_w}={n_d*n_h*n_w}, tokens/step={n_d*n_h*n_w}, total={T*n_d*n_h*n_w}")

    tokens = transformer(tokens, shape_info)
    output = decoder(tokens, shape_info)

    assert output.shape == x.shape
    print(f"  Decoder: {output.shape}")

    loss = F.mse_loss(output, x.detach())
    loss.backward()
    print(f"  Loss: {loss.item():.4f}, Backward: OK")


if __name__ == "__main__":
    print("=" * 60)
    print("Full Pipeline Tests (1D, 2D, 3D)")
    print("=" * 60)

    try:
        test_pipeline_1d()
        test_pipeline_2d()
        test_pipeline_3d()
        test_pipeline_3d_larger()
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("ALL PIPELINE TESTS PASSED!")
    print("=" * 60)
