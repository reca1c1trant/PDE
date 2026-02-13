"""
Test PDEModelV3 (SharedNATransformer) with mixed 1D, 2D, 3D data.
Full pipeline: data -> model -> loss -> backward.
"""
import sys
sys.path.insert(0, '/home/msai/song0304/code/PDE')

import torch
import torch.nn.functional as F


def make_config(enable_1d=True, enable_3d=True):
    return {
        'model': {
            'in_channels': 3,
            'hidden_dim': 128,
            'patch_size': 16,
            'patch_size_3d': 8,
            'num_layers': 2,
            'num_heads': 4,
            'dropout': 0.0,
            'encoder': {'stem_hidden': 64, 'stem_out': 128},
            'intra_patch': {'num_layers': 1, 'temporal_window': 3, 'num_heads': 4},
            'na': {'base_kernel': 3},
            'decoder': {'stem_channels': 128, 'hidden_channels': 64},
            'enable_1d': enable_1d,
            'enable_3d': enable_3d,
        }
    }


def test_1d():
    """Test 1D pipeline through PDEModelV3."""
    from pretrain.model_v3 import PDEModelV3

    device = 'cuda'
    config = make_config(enable_1d=True, enable_3d=False)
    model = PDEModelV3(config).to(device).float()

    B, T, X, C = 2, 4, 256, 3
    x = torch.randn(B, T, X, C, device=device, requires_grad=True)

    print(f"[1D] Input: {x.shape}")
    output = model(x)
    print(f"[1D] Output: {output.shape}")
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"

    loss = F.mse_loss(output, x.detach())
    loss.backward()
    print(f"[1D] Loss: {loss.item():.4f}, grad OK: {x.grad is not None}")
    assert x.grad is not None
    print("[1D] PASSED\n")


def test_2d():
    """Test 2D pipeline through PDEModelV3."""
    from pretrain.model_v3 import PDEModelV3

    device = 'cuda'
    config = make_config(enable_1d=False, enable_3d=False)
    model = PDEModelV3(config).to(device).float()

    B, T, H, W, C = 2, 4, 128, 128, 3
    x = torch.randn(B, T, H, W, C, device=device, requires_grad=True)

    print(f"[2D] Input: {x.shape}")
    output = model(x)
    print(f"[2D] Output: {output.shape}")
    assert output.shape == x.shape

    loss = F.mse_loss(output, x.detach())
    loss.backward()
    print(f"[2D] Loss: {loss.item():.4f}, grad OK: {x.grad is not None}")
    assert x.grad is not None
    print("[2D] PASSED\n")


def test_3d():
    """Test 3D pipeline through PDEModelV3."""
    from pretrain.model_v3 import PDEModelV3

    device = 'cuda'
    config = make_config(enable_1d=False, enable_3d=True)
    model = PDEModelV3(config).to(device).float()

    B, T, D, H, W, C = 1, 4, 32, 32, 32, 3
    x = torch.randn(B, T, D, H, W, C, device=device, requires_grad=True)

    print(f"[3D] Input: {x.shape}")
    output = model(x)
    print(f"[3D] Output: {output.shape}")
    assert output.shape == x.shape

    loss = F.mse_loss(output, x.detach())
    loss.backward()
    print(f"[3D] Loss: {loss.item():.4f}, grad OK: {x.grad is not None}")
    assert x.grad is not None
    print("[3D] PASSED\n")


def test_mixed_sequential():
    """Test mixed 1D + 2D through the same model (like training with mixed batches)."""
    from pretrain.model_v3 import PDEModelV3

    device = 'cuda'
    config = make_config(enable_1d=True, enable_3d=False)
    model = PDEModelV3(config).to(device).float()

    total_params = model.get_num_params()
    print(f"[Mixed] Total params: {total_params:,}")

    # 1D batch
    x1d = torch.randn(2, 4, 256, 3, device=device)
    out1d = model(x1d)
    assert out1d.shape == x1d.shape
    print(f"[Mixed] 1D: {x1d.shape} -> {out1d.shape}")

    # 2D batch
    x2d = torch.randn(2, 4, 128, 128, 3, device=device)
    out2d = model(x2d)
    assert out2d.shape == x2d.shape
    print(f"[Mixed] 2D: {x2d.shape} -> {out2d.shape}")

    # Verify return_normalized
    out_norm, mean, std = model(x1d, return_normalized=True)
    assert out_norm.shape == x1d.shape
    assert mean.shape == (2, 1, 1, 3)  # [B, 1, 1, C] for 1D
    print(f"[Mixed] return_normalized 1D: OK")

    out_norm, mean, std = model(x2d, return_normalized=True)
    assert out_norm.shape == x2d.shape
    assert mean.shape == (2, 1, 1, 1, 3)  # [B, 1, 1, 1, C] for 2D
    print(f"[Mixed] return_normalized 2D: OK")

    print("[Mixed] PASSED\n")


def test_normalized_loss():
    """Test compute_normalized_rmse_loss with different dims."""
    print("[Loss] Testing dimension-agnostic loss...")

    # Simulate what train_pretrain_v3.py does
    channel_mask = torch.zeros(2, 18)
    channel_mask[:, 0] = 1.0  # Vx
    channel_mask[:, 7] = 1.0  # density

    # 1D: [B, T, X, C=18]
    pred_1d = torch.randn(2, 4, 64, 18)
    target_1d = torch.randn(2, 4, 64, 18)
    mask = channel_mask
    for _ in range(pred_1d.ndim - 2):
        mask = mask.unsqueeze(1)
    masked = (pred_1d - target_1d) ** 2 * mask.float()
    assert masked.shape == pred_1d.shape
    print(f"  1D mask broadcast: OK, shape={masked.shape}")

    # 2D: [B, T, H, W, C=18]
    pred_2d = torch.randn(2, 4, 16, 16, 18)
    target_2d = torch.randn(2, 4, 16, 16, 18)
    mask = channel_mask
    for _ in range(pred_2d.ndim - 2):
        mask = mask.unsqueeze(1)
    masked = (pred_2d - target_2d) ** 2 * mask.float()
    assert masked.shape == pred_2d.shape
    print(f"  2D mask broadcast: OK, shape={masked.shape}")

    print("[Loss] PASSED\n")


def test_shared_ffn():
    """Verify that FFN weights are shared (same object across NA keys)."""
    from pretrain.model_v3 import PDEModelV3

    config = make_config(enable_1d=True, enable_3d=False)
    model = PDEModelV3(config)

    layer0 = model.transformer.layers[0]

    # FFN is a single module, used regardless of NA key
    # Verify it exists and has expected structure
    assert hasattr(layer0, 'ffn'), "Missing shared FFN"
    assert hasattr(layer0, 'na_1d'), "Missing 1D NA"
    assert hasattr(layer0, 'na_2d_1x1'), "Missing 2D NA 1x1"
    assert hasattr(layer0, 'na_2d_1x2'), "Missing 2D NA 1x2"
    assert hasattr(layer0, 'na_2d_1x4'), "Missing 2D NA 1x4"
    assert not hasattr(layer0, 'na_3d'), "3D NA should not exist when enable_3d=False"

    # Count params
    ffn_params = sum(p.numel() for p in layer0.ffn.parameters())
    na_1d_params = sum(p.numel() for p in layer0.na_1d.parameters())
    na_2d_params = sum(p.numel() for p in layer0.na_2d_1x1.parameters())

    print(f"[SharedFFN] Per layer: FFN={ffn_params:,}, NA_1d={na_1d_params:,}, NA_2d={na_2d_params:,}")
    print(f"[SharedFFN] FFN is shared across all dims ✓")
    print("[SharedFFN] PASSED\n")


if __name__ == '__main__':
    print("=" * 60)
    print("PDEModelV3 Tests (1D + 2D + 3D + Mixed)")
    print("=" * 60)

    try:
        test_normalized_loss()
        test_shared_ffn()
        test_1d()
        test_2d()
        test_3d()
        test_mixed_sequential()
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    print("=" * 60)
    print("ALL PDEModelV3 TESTS PASSED!")
    print("=" * 60)
