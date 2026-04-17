"""
Muon-LoRA V2: Orthogonal SVD-Parameterized LoRA.

Key innovation: Parameterize ΔW via its SVD form: ΔW = U · Σ · V^T
- U = NS(B): orthonormal columns in output space (d_out × r)
- V^T = NS(A): orthonormal rows in input space (r × d_in)
- Σ = diag(s): per-direction learnable scale, zero-init → ΔW=0 at start

Properties:
- Zero-init scale → exact identity initialization (like standard LoRA's B=0)
- Orthogonal bases prevent direction collapse (standard LoRA's main failure)
- Per-direction scale allows non-uniform capacity allocation (V1's limitation)
- Clean gradient signal: ∂L/∂s_i is projection onto orthogonal basis (no interference)

Vs V1 (Muon-LoRA): all σ forced to 1 → isotropic → can't allocate capacity
Vs standard LoRA: no orthogonal constraint → directions collapse → effective rank < r

Theory:
- OFT (NeurIPS 2023): orthogonal transform preserves hyperspherical energy
- HOFT (2025): two orthogonal matrices Q_U, Q_V needed for full expressivity
- MOFT (2025): bridges OFT and LoRA via orthogonal transform in low-rank subspace

Usage:
    from finetune.model_lora_muon_v2 import PDELoRAModelMuonV2, save_lora_checkpoint, load_lora_checkpoint
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from pretrain.model_v3 import PDEModelV3


def _is_main_process() -> bool:
    return os.environ.get('LOCAL_RANK', '0') == '0'


# ============================================================
# Newton-Schulz Orthogonalization (same as V1)
# ============================================================

def newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz iteration: orthogonalize G → nearest semi-orthogonal UV^T.

    Quintic polynomial φ(x) = 3.4445x - 4.7750x³ + 2.0315x⁵
    Converges singular values σ → 1, preserving U and V directions.
    Scale-invariant: NS(kG) = NS(G) for any k > 0.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)

    X = G / (G.norm() + eps)

    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X


# ============================================================
# Muon-LoRA V2 Linear Layer
# ============================================================

class MuonLoRALinearV2(nn.Module):
    """
    Orthogonal SVD-Parameterized LoRA layer.

    Forward:
        A_orth = NS(A) ∈ R^{r × d_in}   (orthonormal rows)
        B_orth = NS(B) ∈ R^{d_out × r}   (orthonormal columns)
        s = self.scale                     ∈ R^r (per-direction magnitude)
        ΔW = B_orth · diag(s) · A_orth     (rank-r SVD form)
        y = W·x + ΔW·x

    Gradient at init (scale=0):
        ∂L/∂s_i = b_i^T · (∂L/∂y) · (a_i · x)  → nonzero (clean projection)
        ∂L/∂A, ∂L/∂B = 0 (scale is zero)
        → Phase 1: learn magnitudes, Phase 2: refine directions
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 16,
        lora_dropout: float = 0.05,
        ns_steps: int = 5,
        scale_init: float = 0.0,
    ):
        super().__init__()
        self.base_linear = base_linear
        base_linear.weight.requires_grad_(False)
        if base_linear.bias is not None:
            base_linear.bias.requires_grad_(False)

        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.r = r
        self.ns_steps = ns_steps

        # A: input projection basis (r orthonormal directions in d_in space)
        # B: output projection basis (r orthonormal directions in d_out space)
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        nn.init.orthogonal_(self.lora_A)
        nn.init.orthogonal_(self.lora_B)

        # Per-direction scale: ||ΔW||_F = ||s||_2
        # scale_init=0 → exact identity; scale_init>0 → warm start
        self.scale = nn.Parameter(torch.full((r,), scale_init))

        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)

        # Orthogonalize A and B via Newton-Schulz
        A_orth = newtonschulz5(self.lora_A, steps=self.ns_steps)  # [r, d_in]
        B_orth = newtonschulz5(self.lora_B, steps=self.ns_steps)  # [d_out, r]

        # SVD-form ΔW: B_orth @ diag(scale) @ A_orth @ x
        x_drop = self.lora_dropout(x)
        latent = F.linear(x_drop, A_orth)    # [..., r]
        latent = latent * self.scale           # per-direction scaling
        lora_out = F.linear(latent, B_orth)    # [..., d_out]

        return base_out + lora_out

    def extra_repr(self) -> str:
        return (f"in={self.base_linear.in_features}, out={self.base_linear.out_features}, "
                f"r={self.r}, ns_steps={self.ns_steps}")


# ============================================================
# PatchSmoother (reuse from V1)
# ============================================================

class PatchSmoother(nn.Module):
    """Cross-patch conv to fix grid artifacts. Zero-init → identity start."""

    def __init__(self, channels: int = 18, hidden_dim: int = 64,
                 kernel_size: int = 7, num_layers: int = 3):
        super().__init__()
        pad = kernel_size // 2
        layers: list[nn.Module] = []
        for i in range(num_layers):
            in_ch = channels if i == 0 else hidden_dim
            out_ch = channels if i == num_layers - 1 else hidden_dim
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad,
                                    padding_mode='replicate'))
            if i < num_layers - 1:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        x_2d = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
        x_2d = x_2d + self.net(x_2d)
        return x_2d.permute(0, 2, 3, 1).reshape(B, T, H, W, C)


# ============================================================
# Model Wrapper
# ============================================================

class PDELoRAModelMuonV2(nn.Module):
    """
    Muon-LoRA V2 wrapper for PDEModelV3.

    Drop-in replacement for PDELoRAModelMuon.
    Uses MuonLoRALinearV2 (orthogonal SVD parameterization) instead of MuonLoRALinear.
    """

    def __init__(
        self,
        config: Dict,
        pretrained_path: Optional[str] = None,
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,
    ):
        super().__init__()
        self.config = config
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder

        self.model = PDEModelV3(config)

        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

        self._apply_muon_lora_v2(config)

        if freeze_encoder:
            self._freeze_encoders()
        if freeze_decoder:
            self._freeze_decoders()

        # Optional patch smoother
        smoother_cfg = config.get('model', {}).get('patch_smoother', {})
        if smoother_cfg.get('enabled', False):
            in_ch = config.get('model', {}).get('in_channels', 18)
            self.patch_smoother = PatchSmoother(
                channels=in_ch,
                hidden_dim=smoother_cfg.get('hidden_dim', 64),
                kernel_size=smoother_cfg.get('kernel_size', 7),
                num_layers=smoother_cfg.get('num_layers', 3),
            )
            if _is_main_process():
                n_ps = sum(p.numel() for p in self.patch_smoother.parameters())
                print(f"\nPatchSmoother enabled: {n_ps:,} params")

        if _is_main_process():
            self._log_param_summary()

    # ----------------------------------------------------------
    # Pretrained weight loading
    # ----------------------------------------------------------

    def _load_pretrained(self, pretrained_path: str):
        if _is_main_process():
            print(f"\nLoading pretrained weights from: {pretrained_path}")

        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        cleaned = {}
        for k, v in state_dict.items():
            k = k.removeprefix('module.').removeprefix('_orig_mod.')
            cleaned[k] = v

        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)

        if _is_main_process():
            if missing:
                print(f"  Missing keys (randomly init): {len(missing)}")
                for k in missing[:10]:
                    print(f"    - {k}")
            if unexpected:
                print(f"  Unexpected keys (ignored): {len(unexpected)}")
                for k in unexpected[:10]:
                    print(f"    - {k}")
            loaded = len(cleaned) - len(unexpected)
            print(f"  Loaded {loaded} keys successfully")

    # ----------------------------------------------------------
    # Muon-LoRA V2 injection
    # ----------------------------------------------------------

    def _apply_muon_lora_v2(self, config: Dict):
        """Replace target Linear modules with MuonLoRALinearV2."""
        lora_cfg = config.get('model', {}).get('lora', {})
        r = lora_cfg.get('r', 16)
        dropout = lora_cfg.get('dropout', 0.05)
        ns_steps = lora_cfg.get('ns_steps', 5)
        target_modules = set(lora_cfg.get('target_modules', [
            "qkv", "proj", "gate_proj", "up_proj", "down_proj",
        ]))

        scale_init = lora_cfg.get('scale_init', 0.0)

        replaced = 0
        for name, module in list(self.model.transformer.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            short_name = name.split('.')[-1]
            if short_name not in target_modules:
                continue

            # Navigate to parent module
            parts = name.split('.')
            parent = self.model.transformer
            for p in parts[:-1]:
                parent = getattr(parent, p)

            v2_linear = MuonLoRALinearV2(
                base_linear=module,
                r=r,
                lora_dropout=dropout,
                ns_steps=ns_steps,
                scale_init=scale_init,
            )
            setattr(parent, parts[-1], v2_linear)
            replaced += 1

        # Freeze all non-LoRA transformer params
        for name, param in self.model.transformer.named_parameters():
            if 'lora_' not in name and 'scale' not in name:
                param.requires_grad_(False)

        if _is_main_process():
            n_lora_ab = sum(
                p.numel() for n, p in self.model.transformer.named_parameters()
                if p.requires_grad and ('lora_A' in n or 'lora_B' in n)
            )
            n_scale = sum(
                p.numel() for n, p in self.model.transformer.named_parameters()
                if p.requires_grad and 'scale' in n
            )
            n_total_train = n_lora_ab + n_scale
            print(f"\nMuon-LoRA V2 config:")
            print(f"  r={r}, ns_steps={ns_steps}, scale_init={scale_init}")
            print(f"  target_modules={target_modules}")
            print(f"  Replaced {replaced} Linear layers")
            print(f"  Trainable: {n_total_train:,} (A/B: {n_lora_ab:,}, scale: {n_scale:,})")
            print(f"  Identity init: scale=0 → ΔW=0")

    # ----------------------------------------------------------
    # Freeze helpers
    # ----------------------------------------------------------

    def _freeze_module(self, module: nn.Module, name: str):
        count = 0
        for param in module.parameters():
            param.requires_grad = False
            count += param.numel()
        if _is_main_process():
            print(f"  Froze {name}: {count:,} parameters")

    def _freeze_encoders(self):
        for attr in ['encoder', 'encoder_1d', 'encoder_3d']:
            if hasattr(self.model, attr):
                self._freeze_module(getattr(self.model, attr),
                                    f"Encoder ({attr.split('_')[-1].upper() if '_' in attr else '2D'})")

    def _freeze_decoders(self):
        for attr in ['decoder', 'decoder_1d', 'decoder_3d']:
            if hasattr(self.model, attr):
                self._freeze_module(getattr(self.model, attr),
                                    f"Decoder ({attr.split('_')[-1].upper() if '_' in attr else '2D'})")

    # ----------------------------------------------------------
    # Param summary
    # ----------------------------------------------------------

    def _log_param_summary(self):
        components = {}
        if hasattr(self.model, 'encoder'):
            components['Encoder (2D)'] = self.model.encoder
        if hasattr(self.model, 'decoder'):
            components['Decoder (2D)'] = self.model.decoder
        for dim_name in ['1d', '3d']:
            for comp_type, label in [('encoder', 'Encoder'), ('decoder', 'Decoder')]:
                attr = f'{comp_type}_{dim_name}'
                if hasattr(self.model, attr):
                    components[f'{label} ({dim_name.upper()})'] = getattr(self.model, attr)
        components['Transformer'] = self.model.transformer

        total_all, trainable_all = 0, 0
        print(f"\n{'='*60}")
        print(f"PDELoRAModelMuonV2 Parameter Summary")
        print(f"{'='*60}")

        for name, module in components.items():
            t = sum(p.numel() for p in module.parameters())
            tr = sum(p.numel() for p in module.parameters() if p.requires_grad)
            suffix = " (Muon-LoRA V2)" if name == 'Transformer' else ""
            print(f"{name + suffix + ':':<25} {t:>12,} total, {tr:>10,} trainable")
            total_all += t
            trainable_all += tr

        if hasattr(self, 'patch_smoother'):
            t = sum(p.numel() for p in self.patch_smoother.parameters())
            tr = sum(p.numel() for p in self.patch_smoother.parameters() if p.requires_grad)
            print(f"{'PatchSmoother:':<25} {t:>12,} total, {tr:>10,} trainable")
            total_all += t
            trainable_all += tr

        print(f"{'-'*60}")
        print(f"{'Total:':<25} {total_all:>12,} total, {trainable_all:>10,} trainable")
        print(f"Trainable ratio: {100 * trainable_all / total_all:.2f}%")
        print(f"{'='*60}\n")

    # ----------------------------------------------------------
    # Forward
    # ----------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
        return_normalized: bool = False,
    ) -> torch.Tensor:
        result = self.model(x, channel_mask, return_normalized)

        if not hasattr(self, 'patch_smoother'):
            return result

        if return_normalized:
            output_norm, mean, std = result
            if output_norm.ndim == 5:
                output_norm = self.patch_smoother(output_norm)
            return output_norm, mean, std
        else:
            if result.ndim == 5:
                result = self.patch_smoother(result)
            return result

    def get_trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Checkpoint save / load (compatible with V1 interface)
# ============================================================

def save_lora_checkpoint(
    model: PDELoRAModelMuonV2,
    optimizer: torch.optim.Optimizer,
    scheduler,
    global_step: int,
    metrics: Dict,
    save_path: str,
    config: Dict,
    patience_counter: int = 0,
    best_val_loss: float = float('inf'),
):
    """Save V2 checkpoint (trainable params: lora_A, lora_B, scale)."""
    trainable_state_dict = {}
    for name, param in model.model.named_parameters():
        if param.requires_grad:
            trainable_state_dict[name] = param.data.clone()

    checkpoint = {
        'global_step': global_step,
        'trainable_state_dict': trainable_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config,
        'patience_counter': patience_counter,
        'best_val_loss': best_val_loss,
        'model_type': 'muon_lora_v2',
    }

    if hasattr(model, 'patch_smoother'):
        checkpoint['patch_smoother_state_dict'] = model.patch_smoother.state_dict()

    torch.save(checkpoint, save_path)


def load_lora_checkpoint(
    model: PDELoRAModelMuonV2,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> Dict:
    """Load V2 checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'trainable_state_dict' in checkpoint:
        trainable_state_dict = checkpoint['trainable_state_dict']
        model_state = model.model.state_dict()

        matched = {k: v for k, v in trainable_state_dict.items() if k in model_state}
        skipped = set(trainable_state_dict.keys()) - set(matched.keys())

        model_state.update(matched)
        model.model.load_state_dict(model_state)

        if _is_main_process():
            print(f"Loaded {len(matched)}/{len(trainable_state_dict)} trainable parameters")
            if skipped:
                print(f"  Skipped {len(skipped)} keys:")
                for k in sorted(skipped)[:10]:
                    print(f"    - {k}")

    if hasattr(model, 'patch_smoother') and 'patch_smoother_state_dict' in checkpoint:
        try:
            model.patch_smoother.load_state_dict(checkpoint['patch_smoother_state_dict'])
            if _is_main_process():
                print("Loaded patch smoother state dict")
        except RuntimeError as e:
            if _is_main_process():
                print(f"Warning: Smoother shape mismatch: {e}")

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            if _is_main_process():
                print(f"Warning: Failed to load optimizer state: {e}")

    if scheduler is not None and checkpoint.get('scheduler_state_dict'):
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            if _is_main_process():
                print(f"Warning: Failed to load scheduler state: {e}")

    return checkpoint
