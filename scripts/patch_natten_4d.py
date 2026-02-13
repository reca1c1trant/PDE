#!/usr/bin/env python3
"""
Patch NATTEN to support 4D Neighborhood Attention (NeighborhoodAttention4D).

NATTEN natively supports 1D/2D/3D. This script patches the installed NATTEN package
to add 4D support for 3D spatial + temporal PDE data with shape [B, T, D, H, W, C].

Usage:
    python scripts/patch_natten_4d.py

Requirements:
    - natten >= 0.21.0 (with flex backend support)
    - PyTorch >= 2.7 (for flex_attention)

What this does:
    - Adds Dimension4DType / CausalArg4DType to types.py
    - Lifts na_dim <= 3 restrictions to <= 4 in checks
    - Adds 4D tile shapes for flex backend
    - Adds na4d() functional API
    - Adds NeighborhoodAttention4D module
    - Exports new symbols from __init__.py

The 4D attention uses the flex backend (PyTorch flex_attention), which is
dimension-agnostic. No C++ recompilation needed.
"""

import importlib
import os
import re
import shutil
import sys


def get_natten_path():
    """Find the installed natten package path."""
    try:
        import natten
        return os.path.dirname(natten.__file__)
    except ImportError:
        print("ERROR: natten is not installed. Install it first:")
        print("  pip install natten")
        sys.exit(1)


def backup_file(filepath):
    """Create a .bak backup if one doesn't already exist."""
    bak = filepath + ".bak.pre4d"
    if not os.path.exists(bak):
        shutil.copy2(filepath, bak)


def patch_file(filepath, replacements, description=""):
    """Apply a list of (old, new) string replacements to a file."""
    backup_file(filepath)
    with open(filepath, "r") as f:
        content = f.read()

    original = content
    for old, new in replacements:
        if old not in content:
            # Check if already patched
            if new in content:
                continue
            print(f"  WARNING: Expected string not found in {os.path.basename(filepath)}:")
            print(f"    {old[:80]}...")
            continue
        content = content.replace(old, new)

    if content != original:
        with open(filepath, "w") as f:
            f.write(content)
        print(f"  Patched: {os.path.basename(filepath)} {description}")
    else:
        print(f"  Skipped: {os.path.basename(filepath)} (already patched)")


def patch_types(natten_path):
    """Add 4D dimension and causal types."""
    filepath = os.path.join(natten_path, "types.py")
    patch_file(filepath, [
        # Add Dimension4DType
        (
            "Dimension3DType = Tuple[int, int, int]\n",
            "Dimension3DType = Tuple[int, int, int]\n"
            "Dimension4DType = Tuple[int, int, int, int]\n",
        ),
        # Add CausalArg4DType
        (
            "CausalArg3DType = Tuple[bool, bool, bool]\n",
            "CausalArg3DType = Tuple[bool, bool, bool]\n"
            "CausalArg4DType = Tuple[bool, bool, bool, bool]\n",
        ),
        # Add Dimension4DTypeOrDed
        (
            "Dimension3DTypeOrDed = Union[int, Dimension3DType]\n",
            "Dimension3DTypeOrDed = Union[int, Dimension3DType]\n"
            "Dimension4DTypeOrDed = Union[int, Dimension4DType]\n",
        ),
        # Add CausalArg4DTypeOrDed
        (
            "CausalArg3DTypeOrDed = Union[bool, CausalArg3DType]\n",
            "CausalArg3DTypeOrDed = Union[bool, CausalArg3DType]\n"
            "CausalArg4DTypeOrDed = Union[bool, CausalArg4DType]\n",
        ),
        # Extend DimensionType union
        (
            "DimensionType = Union[Dimension1DType, Dimension2DType, Dimension3DType]",
            "DimensionType = Union[Dimension1DType, Dimension2DType, Dimension3DType, Dimension4DType]",
        ),
        # Extend CausalArgType union
        (
            "CausalArgType = Union[CausalArg1DType, CausalArg2DType, CausalArg3DType]",
            "CausalArgType = Union[CausalArg1DType, CausalArg2DType, CausalArg3DType, CausalArg4DType]",
        ),
        # Extend QKTileShapeType
        (
            "    Tuple[Dimension3DType, Dimension3DType],\n]",
            "    Tuple[Dimension3DType, Dimension3DType],\n"
            "    Tuple[Dimension4DType, Dimension4DType],\n]",
        ),
        # Extend CutlassFnaBackwardConfigType
        (
            "    Tuple[Dimension3DType, Dimension3DType, Dimension3DType, bool],\n]",
            "    Tuple[Dimension3DType, Dimension3DType, Dimension3DType, bool],\n"
            "    Tuple[Dimension4DType, Dimension4DType, Dimension4DType, bool],\n]",
        ),
    ], "(4D types)")


def patch_checks(natten_path):
    """Lift dimension restrictions from na_dim<4 to na_dim<5, allow 7D tensors."""
    filepath = os.path.join(natten_path, "utils", "checks.py")
    patch_file(filepath, [
        # Allow 7D tensors in na_tensor_checks
        (
            "if query.dim() not in [4, 5, 6]:",
            "if query.dim() not in [4, 5, 6, 7]:",
        ),
        # Extend na_dim assertions (all occurrences)
        (
            "assert na_dim > 0 and na_dim < 4",
            "assert na_dim > 0 and na_dim < 5",
        ),
        # Allow 7D in check_args_against_input and is_self_attention
        (
            "assert input_tensor.dim() in [4, 5, 6]",
            "assert input_tensor.dim() in [4, 5, 6, 7]",
        ),
        # Extend tile shape check
        (
            "and len(tile_shape) <= 3",
            "and len(tile_shape) <= 4",
        ),
    ], "(4D dimension checks)")

    # Apply replace_all for repeated assertions
    with open(filepath, "r") as f:
        content = f.read()
    backup_file(filepath)
    # Ensure ALL occurrences are replaced
    content = content.replace(
        "assert na_dim > 0 and na_dim < 4",
        "assert na_dim > 0 and na_dim < 5",
    )
    content = content.replace(
        "assert input_tensor.dim() in [4, 5, 6]",
        "assert input_tensor.dim() in [4, 5, 6, 7]",
    )
    with open(filepath, "w") as f:
        f.write(content)


def patch_backend_checks(natten_path):
    """Allow 7D tensors in flex backend checks."""
    filepath = os.path.join(natten_path, "backends", "configs", "checks.py")
    patch_file(filepath, [
        (
            "if query.dim() not in [4, 5, 6]:\n"
            "        target_fn(\n"
            '            "Flex backend expects 4-D, 5-D, or 6-D tensors as inputs (corresponding to FMHA/NA1D, "\n'
            '            f"NA2D, and NA3D), got {query.dim()=}.",',
            "if query.dim() not in [4, 5, 6, 7]:\n"
            "        target_fn(\n"
            '            "Flex backend expects 4-D, 5-D, 6-D, or 7-D tensors as inputs (corresponding to FMHA/NA1D, "\n'
            '            f"NA2D, NA3D, and NA4D), got {query.dim()=}.",',
        ),
    ], "(flex 4D support)")


def patch_flex_configs(natten_path):
    """Add 4D tile shapes and extend dimension assertions in flex config."""
    filepath = os.path.join(natten_path, "backends", "configs", "flex", "__init__.py")
    patch_file(filepath, [
        # Add 4D tile shapes
        (
            "    3: [\n"
            "        # ((4, 4, 8), (4, 4, 8)),\n"
            "        ((4, 4, 4), (4, 4, 4)),\n"
            "        ((2, 4, 8), (2, 4, 8)),\n"
            "        ((2, 4, 8), (4, 4, 4)),\n"
            "    ],\n"
            "}",
            "    3: [\n"
            "        # ((4, 4, 8), (4, 4, 8)),\n"
            "        ((4, 4, 4), (4, 4, 4)),\n"
            "        ((2, 4, 8), (2, 4, 8)),\n"
            "        ((2, 4, 8), (4, 4, 4)),\n"
            "    ],\n"
            "    4: [\n"
            "        ((2, 2, 2, 4), (2, 2, 2, 4)),\n"
            "        ((2, 2, 4, 4), (2, 2, 4, 4)),\n"
            "    ],\n"
            "}",
        ),
        # Extend _get_default_tile_shapes_forward
        (
            "    assert na_dim in [1, 2, 3]\n",
            "    assert na_dim in [1, 2, 3, 4]\n",
        ),
        (
            "    if na_dim == 3:\n"
            "        return ((4, 4, 4), (4, 4, 4))\n"
            "\n"
            "    raise NotImplementedError()",
            "    if na_dim == 3:\n"
            "        return ((4, 4, 4), (4, 4, 4))\n"
            "    if na_dim == 4:\n"
            "        return ((2, 2, 2, 4), (2, 2, 2, 4))\n"
            "\n"
            "    raise NotImplementedError()",
        ),
    ], "(4D tile shapes)")

    # Replace all dim assertions
    with open(filepath, "r") as f:
        content = f.read()
    backup_file(filepath)
    content = content.replace(
        "assert input_tensor.dim() in [4, 5, 6]",
        "assert input_tensor.dim() in [4, 5, 6, 7]",
    )
    with open(filepath, "w") as f:
        f.write(content)


def patch_flex_backend(natten_path):
    """Add 4D type imports to flex backend."""
    filepath = os.path.join(natten_path, "backends", "flex.py")
    patch_file(filepath, [
        (
            "    CausalArg3DTypeOrDed,\n"
            "    CausalArgType,",
            "    CausalArg3DTypeOrDed,\n"
            "    CausalArg4DTypeOrDed,\n"
            "    CausalArgType,",
        ),
        (
            "    Dimension3DTypeOrDed,\n"
            "    DimensionType,",
            "    Dimension3DTypeOrDed,\n"
            "    Dimension4DType,\n"
            "    Dimension4DTypeOrDed,\n"
            "    DimensionType,",
        ),
    ], "(4D type imports)")


def patch_functional(natten_path):
    """Add na4d() function and extend na_dim assertion."""
    filepath = os.path.join(natten_path, "functional.py")
    patch_file(filepath, [
        # Extend na_dim assertion
        (
            "assert na_dim in [1, 2, 3]",
            "assert na_dim in [1, 2, 3, 4]",
        ),
        # Add 4D type imports
        (
            "    CausalArg3DTypeOrDed,\n"
            "    CausalArgTypeOrDed,",
            "    CausalArg3DTypeOrDed,\n"
            "    CausalArg4DTypeOrDed,\n"
            "    CausalArgTypeOrDed,",
        ),
        (
            "    Dimension3DTypeOrDed,\n"
            "    DimensionType,",
            "    Dimension3DTypeOrDed,\n"
            "    Dimension4DType,\n"
            "    Dimension4DTypeOrDed,\n"
            "    DimensionType,",
        ),
    ], "(na4d function)")

    # Append na4d function if not already present
    with open(filepath, "r") as f:
        content = f.read()

    if "def na4d(" not in content:
        na4d_code = '''

def na4d(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: Dimension4DTypeOrDed,
    stride: Dimension4DTypeOrDed = 1,
    dilation: Dimension4DTypeOrDed = 1,
    is_causal: Optional[CausalArg4DTypeOrDed] = False,
    scale: Optional[float] = None,
    additional_keys: Optional[Tensor] = None,
    additional_values: Optional[Tensor] = None,
    attention_kwargs: Optional[Dict] = None,
    backend: Optional[str] = None,
    q_tile_shape: Optional[Dimension4DType] = None,
    kv_tile_shape: Optional[Dimension4DType] = None,
    backward_q_tile_shape: Optional[Dimension4DType] = None,
    backward_kv_tile_shape: Optional[Dimension4DType] = None,
    backward_kv_splits: Optional[Dimension4DType] = None,
    backward_use_pt_reduction: bool = False,
    run_persistent_kernel: bool = True,
    kernel_schedule: Optional[Union[str, KernelSchedule]] = None,
    torch_compile: bool = False,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Computes 4-D neighborhood attention for 3D spatial + temporal data.

    Parameters:
        query: 7-D tensor [batch, T, D, H, W, heads, head_dim]
        key:   same layout as query
        value: same layout as query
        kernel_size: (k_t, k_d, k_h, k_w) or int
        stride: stride per dimension, default 1
        dilation: dilation per dimension, default 1
        is_causal: (causal_t, causal_d, causal_h, causal_w) or bool
        scale: attention scale, default head_dim ** -0.5
    """
    return neighborhood_attention_generic(
        query=query,
        key=key,
        value=value,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        is_causal=is_causal,
        scale=scale,
        additional_keys=additional_keys,
        additional_values=additional_values,
        attention_kwargs=attention_kwargs,
        backend=backend,
        q_tile_shape=q_tile_shape,
        kv_tile_shape=kv_tile_shape,
        backward_q_tile_shape=backward_q_tile_shape,
        backward_kv_tile_shape=backward_kv_tile_shape,
        backward_kv_splits=backward_kv_splits,
        backward_use_pt_reduction=backward_use_pt_reduction,
        run_persistent_kernel=run_persistent_kernel,
        kernel_schedule=kernel_schedule,
        torch_compile=torch_compile,
        return_lse=return_lse,
    )
'''
        with open(filepath, "a") as f:
            f.write(na4d_code)
        print(f"  Appended: na4d() function to functional.py")
    else:
        print(f"  Skipped: na4d() already exists in functional.py")


def patch_modules(natten_path):
    """Add NeighborhoodAttention4D module."""
    filepath = os.path.join(natten_path, "modules.py")
    patch_file(filepath, [
        # Add 4D type imports
        (
            "    CausalArg3DTypeOrDed,\n"
            "    CausalArgTypeOrDed,",
            "    CausalArg3DTypeOrDed,\n"
            "    CausalArg4DTypeOrDed,\n"
            "    CausalArgTypeOrDed,",
        ),
        (
            "    Dimension3DTypeOrDed,\n"
            "    DimensionTypeOrDed,",
            "    Dimension3DTypeOrDed,\n"
            "    Dimension4DTypeOrDed,\n"
            "    DimensionTypeOrDed,",
        ),
    ], "(4D type imports)")

    # Append NeighborhoodAttention4D class if not present
    with open(filepath, "r") as f:
        content = f.read()

    if "class NeighborhoodAttention4D" not in content:
        na4d_class = '''

class NeighborhoodAttention4D(NeighborhoodAttentionGeneric):
    """
    4-D Neighborhood Attention torch module.

    For input shape [B, T, D, H, W, C], applies local attention
    with kernel_size (k_t, k_d, k_h, k_w).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kernel_size: Dimension4DTypeOrDed,
        stride: Dimension4DTypeOrDed = 1,
        dilation: Dimension4DTypeOrDed = 1,
        is_causal: CausalArg4DTypeOrDed = False,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        proj_drop: float = 0.0,
    ):
        super().__init__(
            na_dim=4,
            embed_dim=embed_dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            is_causal=is_causal,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            proj_drop=proj_drop,
        )
'''
        with open(filepath, "a") as f:
            f.write(na4d_class)
        print(f"  Appended: NeighborhoodAttention4D class to modules.py")
    else:
        print(f"  Skipped: NeighborhoodAttention4D already exists in modules.py")


def patch_init(natten_path):
    """Export na4d and NeighborhoodAttention4D from __init__.py."""
    filepath = os.path.join(natten_path, "__init__.py")
    patch_file(filepath, [
        # Add na4d to functional imports
        (
            "from .functional import attention, merge_attentions, na1d, na2d, na3d",
            "from .functional import attention, merge_attentions, na1d, na2d, na3d, na4d",
        ),
        # Add NeighborhoodAttention4D to module imports
        (
            "    NeighborhoodAttention3D,\n)",
            "    NeighborhoodAttention3D,\n"
            "    NeighborhoodAttention4D,\n)",
        ),
        # Add to __all__
        (
            '    "NeighborhoodAttention3D",\n'
            '    "are_deterministic_algorithms_enabled",',
            '    "NeighborhoodAttention3D",\n'
            '    "NeighborhoodAttention4D",\n'
            '    "are_deterministic_algorithms_enabled",',
        ),
        (
            '    "na3d",\n'
            '    "attention",',
            '    "na3d",\n'
            '    "na4d",\n'
            '    "attention",',
        ),
    ], "(4D exports)")


def clear_pycache(natten_path):
    """Remove __pycache__ directories to force recompilation."""
    import shutil
    count = 0
    for root, dirs, files in os.walk(natten_path):
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d))
                count += 1
    print(f"  Cleared {count} __pycache__ directories")


def verify():
    """Verify the patch was applied successfully."""
    # Force reimport
    mods_to_remove = [k for k in sys.modules if k.startswith("natten")]
    for m in mods_to_remove:
        del sys.modules[m]

    try:
        from natten import NeighborhoodAttention4D, na4d
        print("  Import NeighborhoodAttention4D: OK")
        print("  Import na4d: OK")
    except ImportError as e:
        print(f"  Import FAILED: {e}")
        return False

    try:
        from natten import NeighborhoodAttention3D, na3d
        print("  Import NeighborhoodAttention3D: OK (backward compat)")
    except ImportError as e:
        print(f"  3D Import FAILED: {e}")
        return False

    return True


def main():
    print("=" * 60)
    print("NATTEN 4D Patch")
    print("=" * 60)

    natten_path = get_natten_path()

    import natten
    print(f"  NATTEN version: {natten.__version__}")
    print(f"  NATTEN path:    {natten_path}")
    print()

    print("[1/8] Patching types.py ...")
    patch_types(natten_path)

    print("[2/8] Patching utils/checks.py ...")
    patch_checks(natten_path)

    print("[3/8] Patching backends/configs/checks.py ...")
    patch_backend_checks(natten_path)

    print("[4/8] Patching backends/configs/flex/ ...")
    patch_flex_configs(natten_path)

    print("[5/8] Patching backends/flex.py ...")
    patch_flex_backend(natten_path)

    print("[6/8] Patching functional.py ...")
    patch_functional(natten_path)

    print("[7/8] Patching modules.py ...")
    patch_modules(natten_path)

    print("[8/8] Patching __init__.py ...")
    patch_init(natten_path)

    print()
    print("Clearing __pycache__ ...")
    clear_pycache(natten_path)

    print()
    print("Verifying ...")
    success = verify()

    print()
    if success:
        print("=" * 60)
        print("PATCH APPLIED SUCCESSFULLY!")
        print()
        print("Usage:")
        print("  from natten import NeighborhoodAttention4D")
        print()
        print("  model = NeighborhoodAttention4D(")
        print("      embed_dim=256,")
        print("      num_heads=8,")
        print("      kernel_size=(7, 7, 7, 7),")
        print("      is_causal=(True, False, False, False),")
        print("  )")
        print()
        print("  # Input: [B, T, D, H, W, C]")
        print("  x = torch.randn(1, 8, 16, 16, 16, 256)")
        print("  out = model(x)  # Same shape output")
        print("=" * 60)
    else:
        print("PATCH VERIFICATION FAILED!")
        print("Check the warnings above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
