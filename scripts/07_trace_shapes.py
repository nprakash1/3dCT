#!/usr/bin/env python3
"""
07_trace_shapes.py

Print the tensor shape at EVERY step of the baseline pipeline for one volume:
  (A) each MONAI preprocessing transform, applied one at a time
  (B) the reshape into MERLIN's expected input
  (C) each internal stage of the frozen i3D-ResNet-152 image encoder
      (conv1 -> maxpool -> layer1..4 -> avgpool -> contrastive_head)
  (D) the final 512-d embedding, and the pair-level change vector d

This is purely for understanding -- it changes no model behavior.

Usage:
    python scripts/07_trace_shapes.py --depth 32
    python scripts/07_trace_shapes.py --depth 160   # compare: same 512-d out
"""
import argparse
import os
import sys
import warnings

import torch

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, SpatialPadd, CenterSpatialCropd, ToTensord,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merlin_utils import MerlinEmbedder  # noqa: E402

warnings.filterwarnings("ignore")


def shp(x):
    try:
        return tuple(x.shape)
    except Exception:
        return type(x).__name__


def build_step_list(depth):
    """Return the transforms as an ordered (name, transform) list."""
    return [
        ("LoadImaged",           LoadImaged(keys=["image"])),
        ("EnsureChannelFirstd",  EnsureChannelFirstd(keys=["image"])),
        ("Orientationd(RAS)",    Orientationd(keys=["image"], axcodes="RAS")),
        ("Spacingd(1.5,1.5,3)",  Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3),
                                          mode=("bilinear"))),
        ("ScaleIntensityRanged", ScaleIntensityRanged(keys=["image"],
                                 a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0,
                                 clip=True)),
        (f"SpatialPadd(224,224,{depth})",
         SpatialPadd(keys=["image"], spatial_size=[224, 224, depth])),
        (f"CenterSpatialCropd(224,224,{depth})",
         CenterSpatialCropd(roi_size=[224, 224, depth], keys=["image"])),
        ("ToTensord",            ToTensord(keys=["image"])),
    ]


def find_i3res(arch):
    """Locate the i3D-ResNet module (the one owning contrastive_head)."""
    for _name, m in arch.named_modules():
        if hasattr(m, "contrastive_head") and hasattr(m, "layer4"):
            return m
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=32)
    ap.add_argument("--nifti-dir", default="data/nifti_mini")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    # pick any built NIfTI
    files = sorted(f for f in os.listdir(args.nifti_dir) if f.endswith(".nii.gz"))
    sample = os.path.join(args.nifti_dir, files[0])
    print(f"Sample volume: {sample}\n")

    # ---------- (A) transforms, one at a time ----------
    print("=" * 66)
    print("(A) PREPROCESSING  (dict['image'] shape after each transform)")
    print("=" * 66)
    data = {"image": sample}
    for name, t in build_step_list(args.depth):
        data = t(data)
        print(f"  {name:34s} -> {shp(data['image'])}")
    img = data["image"]  # (C=1, 224, 224, depth)

    # ---------- (B) reshape into MERLIN input ----------
    print("\n" + "=" * 66)
    print("(B) BATCHING  (what encode_image receives)")
    print("=" * 66)
    x = img.unsqueeze(0).to(args.device)  # (B=1, C=1, 224, 224, depth)
    print(f"  after unsqueeze(0)                 -> {shp(x)}   (B, C, H, W, D)")
    print( "  NOTE: inside forward it will permute -> (B, C, D, H, W)")
    print( "        then triple the channel       -> (B, 3, D, H, W)")

    # ---------- load MERLIN + hooks ----------
    print("\n==> Loading frozen MERLIN...")
    embedder = MerlinEmbedder(device=args.device)
    enc = find_i3res(embedder.arch)
    if enc is None:
        print("!! could not locate i3res encoder; printing final shape only")

    logs = []
    handles = []
    if enc is not None:
        # pre-hook on conv1 shows the (B,3,D,H,W) tensor after permute+cat
        def pre_hook(mod, inp):
            logs.append(("conv1 INPUT (after permute+cat)", shp(inp[0])))
        handles.append(enc.conv1.register_forward_pre_hook(pre_hook))

        watch = ["conv1", "bn1", "relu", "maxpool",
                 "layer1", "layer2", "layer3", "layer4",
                 "avgpool", "contrastive_head", "classifier"]
        for wname in watch:
            mod = getattr(enc, wname, None)
            if mod is None:
                continue

            def mk(nm):
                def hook(mod, inp, out):
                    o = out[0] if isinstance(out, (tuple, list)) else out
                    logs.append((nm, shp(o)))
                return hook
            handles.append(mod.register_forward_hook(mk(wname)))

    # ---------- (C) run encoder ----------
    print("\n" + "=" * 66)
    print("(C) IMAGE ENCODER  (i3D-ResNet-152 internal stages)")
    print("=" * 66)
    with torch.no_grad():
        feats, ehr = embedder.arch.encode_image(x)
    for nm, s in logs:
        print(f"  {nm:34s} -> {s}")
    for h in handles:
        h.remove()

    # ---------- (D) outputs ----------
    print("\n" + "=" * 66)
    print("(D) OUTPUTS")
    print("=" * 66)
    print(f"  encode_image image_features        -> {shp(feats)}")
    print(f"  encode_image ehr/phenotype         -> {shp(ehr)}")
    v = feats.detach().float().cpu().reshape(-1)[:512]
    print(f"  flattened contrastive embedding    -> {shp(v)}   (this is v)")
    print(f"  pair change vector d = v_cur-v_pri -> (512,)  (same 512-d)")

    print("\n==> KEY TAKEAWAY: regardless of input depth (try --depth 160), the "
          "AdaptiveAvgPool3d((1,1,1)) collapses D/H/W, so the embedding is "
          "always 512-d. Depth changes the INTERMEDIATE feature-map sizes only.")


if __name__ == "__main__":
    main()
