#!/usr/bin/env python3
"""Dump the exact trainable-parameter breakdown of Model v1 to data/param_breakdown.txt."""
import importlib
import os

mod = importlib.import_module("09_global_temporal_model")
cfg = mod.Config()  # v1 defaults: d_model=512, 2 layers, flags off
model = mod.GlobalTemporalTransformer(cfg)

lines = []
total = 0
groups = {}
for name, p in model.named_parameters():
    n = p.numel()
    total += n
    top = name.split(".")[0]
    groups[top] = groups.get(top, 0) + n
    lines.append(f"  {name:45s} {tuple(p.shape)!s:20s} {n:>10,}")

out = []
out.append("MODEL v1 trainable parameters (per-fold, fresh init)")
out.append("config: " + cfg.tag())
out.append("=" * 78)
out.append("PER-PARAMETER:")
out += lines
out.append("-" * 78)
out.append("BY MODULE:")
for k, v in groups.items():
    out.append(f"  {k:20s} {v:>12,}")
out.append("  logit_scale (sesparate) {:>12,}".format(1))
out.append("-" * 78)
out.append(f"TOTAL model params      {total:>12,}")
out.append(f"TOTAL incl logit_scale  {total + 1:>12,}")

os.makedirs("data", exist_ok=True)
open("data/param_breakdown.txt", "w").write("\n".join(out))
print("wrote data/param_breakdown.txt")
