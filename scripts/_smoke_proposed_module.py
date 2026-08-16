#!/usr/bin/env python3
"""Smoke-test model + masked SupCon from the proposed notebook (no Colab/CT-CLIP)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "train_proposed_finding_conditioned_colab.ipynb"


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    assert len(nb["cells"]) == 28, len(nb["cells"])
    print("notebook cells OK:", len(nb["cells"]))

    g = {
        "np": np,
        "torch": torch,
        "nn": nn,
        "F": F,
        "FINDINGS": [f"f{i}" for i in range(18)],
        "D_MODEL": 256,
        "ANTISYM": False,
        "USE_MAGNITUDE": True,
        "FINDING_CONDITIONING": True,
        "FINDING_AS_4TH_TOKEN": False,
        "USE_LEARNED_FINDING_EMB": True,
        "FINDING_NAME_EMB": None,
        "TAU_CON_INIT": 0.07,
    }
    src = "".join(nb["cells"][16]["source"])
    exec(src, g)

    DifferenceTransformer = g["DifferenceTransformer"]
    logits_from = g["logits_from"]
    masked_supcon_loss = g["masked_supcon_loss"]
    npar = g["npar"]

    B = 32
    vp = torch.randn(B, 512)
    vc = torch.randn(B, 512)
    fid = torch.randint(0, 18, (B,))
    y = torch.randint(0, 3, (B,))
    fid[:8] = 0
    y[:4] = 0
    y[4:8] = 1
    fid[8:16] = 1
    y[8:12] = 0
    y[12:16] = 2

    model = DifferenceTransformer(
        n_findings=18,
        d_model=256,
        magnitude=True,
        finding_conditioning=True,
        finding_as_4th_token=False,
        use_learned_finding_emb=True,
        tau_con_init=0.07,
    )
    vd, mag = model(vp, vc, fid)
    assert vd.shape == (B, 512)
    assert mag.shape == (B,)
    proto = F.normalize(torch.randn(B, 3, 512), dim=-1)
    lg = logits_from(vd, proto, model.logit_scale)
    assert lg.shape == (B, 3)
    loss_ce = F.cross_entropy(lg, y)
    loss_mag = F.binary_cross_entropy_with_logits(mag, (y != 1).float())
    loss_con = masked_supcon_loss(vd, y, fid, model.tau_con())
    loss = loss_ce + 0.5 * loss_mag + 0.5 * loss_con
    loss.backward()
    print(
        "forward/backward OK",
        float(loss),
        float(loss_con),
        "params",
        npar(model),
    )

    m2 = DifferenceTransformer(
        n_findings=18, finding_as_4th_token=True, magnitude=False
    )
    vd2, mag2 = m2(vp, vc, fid)
    assert vd2.shape == (B, 512) and mag2 is None
    print("4th-token path OK")

    m3 = DifferenceTransformer(n_findings=18, finding_conditioning=False)
    vd3, _ = m3(vp, vc, fid)
    assert vd3.shape == (B, 512)
    print("no-conditioning path OK")

    l0 = masked_supcon_loss(
        torch.randn(B, 512),
        torch.arange(B) % 3,
        torch.arange(B),
        torch.tensor(0.07),
    )
    print("supcon degenerate OK", float(l0))
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
