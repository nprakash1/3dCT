#!/usr/bin/env python3
"""
09_global_temporal_model.py

MODEL v1 -- "Global Temporal Transformer" (GTT).

The first TRAINABLE model for CT interval-change, kept deliberately minimal so it
is a clean ablation over the zero-shot baseline (step 08). It reuses the SAME
cached 512-d MERLIN image embeddings and the SAME text-prompt supervision, but
replaces the hard-coded  d = v_current - v_prior  subtraction with a small
*learned* temporal transformer over two global vectors.

--------------------------------------------------------------------------------
PIPELINE (all encoders FROZEN; only the small module below is trained)
--------------------------------------------------------------------------------
  cached 512-d  v_prior, v_current            (data/emb_mini_d32.pt)
        |
        |  W: Linear(512 -> d_model)   + learnable role embeddings
        v
  tokens = [ e_diff ,  W v_prior + role_prior ,  W v_current + role_current ]
        |
        |  tiny Transformer encoder (N_LAYERS, N_HEADS)
        v
  H[0] (the e_diff token)  ->  Linear(d_model -> 512)  ->  v_d   (back in
                                                                MERLIN text space)
        |
        |  cosine( v_d , { t_worse, t_stable, t_improved } ) * logit_scale
        v
  logits over {improved, stable, worse}  ->  weighted cross-entropy

The 3 text prototypes t_* are frozen MERLIN text encodings of the SAME paraphrase
prompt bank used by step 08 (averaged per class). Because there are only 3 fixed
prototypes, contrastive alignment reduces to supervised cosine-classification.

--------------------------------------------------------------------------------
ABLATION FLAGS (see CONFIG below) -- v1 keeps BOTH OFF
--------------------------------------------------------------------------------
  USE_ANTISYMMETRY   : v_d = readout(c,p) - readout(p,c)  =>  "no change" == 0
                       vector by construction (fixes cosine's inability to
                       represent "stable"). Off = single forward pass.
  USE_MAGNITUDE_HEAD : extra scalar head predicting log(diam_curr/diam_prior)
                       from RECIST, with an aux regression loss; "stable" can be
                       routed by predicted magnitude. Off = pure 3-way cosine-CE.

Everything is organized so flipping a flag is the ONLY change between ablations.

Usage:
    python scripts/09_global_temporal_model.py
    python scripts/09_global_temporal_model.py --antisymmetry --magnitude-head
    python scripts/09_global_temporal_model.py --d-model 256 --epochs 400
"""
import argparse
import csv
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# NOTE: MerlinEmbedder is imported LAZILY inside build_text_prototypes() so that
# once data/text_prototypes.pt is cached, this whole script runs with NO MERLIN
# dependency (pure CPU on cached tensors).

warnings.filterwarnings("ignore")


# order is FIXED everywhere (matches step 06/08)
CLASSES = ["improved", "stable", "worse"]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}

# reuse step 08's paraphrase bank so text supervision is identical to the baseline
ENSEMBLE_PROMPTS = {
    "worse": [
        "the lesion has increased in size and worsened compared to the prior study",
        "the finding is larger and has progressed since the previous exam",
        "interval enlargement of the lesion, consistent with disease progression",
        "the mass has grown compared to the prior study",
        "increased size of the lesion indicating worsening",
    ],
    "stable": [
        "the lesion is unchanged and stable compared to the prior study",
        "no significant interval change in the lesion since the previous exam",
        "the finding is stable with no change in size",
        "the lesion appears similar to the prior study",
        "no measurable change in the lesion compared to prior",
    ],
    "improved": [
        "the lesion has decreased in size and improved compared to the prior study",
        "the finding is smaller and has regressed since the previous exam",
        "interval decrease in the lesion, consistent with treatment response",
        "the mass has shrunk compared to the prior study",
        "decreased size of the lesion indicating improvement",
    ],
}


# ============================================================================
# CONFIG  --  the single place that defines an experiment / ablation
# ============================================================================
@dataclass
class Config:
    # --- data ---
    manifest: str = "data/dlt/manifest_mini.csv"
    emb_cache: str = "data/emb_mini_d32.pt"
    proto_cache: str = "data/text_prototypes.pt"

    # --- model (v1 defaults) ---
    d_model: int = 512          # transformer width (512 = no down-projection)
    n_layers: int = 2
    n_heads: int = 8
    ffn_mult: int = 2
    dropout: float = 0.1

    # --- ABLATION FLAGS (v1 = both OFF) ---
    use_antisymmetry: bool = False
    use_magnitude_head: bool = False
    lambda_recist: float = 0.5   # weight of the magnitude aux loss (if enabled)

    # --- training ---
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-4
    n_folds: int = 5
    seed: int = 0
    class_weighted: bool = True  # weight CE by inverse class frequency (84% stable)

    emb_dim: int = 512           # frozen MERLIN joint dim (fixed)

    def tag(self) -> str:
        """A short human-readable name for this configuration (for logs/ablations)."""
        parts = [f"d{self.d_model}", f"L{self.n_layers}"]
        parts.append("antisym" if self.use_antisymmetry else "plain")
        if self.use_magnitude_head:
            parts.append("mag")
        return "GTT_" + "_".join(parts)


# ============================================================================
# MODEL
# ============================================================================
class GlobalTemporalTransformer(nn.Module):
    """Two global vectors -> tiny transformer with a learned diff token -> v_d."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # 512 -> d_model input projection
        self.in_proj = nn.Linear(cfg.emb_dim, d)

        # learnable tokens / embeddings
        self.diff_token = nn.Parameter(torch.randn(d) * 0.02)      # the "e_diff" query
        self.role_prior = nn.Parameter(torch.randn(d) * 0.02)      # "which exam" = prior
        self.role_current = nn.Parameter(torch.randn(d) * 0.02)    # "which exam" = current

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_heads, dim_feedforward=d * cfg.ffn_mult,
            dropout=cfg.dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)

        # d_model -> 512 output head (re-enters MERLIN's text space)
        self.out_head = nn.Linear(d, cfg.emb_dim)

        # optional magnitude head (predicts log diameter-ratio); built but unused if flag off
        self.mag_head = nn.Linear(d, 1) if cfg.use_magnitude_head else None

    def _readout(self, v_cur: torch.Tensor, v_pri: torch.Tensor):
        """One forward pass. Returns (v_d unnormalized [B,512], mag [B] or None)."""
        B = v_cur.shape[0]
        tok_pri = self.in_proj(v_pri) + self.role_prior
        tok_cur = self.in_proj(v_cur) + self.role_current
        tok_diff = self.diff_token.unsqueeze(0).expand(B, -1)
        seq = torch.stack([tok_diff, tok_pri, tok_cur], dim=1)   # (B, 3, d_model)
        h = self.encoder(seq)                                    # (B, 3, d_model)
        h_diff = h[:, 0]                                         # diff-token output
        v_d = self.out_head(h_diff)                             # (B, 512)
        mag = self.mag_head(h_diff).squeeze(-1) if self.mag_head is not None else None
        return v_d, mag

    def forward(self, v_prior: torch.Tensor, v_current: torch.Tensor):
        v_d, mag = self._readout(v_current, v_prior)
        if self.cfg.use_antisymmetry:
            # subtract the time-reversed readout => "no change" maps to 0 vector
            v_d_rev, _ = self._readout(v_prior, v_current)
            v_d = v_d - v_d_rev
        v_d = F.normalize(v_d, dim=-1)
        return v_d, mag


# ============================================================================
# TEXT PROTOTYPES (frozen MERLIN text encoder; cached to disk)
# ============================================================================
def build_text_prototypes(cfg: Config) -> torch.Tensor:
    """Return (3, 512) unit prototypes in CLASSES order, cached to proto_cache."""
    if os.path.exists(cfg.proto_cache):
        protos = torch.load(cfg.proto_cache, weights_only=False)
        print(f"loaded text prototypes from {cfg.proto_cache}")
        return protos
    print("==> Building text prototypes with frozen MERLIN text encoder...")
    from merlin_utils import MerlinEmbedder  # lazy: only needed to build the cache
    embedder = MerlinEmbedder(device="cpu")

    rows = []
    for c in CLASSES:
        embs = embedder.embed_texts(ENSEMBLE_PROMPTS[c], normalize=True)  # (k,512)
        rows.append(F.normalize(embs.mean(dim=0), dim=-1))
    protos = torch.stack(rows, dim=0)  # (3, 512)
    torch.save(protos, cfg.proto_cache)
    print(f"cached text prototypes -> {cfg.proto_cache}")
    return protos


# ============================================================================
# DATA
# ============================================================================
def patient_id(nii_name: str) -> str:
    """DeepLesion nii names start with a 6-digit patient index."""
    return nii_name[:6]


def load_pairs(cfg: Config, emb: Dict[str, torch.Tensor]) -> List[dict]:
    """Load manifest rows whose BOTH volumes have a cached embedding."""
    pairs = []
    with open(cfg.manifest) as f:
        for r in csv.DictReader(f):
            s, t = r["source_nii"], r["target_nii"]
            if s in emb and t in emb and r["label"] in CLASS_IDX:
                pairs.append(r)
    return pairs


def group_kfold(pairs: List[dict], n_folds: int, seed: int) -> List[int]:
    """Assign each pair a fold id, grouping by PATIENT (no patient across folds)."""
    patients = sorted({patient_id(r["source_nii"]) for r in pairs})
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    fold_of_patient = {p: (i % n_folds) for i, p in enumerate(patients)}
    return [fold_of_patient[patient_id(r["source_nii"])] for r in pairs]


def pairs_to_tensors(pairs, emb, device):
    vp = torch.stack([emb[r["source_nii"]] for r in pairs]).to(device).float()
    vc = torch.stack([emb[r["target_nii"]] for r in pairs]).to(device).float()
    y = torch.tensor([CLASS_IDX[r["label"]] for r in pairs], device=device)
    # RECIST log diameter-ratio target for the (optional) magnitude head
    logratio = []
    for r in pairs:
        pl, cl = float(r["prior_long_mm"]), float(r["curr_long_mm"])
        logratio.append(np.log(cl / pl) if pl > 0 and cl > 0 else 0.0)
    lr = torch.tensor(logratio, device=device).float()
    return vp, vc, y, lr


# ============================================================================
# METRICS
# ============================================================================
def confusion(y_true, y_pred):
    M = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[t, p] += 1
    return M


def metrics(M):
    per, f1s = {}, []
    for i, c in enumerate(CLASSES):
        tp = M[i, i]; fp = M[:, i].sum() - tp; fn = M[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[c] = (prec, rec, f1); f1s.append(f1)
    acc = np.trace(M) / M.sum() if M.sum() else 0.0
    return acc, float(np.mean(f1s)), per


# ============================================================================
# TRAIN / EVAL (one fold)
# ============================================================================
def train_one_fold(cfg, protos, tr, va, device):
    model = GlobalTemporalTransformer(cfg).to(device)
    # CLIP-style learnable temperature
    logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32,
                                             device=device))
    params = list(model.parameters()) + [logit_scale]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    vp, vc, y, lr = tr
    # class weights from TRAIN distribution (handles 84% stable)
    if cfg.class_weighted:
        counts = torch.bincount(y, minlength=3).float()
        w = (counts.sum() / (3 * counts.clamp(min=1))).to(device)
    else:
        w = None

    protos = protos.to(device)
    model.train()
    for _ in range(cfg.epochs):
        opt.zero_grad()
        v_d, mag = model(vp, vc)
        logits = logit_scale.exp() * (v_d @ protos.t())    # (B,3)
        loss = F.cross_entropy(logits, y, weight=w)
        if cfg.use_magnitude_head and mag is not None:
            loss = loss + cfg.lambda_recist * F.smooth_l1_loss(mag, lr)
        loss.backward()
        opt.step()

    # ---- eval on val ----
    model.eval()
    vpv, vcv, yv, _ = va
    with torch.no_grad():
        v_d, _ = model(vpv, vcv)
        logits = logit_scale.exp() * (v_d @ protos.t())
        pred = logits.argmax(dim=-1).cpu().numpy()
    return yv.cpu().numpy(), pred


# ============================================================================
# MAIN
# ============================================================================
def main():
    cfg = Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=cfg.d_model)
    ap.add_argument("--n-layers", type=int, default=cfg.n_layers)
    ap.add_argument("--epochs", type=int, default=cfg.epochs)
    ap.add_argument("--seed", type=int, default=cfg.seed)
    ap.add_argument("--antisymmetry", action="store_true")
    ap.add_argument("--magnitude-head", action="store_true")
    ap.add_argument("--no-class-weight", action="store_true")
    args = ap.parse_args()

    # fold CLI into CONFIG (keeps CONFIG the single source of truth)
    cfg.d_model = args.d_model
    cfg.n_layers = args.n_layers
    cfg.epochs = args.epochs
    cfg.seed = args.seed
    cfg.use_antisymmetry = args.antisymmetry
    cfg.use_magnitude_head = args.magnitude_head
    cfg.class_weighted = not args.no_class_weight

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print(f"MODEL v1: Global Temporal Transformer   [{cfg.tag()}]")
    print(f"  antisymmetry={cfg.use_antisymmetry}  magnitude_head={cfg.use_magnitude_head}"
          f"  d_model={cfg.d_model}  layers={cfg.n_layers}  device={device}")
    print("=" * 70)

    if not os.path.exists(cfg.emb_cache):
        print(f"!! missing {cfg.emb_cache}; run step 06 first.")
        return
    emb = torch.load(cfg.emb_cache, weights_only=False)
    print(f"loaded {len(emb)} cached image embeddings from {cfg.emb_cache}")

    protos = build_text_prototypes(cfg)
    pairs = load_pairs(cfg, emb)
    print(f"usable pairs: {len(pairs)}")

    folds = group_kfold(pairs, cfg.n_folds, cfg.seed)

    # ---- reference: always-'stable' ----
    y_all = np.array([CLASS_IDX[r["label"]] for r in pairs])
    stable_i = CLASS_IDX["stable"]
    ref_pred = np.full_like(y_all, stable_i)
    ref_acc, ref_f1, _ = metrics(confusion(y_all, ref_pred))

    # ---- out-of-fold predictions for the trained model ----
    oof_true = np.zeros(len(pairs), dtype=int)
    oof_pred = np.zeros(len(pairs), dtype=int)
    fold_f1s = []
    for k in range(cfg.n_folds):
        tr_idx = [i for i, f in enumerate(folds) if f != k]
        va_idx = [i for i, f in enumerate(folds) if f == k]
        tr = pairs_to_tensors([pairs[i] for i in tr_idx], emb, device)
        va = pairs_to_tensors([pairs[i] for i in va_idx], emb, device)
        yv, pv = train_one_fold(cfg, protos, tr, va, device)
        oof_true[va_idx] = yv
        oof_pred[va_idx] = pv
        _, f1_k, _ = metrics(confusion(yv, pv))
        fold_f1s.append(f1_k)
        print(f"  fold {k}: n_val={len(va_idx):3d}  macro-F1={f1_k:.3f}")

    # ---- pooled out-of-fold metrics ----
    M = confusion(oof_true, oof_pred)
    acc, mf1, per = metrics(M)

    print("\n" + "=" * 70)
    print(f"RESULTS  [{cfg.tag()}]   (pooled out-of-fold over {cfg.n_folds} folds)")
    print("=" * 70)
    print(f"  accuracy = {acc:.3f}   macro-F1 = {mf1:.3f}"
          f"   (per-fold macro-F1 = {np.mean(fold_f1s):.3f} +/- {np.std(fold_f1s):.3f})")
    for c in CLASSES:
        p, r, f1 = per[c]
        print(f"  {c:9s} P={p:.3f} R={r:.3f} F1={f1:.3f}")
    print("  confusion (rows=true, cols=pred)", CLASSES)
    print(M)

    print("\n----------------- SUMMARY (macro-F1 / accuracy) -----------------")
    print(f"  always-stable (ref)     : {ref_f1:.3f} / {ref_acc:.3f}")
    print(f"  {cfg.tag():23s} : {mf1:.3f} / {acc:.3f}")
    print("\nCompare against zero-shot baselines from step 08 "
          "(A/B/C) reported in docs/baseline_results.png.")


if __name__ == "__main__":
    main()
