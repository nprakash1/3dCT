import numpy as np
try:
    import torch, torch.nn as nn
except Exception as e:
    print("torch NOT installed locally -> skip runtime dry-run (Colab has it):", e); raise SystemExit
from sklearn.metrics import f1_score, confusion_matrix
from collections import defaultdict


class TemporalModule(nn.Module):
    def __init__(self, d_in=512, d_model=256, n_layers=2, n_heads=4, dropout=0.1, antisym=False):
        super().__init__()
        self.W = nn.Linear(d_in, d_model)
        self.role = nn.Parameter(torch.randn(2, d_model) * 0.02)
        self.e_diff = nn.Parameter(torch.randn(1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, d_in)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07)))
        self.antisym = antisym

    def _pass(self, vp, vc):
        B = vp.size(0)
        tp = self.W(vp) + self.role[0]
        tc = self.W(vc) + self.role[1]
        ed = self.e_diff.expand(B, -1)
        toks = torch.stack([ed, tp, tc], dim=1)
        return self.head(self.enc(toks)[:, 0])

    def forward(self, vp, vc):
        return self._pass(vp, vc) - self._pass(vc, vp) if self.antisym else self._pass(vp, vc)


def logits_from(vd, proto, ls):
    vd = nn.functional.normalize(vd, dim=-1)
    pr = nn.functional.normalize(proto, dim=-1)
    return ls.exp().clamp(max=100) * torch.einsum('bd,bkd->bk', vd, pr)


torch.manual_seed(0); np.random.seed(0)
N = 800
VP = torch.randn(N, 512); VC = torch.randn(N, 512); PR = torch.randn(N, 3, 512)
Y = torch.randint(0, 3, (N,)); F = [f"finding_{i%6}" for i in range(N)]

for antisym in (False, True):
    m = TemporalModule(antisym=antisym)
    opt = torch.optim.AdamW(m.parameters(), 1e-3); ce = nn.CrossEntropyLoss()
    for ep in range(3):
        m.train(); lg = logits_from(m(VP, VC), PR, m.logit_scale); loss = ce(lg, Y)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        pred = logits_from(m(VP, VC), PR, m.logit_scale).argmax(1).numpy()
    y = Y.numpy()
    macro = f1_score(y, pred, labels=[0, 1, 2], average='macro', zero_division=0)
    print(f"antisym={antisym}: loss={loss.item():.3f} macroF1={macro:.3f} logits_shape={tuple(lg.shape)}")
    if antisym:
        with torch.no_grad():
            a = m(VP[:4], VC[:4]); b = m(VC[:4], VP[:4])
        print("   antisymmetry check max|a+b| =", round(float((a + b).abs().max()), 6), "(should be ~0)")

by = defaultdict(lambda: {'y': [], 'p': []})
for yi, pi, fi in zip(y, pred, F):
    by[fi]['y'].append(yi); by[fi]['p'].append(pi)
for f, d in list(by.items())[:2]:
    yy, pp = np.array(d['y']), np.array(d['p']); pres = sorted(set(yy.tolist()))
    print(f"  {f}: n={len(yy)} macroF1*={f1_score(yy, pp, labels=pres, average='macro', zero_division=0):.3f}")
print("confusion:\n", confusion_matrix(y, pred, labels=[0, 1, 2]))
print("DRY-RUN OK — module, einsum logits, training step, per-class + per-disease metrics all execute.")
