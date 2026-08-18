class DifferenceTransformer(nn.Module):
    def __init__(self, n_findings=18, d_in=512, d_model=256, n_layers=2, n_heads=4,
                 dropout=0.1, antisym=False, magnitude=False,
                 finding_conditioning=True, finding_as_4th_token=False,
                 use_learned_finding_emb=True, frozen_finding_emb=None,
                 tau_con_init=0.07, learnable_tau_con=True):
        super().__init__()
        self.finding_conditioning = finding_conditioning
        self.finding_as_4th_token = finding_as_4th_token and finding_conditioning
        self.antisym = antisym
        self.W = nn.Linear(d_in, d_model)
        self.role = nn.Parameter(torch.randn(2, d_model) * 0.02)
        self.e_diff = nn.Parameter(torch.randn(1, d_model) * 0.02)
        if finding_conditioning and use_learned_finding_emb:
            self.finding_emb = nn.Embedding(n_findings, d_model)
            nn.init.normal_(self.finding_emb.weight, std=0.02)
        else:
            self.finding_emb = None
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout=dropout,
            batch_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, d_in)
        self.mag_head = nn.Linear(d_model, 1) if magnitude else None
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(1 / 0.07))))
        log_tau = float(np.log(tau_con_init))
        if learnable_tau_con:
            self.log_tau_con = nn.Parameter(torch.tensor(log_tau))
        else:
            self.register_buffer('log_tau_con', torch.tensor(log_tau))

    def _pass(self, vp, vc, fid):
        B = vp.size(0)
        tp = self.W(vp) + self.role[0]
        tc = self.W(vc) + self.role[1]
        ed = self.e_diff.expand(B, -1).clone()
        if self.finding_emb is not None and not self.finding_as_4th_token:
            ed = ed + self.finding_emb(fid)
            seq = torch.stack([ed, tp, tc], 1)
        elif self.finding_emb is not None and self.finding_as_4th_token:
            seq = torch.stack([ed, tp, tc, self.finding_emb(fid)], 1)
        else:
            seq = torch.stack([ed, tp, tc], 1)
        h = self.enc(seq)[:, 0]
        mag = self.mag_head(h).squeeze(-1) if self.mag_head is not None else None
        return self.head(h), mag

    def forward(self, vp, vc, fid=None):
        if fid is None:
            fid = torch.zeros(vp.size(0), dtype=torch.long, device=vp.device)
        vd, mag = self._pass(vp, vc, fid)
        if self.antisym:
            vr, _ = self._pass(vc, vp, fid)
            vd = vd - vr
        return vd, mag

    def tau_con(self):
        return self.log_tau_con.exp().clamp(min=1e-3, max=1.0)


def logits_from_proto(vd, proto, logit_scale):
    vd = F.normalize(vd, dim=-1)
    pr = F.normalize(proto, dim=-1)
    cos = torch.einsum('bd,bkd->bk', vd, pr)
    return logit_scale.exp().clamp(max=100) * cos


def sample_eval_prototypes(fn_list, rng):
    rows = []
    for f in fn_list:
        bank = EVAL_BANK_EMB[f]
        chosen = [bank[c, rng.randrange(bank.size(1))] for c in range(3)]
        rows.append(torch.stack(chosen, 0))
    return torch.stack(rows, 0)


def logits_from_holdout(vd, fn_list, logit_scale, rng):
    proto = sample_eval_prototypes(fn_list, rng).to(vd.device)
    return logits_from_proto(vd, proto, logit_scale)
