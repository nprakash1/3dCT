def _masked_crossmodal_one_way(anchor, other, y_a, y_o, fid_a, fid_o, tau):
    anchor = F.normalize(anchor, dim=-1)
    other = F.normalize(other, dim=-1)
    B, M = anchor.size(0), other.size(0)
    if B == 0 or M == 0:
        return anchor.new_zeros(()), {'n_valid': 0}
    sim = (anchor @ other.t()) / tau
    same_f = fid_a.unsqueeze(1).eq(fid_o.unsqueeze(0))
    same_y = y_a.unsqueeze(1).eq(y_o.unsqueeze(0))
    pos_mask = same_f & same_y
    neg_mask = same_f & ~same_y
    allowed = same_f
    pos_counts = pos_mask.sum(1).float()
    neg_counts = neg_mask.sum(1).float()
    valid = (pos_counts >= 1) & (neg_counts >= 1)
    if not valid.any():
        return anchor.new_zeros(()), {'n_valid': 0}
    neg_large = torch.finfo(sim.dtype).min / 4
    logits = sim.masked_fill(~allowed, neg_large)
    logits = logits - logits.max(1, keepdim=True).values.detach()
    exp_logits = logits.exp() * allowed.float()
    log_prob = logits - exp_logits.sum(1, keepdim=True).clamp(min=1e-8).log()
    pos_log = torch.where(pos_mask, log_prob, torch.zeros_like(log_prob))
    mean_pos = pos_log.sum(1) / pos_counts.clamp(min=1.0)
    loss = -mean_pos[valid].mean()
    if not torch.isfinite(loss):
        loss = anchor.new_zeros(())
    return loss, {'n_valid': int(valid.sum())}


def masked_crossmodal_supcon_loss(d, t, y, fid, tau, symmetric=False):
    t = t.detach()
    li, st = _masked_crossmodal_one_way(d, t, y, y, fid, fid, tau)
    if not symmetric:
        return li, st
    lt, st2 = _masked_crossmodal_one_way(t, d, y, y, fid, fid, tau)
    return 0.5 * (li + lt), {**st, 'n_valid_t2i': st2.get('n_valid', 0)}


def build_buckets(split):
    y, fid = DATA[split]['y'], DATA[split]['fid']
    buckets = defaultdict(list)
    for i in range(len(y)):
        buckets[(int(fid[i]), int(y[i]))].append(i)
    return buckets

TRAIN_BUCKETS = build_buckets('train')

def sample_contrastive_batch(buckets, k_findings=K_FINDINGS_PER_BATCH,
                             n_per_class=N_PER_CLASS, max_bs=MAX_BATCH_SIZE, rng=None):
    rng = rng or random
    by_f = defaultdict(set)
    for (f, y), idxs in buckets.items():
        if idxs:
            by_f[f].add(y)
    eligible = [f for f, ys in by_f.items() if len(ys) >= 2] or list(by_f.keys())
    if not eligible:
        return []
    k = min(k_findings, len(eligible))
    batch = []
    for f in rng.sample(eligible, k):
        for y in range(3):
            pool = buckets.get((f, y), [])
            if not pool:
                continue
            take = min(n_per_class, len(pool))
            batch.extend(rng.sample(pool, take) if len(pool) >= take else list(pool))
    if len(batch) > max_bs:
        batch = rng.sample(batch, max_bs)
    rng.shuffle(batch)
    return batch

_bs = [len(sample_contrastive_batch(TRAIN_BUCKETS)) for _ in range(20)]
APPROX_BS = max(int(sum(_bs) / len(_bs)), 32)
STEPS = max(1, math.ceil(len(DATA['train']['y']) / APPROX_BS))
print('steps', STEPS, 'approx_bs', APPROX_BS)
