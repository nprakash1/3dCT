import time
from itertools import product

SKIP_ALL_OFF = True
QUICK_ABLATION = False
ABLATION_EPOCHS = EPOCHS
ABLATION_PATIENCE = PATIENCE
ABLATION_SEED = 0
RUN_TEST_EACH = True
N_EVAL_PASSES = 3

ABL_DIR = f'{DRIVE}/ctclip_cache/ablations_holdout_eval'
os.makedirs(ABL_DIR, exist_ok=True)
ts = time.strftime('%Y%m%d_%H%M%S')
_SYM = globals().get('CONTRASTIVE_SYMMETRIC', False)

def _set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def _make_model():
    return DifferenceTransformer(
        n_findings=len(FINDINGS), d_model=D_MODEL, antisym=ANTISYM,
        magnitude=True, finding_conditioning=FINDING_CONDITIONING,
        finding_as_4th_token=FINDING_AS_4TH_TOKEN,
        use_learned_finding_emb=USE_LEARNED_FINDING_EMB,
        tau_con_init=TAU_CON_INIT, learnable_tau_con=LEARNABLE_TAU_CON,
    ).to(DEVICE)

@torch.no_grad()
def evaluate_holdout(model, split, base_seed, n_passes=N_EVAL_PASSES):
    """Eval with holdout sentences; average macro-F1 over n_passes of resampling."""
    model.eval()
    D = DATA[split]
    macros, pers = [], []
    all_last_y, all_last_p = None, None
    for p in range(n_passes):
        rng = random.Random(base_seed + 10007 * p + 17)
        ys, ps = [], []
        bs = 256
        for i in range(0, len(D['y']), bs):
            sl = slice(i, i + bs)
            vp = D['vp'][sl].to(DEVICE)
            vc = D['vc'][sl].to(DEVICE)
            y = D['y'][sl].to(DEVICE)
            fid = D['fid'][sl].to(DEVICE)
            fn = D['fn'][i:i+bs]
            vd, _ = model(vp, vc, fid)
            lg = logits_from_holdout(vd, fn, model.logit_scale, rng)
            ys += y.cpu().tolist()
            ps += lg.argmax(1).cpu().tolist()
        y_true, y_pred = np.array(ys), np.array(ps)
        macros.append(f1_score(y_true, y_pred, labels=[0,1,2], average='macro', zero_division=0))
        pers.append(f1_score(y_true, y_pred, labels=[0,1,2], average=None, zero_division=0))
        all_last_y, all_last_p = y_true, y_pred
    per = np.mean(np.stack(pers, 0), 0)
    return {
        'macro_f1': float(np.mean(macros)),
        'macro_f1_std': float(np.std(macros)),
        'f1_worsened': float(per[0]),
        'f1_stable': float(per[1]),
        'f1_improved': float(per[2]),
    }

def _train_one(use_ce, use_mag, use_supcon, seed):
    _set_seed(seed)
    model = _make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ce_fn = nn.CrossEntropyLoss(weight=W_cls.to(DEVICE))
    best, best_state, best_ep, bad = -1.0, None, 0, 0
    n_train = len(DATA['train']['y'])

    for ep in range(1, ABLATION_EPOCHS + 1):
        model.train()
        tot, n = 0.0, 0
        for _ in range(STEPS):
            if use_supcon:
                idxs = sample_contrastive_batch(TRAIN_BUCKETS)
                if len(idxs) < 4:
                    idxs = random.sample(range(n_train), min(MAX_BATCH_SIZE, n_train))
            else:
                idxs = random.sample(range(n_train), min(MAX_BATCH_SIZE, n_train))
            idxs_t = torch.tensor(idxs, dtype=torch.long)
            vp = DATA['train']['vp'][idxs_t].to(DEVICE)
            vc = DATA['train']['vc'][idxs_t].to(DEVICE)
            pr = DATA['train']['pr'][idxs_t].to(DEVICE)
            y = DATA['train']['y'][idxs_t].to(DEVICE)
            fid = DATA['train']['fid'][idxs_t].to(DEVICE)
            tt = DATA['train']['tt'][idxs_t].to(DEVICE)

            vd, mag = model(vp, vc, fid)
            # TRAIN CE uses fixed PROTO_TRAIN (in pr)
            lg = logits_from_proto(vd, pr, model.logit_scale)

            loss = vd.new_zeros(())
            if use_ce:
                loss = loss + LAMBDA_CE * ce_fn(lg, y)
            if use_mag and mag is not None:
                is_change = (y != C2I['stable']).float()
                loss = loss + LAMBDA_MAG * F.binary_cross_entropy_with_logits(mag, is_change)
            if use_supcon:
                l_con, _ = masked_crossmodal_supcon_loss(
                    vd, tt, y, fid, model.tau_con(), symmetric=_SYM)
                loss = loss + LAMBDA_CON * l_con
            if not torch.isfinite(loss) or (
                loss.item() == 0.0 and not (use_ce or use_mag or use_supcon)
            ):
                loss = vd.sum() * 0.0

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item()) * len(idxs)
            n += len(idxs)

        # early-stop on HOLDOUT eval (not train prototypes)
        tune = evaluate_holdout(model, 'tune', base_seed=seed + ep)
        vf1 = tune['macro_f1']
        if vf1 > best:
            best, best_ep = vf1, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if ep % 10 == 0 or ep == 1:
            print('  ep %3d  loss %.3f  tuneHoldoutF1 %.3f+/-%.3f  best %.3f' % (
                ep, tot/max(n,1), vf1, tune['macro_f1_std'], best))
        if bad >= ABLATION_PATIENCE:
            print('  early stop @ ep %d  best %.3f' % (ep, best))
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, best_state, best, best_ep, ep
