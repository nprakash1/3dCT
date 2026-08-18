# ---- TRAIN templates (CE only; must not overlap eval bank strings) ----
TRAIN_TEMPLATES = {
    'worsened': [
        '{f} has increased compared to the prior study',
        '{f} has worsened since the previous exam',
        'interval enlargement of {f}',
        'new {f}',
        'increased {f}',
    ],
    'stable': [
        '{f} is unchanged compared to the prior study',
        'stable {f} with no interval change',
        'no significant change in {f}',
        '{f} appears similar to prior',
    ],
    'improved': [
        '{f} has decreased compared to the prior study',
        '{f} has improved since the previous exam',
        'interval decrease of {f}',
        '{f} has resolved',
        'decreased {f}',
    ],
}

# Holdout eval bank: 5 MedGemma-style paraphrases/class, disjoint from TRAIN_TEMPLATES
EVAL_SENTENCE_STEMS = {
    'worsened': [
        'There is interval progression of the {f}.',
        'The {f} is more conspicuous than on the prior examination.',
        'Findings suggest worsening of the {f} relative to prior.',
        'Compared with the previous study, the {f} has enlarged.',
        'Progressive {f} is noted since the last CT.',
    ],
    'stable': [
        'No appreciable interval change in the {f}.',
        'The {f} is similar in appearance to the prior exam.',
        'Chronic-appearing {f} without definite interval change.',
        'The {f} remains essentially unchanged from prior.',
        'Stability of the {f} is demonstrated compared with prior imaging.',
    ],
    'improved': [
        'There is interval improvement of the {f}.',
        'The {f} has decreased in extent since the prior study.',
        'Partial resolution of the {f} compared with prior.',
        'Findings indicate improving {f} relative to the previous CT.',
        'The {f} is less pronounced than on the prior examination.',
    ],
}

STABLE_SYNTH_TEMPLATES = [
    'The {f} is unchanged from the prior examination.',
    'There has been no significant interval change in the {f}.',
    'The {f} remains stable compared with the prior examination.',
    'The {f} is stable compared with the previous study.',
]

def l2np(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

def embed_mean(texts):
    texts = [t for t in texts if t and str(t).strip()]
    if not texts:
        return None
    vecs = []
    for i in range(0, len(texts), 64):
        vecs.append(emb.embed_texts(texts[i:i+64], normalize=True).numpy())
    return l2np(np.concatenate(vecs, 0).mean(0))

def embed_list(texts, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        out.append(emb.embed_texts(texts[i:i+bs], normalize=True).float().cpu())
    return torch.cat(out, 0) if out else torch.zeros(0, 512)

PROTO_TRAIN = {}
for f in FINDINGS:
    rows = []
    for c in CLASSES:
        prompts = [t.format(f=f.lower()) for t in TRAIN_TEMPLATES[c]]
        rows.append(embed_mean(prompts))
    PROTO_TRAIN[f] = torch.tensor(np.stack(rows)).float()
print('PROTO_TRAIN', len(PROTO_TRAIN), PROTO_TRAIN[FINDINGS[0]].shape)

train_string_set = set()
for c, ts_ in TRAIN_TEMPLATES.items():
    for t in ts_:
        for f in FINDINGS:
            train_string_set.add(t.format(f=f.lower()).strip().lower())

EVAL_BANK = {}
EVAL_BANK_EMB = {}
overlap = 0
for f in FINDINGS:
    per_c_txt, per_c_emb = [], []
    for c in CLASSES:
        stems = EVAL_SENTENCE_STEMS[c]
        assert len(stems) >= N_EVAL_SENTENCES
        sents = [stems[i].format(f=f.lower()) for i in range(N_EVAL_SENTENCES)]
        for s in sents:
            if s.strip().lower() in train_string_set:
                overlap += 1
        per_c_txt.append(sents)
        per_c_emb.append(embed_list(sents))
    EVAL_BANK[f] = per_c_txt
    EVAL_BANK_EMB[f] = torch.stack(per_c_emb, 0)
assert overlap == 0, 'train/eval sentence overlap=%d' % overlap
torch.save(dict(bank=EVAL_BANK, emb=EVAL_BANK_EMB), EVAL_BANK_PT)
print('EVAL_BANK ok; overlap=', overlap, 'emb', EVAL_BANK_EMB[FINDINGS[0]].shape)

# Temporal text for SupCon
rng = random.Random(STABLE_TEXT_SEED)

def _real(e):
    return (e.get("evidence") or "").strip() or (e.get("dynamic") or "").strip()

for sp, exs in examples.items():
    for e in exs:
        real = _real(e)
        if real:
            e["temporal_text"] = real
            e["temporal_src"] = "real"
        elif e["y"] == C2I["stable"]:
            tid = rng.randrange(len(STABLE_SYNTH_TEMPLATES))
            e["temporal_text"] = STABLE_SYNTH_TEMPLATES[tid].format(f=e["finding"].lower())
            e["temporal_src"] = "synthetic"
        else:
            c = I2C[e["y"]]
            e["temporal_text"] = TRAIN_TEMPLATES[c][0].format(f=e["finding"].lower())
            e["temporal_src"] = "template_fallback"

all_t, meta = [], []
for sp in ["train", "tune", "test"]:
    for i, e in enumerate(examples[sp]):
        all_t.append(e["temporal_text"])
        meta.append((sp, i))
uniq, uix = [], {}
for t in all_t:
    if t not in uix:
        uix[t] = len(uniq)
        uniq.append(t)
print("encoding", len(uniq), "unique temporal texts...")
U = embed_list(uniq)
by = {sp: [None]*len(examples[sp]) for sp in examples}
for (sp, i), t in zip(meta, all_t):
    by[sp][i] = U[uix[t]]
TEMPORAL_TEXT_EMB = {sp: torch.stack(by[sp], 0) for sp in by}

def tensorize(exs, tt):
    return {
        "vp": torch.stack([POOLED[e["vp"]] for e in exs]),
        "vc": torch.stack([POOLED[e["vc"]] for e in exs]),
        "pr": torch.stack([PROTO_TRAIN[e["finding"]] for e in exs]),
        "y": torch.tensor([e["y"] for e in exs], dtype=torch.long),
        "fid": torch.tensor([e["fid"] for e in exs], dtype=torch.long),
        "tt": tt.float().clone(),
        "fn": [e["finding"] for e in exs],
    }

DATA = {sp: tensorize(examples[sp], TEMPORAL_TEXT_EMB[sp]) for sp in ["train", "tune", "test"]}
cnt = Counter(DATA["train"]["y"].tolist()); tot = sum(cnt.values())
W_cls = torch.tensor([tot / (3 * max(cnt[i], 1)) for i in range(3)], dtype=torch.float32)
print("weights", [round(x,3) for x in W_cls.tolist()])
for sp in DATA:
    print(sp, {k: (tuple(v.shape) if torch.is_tensor(v) else len(v)) for k,v in DATA[sp].items()})
