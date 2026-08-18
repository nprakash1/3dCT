import glob
import sys
sys.path.insert(0, '/content/3dCT/scripts')
from huggingface_hub import hf_hub_download
from ctclip_utils import CTCLIPEmbedder, REPO_ID, CTCLIP_WEIGHTS_HF

POOLED = {}
for fp in glob.glob(f'{IMG_DIR}/*.pt'):
    key = os.path.basename(fp).replace('.pt', '')
    POOLED[key] = torch.load(fp, map_location='cpu').float().view(-1)
print('pooled volumes', len(POOLED))
assert len(POOLED) > 100

wp = f'{WEIGHTS}/{os.path.basename(CTCLIP_WEIGHTS_HF)}'
if not os.path.exists(wp):
    wp = hf_hub_download(REPO_ID, CTCLIP_WEIGHTS_HF, repo_type='dataset', local_dir=WEIGHTS)
emb = CTCLIPEmbedder(wp)
print('text tower on', emb.device)

def vkey(v):
    return v.replace('.nii.gz', '').replace('.nii', '')

def volume_domain(v):
    k = vkey(v).lower()
    if k.startswith('train_'): return 'hub_train'
    if k.startswith('valid_'): return 'hub_valid'
    return 'unknown'

def pair_domain(pv, cv):
    a, b = volume_domain(pv), volume_domain(cv)
    return a if a == b else 'cross_domain'

def dev_partition(patient):
    raw = f'{SPLIT_SEED}|{patient}'.encode()
    u = int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') / 2**64
    return 'tune' if u < TUNE_FRAC else 'train'

recs = {}
for line in open(LAB, encoding='utf-8'):
    if not line.strip(): continue
    x = json.loads(line)
    recs[(x['patient'], x['prior_volume'], x['curr_volume'])] = x

dyn_of = {}
try:
    for line in open(LAB_DS, encoding='utf-8'):
        if not line.strip(): continue
        x = json.loads(line)
        ds = x.get('dynamic_sentences') or []
        if isinstance(ds, list):
            ds = ' '.join(s for s in ds if isinstance(s, str))
        dyn_of[(x['patient'], x['prior_volume'], x['curr_volume'])] = (ds or '').strip()
    print('dynamic for', len(dyn_of), 'pairs')
except FileNotFoundError:
    print('WARN no dynamic file')

pair_ids = {}
examples = {'train': [], 'tune': [], 'test': []}
skipped = Counter()
missing_hub_valid = []
for key, rec in recs.items():
    patient, pv, cv = key
    domain = pair_domain(pv, cv)
    if domain == 'hub_train':
        sp = dev_partition(patient)
    elif domain == 'hub_valid':
        sp = 'test'
    else:
        skipped[domain] += 1
        continue
    if not rec.get('parse_ok'):
        skipped['no_label'] += 1
        continue
    if vkey(pv) not in POOLED or vkey(cv) not in POOLED:
        skipped['no_feat'] += 1
        if sp == 'test':
            missing_hub_valid.append(key)
        continue
    for fd in rec.get('findings', []):
        if fd.get('tier') != 'explicit':
            continue
        d, f = fd.get('direction'), fd.get('finding')
        if d not in C2I or not f:
            continue
        pid = pair_ids.setdefault(key, len(pair_ids))
        examples[sp].append(dict(
            vp=vkey(pv), vc=vkey(cv), patient=patient, finding=f,
            hub_domain=domain, y=C2I[d], pid=pid,
            evidence=fd.get('evidence', '') or '',
            dynamic=dyn_of.get(key, ''),
        ))

assert examples['train'] and examples['tune'] and examples['test']
if REQUIRE_COMPLETE_HUB_VALID_FEATURES:
    assert not missing_hub_valid

FINDINGS = [
    'Medical material', 'Arterial wall calcification', 'Cardiomegaly',
    'Pericardial effusion', 'Coronary artery wall calcification', 'Hiatal hernia',
    'Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule', 'Lung opacity',
    'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern',
    'Peribronchial thickening', 'Consolidation', 'Bronchiectasis',
    'Interlobular septal thickening',
]
F2I = {f: i for i, f in enumerate(FINDINGS)}
for sp in examples:
    for e in examples[sp]:
        e['fid'] = F2I[e['finding']]
for sp in ['train', 'tune', 'test']:
    cc = Counter(e['y'] for e in examples[sp])
    print(f"{sp:5}: {len(examples[sp]):5}  w/s/i={cc[0]}/{cc[1]}/{cc[2]}")
print('skipped', dict(skipped))
