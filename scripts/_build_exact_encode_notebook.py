#!/usr/bin/env python3
"""Build the full-scale CT-RATE -> EXACT Y-Mamba Colab encoder notebook."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "exact_encode_ctrate_cache_colab.ipynb"


def lines(text: str) -> list[str]:
    text = text.strip("\n")
    parts = text.splitlines()
    return [line + "\n" for line in parts[:-1]] + ([parts[-1]] if parts else [])


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


cells = [
    markdown(r"""
# CT-RATE → official EXACT Y-Mamba image encodings → Google Drive

This notebook creates **EXACT image features at the same scale as the existing
CT-CLIP cache**. It does **not** use CT-CLIP to encode images. The CT-CLIP cache is
only read to obtain the exact set of already-processed `train_*` and `valid_*`
volume names.

For each target volume it:

1. downloads one gated CT-RATE NIfTI from Hugging Face;
2. preprocesses it into a configurable 3D tensor;
3. runs the **official `JasonW375/EXACT` `YMamba` model** loaded from
   `ymamba_pretrain_best.pth`;
4. extracts `encoder5(vit(x)[3])`, globally pools it to an **official-encoder-derived
   768-d image vector**, and saves it immediately to Drive;
5. deletes the raw volume and continues.

The loop is resumable and processes validation before training.

## Does EXACT have image and text encoders?

**EXACT has a 3D image encoder (Y-Mamba), not a CLIP-style paired text encoder.**
Reports supply weak disease labels during pretraining, but EXACT does not produce text
vectors in the same space as its image features. The saved vectors here are image-only.

## Reproducibility caveat

The public EXACT README references `EXACT_Pretrain/data_preprocessed/*.py`, but those
raw-NIfTI preprocessing scripts are absent from the current public repository. This
notebook therefore uses the **exact public architecture and exact checkpoint**, while
making the reconstructed NIfTI preprocessing explicit and recording it in every output.
Do not describe preprocessing as bit-exact to the paper unless the authors provide the
missing scripts.

**Required Colab runtime:** before running any cell, open **Runtime → Change runtime
type**, select an NVIDIA GPU, and set **Runtime Version → 2025.07**. That past runtime
provides Python 3.11 and PyTorch 2.6, for which matching prebuilt Mamba CUDA wheels exist.
A100/L4 (24 GB+) is recommended.
"""),
    code(r"""
# GPU/runtime check. PyTorch 2.4.1 has no Python 3.13 wheel, so it cannot be installed
# into the current default Colab runtime. Use Colab's 2025.07 past runtime instead.
!nvidia-smi || echo 'NO GPU — Runtime > Change runtime type > GPU'
import platform, sys, torch
print('python:', platform.python_version())
print('torch:', torch.__version__, '| CUDA:', torch.version.cuda,
      '| available:', torch.cuda.is_available())
assert sys.version_info[:2] == (3, 11) and torch.__version__.split('+')[0] == '2.6.0', (
    f'Unsupported Colab image: Python {platform.python_version()}, torch {torch.__version__}.\n'
    'Do not try to pip-install torch 2.4.1 under Python 3.13; no such wheel exists.\n'
    'Select Runtime > Change runtime type > Runtime Version > 2025.07, choose a GPU, '
    'save, and then use Runtime > Run all.'
)
assert torch.cuda.is_available(), 'A CUDA GPU is required for full-scale EXACT encoding.'
"""),
    markdown(r"""
## 1. Install the official EXACT foundation-model environment

EXACT documents PyTorch 2.4.1 + CUDA 12.1, but that PyTorch release has no Python 3.13
wheel. The current default Colab runtime therefore cannot be repaired by pip-downgrading
PyTorch. This notebook uses Colab's **2025.07 past runtime** (Python 3.11 / PyTorch 2.6)
and downloads matching prebuilt CUDA wheels directly. It never compiles Mamba from source.

Colab keeps past runtime versions for one year. If `2025.07` is no longer offered, stop:
the wheel matrix below must be updated and smoke-tested for a newer runtime rather than
silently compiling or mixing binary-incompatible packages.
"""),
    code(r"""
# Deliberately do not replace Colab's core PyTorch package. Compiled extension wheels
# below are selected for the 2025.07 runtime's Python and PyTorch versions.
import platform, sys, torch
assert sys.version_info[:2] == (3, 11), platform.python_version()
assert torch.__version__.split('+')[0] == '2.6.0', torch.__version__
print('Supported base runtime:', platform.python_version(), torch.__version__, torch.version.cuda)
"""),
    code(r"""
# Install binary-compatible prebuilt Mamba wheels (no local CUDA compilation).
%cd /content
![ -d EXACT/.git ] || git clone --depth 1 https://github.com/JasonW375/EXACT.git EXACT

import os, subprocess, sys, torch
from pathlib import Path

assert torch.__version__.split('+')[0] == '2.6.0', 'Select Colab Runtime Version 2025.07.'
assert sys.version_info[:2] == (3, 11), 'Select Colab Runtime Version 2025.07.'

subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-q',
    'ninja', 'packaging', 'einops==0.8.0', 'transformers',
    'huggingface_hub', 'nibabel==5.3.2', 'scipy', 'tqdm', 'gdown', 'monai==1.3.0'
], check=True)

py_tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
abi = 'TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE'
platform_tag = 'linux_x86_64'
causal_name = f'causal_conv1d-1.5.0.post8+cu12torch2.6cxx11abi{abi}-{py_tag}-{py_tag}-{platform_tag}.whl'
mamba_name = f'mamba_ssm-2.2.4+cu12torch2.6cxx11abi{abi}-{py_tag}-{py_tag}-{platform_tag}.whl'
causal_url = 'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/' + causal_name.replace('+', '%2B')
mamba_url = 'https://github.com/state-spaces/mamba/releases/download/v2.2.4/' + mamba_name.replace('+', '%2B')

print('Python tag:', py_tag, '| CXX11 ABI:', abi)
print('Installing prebuilt causal-conv1d wheel:', causal_name)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', causal_url], check=True)
print('Installing prebuilt mamba-ssm wheel:', mamba_name)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', mamba_url], check=True)

import causal_conv1d, mamba_ssm
print('causal_conv1d:', causal_conv1d.__version__)
print('mamba_ssm:', mamba_ssm.__version__)

EXACT_ROOT = '/content/EXACT'
EXACT_PRETRAIN = os.path.join(EXACT_ROOT, 'EXACT_Pretrain')
if EXACT_PRETRAIN not in sys.path:
    sys.path.insert(0, EXACT_PRETRAIN)
from models.ymamba.ymamba import YMamba
print('Imported official YMamba from:', sys.modules[YMamba.__module__].__file__)
"""),
    markdown(r"""
## 2. Mount Drive and configure paths

The target inventory is `MyDrive/3dCT/ctclip_cache/img/*.pt`. EXACT results are kept
separately under `MyDrive/3dCT/exact_cache/pooled/`.

Place the official checkpoint at:

`MyDrive/3dCT/exact_weights/ymamba_pretrain_best.pth`

Official checkpoint folder: https://drive.google.com/drive/folders/1i2J6XUqTm2G8m3-OlbH7Wt00aBxIpClf
"""),
    code(r"""
from google.colab import drive
drive.mount('/content/drive')

import glob, json, os, shutil, time
from pathlib import Path

DRIVE_3DCT = Path('/content/drive/MyDrive/3dCT')
CTCLIP_IMG_DIR = DRIVE_3DCT / 'ctclip_cache' / 'img'
EXACT_CACHE = DRIVE_3DCT / 'exact_cache'
EXACT_POOLED_DIR = EXACT_CACHE / 'pooled'
EXACT_SPATIAL_DIR = EXACT_CACHE / 'stage4_spatial'
EXACT_MANIFEST_DIR = EXACT_CACHE / 'manifests'
CHECKPOINT_PATH = DRIVE_3DCT / 'exact_weights' / 'ymamba_pretrain_best.pth'
TMP = Path('/content/_exact_volume_tmp')

for p in [EXACT_POOLED_DIR, EXACT_MANIFEST_DIR, CHECKPOINT_PATH.parent, TMP]:
    p.mkdir(parents=True, exist_ok=True)

# Full-scale defaults. CT-CLIP determines the target set; there is no volume limit.
PROCESS_VALID_FIRST = True
SAVE_STAGE4_SPATIAL = False  # much larger cache; pooled 768-d is always saved
USE_AMP = True

# Reconstructed raw-NIfTI preprocessing. The missing official scripts prevent a
# bit-exact claim. Keep these fixed across all volumes in an experiment.
TARGET_DHW = (64, 64, 64)  # matches official config block_size=64; 16x downsampling
HU_MIN, HU_MAX = -1000.0, 1000.0
NORMALIZE_TO = 'zero_one'
PREPROCESSING_ID = 'reconstructed_ctrate_canonical_resize64_hu-1000_1000_zero_one_v1'

print('CT-CLIP inventory:', CTCLIP_IMG_DIR)
print('EXACT output:', EXACT_CACHE)
print('Checkpoint:', CHECKPOINT_PATH)
assert CTCLIP_IMG_DIR.is_dir(), f'Missing CT-CLIP cache: {CTCLIP_IMG_DIR}'
"""),
    markdown(r"""
## 3. Official checkpoint

The authors publish the checkpoint inside a shared ~6.3 GB Drive folder rather than a
direct model URL. Download only `01_pretrain/ymamba_pretrain_best.pth` from the official
folder and put it at `CHECKPOINT_PATH` above. The notebook refuses to run with a missing
or suspiciously small file.
"""),
    code(r"""
MIN_CHECKPOINT_BYTES = 800 * 1024 * 1024
if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f'Official EXACT checkpoint not found at {CHECKPOINT_PATH}.\n'
        'Download 01_pretrain/ymamba_pretrain_best.pth from:\n'
        'https://drive.google.com/drive/folders/1i2J6XUqTm2G8m3-OlbH7Wt00aBxIpClf\n'
        'and copy it to the path above.'
    )
size_gb = CHECKPOINT_PATH.stat().st_size / 1024**3
assert CHECKPOINT_PATH.stat().st_size >= MIN_CHECKPOINT_BYTES, (
    f'Checkpoint is unexpectedly small ({size_gb:.2f} GiB): {CHECKPOINT_PATH}')
print(f'Official checkpoint present: {size_gb:.2f} GiB')
"""),
    markdown(r"""
## 4. Enumerate exactly the volumes already cached by CT-CLIP

Only basenames beginning with `train_` or `valid_` are selected. Existing valid EXACT
outputs are skipped, so the operation is safely resumable.
"""),
    code(r"""
def cache_key_to_volume(path):
    return Path(path).stem + '.nii.gz'

ctclip_files = sorted(CTCLIP_IMG_DIR.glob('*.pt'))
train_targets = [cache_key_to_volume(p) for p in ctclip_files if p.stem.startswith('train_')]
valid_targets = [cache_key_to_volume(p) for p in ctclip_files if p.stem.startswith('valid_')]
unknown = [p.name for p in ctclip_files
           if not (p.stem.startswith('train_') or p.stem.startswith('valid_'))]

assert train_targets, 'No train_* tensors found in the CT-CLIP cache.'
assert valid_targets, 'No valid_* tensors found in the CT-CLIP cache.'
assert not (set(train_targets) & set(valid_targets)), 'Train/valid target overlap.'

targets_payload = {
    'created_unix': time.time(),
    'source': str(CTCLIP_IMG_DIR),
    'n_train': len(train_targets),
    'n_valid': len(valid_targets),
    'unknown_cache_files': unknown,
    'train': train_targets,
    'valid': valid_targets,
}
with open(EXACT_MANIFEST_DIR / 'target_volumes.json', 'w') as f:
    json.dump(targets_payload, f, indent=2)

print('CT-CLIP-backed targets')
print('  train:', len(train_targets))
print('  valid:', len(valid_targets))
print('  unknown:', len(unknown))
print('  total:', len(train_targets) + len(valid_targets))
"""),
    markdown(r"""
## 5. Authenticate to gated CT-RATE and define disk-safe downloads

Accept the CT-RATE terms on Hugging Face first. Each raw NIfTI is downloaded into a
temporary directory that is completely removed after the encoding attempt, including
Hugging Face's local metadata/cache files.
"""),
    code(r"""
from huggingface_hub import login, hf_hub_download

login()  # paste a Hugging Face READ token with CT-RATE access
HF_TOKEN = os.environ.get('HF_TOKEN') or True
CTRATE_REPO = 'ibrahimhamamci/CT-RATE'

def remote_candidates(volume):
    base = volume.removesuffix('.nii.gz').removesuffix('.nii')
    split, patient_id, scan = base.split('_')[:3]
    patient = f'{split}_{patient_id}'
    scan_folder = f'{split}_{patient_id}_{scan}'
    folders = [f'{split}_fixed', split] if split in {'train', 'valid'} else [split]
    return [f'dataset/{folder}/{patient}/{scan_folder}/{volume}' for folder in folders]

def reset_tmp():
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)

def download_volume(volume):
    reset_tmp()
    errors = []
    for remote_path in remote_candidates(volume):
        try:
            return hf_hub_download(
                CTRATE_REPO, remote_path, repo_type='dataset', token=HF_TOKEN,
                local_dir=str(TMP), force_download=False)
        except Exception as exc:
            errors.append(f'{remote_path}: {type(exc).__name__}')
    print('DOWNLOAD FAIL', volume, '|', ' ; '.join(errors))
    return None

print('Example candidates:', remote_candidates(valid_targets[0]))
"""),
    markdown(r"""
## 6. Reconstructed NIfTI preprocessing

The operation is deterministic:

- canonicalize orientation with nibabel;
- read HU as float32;
- clip to `[HU_MIN, HU_MAX]`;
- resize the complete canonical volume to `TARGET_DHW` with trilinear interpolation;
- scale to `[0,1]`;
- return `[1,1,D,H,W]`.

This is deliberately isolated in one function so it can be replaced if the authors
release the missing official scripts. Changing it after starting a cache requires a new
`PREPROCESSING_ID` and output directory.
"""),
    code(r"""
import nibabel as nib
import numpy as np
import torch.nn.functional as F

def preprocess_exact_nifti(path):
    nii = nib.load(path)
    original_shape = tuple(int(x) for x in nii.shape[:3])
    original_axcodes = tuple(str(x) for x in nib.aff2axcodes(nii.affine))
    canonical = nib.as_closest_canonical(nii)
    canonical_axcodes = tuple(str(x) for x in nib.aff2axcodes(canonical.affine))
    xyz = canonical.get_fdata(dtype=np.float32)
    if xyz.ndim != 3:
        raise ValueError(f'Expected 3D NIfTI, got {xyz.shape}')
    xyz = np.nan_to_num(xyz, nan=HU_MIN, posinf=HU_MAX, neginf=HU_MIN)
    xyz = np.clip(xyz, HU_MIN, HU_MAX)

    # nibabel array is X,Y,Z; model tensor is D,H,W = Z,Y,X.
    dhw = np.ascontiguousarray(xyz.transpose(2, 1, 0))
    x = torch.from_numpy(dhw).unsqueeze(0).unsqueeze(0)
    x = F.interpolate(x, size=TARGET_DHW, mode='trilinear', align_corners=False)
    x = (x - HU_MIN) / (HU_MAX - HU_MIN)
    x = x.contiguous().float()
    if not torch.isfinite(x).all():
        raise ValueError('Non-finite values after preprocessing.')
    metadata = {
        'original_shape_xyz': original_shape,
        'original_axcodes': original_axcodes,
        'canonical_shape_xyz': tuple(int(v) for v in canonical.shape[:3]),
        'canonical_axcodes': canonical_axcodes,
        'model_shape_bcdhw': tuple(int(v) for v in x.shape),
        'target_dhw': TARGET_DHW,
        'hu_clip': (HU_MIN, HU_MAX),
        'normalization': NORMALIZE_TO,
        'preprocessing_id': PREPROCESSING_ID,
    }
    return x, metadata
"""),
    markdown(r"""
## 7. Build official Y-Mamba, load weights, and expose encoder features

The model is the official `YMamba` defaults: feature channels `48/96/192/384` and
`encoder5: 384→768`. The checkpoint must cover the model exactly after removing an
optional `module.` prefix. No randomly initialized encoder parameters are accepted.
"""),
    code(r"""
import hashlib, gc

DEVICE = torch.device('cuda')
model = YMamba(
    in_chans=1, num_classes=7, num_abnormal_classes=18,
    depths=[2, 2, 2, 2], feat_size=[48, 96, 192, 384], hidden_size=768,
).to(DEVICE)

checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state = checkpoint['model_state_dict']
elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state = checkpoint['state_dict']
elif isinstance(checkpoint, dict) and all(torch.is_tensor(v) for v in checkpoint.values()):
    state = checkpoint
else:
    raise ValueError(f'Unrecognized checkpoint structure: {type(checkpoint)}')
state = {k.removeprefix('module.'): v for k, v in state.items()}
missing, unexpected = model.load_state_dict(state, strict=False)
print('checkpoint keys:', len(state), '| missing:', len(missing), '| unexpected:', len(unexpected))
if missing: print('missing[:20]:', missing[:20])
if unexpected: print('unexpected[:20]:', unexpected[:20])
assert not missing and not unexpected, 'Official checkpoint does not exactly cover official YMamba.'
del checkpoint, state

for p in model.parameters():
    p.requires_grad_(False)
model.eval()

def sha256_file(path, chunk=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            block = f.read(chunk)
            if not block: break
            digest.update(block)
    return digest.hexdigest()

CHECKPOINT_SHA256 = sha256_file(CHECKPOINT_PATH)
print('checkpoint SHA256:', CHECKPOINT_SHA256)

@torch.inference_mode()
def encode_exact(x_cpu):
    x = x_cpu.to(DEVICE, non_blocking=True)
    amp_dtype = torch.float16
    with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=USE_AMP):
        stages = model.vit(x)
        hidden = model.encoder5(stages[3])
        pooled = hidden.mean(dim=(2, 3, 4))
    expected_channels = [48, 96, 192, 384]
    assert len(stages) == 4
    assert [int(t.shape[1]) for t in stages] == expected_channels
    assert hidden.ndim == 5 and hidden.shape[1] == 768
    assert pooled.shape == (x.shape[0], 768)
    assert torch.isfinite(pooled).all()
    result = {
        'pooled_encoder5': pooled.float().cpu(),
        'stage_shapes': [tuple(int(v) for v in t.shape) for t in stages],
        'encoder5_shape': tuple(int(v) for v in hidden.shape),
    }
    if SAVE_STAGE4_SPATIAL:
        result['stage4_spatial'] = stages[3].half().cpu()
    return result

print('Official EXACT Y-Mamba is frozen and ready.')
"""),
    markdown(r"""
## 8. Resumable encoder loop

Outputs are written atomically. Existing files are skipped only if they contain a finite
768-d vector with the same checkpoint hash and preprocessing ID. Failures are appended to
JSONL and retried on the next run.
"""),
    code(r"""
from tqdm.auto import tqdm

RESULTS_JSONL = EXACT_MANIFEST_DIR / 'encoding_results.jsonl'
FAILURES_JSONL = EXACT_MANIFEST_DIR / 'failures.jsonl'

def output_path(volume):
    key = volume.removesuffix('.nii.gz').removesuffix('.nii')
    return EXACT_POOLED_DIR / f'{key}.pt'

def valid_existing(path):
    if not path.exists(): return False
    try:
        payload = torch.load(path, map_location='cpu')
        vector = payload['pooled_encoder5']
        return (
            tuple(vector.shape) == (768,) and torch.isfinite(vector).all().item()
            and payload.get('checkpoint_sha256') == CHECKPOINT_SHA256
            and payload.get('preprocessing', {}).get('preprocessing_id') == PREPROCESSING_ID
        )
    except Exception:
        return False

def append_jsonl(path, record):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, default=str) + '\n')

def atomic_torch_save(payload, destination):
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, destination)

def encode_targets(volumes, split):
    progress_path = EXACT_CACHE / f'progress_{split}.json'
    new = skipped = missing = failed = 0
    for i, volume in enumerate(tqdm(volumes, desc=f'EXACT {split}')):
        out = output_path(volume)
        if valid_existing(out):
            skipped += 1
            continue
        if out.exists():
            out.unlink()  # corrupt or incompatible prior result

        downloaded = download_volume(volume)
        if downloaded is None:
            missing += 1
            append_jsonl(FAILURES_JSONL, {
                'time': time.time(), 'split': split, 'volume': volume,
                'kind': 'download_missing'})
            reset_tmp()
            continue

        started = time.time()
        try:
            x, prep = preprocess_exact_nifti(downloaded)
            encoded = encode_exact(x)
            pooled = encoded['pooled_encoder5'].squeeze(0)
            payload = {
                'format_version': 1,
                'encoder': 'official_EXACT_YMamba_encoder5_global_average_pool',
                'volume': volume,
                'split': split,
                'pooled_encoder5': pooled.half().clone(),
                'stage_shapes': encoded['stage_shapes'],
                'encoder5_shape': encoded['encoder5_shape'],
                'preprocessing': prep,
                'checkpoint_name': CHECKPOINT_PATH.name,
                'checkpoint_sha256': CHECKPOINT_SHA256,
                'exact_git_remote': 'https://github.com/JasonW375/EXACT',
                'seconds': time.time() - started,
            }
            if 'stage4_spatial' in encoded:
                payload['stage4_spatial'] = encoded['stage4_spatial'].squeeze(0)
            atomic_torch_save(payload, out)
            assert valid_existing(out), f'Post-save validation failed: {out}'
            new += 1
            append_jsonl(RESULTS_JSONL, {
                'time': time.time(), 'split': split, 'volume': volume,
                'output': str(out), 'seconds': payload['seconds'],
                'stage_shapes': payload['stage_shapes'],
                'encoder5_shape': payload['encoder5_shape']})
        except Exception as exc:
            failed += 1
            print('ENCODE FAIL', volume, repr(exc))
            append_jsonl(FAILURES_JSONL, {
                'time': time.time(), 'split': split, 'volume': volume,
                'kind': 'encode_failure', 'error_type': type(exc).__name__,
                'error': str(exc)})
        finally:
            reset_tmp()
            gc.collect()
            torch.cuda.empty_cache()

        if (i + 1) % 10 == 0:
            with open(progress_path, 'w') as f:
                json.dump({'i': i + 1, 'total': len(volumes), 'new': new,
                           'skipped': skipped, 'missing': missing, 'failed': failed,
                           'updated_unix': time.time()}, f, indent=2)

    summary = {'i': len(volumes), 'total': len(volumes), 'new': new,
               'skipped': skipped, 'missing': missing, 'failed': failed,
               'updated_unix': time.time()}
    with open(progress_path, 'w') as f: json.dump(summary, f, indent=2)
    print(split, summary)
    return summary
"""),
    markdown(r"""
## 9. Smoke test one validation CT

This performs a real Hugging Face download and official Y-Mamba forward pass before the
large run. It writes the first valid result to the same resumable cache.
"""),
    code(r"""
smoke_summary = encode_targets(valid_targets[:1], 'valid_smoke')
assert valid_existing(output_path(valid_targets[0])), 'Smoke-test output is invalid.'
smoke = torch.load(output_path(valid_targets[0]), map_location='cpu')
print('Smoke volume:', smoke['volume'])
print('Stage shapes:', smoke['stage_shapes'])
print('encoder5:', smoke['encoder5_shape'])
print('pooled:', tuple(smoke['pooled_encoder5'].shape), smoke['pooled_encoder5'].dtype)
"""),
    markdown(r"""
## 10. Encode all validation targets

Run this cell to completion. Re-run it after any disconnect; completed outputs are
validated and skipped.
"""),
    code(r"""
valid_summary = encode_targets(valid_targets, 'valid')
"""),
    markdown(r"""
## 11. Encode every training target already present in the CT-CLIP cache

This can take many Colab sessions. It does not enumerate the entire CT-RATE training
split—it exactly follows the existing CT-CLIP cache inventory.
"""),
    code(r"""
train_summary = encode_targets(train_targets, 'train')
"""),
    markdown(r"""
## 12. Final one-to-one coverage audit

The audit checks every CT-CLIP-backed target, validates each EXACT payload, and writes a
machine-readable report to Drive.
"""),
    code(r"""
def audit(volumes):
    good, bad = [], []
    for volume in tqdm(volumes, desc='audit'):
        (good if valid_existing(output_path(volume)) else bad).append(volume)
    return good, bad

valid_good, valid_bad = audit(valid_targets)
train_good, train_bad = audit(train_targets)
audit_payload = {
    'created_unix': time.time(),
    'checkpoint_sha256': CHECKPOINT_SHA256,
    'preprocessing_id': PREPROCESSING_ID,
    'valid': {'target': len(valid_targets), 'good': len(valid_good), 'bad': valid_bad},
    'train': {'target': len(train_targets), 'good': len(train_good), 'bad': train_bad},
}
with open(EXACT_MANIFEST_DIR / 'coverage_audit.json', 'w') as f:
    json.dump(audit_payload, f, indent=2)

print('VALID:', len(valid_good), '/', len(valid_targets), '| missing/invalid:', len(valid_bad))
print('TRAIN:', len(train_good), '/', len(train_targets), '| missing/invalid:', len(train_bad))
if valid_bad: print('valid bad[:10]:', valid_bad[:10])
if train_bad: print('train bad[:10]:', train_bad[:10])
print('Audit:', EXACT_MANIFEST_DIR / 'coverage_audit.json')
"""),
    markdown(r"""
## Output and downstream use

Each `.pt` payload contains a 768-d `pooled_encoder5` vector from the frozen official
EXACT image encoder. EXACT has no paired text encoder. For the temporal model, either set
`d_in=768` or learn a `Linear(768,512)` bridge before the current Difference Transformer.
The final `d_f` can still be trained against frozen CT-CLIP text prototypes, but raw EXACT
vectors are **not** already in CT-CLIP's text space.
"""),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "exact_encode_ctrate_cache_colab.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {OUT} ({len(cells)} cells)")