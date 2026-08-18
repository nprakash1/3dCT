from google.colab import drive
drive.mount('/content/drive')
import os, json, csv, math, random, hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, defaultdict
from sklearn.metrics import f1_score
from itertools import product
import pandas as pd
csv.field_size_limit(10**9)

DRIVE = '/content/drive/MyDrive/3dCT'
IMG_DIR = f'{DRIVE}/ctclip_cache/img'
WEIGHTS = f'{DRIVE}/ctclip_weights'
LAB = '/content/3dCT/medgemma_labels_v3.jsonl'
LAB_DS = '/content/3dCT/medgemma_labels (2).jsonl'
EVAL_BANK_PT = f'{DRIVE}/ctclip_cache/eval_holdout_sentence_bank.pt'

TUNE_FRAC, SPLIT_SEED = 0.15, 2026
REQUIRE_COMPLETE_HUB_VALID_FEATURES = True
CLASSES = ['worsened', 'stable', 'improved']
C2I = {c: i for i, c in enumerate(CLASSES)}
I2C = {i: c for c, i in C2I.items()}

FINDING_CONDITIONING = True
FINDING_AS_4TH_TOKEN = False
USE_LEARNED_FINDING_EMB = True
ANTISYM = False

USE_CE, USE_MAGNITUDE, USE_SUPCON = True, True, False
CONTRASTIVE_SYMMETRIC = False
LEARNABLE_TAU_CON = True
TAU_CON_INIT = 0.07
STABLE_TEXT_SEED = 2026
EVAL_SAMPLE_SEED = 2027
N_EVAL_SENTENCES = 5

D_MODEL, EPOCHS, LR, PATIENCE = 256, 120, 1e-3, 20
WEIGHT_DECAY = 1e-2
LAMBDA_CE, LAMBDA_MAG, LAMBDA_CON = 1.0, 0.5, 0.5
K_FINDINGS_PER_BATCH, N_PER_CLASS, MAX_BATCH_SIZE = 8, 4, 256

torch.manual_seed(0); np.random.seed(0); random.seed(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device', DEVICE, 'img', os.path.isdir(IMG_DIR))
print('EVAL: holdout', N_EVAL_SENTENCES, 'sents/class; random sample at eval')
