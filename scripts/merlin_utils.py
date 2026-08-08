"""
Shared helpers for embedding CT scans and text prompts with a FROZEN MERLIN.

We call MERLIN's sub-encoders directly:
  - arch.encode_image(volume) -> (image_features[512], ehr_features[1692])
  - arch.encode_text(list[str]) -> text_features[512]

Both are returned UN-normalized here; callers decide when to L2-normalize.
The MERLIN weights are never updated (we always run under torch.no_grad()).
"""

import os
from typing import List, Optional

import torch
import torch.nn.functional as F

from merlin import Merlin
from merlin.data import DataLoader


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class MerlinEmbedder:
    """Thin wrapper that turns CT volumes and text into 512-d embeddings."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or get_device()
        # Full (default) Merlin: gives us BOTH encode_image and encode_text.
        self.model = Merlin()
        self.model.eval()
        self.model.to(self.device)
        # The underlying MerlinArchitecture (exposes encode_image/encode_text).
        self.arch = self.model.model

    # ---- Images ---------------------------------------------------------
    @torch.no_grad()
    def embed_images(
        self,
        image_paths: List[str],
        cache_dir: str,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Return a (N, 512) tensor of contrastive image embeddings."""
        datalist = [{"image": p, "text": ""} for p in image_paths]
        loader = DataLoader(
            datalist=datalist,
            cache_dir=cache_dir,
            batchsize=1,
            shuffle=False,
            num_workers=0,
        )
        feats = []
        for batch in loader:
            image_features, _ehr = self.arch.encode_image(
                batch["image"].to(self.device)
            )
            if image_features.dim() == 1:
                image_features = image_features.unsqueeze(0)
            feats.append(image_features.detach().cpu())
        out = torch.cat(feats, dim=0)
        if normalize:
            out = F.normalize(out, dim=-1)
        return out

    # ---- Text -----------------------------------------------------------
    @torch.no_grad()
    def embed_texts(
        self,
        texts: List[str],
        normalize: bool = True,
    ) -> torch.Tensor:
        """Return a (N, 512) tensor of text embeddings for prompts/sentences."""
        text_features = self.arch.encode_text(texts)
        if text_features.dim() == 1:
            text_features = text_features.unsqueeze(0)
        out = text_features.detach().cpu()
        if normalize:
            out = F.normalize(out, dim=-1)
        return out


def project_paths():
    """Return (project_dir, cache_dir, data_dir) as absolute paths."""
    here = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(here)
    cache_dir = os.path.join(project_dir, "cache", "abct_data_cache")
    data_dir = os.path.join(project_dir, "data", "abct_data")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    return project_dir, cache_dir, data_dir
