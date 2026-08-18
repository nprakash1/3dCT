#!/usr/bin/env python3
"""Build notebooks/train_holdout_eval_sentence_ablation_colab.ipynb."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "train_holdout_eval_sentence_ablation_colab.ipynb"
FRAG = ROOT / "scripts" / "_holdout_eval_fragments"


def src(text: str) -> list[str]:
    text = text.strip("\n")
    if not text:
        return []
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def md(t: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src(t)}


def code(t: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src(t),
    }


def load(name: str) -> str:
    return (FRAG / name).read_text(encoding="utf-8")


def main() -> None:
    order = [
        ("md", "00_intro.md"),
        ("md", "01_setup.md"),
        ("code", "02_setup.py"),
        ("md", "03_config.md"),
        ("code", "04_config.py"),
        ("md", "05_data.md"),
        ("code", "06_data.py"),
        ("md", "07_banks.md"),
        ("code", "08_banks.py"),
        ("md", "09_model.md"),
        ("code_join", ["10_model.py", "10b_loss_sampler.py"]),
        ("md", "11_ablation.md"),
        ("code_join", ["12_ablation.py", "12b_ablation_run.py"]),
    ]
    cells = []
    for kind, name in order:
        if kind == "md":
            cells.append(md(load(name)))
        elif kind == "code":
            cells.append(code(load(name)))
        else:
            body = "\n\n".join(load(n) for n in name)
            cells.append(code(body))

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "colab": {
                "provenance": [],
                "name": "train_holdout_eval_sentence_ablation_colab.ipynb",
            },
        },
        "cells": cells,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(cells)} cells)")
    for i, c in enumerate(cells):
        head = "".join(c["source"])[:70].replace("\n", " ")
        print(f"  {i:02d} {c['cell_type']:8} {head}")


if __name__ == "__main__":
    main()
