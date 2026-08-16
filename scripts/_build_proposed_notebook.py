#!/usr/bin/env python3
"""Assemble notebooks/train_proposed_finding_conditioned_colab.ipynb from cell_*.txt files."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLS_DIR = ROOT / "scripts" / "_proposed_nb_cells"
OUT = ROOT / "notebooks" / "train_proposed_finding_conditioned_colab.ipynb"


def src(text: str) -> list[str]:
    text = text.strip("\n")
    if not text:
        return []
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def main() -> None:
    files = sorted(CELLS_DIR.glob("cell_*.txt"))
    if not files:
        raise SystemExit(f"no cell_*.txt in {CELLS_DIR}")

    cells: list[dict] = []
    for fp in files:
        # cell_00_md.txt / cell_01_code.txt
        m = re.match(r"cell_\d+_(md|code)\.txt$", fp.name)
        if not m:
            raise SystemExit(f"bad cell filename: {fp.name}")
        kind = m.group(1)
        text = fp.read_text(encoding="utf-8")
        if kind == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": src(text)})
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": src(text),
                }
            )

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
                "name": "train_proposed_finding_conditioned_colab.ipynb",
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
