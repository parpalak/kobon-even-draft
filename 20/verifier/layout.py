"""Canonical line order: Y_0, grid lines by increasing x-intercept, pair (-eps, +eps), R."""
from __future__ import annotations

from arrangement import Intercept

LAYOUT = ("0 = Y_0; 1..n-2 = grid lines L_1..L_{n-2} by increasing x-intercept; "
          "n-1 = L_{n-1} (-epsilon), n = L_n (+epsilon); n+1 = R")


def canonical_order(structure: dict) -> list[int]:
    labels = {int(k): v for k, v in structure["labels"].items()}
    r_line = structure["r_line"]
    pair = structure["special_pair"]
    neg = [i for i in pair if labels[i] == "-epsilon"]
    pos = [i for i in pair if labels[i] == "+epsilon"]
    if len(neg) != 1 or len(pos) != 1:
        raise ValueError("the special pair must have the labels -epsilon and +epsilon")
    grid = [i for i in labels if i not in pair and i != r_line]
    for i in grid:
        if Intercept.parse(labels[i]).depends_on_parameters():
            raise ValueError(f"grid line {i} moves with epsilon/delta")
    grid.sort(key=lambda i: Intercept.parse(labels[i]).node)
    return [structure["y0_line"]] + grid + neg + pos + [r_line]


def check_layout(structure: dict) -> list[str]:
    order = canonical_order(structure)
    if order != list(range(structure["n_lines"])):
        return [f"lines are not in the canonical order ({LAYOUT}); canonical order of current indices: {order}"]
    return []

