"""Lines Y_0 : y = 0 and L_i : y = m_i (x - a_i) in the BBL chart.

a_i = tan(q pi) + c epsilon + d delta.  Slopes: free (exact, from lines.csv), triple (through
L_i cap L_j) or bbl (new line of a doubling step).
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from intervals import Interval, ZERO, sin_interval, tan_interval

LABEL_RE = re.compile(r"^(?:(?P<sign>[+-])tan\((?P<k>\d+)pi/(?P<n>\d+)\))?"
                      r"(?P<eps>[+-]epsilon)?(?P<delta>[+-]delta)?$")


@dataclass(frozen=True)
class Intercept:
    node: Fraction | None
    eps_coef: int = 0
    delta_coef: int = 0

    @staticmethod
    def parse(label: str) -> "Intercept":
        match = LABEL_RE.match(label)
        if not match or label == "":
            raise ValueError(f"unrecognized intercept label {label!r}")
        node = None
        if match.group("k") is not None:
            node = Fraction(int(match.group("k")), int(match.group("n")))
            if match.group("sign") == "-":
                node = -node
        eps = match.group("eps"); delta = match.group("delta")
        return Intercept(node,
                         0 if eps is None else (1 if eps[0] == "+" else -1),
                         0 if delta is None else (1 if delta[0] == "+" else -1))

    def node_interval(self) -> Interval:
        if self.node is None or self.node == 0:
            return ZERO
        return tan_interval(self.node.numerator, self.node.denominator)

    def value(self, epsilon: Fraction, delta: Fraction) -> Interval:
        return self.node_interval() + Interval.point(self.eps_coef * epsilon + self.delta_coef * delta)

    def minus(self, other: "Intercept", epsilon: Fraction, delta: Fraction) -> Interval:
        # equal nodes cancel exactly (R and the grid line at a_*)
        offset = Interval.point((self.eps_coef - other.eps_coef) * epsilon + (self.delta_coef - other.delta_coef) * delta)
        if self.node == other.node or (self.node in (None, 0) and other.node in (None, 0)):
            return offset
        return self.node_interval() - other.node_interval() + offset

    def depends_on_parameters(self) -> bool:
        return self.eps_coef != 0 or self.delta_coef != 0


def load_lines(path: Path) -> list[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["m", "b"]:
            raise ValueError(f"{path} must have the CSV header m,b")
        rows = [(row["m"].strip(), row["b"].strip()) for row in reader]
    if rows[0] != ("0", "0"):
        raise ValueError(f"{path}: the first row must be Y_0 = (0, 0)")
    return rows


def load_omatrix(path: Path, n_lines: int) -> list[list[list[int]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if len(raw) != n_lines:
        raise ValueError(f"{path}: expected {n_lines} rows, got {len(raw)}")
    rows = []
    for i, row in enumerate(raw):
        groups = [entry if isinstance(entry, list) else [entry] for entry in row]
        flat = [j for g in groups for j in g]
        if sorted(flat) != [j for j in range(n_lines) if j != i]:
            raise ValueError(f"{path}: row {i} must contain every other line exactly once")
        rows.append(groups)
    return rows


class Arrangement:
    def __init__(self, structure: dict, lines_csv: list[tuple[str, str]]):
        self.structure = structure
        self.n = structure["n_lines"]
        if len(lines_csv) != self.n:
            raise ValueError(f"lines.csv has {len(lines_csv)} rows, structure says {self.n}")
        self.csv = lines_csv
        self.intercepts = {0: Intercept(Fraction(0))}
        for key, label in structure["labels"].items():
            self.intercepts[int(key)] = Intercept.parse(label)
        if sorted(self.intercepts) != list(range(self.n)):
            raise ValueError("every line 1..n-1 needs an intercept label")
        self.triple_rules = {d["line"]: tuple(d["through"]) for d in structure.get("dependent_slopes", [])}
        self.bbl_rules = {d["line"]: d for d in structure.get("bbl_lines", [])}
        self._slopes: dict[int, Interval] | None = None

    def last_step_lines(self) -> list[int]:
        if not self.bbl_rules:
            return []
        n = max(r["n"] for r in self.bbl_rules.values())
        return sorted(i for i, r in self.bbl_rules.items() if r["n"] == n)

    def free_lines(self) -> list[int]:
        return [i for i in range(1, self.n) if i not in self.triple_rules and i not in self.bbl_rules]

    def slopes(self) -> dict[int, Interval]:
        if self._slopes is not None:
            return self._slopes
        m: dict[int, Interval] = {0: ZERO}
        for i in self.free_lines():
            m[i] = Interval.point(Fraction(self.csv[i][0]))
        for t, (i, j) in self.triple_rules.items():
            for line in (t, i, j):
                if self.intercepts[line].depends_on_parameters():
                    raise ValueError(f"triple ({t}, {i}, {j}) involves a line moving with epsilon/delta")
            ai, aj, at = (self.intercepts[x].value(Fraction(0), Fraction(0)) for x in (i, j, t))
            x = (m[i] * ai - m[j] * aj) / (m[i] - m[j])
            y = m[i] * (x - ai)
            m[t] = y / (x - at)
        for t, rule in sorted(self.bbl_rules.items(), key=lambda kv: kv[1]["n"]):
            n = rule["n"]; j = rule["j"]; sigma = rule["sigma"]
            m_min = m[rule["m_min_line"]]
            if m_min.hi < 0:
                m_min = -m_min
            elif m_min.lo <= 0:
                raise ValueError("m_min interval contains 0")
            # beta_j = (2j - 1 - n) pi / (2n)
            b = tan_interval(2 * j - 1 - n, 2 * n)
            s = sin_interval(2 * j - 1 - n, n)
            n6 = Interval.point(Fraction(n) ** 6); n10 = Interval.point(Fraction(n) ** 10)
            mu = (m_min / n10) * (s + Interval.point(1) / (n6 * b))
            m[t] = mu if sigma > 0 else -mu
        self._slopes = m
        return m

    def y_coordinate(self, i: int, j: int, epsilon: Fraction, delta: Fraction) -> Interval:
        if i == 0 or j == 0:
            return ZERO
        m = self.slopes()
        diff = self.intercepts[i].minus(self.intercepts[j], epsilon, delta)
        if diff.is_zero():
            return ZERO
        return (m[i] * m[j] * diff) / (m[i] - m[j])
