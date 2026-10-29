"""Check that the lines realize omatrix.json for all epsilon in (0, E], delta in (0, D].

Each order condition is affine in (epsilon, delta), so the four corners suffice:
>= 0 at the corners with a zero parameter, > 0 at (E, D).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

from arrangement import Arrangement, Intercept, load_lines, load_omatrix
from intervals import Interval
from layout import check_layout

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_TOLERANCE = Fraction(1, 10**25)


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_normalization(A: Arrangement) -> list[str]:
    corner = A.structure["corner_of_lines_csv"]
    eps, delta = Fraction(corner["epsilon"]), Fraction(corner["delta"])
    m = A.slopes()
    problems = []
    for i in range(1, A.n):
        m_csv, b_csv = (Fraction(v) for v in A.csv[i])
        if i in A.triple_rules or i in A.bbl_rules:
            if not (m[i].lo - CSV_TOLERANCE <= m_csv <= m[i].hi + CSV_TOLERANCE):
                problems.append(f"line {i}: slope in lines.csv differs from its rule")
        anchor = -b_csv / m_csv
        a = A.intercepts[i].value(eps, delta)
        if not (a.lo - CSV_TOLERANCE * 10**10 <= anchor <= a.hi + CSV_TOLERANCE * 10**10):
            problems.append(f"line {i}: x-intercept {float(anchor)!r} does not match {A.structure['labels'][str(i)]}")
    return problems


def check_groups(A: Arrangement, rows: list[list[list[int]]]) -> list[str]:
    declared = {frozenset((t,) + pair) for t, pair in A.triple_rules.items()}
    found = set()
    problems = []
    for i, row in enumerate(rows):
        for group in row:
            if len(group) > 1:
                point = frozenset([i] + group)
                if point not in declared:
                    problems.append(f"row {i}: multiple point {sorted(point)} is not a declared triple point")
                found.add(point)
    for point in declared - found:
        problems.append(f"declared triple point {sorted(point)} is missing in the O-matrix")
    return problems


def check_m_min(A: Arrangement) -> list[str]:
    problems = []
    m = A.slopes()
    r_line = A.structure.get("r_line")
    last = A.last_step_lines()
    for t in last:
        rule = A.bbl_rules[t]
        old = [i for i in range(1, A.n) if i not in last and i != r_line and i != 0]
        k = rule["m_min_line"]
        absk_hi = max(abs(m[k].lo), abs(m[k].hi))
        for i in old:
            if i == k:
                continue
            lo_i = min(abs(m[i].lo), abs(m[i].hi)) if m[i].lo * m[i].hi > 0 else Fraction(0)
            if not lo_i > absk_hi:
                problems.append(f"line {t}: m_min is not attained only at line {k} (compare line {i})")
        break
    return problems


def corner_conditions(A: Arrangement, rows, eps: Fraction, delta: Fraction):
    out = []
    for i, row in enumerate(rows):
        for pos, (g, h) in enumerate(zip(row, row[1:])):
            if i == 0:
                value = A.intercepts[h[0]].minus(A.intercepts[g[0]], eps, delta)
            else:
                value = A.y_coordinate(i, g[0], eps, delta) - A.y_coordinate(i, h[0], eps, delta)
            out.append((i, pos, g, h, value))
    return out


def check_parent(A: Arrangement, parent_dir: Path) -> list[str]:
    """Old lines coincide with the parent level (via parent_index), new ones are M_1..M_n."""
    ps = json.loads((parent_dir / "structure.json").read_text(encoding="utf-8"))
    P = Arrangement(ps, load_lines(parent_dir / "lines.csv"))
    problems = []
    parent = {int(k): v for k, v in A.structure.get("parent_index", {}).items()}
    if sorted(parent.values()) != list(range(P.n)) or parent.get(0) != 0:
        return ["parent_index must map the old lines one-to-one onto the parent level, Y_0 to Y_0"]
    for i, p in parent.items():
        if i == 0:
            continue
        if A.structure["labels"][str(i)] != ps["labels"][str(p)]:
            problems.append(f"line {i} (parent {p}): label differs from the parent level")
        if (i in A.free_lines()) != (p in P.free_lines()) or (p in P.free_lines() and Fraction(A.csv[i][0]) != Fraction(P.csv[p][0])):
            problems.append(f"line {i} (parent {p}): slope differs from the parent level")
    if {parent[t]: tuple(parent[x] for x in pair) for t, pair in A.triple_rules.items()} != P.triple_rules:
        problems.append("triple-point rules differ from the parent level")
    if parent[A.structure["r_line"]] != ps["r_line"]:
        problems.append("R is not the line R of the parent level")
    if sorted(parent[i] for i in A.structure["special_pair"]) != sorted(ps["special_pair"]):
        problems.append("the special pair is not the special pair of the parent level")
    new = sorted(set(range(A.n)) - set(parent))
    last = A.last_step_lines()
    if new != last or sorted(A.bbl_rules[i]["j"] for i in new) != list(range(1, len(new) + 1)):
        problems.append("the new lines are not exactly M_1..M_n of one BBL step")
    for i in new:
        rule = A.bbl_rules[i]
        if rule["n"] != P.n - 2 or rule["sigma"] != -1 or rule["m_min_line"] not in parent:
            problems.append("BBL rule with wrong n, sigma or m_min line (expected n = old lines without R and Y_0, sigma = -1)")
            break
    for i, p in parent.items():
        if (i in A.bbl_rules) != (p in P.bbl_rules):
            problems.append(f"line {i} (parent {p}): BBL rule differs from the parent level")
        elif i in A.bbl_rules:
            r, rp = A.bbl_rules[i], P.bbl_rules[p]
            if (r["j"], r["n"], r["sigma"], parent[r["m_min_line"]]) != (rp["j"], rp["n"], rp["sigma"], rp["m_min_line"]):
                problems.append(f"line {i} (parent {p}): BBL rule differs from the parent level")
    return problems


def certify(data_dir: Path, parent: Path | None = None) -> dict:
    structure = json.loads((data_dir / "structure.json").read_text(encoding="utf-8"))
    A = Arrangement(structure, load_lines(data_dir / "lines.csv"))
    rows = load_omatrix(data_dir / "omatrix.json", A.n)

    problems = check_layout(structure) + check_normalization(A) + check_groups(A, rows) + check_m_min(A)
    if parent is not None:
        problems += check_parent(A, parent)

    E = Fraction(structure["parameters"]["epsilon"][1])
    D = Fraction(structure["parameters"]["delta"][1])
    corners = {"(E,D)": (E, D), "(E,0)": (E, Fraction(0)), "(0,D)": (Fraction(0), D), "(0,0)": (Fraction(0), Fraction(0))}
    corner_report = {}
    min_margin = None
    count = 0
    for name, (eps, delta) in corners.items():
        conds = corner_conditions(A, rows, eps, delta)
        count = len(conds)
        zeros = [dict(row=i, position=p, left=g, right=h) for i, p, g, h, v in conds if v.is_zero()]
        bad = [dict(row=i, position=p, left=g, right=h, lower=str(v.lo)) for i, p, g, h, v in conds
               if (v.lo <= 0 and not v.is_zero()) or (name == "(E,D)" and v.lo <= 0)]
        corner_report[name] = dict(zero_conditions=zeros, failed_conditions=bad)
        if name == "(E,D)":
            min_margin = min(v.lo for *_, v in conds)
        if bad:
            problems.append(f"corner {name}: {len(bad)} order conditions not certified")

    m = A.slopes()
    lines = list(range(A.n))
    order = sorted(lines, key=lambda i: m[i].lo)
    for i, j in zip(order, order[1:]):
        if not m[i].hi < m[j].lo:
            problems.append(f"slopes of lines {i} and {j} are not separated")
    direction = dict(rule="lines sorted by increasing slope; Y_0 has slope 0", order=order,
                     negative=[i for i in lines if m[i].hi < 0], positive=[i for i in lines if m[i].lo > 0])

    return dict(
        n_lines=A.n,
        lines_checksum=checksum(data_dir / "lines.csv"),
        omatrix_checksum=checksum(data_dir / "omatrix.json"),
        structure_checksum=checksum(data_dir / "structure.json"),
        parameters=structure["parameters"],
        free_slopes={str(i): A.csv[i][0] for i in A.free_lines()},
        dependent_slopes=structure.get("dependent_slopes", []),
        bbl_lines=structure.get("bbl_lines", []),
        slope_enclosures={str(i): [str(m[i].lo), str(m[i].hi)] for i in range(1, A.n) if i not in A.free_lines()},
        order_conditions_per_corner=count,
        min_margin_at_E_D=str(min_margin),
        corners=corner_report,
        direction_order=direction,
        problems=problems,
        verdict=not problems,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Interval certification of a normalized BBL arrangement")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "n20")
    parser.add_argument("--certificate", type=Path, default=None)
    parser.add_argument("--parent", type=Path, default=None, help="data directory of the level this one was doubled from")
    args = parser.parse_args()
    cert = certify(args.data, args.parent)
    out = args.certificate or args.data / "interval_certificate.json"
    out.write_text(json.dumps(cert, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: cert[k] for k in ("n_lines", "order_conditions_per_corner", "min_margin_at_E_D", "problems", "verdict")}, indent=1))
    for name, rep in cert["corners"].items():
        print(name, "zero conditions:", len(rep["zero_conditions"]), "failed:", len(rep["failed_conditions"]))
    return 0 if cert["verdict"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
