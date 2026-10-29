"""Faces from omatrix.json: bounded triangles, external digons (side on L_inf), p3, p3(R),
a3 = p3 - p3(R) (valid if R carries no multiple points).  On A = without R also checks the
conditions of the doubling lemma (D1)-(D3) and of the wedge lemma (W1), (W1').
"""
from __future__ import annotations

import argparse
import itertools
import json
from fractions import Fraction
from pathlib import Path

from arrangement import Intercept, load_omatrix
from certify_interval import checksum

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def delete_line(rows, r):
    out = []
    for i, row in enumerate(rows):
        if i == r:
            out.append(None)
            continue
        new = []
        for g in row:
            g2 = [j for j in g if j != r]
            if g2:
                new.append(g2)
        out.append(new)
    return out


def position(rows):
    pos = {}
    for i, row in enumerate(rows):
        if row is None:
            continue
        for p, g in enumerate(row):
            for j in g:
                pos[(i, j)] = p
    return pos


def bounded_triangles(rows, lines):
    pos = position(rows)
    tri = []
    for a, b, c in itertools.combinations(lines, 3):
        ok = True
        for x, y, z in ((a, b, c), (b, a, c), (c, a, b)):
            p, q = pos[(x, y)], pos[(x, z)]
            if abs(p - q) != 1:
                ok = False
                break
        if ok:
            tri.append((a, b, c))
    return tri


def end_group(rows, i, end, slope_sign):
    if i == 0:
        return rows[0][-1] if end > 0 else rows[0][0]
    up = (end > 0) == (slope_sign[i] > 0)
    return rows[i][0] if up else rows[i][-1]


def digons(rows, lines, order, slope_sign):
    order = [i for i in order if i in lines]
    rays = [(i, 1) for i in order] + [(i, -1) for i in order]
    out = []
    for (i, e), (j, f) in zip(rays, rays[1:] + rays[:1]):
        if j in end_group(rows, i, e, slope_sign) and i in end_group(rows, j, f, slope_sign):
            out.append((i, j, "right" if e > 0 and f > 0 else "left" if e < 0 and f < 0 else "vertical-turn"))
    return out


def wedges(rows, lines, slope_sign, end):
    out = []
    for i, j in itertools.combinations(lines, 2):
        if j in end_group(rows, i, end, slope_sign) and i in end_group(rows, j, end, slope_sign):
            out.append((i, j))
    return out


def certify(data_dir: Path, interval_cert: dict) -> dict:
    structure = json.loads((data_dir / "structure.json").read_text(encoding="utf-8"))
    n = structure["n_lines"]
    rows = load_omatrix(data_dir / "omatrix.json", n)
    r_line = structure["r_line"]
    labels = {int(k): v for k, v in structure["labels"].items()}
    order = interval_cert["direction_order"]["order"]
    sign = {i: 0 for i in range(n)}
    for i in interval_cert["direction_order"]["negative"]:
        sign[i] = -1
    for i in interval_cert["direction_order"]["positive"]:
        sign[i] = 1

    lines = list(range(n))
    tri = bounded_triangles(rows, lines)
    dig = digons(rows, lines, order, sign)
    p3 = len(tri) + len(dig)
    p3_r = sum(1 for t in tri if r_line in t) + sum(1 for d in dig if r_line in d[:2])
    a3 = p3 - p3_r
    target = Fraction(n * (2 * n - 5) + 2, 6)
    multiple = sorted({tuple(sorted([i] + g)) for i, row in enumerate(rows) for g in row if len(g) > 1})

    # A = without R
    A_rows = delete_line(rows, r_line)
    A_lines = [i for i in lines if i != r_line]
    A_tri = bounded_triangles(A_rows, A_lines)
    y0 = A_rows[0]
    segs = []
    for g, h in zip(y0, y0[1:]):
        l, r = g[0], h[0]
        on = [t for t in A_tri if 0 in t and l in t and r in t]
        side = None
        if len(on) == 1:
            pos_l = {j: p for p, gg in enumerate(A_rows[l]) for j in gg}
            side = "above" if pos_l[r] < pos_l[0] else "below"
        segs.append(dict(between=[l, r], triangles=len(on), side=side))
    d3 = all(s["triangles"] == 1 for s in segs) and all(a["side"] != b["side"] for a, b in zip(segs, segs[1:]))
    grid = sorted((Intercept.parse(labels[i]).node, i) for i in A_lines if i != 0 and labels[i] not in ("-epsilon", "+epsilon"))
    pair = structure["special_pair"]
    n_old = len(A_lines) - 1
    expected_nodes = sorted([Fraction(k, n_old) for k in range(1, n_old // 2)] + [Fraction(-k, n_old) for k in range(1, n_old // 2)])
    d1 = [node for node, _ in grid] == expected_nodes and sorted(labels[i] for i in pair) == ["+epsilon", "-epsilon"]
    left_pair, right_pair = sorted(pair, key=lambda i: labels[i] != "-epsilon")
    d2 = sign[left_pair] < 0 < sign[right_pair]
    l_first, l_last = grid[0][1], grid[-1][1]
    right_w = wedges(A_rows, A_lines, sign, 1)
    left_w = wedges(A_rows, A_lines, sign, -1)
    w1 = (min(0, l_last), max(0, l_last)) in right_w
    w1_prime = (min(0, l_first), max(0, l_first)) in left_w

    node_line = next((i for i in A_lines if labels.get(i) == structure.get("node")), None)
    node_right = [sg for sg in segs if node_line is not None and sg["between"][0] == node_line]
    all_checks = dict(a3_equals_bound=(a3 == target), D1=d1, D2=d2, D3=d3, W1=w1, W1_prime=w1_prime,
                      slope_L_n_minus_2_negative=sign[l_last] < 0, slope_L1_positive=sign[l_first] > 0,
                      node_right_triangle_above=len(node_right) == 1 and node_right[0]["side"] == "above",
                      wedge_apices_simple=all(len(end_group(A_rows, i, e, sign)) == 1
                                              for e, ws in ((1, right_w), (-1, left_w)) for w in ws for i in w))
    required = structure.get("requirements", list(all_checks))
    checks = {k: all_checks[k] for k in required}
    checks["counting_line_simple"] = not any(r_line in point for point in multiple)
    informative = {k: v for k, v in all_checks.items() if k not in required}
    return dict(
        n_lines=n,
        bounded_triangles_bbl_chart=len(tri),
        external_digons=len(dig),
        p3=p3, p3_R=p3_r, a3_counting_chart=a3, bound=str(target),
        multiple_points=multiple,
        A_without_R=dict(bounded_triangles=len(A_tri), y0_segments=segs, L1=l_first, L_n_minus_2=l_last,
                         right_wedges=right_w, left_wedges=left_w),
        external_digon_list=dig,
        checks=checks,
        not_required=informative,
        verdict=all(checks.values()),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Combinatorial counts of a certified BBL arrangement")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "n20")
    args = parser.parse_args()
    interval_cert = json.loads((args.data / "interval_certificate.json").read_text(encoding="utf-8"))
    if not interval_cert["verdict"]:
        raise SystemExit("interval certificate has no positive verdict")
    stale = [name for name in ("lines.csv", "omatrix.json", "structure.json")
             if interval_cert[name.split(".")[0] + "_checksum"] != checksum(args.data / name)]
    if stale:
        raise SystemExit(f"interval certificate was made for other data ({', '.join(stale)} changed); rerun certify_interval.py")
    cert = certify(args.data, interval_cert)
    (args.data / "combinatorial_certificate.json").write_text(json.dumps(cert, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: cert[k] for k in ("n_lines", "bounded_triangles_bbl_chart", "external_digons", "p3", "p3_R",
                                            "a3_counting_chart", "bound", "multiple_points", "checks", "not_required", "verdict")}, indent=1))
    print("right wedges of A:", cert["A_without_R"]["right_wedges"], " left:", cert["A_without_R"]["left_wedges"])
    return 0 if cert["verdict"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
