"""Rational interval arithmetic; tan(k pi/n) is enclosed by a sign change of an integer polynomial."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import comb

import mpmath as mp


@dataclass(frozen=True)
class Interval:
    lo: Fraction
    hi: Fraction

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError(f"invalid interval [{self.lo}, {self.hi}]")

    @staticmethod
    def point(value) -> "Interval":
        value = Fraction(value)
        return Interval(value, value)

    def __add__(self, other: "Interval") -> "Interval":
        return Interval(self.lo + other.lo, self.hi + other.hi)

    def __sub__(self, other: "Interval") -> "Interval":
        return Interval(self.lo - other.hi, self.hi - other.lo)

    def __neg__(self) -> "Interval":
        return Interval(-self.hi, -self.lo)

    def __mul__(self, other: "Interval") -> "Interval":
        products = (self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi)
        return Interval(min(products), max(products))

    def __truediv__(self, other: "Interval") -> "Interval":
        if other.lo <= 0 <= other.hi:
            raise ZeroDivisionError(f"division by an interval containing 0: {other}")
        return self * Interval(1 / other.hi, 1 / other.lo)

    def is_zero(self) -> bool:
        return self.lo == 0 and self.hi == 0


ZERO = Interval.point(0)


def tan_numerator(n: int) -> list[int]:
    """P_n with tan(nx) = P_n(t)/Q_n(t), t = tan x, coefficients from low to high."""
    coefficients = [0] * (n + 1)
    for j in range(1, n + 1, 2):
        coefficients[j] = (-1) ** ((j - 1) // 2) * comb(n, j)
    return coefficients


def evaluate(coefficients: list[int], x: Fraction) -> Fraction:
    value = Fraction(0)
    for c in reversed(coefficients):
        value = value * x + c
    return value


DIGITS = 60


@lru_cache(maxsize=None)
def positive_tan_roots(n: int) -> tuple[Interval, ...]:
    """P_n has exactly m = ceil(n/2) - 1 positive roots tan(j pi/n).  m disjoint increasing
    intervals with a sign change each therefore enclose them in order.  mpmath only picks centres."""
    poly = tan_numerator(n)
    m = (n + 1) // 2 - 1
    out = []
    with mp.workdps(DIGITS + 20):
        radius = mp.mpf(10) ** (-DIGITS)
        for j in range(1, m + 1):
            centre = mp.tan(j * mp.pi / n)
            lo = Fraction(mp.nstr(centre - radius, DIGITS + 15, min_fixed=-10**9, max_fixed=10**9))
            hi = Fraction(mp.nstr(centre + radius, DIGITS + 15, min_fixed=-10**9, max_fixed=10**9))
            if lo <= 0 or evaluate(poly, lo) * evaluate(poly, hi) >= 0:
                raise ValueError(f"no certified sign change for tan({j}pi/{n})")
            if out and lo <= out[-1].hi:
                raise ValueError(f"enclosures of tan(({j}-1)pi/{n}) and tan({j}pi/{n}) overlap")
            out.append(Interval(lo, hi))
    return tuple(out)


def tan_interval(k: int, n: int) -> Interval:
    if k == 0:
        return ZERO
    if 2 * abs(k) >= n:
        raise ValueError(f"tan({k}pi/{n}) is not finite or outside (-pi/2, pi/2)")
    if k < 0:
        return -tan_interval(-k, n)
    return positive_tan_roots(n)[k - 1]


@lru_cache(maxsize=None)
def sin_interval(k: int, n: int) -> Interval:
    """sin(k pi/n) = 2t/(1 + t^2), t = tan(k pi/(2n))."""
    if 2 * abs(k) == n:
        return Interval.point(1 if k > 0 else -1)
    t = tan_interval(k, 2 * n)
    if t.lo <= 1 <= t.hi or t.lo <= -1 <= t.hi:
        raise ValueError("t = +-1 must be handled exactly")
    f = lambda x: 2 * x / (1 + x * x)
    a, b = f(t.lo), f(t.hi)
    return Interval(min(a, b), max(a, b))
