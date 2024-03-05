import math
from sys import maxsize
from typing import Callable
from scipy.special import erfcinv

INT_MAX = maxsize
TANH_MULTIPLIER: float = math.pi / 1.7320508075688772
FRAC_2_SQRT_PI = 1.12837916709551257389615890312154517
SQRT_2 = 1.41421356237309504880168872420969808

BOUNDS = (-6000., 9000.)
LN_10: float = 2.30258509299404568401799145468436421
DEFAULT_MU = 1500.
DEFAULT_SIG = 500.
DEFAULT_SIG_LIMIT = 40.
FLOAT_MAX: float = float('inf')

SECS_PER_DAY = 86400
DEFAULT_WEIGHT_LIMIT = 0.2
DEFAULT_DRIFTS_PER_DAY = 0.
DEFAULT_SPLIT_TIES = False
DEFAULT_TRANSFER_SPEED = 1.

TS_MU = 1500.
TS_SIG = 500.
TS_BETA = 250.
TS_TAU = 5.
TS_DRAW_PROP = 0.001
TS_BACKEND = 'scipy'

DEFAULT_BETA = 400. * TANH_MULTIPLIER / LN_10

DRAW_PROBABILITY = 0.0001

GLICKO_Q = math.log(10) / 400.


def standard_logistic_pdf(z: float) -> float:
    return 0.25 * TANH_MULTIPLIER * math.pow(math.cosh(0.5 * TANH_MULTIPLIER * z), -2)


def standard_logistic_cdf(z: float) -> float:
    return 0.5 + 0.5 * math.tanh(0.5 * TANH_MULTIPLIER * z)


def standard_logistic_cdf_inv(prob: float) -> float:
    return math.atanh(2. * prob - 1.) * 2. / TANH_MULTIPLIER


def standard_normal_pdf(z: float) -> float:
    NORMALIZE: float = 0.5 * FRAC_2_SQRT_PI / SQRT_2
    return NORMALIZE * math.exp(-0.5 * z * z)


def standard_normal_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / SQRT_2)


def standard_normal_cdf_inv(prob: float) -> float:
    return -SQRT_2 * erfcinv(2. * prob)


def solve_bisection(bounds: tuple[float, float], f: Callable[..., float]) -> float:
    lo, hi = bounds
    while True:
        flo = f(lo)
        guess = 0.5 * (lo + hi)
        if lo >= guess or guess >= hi:
            return guess
        if f(guess) * flo > 0.:
            lo = guess
        else:
            hi = guess


def solve_illinois(bounds: tuple[float, float], f: Callable[..., float]) -> float:
    lo, hi = bounds
    (flo, fhi, side) = (f(lo), f(hi), 0)
    while True:
        guess = (flo * hi - fhi * lo) / (flo - fhi)
        if lo >= guess or guess >= hi:
            return 0.5 * (lo + hi)
        fguess = f(guess)
        if fguess * flo > 0.:
            lo = guess
            flo = fguess
            if side == -1:
                fhi *= 0.5
            side = -1
        elif fguess * fhi > 0.:
            hi = guess
            fhi = fguess
            if side == 1:
                flo *= 0.5
            side = 1
        else:
            return guess


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


def solve_newton(lo_hi, f):
    lo, hi = lo_hi
    guess = 0.5 * (lo + hi)
    while True:
        sum, sum_prime = f(guess)
        extrapolate = guess - sum / sum_prime
        if extrapolate < guess:
            hi = guess
            guess = max(extrapolate, hi - 0.75 * (hi - lo))
        else:
            lo = guess
            guess = min(extrapolate, lo + 0.75 * (hi - lo))
        if lo >= guess or guess >= hi:
            if abs(sum) > 1e-10:
                print(f"Possible failure to converge @ {guess}: s={sum}, s'={sum_prime}")
            return guess


def main():
    import numpy as np
    import scipy.stats

    def almost_equal(arg1, *args, tol=1e-5):
        for arg in args:
            if abs(arg1 - arg) > tol:
                return False
        return True

    assert almost_equal(standard_logistic_pdf(0.123),
                        TANH_MULTIPLIER * scipy.stats.logistic.pdf(0.123 * TANH_MULTIPLIER),
                        TANH_MULTIPLIER * scipy.stats.logistic.pdf(np.array([2.]) * TANH_MULTIPLIER)[0])

    assert almost_equal(standard_logistic_cdf(0.123),
                        scipy.stats.logistic.cdf(0.123 * TANH_MULTIPLIER),
                        scipy.stats.logistic.cdf(np.array([2.]) * TANH_MULTIPLIER)[0])

    assert almost_equal(standard_logistic_cdf_inv(0.123),
                        scipy.stats.logistic.ppf(0.123) / TANH_MULTIPLIER,
                        scipy.stats.logistic.ppf(np.array([0.123]))[0] / TANH_MULTIPLIER)

    assert almost_equal(standard_normal_pdf(0.123),
                        scipy.stats.norm.pdf(0.123),
                        scipy.stats.norm.pdf(np.array([0.123]))[0])

    assert almost_equal(standard_normal_cdf(0.123),
                        scipy.stats.norm.cdf(0.123),
                        scipy.stats.norm.cdf(np.array([0.123]))[0])

    assert almost_equal(standard_normal_cdf_inv(0.123),
                        scipy.stats.norm.ppf(0.123),
                        scipy.stats.norm.ppf(np.array([0.123]))[0])
