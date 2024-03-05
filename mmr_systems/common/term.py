import math
from abc import abstractmethod
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from functools import reduce
from typing import Generator, Iterable, Iterator, Self

from mmr_systems.common.numericals import (solve_newton,
                                           standard_normal_cdf,
                                           standard_normal_pdf,
                                           TANH_MULTIPLIER, BOUNDS)
from mmr_systems.common.ordering import Ordering


class Term:
    @abstractmethod
    def eval(self, x: float, order: Ordering, split_ties: bool) -> tuple[float, float]:
        pass

    def evals(self, x: float, ranks: list[int], my_rank: int, split_ties: bool) -> tuple[float, float]:
        if len(ranks) == 1:
            return self.eval(x, Ordering.cmp(ranks[0], my_rank), split_ties)
        start, end = self.equal_range(ranks, my_rank)
        equal = end - start
        greater = len(ranks) - end
        value = 0.0
        deriv = 0.0
        if start > 0:
            v, p = self.eval(x, Ordering.LESS, split_ties)
            value += start * v
            deriv += start * p
        if equal > 0:
            v, p = self.eval(x, Ordering.EQUAL, split_ties)
            value += equal * v
            deriv += equal * p
        if greater > 0:
            v, p = self.eval(x, Ordering.GREATER, split_ties)
            value += greater * v
            deriv += greater * p
        return value, deriv

    def equal_range(self, ranks: list[int], my_rank: int) -> tuple[int, int]:
        left = bisect_left(ranks, my_rank)
        right = bisect_right(ranks, my_rank, lo=left)
        return left, right


@dataclass
class Rating(Term):
    mu: float
    sig: float

    def __hash__(self) -> int:
        return hash((self.mu, self.sig))

    def eval(self, x: float, order: Ordering, split_ties: bool) -> tuple[float, float]:
        z = (x - self.mu) / self.sig
        pdf = standard_normal_pdf(z) / self.sig
        pdf_prime = -z * pdf / self.sig
        match order:
            case Ordering.LESS:
                cdf_m1 = -standard_normal_cdf(-z)
                val = pdf / cdf_m1
                return (val, pdf_prime / cdf_m1 - val * val)
            case Ordering.GREATER:
                cdf = standard_normal_cdf(z)
                val = pdf / cdf
                return (val, pdf_prime / cdf - val * val)
            case Ordering.EQUAL:
                if split_ties:
                    cdf = standard_normal_cdf(z)
                    cdf_m1 = -standard_normal_cdf(-z)
                    val0 = pdf / cdf
                    val1 = pdf / cdf_m1
                    return (
                        0.5 * (val0 + val1),
                        0.5 * (pdf_prime * (1. / cdf + 1. / cdf_m1) - val0 * val0 - val1 * val1),
                    )
                else:
                    pdf_pp = -(pdf / self.sig + z * pdf_prime) / self.sig
                    val = pdf_prime / pdf
                    return (val, pdf_pp / pdf - val * val)

    def with_noise(self, sig_noise: float) -> 'Rating':
        return Rating(self.mu, (self.sig ** 2 + sig_noise ** 2) ** 0.5)

    def towards_noise(self, decay: float, limit: Self) -> 'Rating':
        mu_diff = self.mu - limit.mu
        sig_sq_diff = self.sig ** 2 - limit.sig ** 2
        return Rating(limit.mu + mu_diff * decay, (limit.sig ** 2 + sig_sq_diff * decay ** 2) ** 0.5)


@dataclass
class TanhTerm(Term):
    mu: float
    w_arg: float
    w_out: float

    def eval(self, x: float, order: Ordering, split_ties: bool):
        val, val_prime = self.base_values(x)
        match order:
            case Ordering.LESS:
                return val - self.w_out, val_prime
            case Ordering.GREATER:
                return val + self.w_out, val_prime
            case Ordering.EQUAL:
                if split_ties:
                    return val, val_prime
                else:
                    return 2. * val, 2. * val_prime

    @classmethod
    def from_rating(cls, rating: Rating) -> Self:
        w = TANH_MULTIPLIER / rating.sig
        return cls(rating.mu, w * 0.5, w)

    def get_weight(self) -> float:
        return self.w_arg * self.w_out * 2. / TANH_MULTIPLIER ** 2

    def base_values(self, x: float) -> tuple[float, float]:
        z = (x - self.mu) * self.w_arg
        val = -math.tanh(z) * self.w_out
        val_prime = -math.cosh(z) ** -2 * self.w_arg * self.w_out
        return val, val_prime

    def evals(self, x: float, ranks: list[int], my_rank: int, split_ties: bool) -> tuple[float, float]:
        if len(ranks) == 1:
            return self.eval(x, Ordering.cmp(ranks[0], my_rank), split_ties)
        val, val_prime = self.base_values(x)
        start, end = self.equal_range(ranks, my_rank)
        total = float(len(ranks))
        win_minus_loss = total - float(start + end)
        if not split_ties:
            equal = end - start
            total += float(equal)
        value = val * total + self.w_out * win_minus_loss
        deriv = val_prime * total
        return value, deriv


def robust_average(all_ratings: Iterable[TanhTerm], offset: float, slope: float) -> float:
    if isinstance(all_ratings, (Iterator, Generator)):
        raise TypeError('robust_average must accept an iterable which isn\'t consumed (i.e. a list or tuple).')
    bounds = BOUNDS

    def f(x: float) -> tuple[float, float]:
        def inner(term: TanhTerm):
            tanh_z = math.tanh((x - term.mu) * term.w_arg)
            return (
                tanh_z * term.w_out,
                (1. - tanh_z * tanh_z) * term.w_arg * term.w_out
            )
        return reduce(lambda acc, val: (acc[0] + val[0], acc[1] + val[1]), map(inner, all_ratings), (offset + slope * x, slope))
    return solve_newton(bounds, f)
