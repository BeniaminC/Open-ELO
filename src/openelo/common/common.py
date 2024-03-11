from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

from .player import Player
from .term import TanhTerm


__all__ = ['convert_placement_to_standings',
           'ContestRatingParams',
           'EloMMRVariant']


Standings = list[tuple[Player, int, int]]


def convert_placement_to_standings(placements: list[tuple[Player, int]]) -> Standings:
    '''
    Given a list player objects `Player` with their respective placement
    of any order, return an ordered list of placements according to the
    format of the elo rating systems. If the placement are unique, then
    the starting placements will equal the ending placements for all 
    players. If they are equal in placement, then their end placement will
    be the starting placement plus the number of players in the same placement.

    Args:
        placements (:obj:`list[tuple[Player, int]]): A list of tuples of Player
        objects with their respective placement.

    Returns:
        :obj:`Standings: A list of tuples of Player objects
        with their respective start and end placement.
    
    Example::

        placements = [('A', 0), ('B', 1), ('C', 1), ('D', 2)]
        conv_placements = convert_placement_to_standings(placements)
        conv_placements  # --> [('A', 0, 0), ('B', 1, 2), ('C', 1, 2), ('D', 3, 3)]
    '''
    unique_placements = set()
    player_placements = defaultdict(list)

    for player, placement in placements:
        unique_placements.add(placement)
        player_placements[placement].append(player)

    sorted_unique_placements = sorted(unique_placements)
    standings = []
    starting_placement = 0
    for unique_placement in sorted_unique_placements:
        num_same = len(player_placements[unique_placement])
        for player in player_placements[unique_placement]:
            standings.append((player, starting_placement, starting_placement + num_same - 1))
        starting_placement += num_same
    return standings


@dataclass
class ContestRatingParams:
    '''
    Dataclass for the individual contests of the elo rating systems.

    Args:
        weight (:obj:`float`): Weight of the contest.
        per_ceiling (:obj:`float`): Performance ceiling (maximum performance).
        per_floor (:obj:`float`): Performance floor (minimum performance).

    '''
    weight: float = 1.
    perf_ceiling: float = float('inf')
    perf_floor: float = -float('inf')


def eval_less(term: TanhTerm, x: float) -> tuple[float, float]:
    '''
    Given a tanh term and a value, compute the new val and derivative. Generally
    used to compute the minimum of a function. This is used for rating less than.

    Args:
        term (:obj:`TanhTerm`): Tanh term object.

        x (:obj:`float`): Value of x in minimization function.

    Returns:
        :obj:`tuple[float, float]`: the compute value and the derivative.
    '''
    val, val_prime = term.base_values(x)
    return val - term.w_out, val_prime


def eval_grea(term: TanhTerm, x: float) -> tuple[float, float]:
    '''
    Given a tanh term and a value, compute the new val and derivative. Generally
    used to compute the minimum of a function. This is used for rating greater than.

    Args:
        term (:obj:`TanhTerm`): Tanh term object.

        x (:obj:`float`): Value of x in minimization function.

    Returns:
        :obj:`tuple[float, float]`: the compute value and the derivative.
    '''
    val, val_prime = term.base_values(x)
    return val + term.w_out, val_prime


def eval_equal(term: TanhTerm, x: float, mul: float) -> tuple[float, float]:
    '''
    Given a tanh term and a value, compute the new val and derivative. Generally
    used to compute the minimum of a function. This is used for rating equal.

    Args:
        term (:obj:`TanhTerm`): Tanh term object.

        x (:obj:`float`): Value of x in minimization function.

    Returns:
        :obj:`tuple[float, float]`: the compute value and the derivative.
    '''
    val, val_prime = term.base_values(x)
    return mul * val, mul * val_prime


@dataclass
class EloMMRVariant:
    '''
    Dataclass to mimic a enum class with "Gaussian" and "Logistic variants.
    Only logistic has a value.

    Args:
        variant_type (:obj:`Literal['Gaussian'] | Literal['Logistic']`): Either
        Gaussian or Logistic variant.

        value (:obj:`float`): Value for logistic variant. Does not apply to Gaussian.
    '''
    variant_type: Literal['Gaussian'] | Literal['Logistic']
    value: Optional[float] = field(default=None, compare=False)

    @classmethod
    def gaussian(cls):
        '''
        Class method to return a Gaussian variant.
        '''
        return cls("Gaussian")

    @classmethod
    def logistic(cls, value: Optional[float] = None):
        '''
        Class method to return a logistic variant with a value.
        '''
        return cls("Logistic", value)


def find_left_partial(rankings: list[Any], rank: int, key: Callable[..., int]) -> list[Any]:
    '''
    Given a list of rankings and a rank with a key, return a sub array of rankings
    of the left partial (highest rank of a set of team that is lower than rank). 
    This means teams that are tied in this partial will be included in this set.

    Args:
        rankings (:obj:`list[Any]`): List of rankings.

        rank (:obj:`int`): Rank to find the partial.

        key (:obj:`Callable[..., int]`): function to compare the rankings and rank.

    Returns:
        :obj:`list[Any]`: Sub array of ranking of the left partial.
    '''
    if not rankings:
        return []
    left_i = bisect_left(rankings, rank, key=key)
    if left_i == 0:
        return []
    else:
        left_val = key(rankings[left_i - 1])
        left = bisect_left(rankings, left_val, key=key)
        right = bisect_right(rankings, left_val, key=key)
        return rankings[left:right]


def find_right_partial(rankings: list[Any], rank: int, key: Callable[..., int]) -> list[Any]:
    '''
    Given a list of fankings and a rank with a key, reutrn a sub array of rankings
    of the right partial (lowest rank of a set of team that is higher than rank).
    This means teams that are tied in this partial will be included in the set.

    Args:
        rankings (:obj:`list[Any]`): List of rankings.

        rank (:obj:`int`): Rank to find the partial.

        key (:obj:`Callable[..., int]`): function to compare the rankings and rank.

    Returns:
        :obj:`list[Any]`: Sub array of ranking of the right partial.
    '''
    if not rankings:
        return []
    right_i = bisect_right(rankings, rank, key=key)
    if right_i == len(rankings):
        return []
    else:
        right_value = key(rankings[right_i])
        left = bisect_left(rankings, right_value, key=key)
        right = bisect_right(rankings, right_value, key=key)
        return rankings[left:right]


def total_partial(rankings: list[Any], rank: int, key: Callable[..., int]) -> list[Any]:
    '''
    Given a list of rankings and a rank with a key, return a sub array of rankings
    of the left and right partial (lowest/highest of higher/lower ranks, respectively).
    This means teams that are tied in this partial will be included in the set.

    Args:
        rankings (:obj:`list[Any]`): List of rankings.

        rank (:obj:`int`): Rank to find the partial.

        key (:obj:`Callable[..., int]`): function to compare the rankings and rank.

    Returns:
        :obj:`list[Any]`: Sub array of ranking of the left and right partials.
    '''
    return find_left_partial(rankings, rank, key) + find_right_partial(rankings, rank, key)


def ranks_lt(rankings: list[Any], rank: int, key: Callable[..., int]) -> list[Any]:
    '''
    Given a list of rankings and a rank with a key, return a sub array of rankings
    of the ranking that are less than rank.

    Args:
        rankings (:obj:`list[Any]`): List of rankings.

        rank (:obj:`int`): Rank to find sub array less than this value.

        key (:obj:`Callable[..., int]`): function to compare the rankings and rank.

    Returns:
        :obj:`list[Any]`: Sub array of ranking less than rank.
    '''
    left_i = bisect_left(rankings, rank, key=key)
    if left_i == 0:
        return []
    return rankings[0:left_i]


def ranks_le(rankings: list[Any], rank: int, key: Callable[..., int]) -> list[Any]:
    '''
    Given a list of rankings and a rank with a key, return a sub array of rankings
    of the ranking that are less than or equal to rank.

    Args:
        rankings (:obj:`list[Any]`): List of rankings.

        rank (:obj:`int`): Rank to find sub array less than or equal to this value.

        key (:obj:`Callable[..., int]`): function to compare the rankings and rank.

    Returns:
        :obj:`list[Any]`: Sub array of ranking less than or equal to rank.
    '''
    right_i = bisect_right(rankings, rank, key=key)
    if right_i == 0:
        return []
    return rankings[:right_i]


def ranks_gt(rankings: list[Any], rank: int, key: Callable[..., int]) -> list[Any]:
    '''
    Given a list of rankings and a rank with a key, return a sub array of rankings
    of the ranking that are greater than rank.

    Args:
        rankings (:obj:`list[Any]`): List of rankings.

        rank (:obj:`int`): Rank to find sub array greater than this value.

        key (:obj:`Callable[..., int]`): function to compare the rankings and rank.

    Returns:
        :obj:`list[Any]`: Sub array of ranking greater than rank.
    '''
    right_i = bisect_right(rankings, rank, key=key)
    if right_i == len(rankings):
        return []
    return rankings[right_i:]


def ranks_ge(rankings: list[Any], rank: int, key: Callable[..., int]) -> list[Any]:
    '''
    Given a list of rankings and a rank with a key, return a sub array of rankings
    of the ranking that are greater than or equal to rank.

    Args:
        rankings (:obj:`list[Any]`): List of rankings.

        rank (:obj:`int`): Rank to find sub array greater than or equal to this value.

        key (:obj:`Callable[..., int]`): function to compare the rankings and rank.

    Returns:
        :obj:`list[Any]`: Sub array of ranking greater than or eqaul to rank.
    '''
    left_i = bisect_left(rankings, rank, key=key)
    if left_i == len(rankings):
        return []
    return rankings[left_i:]


def ranks_eq(rankings: list[Any], rank: int, key: Callable[..., int]) -> list[Any]:
    '''
    Given a list of rankings and a rank with a key, return a sub array of rankings
    of the ranking that are less than rank.

    Args:
        rankings (:obj:`list[Any]`): List of rankings.

        rank (:obj:`int`): Rank to find sub array less than this value.

        key (:obj:`Callable[..., int]`): function to compare the rankings and rank.

    Returns:
        :obj:`list[Any]`: Sub array of ranking equal to rank.
    '''
    left_i = bisect_left(rankings, rank, key=key)
    right_i = bisect_right(rankings, rank, key=key)
    if left_i == len(rankings):
        return []
    return rankings[left_i:right_i]
