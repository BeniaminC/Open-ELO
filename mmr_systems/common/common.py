import concurrent.futures
from abc import ABC, abstractmethod
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, TypedDict

from mmr_systems.common.numericals import SECS_PER_DAY
from mmr_systems.common.player import Player
from mmr_systems.common.term import Rating, TanhTerm


def convert_placement_to_standings(placements: list[tuple[Player, int]]) -> list[tuple[Player, int, int]]:
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
    weight: float = 1.
    perf_ceiling: float = float('inf')
    perf_floor: float = -float('inf')


class RatingSystem(ABC):

    @abstractmethod
    def round_update(self, params: ContestRatingParams, standings: list[tuple[Player, int, int]]):
        pass

    @staticmethod
    def init_players_event(standings: list[tuple[Player, int, int]], contest_time: int = 0):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for player, lo, _ in standings:
                executor.submit(player.init_player_event, lo, contest_time)

    def compute_weight(self, contest_weight: float, weight_limit: float, noob_delay: list[float], n: int) -> float:
        computed_weight = contest_weight * weight_limit
        if n < len(noob_delay):
            computed_weight *= noob_delay[n]
        return computed_weight

    def compute_sig_perf(self, weight: float, sig_limit: float, drift_per_day: float) -> float:
        discrete_perf = (1. + 1. / weight) * sig_limit * sig_limit
        continuous_perf = drift_per_day / weight
        return (discrete_perf + continuous_perf) ** 0.5

    def compute_sig_drift(self, weight: float, sig_limit: float, drift_per_day: float, delta_secs: float, ) -> float:
        discrete_drift = weight * sig_limit * sig_limit
        continuous_drift = drift_per_day * delta_secs / SECS_PER_DAY
        return (discrete_drift + continuous_drift) ** 0.5


class TeamRatingAggregation(ABC):
    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Rating:
        pass


class TeamSumAggregation(TeamRatingAggregation):
    def __call__(self, players: list[Player]) -> Rating:
        return Rating(sum(player.approx_posterior.mu for player in players),
                      sum(player.approx_posterior.sig ** 2 for player in players))


class TeamAverageAggregation(TeamRatingAggregation):
    def __call__(self, players: list[Player]) -> Rating:
        return Rating(sum(player.approx_posterior.mu for player in players) / len(players),
                      sum(player.approx_posterior.sig ** 2 for player in players) / len(players))


class TeamMaxAggregation(TeamRatingAggregation):
    def __call__(self, players: list[Player]) -> Rating:
        max_player = max(players, key=lambda player: player.approx_posterior.mu)
        return Rating(max_player.approx_posterior.mu,
                      max_player.approx_posterior.sig ** 2)


class TeamMinAggregation(TeamRatingAggregation):
    def __call__(self, players: list[Player]) -> Rating:
        min_player = min(players, key=lambda player: player.approx_posterior.mu)
        return Rating(min_player.approx_posterior.mu,
                      min_player.approx_posterior.sig ** 2)


class TeamAverageAggregationN(TeamRatingAggregation):
    def __init__(self, n: int = 3, best: bool = True) -> None:
        self.n = n
        self.best = best

    def __call__(self, players: list[Player]) -> Rating:
        if self.n > len(players):
            raise IndexError()
        sorted_players = sorted(players, key=lambda player: player.approx_posterior.mu)
        n_players = sorted_players[-self.n:] if self.best else sorted_players[:self.n]
        return Rating(sum(player.approx_posterior.mu for player in n_players) / len(n_players),
                      sum(player.approx_posterior.sig ** 2 for player in n_players) / len(n_players))


class TeamSumAggregationN(TeamRatingAggregation):
    def __init__(self, n: int = 3, best: bool = True) -> None:
        self.n = n
        self.best = best

    def __call__(self, players: list[Player]) -> Rating:
        if self.n > len(players):
            raise IndexError()
        sorted_players = sorted(players, key=lambda player: player.approx_posterior.mu)
        n_players = sorted_players[-self.n:] if self.best else sorted_players[:self.n]
        return Rating(sum(player.approx_posterior.mu for player in n_players),
                      sum(player.approx_posterior.sig ** 2 for player in n_players))


TeamInfo = TypedDict('TeamInfo', {'players': list, 'rank': int})


class TeamRatingSystem(ABC):
    @abstractmethod
    def team_round_update(self,
                          params: ContestRatingParams,
                          standings: list[tuple[Player, int, int]],
                          agg: TeamRatingAggregation):
        pass

    @staticmethod
    def convert_to_teams(standings: list[tuple[Player, int, int]]) -> dict[int, TeamInfo]:

        teams: dict[int, Any] = {}
        for player, team, rank in standings:
            if team not in teams:
                teams[team] = {}
                teams[team]['players'] = []
                teams[team]['rank'] = rank
            teams[team]['players'].append(player)
        return teams

    @staticmethod
    def team_individual_update(player: Player, team_sig_sq: float, omega: float, delta_i: float, kappa: float):
        old_mu = player.approx_posterior.mu
        old_sig_sq = player.approx_posterior.sig ** 2
        sig_ratio = (old_sig_sq / team_sig_sq)
        new_mu = old_mu + (sig_ratio) * omega
        new_sig = ((old_sig_sq) * max(1 - sig_ratio * delta_i, kappa)) ** 0.5
        player.update_rating(Rating(new_mu, new_sig), 0.)

def eval_less(term: TanhTerm, x: float) -> tuple[float, float]:
    val, val_prime = term.base_values(x)
    return val - term.w_out, val_prime


def eval_grea(term: TanhTerm, x: float) -> tuple[float, float]:
    val, val_prime = term.base_values(x)
    return val + term.w_out, val_prime


def eval_equal(term: TanhTerm, x: float, mul: float) -> tuple[float, float]:
    val, val_prime = term.base_values(x)
    return mul * val, mul * val_prime


@dataclass
class EloMMRVariant:
    variant_type: Literal['Gaussian'] | Literal['Logistic']
    value: Optional[float] = field(default=None, compare=False)

    @classmethod
    def gaussian(cls):
        return cls("Gaussian")

    @classmethod
    def logistic(cls, value: Optional[float] = None):
        return cls("Logistic", value)


def find_left_partial(rankings: list[Any], rank: int, key: Callable[..., int]) -> list[Any]:
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
    return find_left_partial(rankings, rank, key) + find_right_partial(rankings, rank, key)


def ranks_lt(rankings: list[Any], rank: int, key: Callable[..., int]) -> list[Any]:
    left_i = bisect_left(rankings, rank, key=key)
    if left_i == 0:
        return []
    return rankings[0:left_i]


def ranks_le(rankings: list[Any], rank: int, key: Callable[..., int]) -> list[Any]:
    right_i = bisect_right(rankings, rank, key=key)
    if right_i == 0:
        return []
    return rankings[:right_i]


def ranks_gt(rankings: list[Any], rank: int, key: Callable[..., int]) -> list[Any]:
    right_i = bisect_right(rankings, rank, key=key)
    if right_i == len(rankings):
        return []
    return rankings[right_i:]


def ranks_ge(rankings: list[Any], rank: int, key: Callable[..., int]) -> list[Any]:
    left_i = bisect_left(rankings, rank, key=key)
    if left_i == len(rankings):
        return []
    return rankings[left_i:]


def ranks_eq(rankings: list[Any], rank: int, key: Callable[..., int]) -> list[Any]:
    left_i = bisect_left(rankings, rank, key=key)
    right_i = bisect_right(rankings, rank, key=key)
    if left_i == len(rankings):
        return []
    return rankings[left_i:right_i]

