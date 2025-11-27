"""
Elo Team Balancer
Author: Beniamin#9574 (Beniamin Condrea)
Written 3/2/2023
"""

from collections import Counter
from dataclasses import dataclass
from itertools import batched
from typing import Final, Hashable, Literal, Sequence, TypedDict
from warnings import deprecated

import numpy as np
import numpy.typing as npt

from team_system import TeamSystem, TeamSystemInfoDict, TeamSystemSettings


__all__ = ['TeamBalancerParams',
           'TeamBalancer']

# default of platform
default_float = np.float64
default_int = np.int64
default_uint = np.uint64

# non-numpy types
type Player = tuple[Hashable, int | float]
type Players = Sequence[tuple[Hashable, int | float]]
# numpy types
type Rating = np.integer | np.floating  # signed, unsigned, floating (not complex)
type Index = np.integer
type RatingDType[T: Rating = default_float] = np.dtype[T]
type IndexDtype[T: Index = default_uint] = np.dtype[T]

type PlayerStructDType[T: Rating = default_float] = np.dtype[
    np.void
    # list[tuple[Literal["player"], np.object_], tuple[Literal["rating"], T]]
]
# team 0 sum, team 1 sum, team0 avg, team1 avg, elo diff, elo abs
type GameStatsDType[T: Rating = default_float] = np.dtype[
    np.void
    # list[
    #     tuple[Literal["team0_sum"], T],
    #     tuple[Literal["team1_sum"], T],
    #     tuple[Literal["team0_avg"], T],
    #     tuple[Literal["team1_avg"], T],
    #     tuple[Literal["rating_diff"], T],
    #     tuple[Literal["rating_abs"], T],
    # ]
]
type PlayersStruct[T: Rating = default_float] = np.ndarray[tuple[int], PlayerStructDType[T]]
type Team[T: Rating = default_float] = np.ndarray[tuple[int], PlayerStructDType[T]]
type Teams[T: Rating = default_float] = np.ndarray[tuple[int, int], PlayerStructDType[T]]
type Game[T: Rating = default_float] = np.ndarray[tuple[Literal[2], int], PlayerStructDType[T]]
type Games[T: Rating = default_float] = np.ndarray[tuple[int, Literal[2], int], PlayerStructDType[T]]
type GameStats[T: Rating = default_float] = np.ndarray[tuple[int], GameStatsDType[T]]
type Indices[T: np.integer = default_uint] = np.ndarray[tuple[int], np.dtype[T]]
# default player struct
default_player_np_dt: Final = np.dtype([("player", np.object_), ("rating", default_float)])
default_game_stats_np_dt: Final = np.dtype(
    [
        ("team0_sum", default_float),
        ("team1_sum", default_float),
        ("team0_avg", default_float),
        ("team1_avg", default_float),
        ("rating_diff", default_float),
        ("rating_abs", default_float),
    ]
)


class TeamBalancerParamsProtocol(TeamSystemSettings):
    top_k: int
    elo_diff_range: tuple[float, float]
    player_balance: bool


@dataclass
class TeamBalancerParams(TeamSystemSettings):
    top_k: int = 50
    elo_diff_range: tuple[float, float] = (0.0, 200.0)
    player_balance: bool = True


class TeamBalancerInfoDict[T_Rating: Rating = default_float, T_Index: Index = default_uint](
    TeamSystemInfoDict, TypedDict
):
    best_games_partition_ind: Indices[T_Index]
    player_balance_ind: Indices[T_Index]
    rating_abs_ind: Indices[T_Index]
    all_constraints_ind: Indices[T_Index]


class TeamBalancer[T_Rating: Rating = default_float, T_Index: Index = default_uint](TeamSystem[T_Rating, T_Index]):
    def __init__(
        self,
        settings: TeamBalancerParams,
        players: npt.ArrayLike | None = None,
    ) -> None:
        super().__init__(settings, players)

    def _sets_of_two_balance_constraint(self, team0: Team[T_Rating], team1: Team[T_Rating]) -> bool:
        """
        Check if two teams are balanced according to the sets-of-two constraint.

        Parameters
        ----------
        team0 : Team[T_Rating]
            First team to balance against.
        team1 : Team[T_Rating]
            Second team to balance against.

        Returns
        -------
        bool
            True if the teams meet the balance constraint, False otherwise.
        """
        # get counters
        c0 = Counter(team0["rating"])
        c1 = Counter(team1["rating"])
        all_c = c0 + c1
        dups = {k for k, v in all_c.items() if v > 1}
        # sort players by rating and player id (to ensure consistent ordering)
        all_players = np.concatenate((team0, team1), dtype=self.struct_dtype)
        all_players.sort(order=("rating", "player"))
        for p1, p2 in batched(all_players, 2):
            # if they are in the same team
            if ((p1 in team0) and (p2 in team0)) or ((p1 in team1) and (p2 in team1)):
                # if either are not in duplicates (if both are uniuqe), return false
                if not (p1["rating"] in dups or p2["rating"] in dups):
                    return False
        # if there are duplicates, they must be "evenly spread out" to each team
        c0.subtract(c1)
        if any(abs(diff) > 1 for diff in c0.values()):
            return False
        return True

    def _players_balance_partition_constraint(self, games: Games[T_Rating]) -> Indices[T_Index]:
        balanced_ind = np.where([self._sets_of_two_balance_constraint(game[0], game[1]) for game in games])
        return np.array(balanced_ind[0], dtype=self.index_dtype)

    def _partition_k_teams_ind(self, games: Games[T_Rating], k: int) -> Indices[T_Index]:
        k = k if k < len(games) else len(games)
        elos_abs = self.create_games_stats(games)["rating_abs"]
        best_k_elos_partition = np.argpartition(elos_abs, k - 1)[:k]
        return np.array(best_k_elos_partition, dtype=self.index_dtype)

    def _teams_abs_constraint(self, games: Games[T_Rating], diff_range: tuple[float, float]) -> Indices[T_Index]:
        elos_abs = self.create_games_stats(games)["rating_abs"]
        low, high = diff_range
        diff_ind = np.where((low <= elos_abs) & (elos_abs <= high))
        return np.array(diff_ind[0], dtype=self.index_dtype)

    def info(self) -> TeamBalancerInfoDict[T_Rating, T_Index]:
        info = super().info()
        game_combs = info["game_combinations"]
        all_ind = np.arange(len(game_combs), dtype=self.index_dtype)
        indices = [all_ind]

        if self.settings.top_k:
            info["best_games_partition_ind"] = self._partition_k_teams_ind(game_combs, self.settings.top_k)
            indices.append(info["best_games_partition_ind"])
        else:
            info["best_games_partition_ind"] = None
        if self.settings.player_balance is not False:
            info["player_balance_ind"] = self._players_balance_partition_constraint(game_combs)
            indices.append(info["player_balance_ind"])
        else:
            info["player_balance_ind"] = None
        if self.settings.elo_diff_range is not None:
            info["rating_abs_ind"] = self._teams_abs_constraint(game_combs, self.settings.elo_diff_range)
            indices.append(info["rating_abs_ind"])
        else:
            info["rating_abs_ind"] = None
        info["all_constraints_ind"] = self.intersection(*indices)
        return info

    @deprecated("Redundant method. Use create_game_stats instead")
    def get_elo_game_statistics(self, game: Game[T_Rating]) -> GameStats[T_Rating]:
        return self.create_game_stats(game)

    def get_best_game(
        self,
    ) -> tuple[Game[T_Rating], GameStats[T_Rating]] | tuple[None, None]:
        info = self.info()
        if len(info["all_constraints_ind"]) > 0:
            best_ind = np.argmin(info["game_statistics"][info["all_constraints_ind"]]["rating_abs"])
            best_game = info["game_combinations"][info["all_constraints_ind"]][best_ind]
            best_stats = info["game_statistics"][info["all_constraints_ind"]][best_ind]
            return best_game, best_stats
        return None, None
