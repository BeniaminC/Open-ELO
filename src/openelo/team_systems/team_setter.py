"""
Elo Team Balancer
Author: Beniamin#9574 (Beniamin Condrea)
Written 3/2/2023
"""

from dataclasses import dataclass, field
from typing import Final, Hashable, Literal, Sequence, TypedDict
from warnings import deprecated

import numpy as np

from team_system import TeamSystem, TeamSystemInfoDict, TeamSystemSettings

__all__ = ['TeamSetterParams',
           'TeamSetter']

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


@dataclass
class TeamSetterParams(TeamSystemSettings):
    subset_player_sets: list[set[str]] | tuple[set[str]] = field(default_factory=list)
    not_subset_player_sets: list[set[str]] | tuple[set[str]] = field(default_factory=list)


class TeamSetterInfoDict[T_Rating: Rating = default_float, T_Index: Index = default_uint](
    TeamSystemInfoDict, TypedDict
):
    subset_player_sets_ind: Indices[T_Index]
    not_subset_player_sets_ind: Indices[T_Index]
    all_constraints_ind: Indices[T_Index]


# class for set operations
class TeamSetter[T_Rating: Rating = default_float, T_Index: Index = default_uint](TeamSystem[T_Rating, T_Index]):
    def __init__(
        self,
        settings: TeamSetterParams,
        players: Players | PlayersStruct[T_Rating] | None = None,
    ) -> None:
        super().__init__(settings, players)

    # NOTE: returns true if there are no player subsets in the game
    def _subset_set_in_team(
        self, team0: Team[T_Rating], team1: Team[T_Rating], player_sets: Sequence[set[str]]
    ) -> bool:
        team0_set, team1_set = set(team0["player"]), set(team1["player"])
        game_set = team0_set | team1_set
        filtered_player_sets = (player_set for player_set in player_sets if player_set <= game_set)
        return all(player_set <= team0_set or player_set <= team1_set for player_set in filtered_player_sets)

    def _subset_player_set_constraint(
        self, games: Games[T_Rating], player_sets: Sequence[set[str]]
    ) -> Indices[T_Index]:
        subset_player_set_ind = np.where([self._subset_set_in_team(game[0], game[1], player_sets) for game in games])
        return np.array(subset_player_set_ind[0], dtype=self.index_dtype)

    def _not_subset_set_in_team(
        self, team0: Team[T_Rating], team1: Team[T_Rating], player_sets: Sequence[set[str]]
    ) -> bool:
        team0_set, team1_set = set(team0["player"]), set(team1["player"])
        game_set = team0_set | team1_set
        filtered_player_sets = (player_set for player_set in player_sets if player_set <= game_set)
        return not any((player_set <= team0_set or player_set <= team1_set) for player_set in filtered_player_sets)

    def _not_subset_player_set_constraint(
        self, games: Games[T_Rating], not_player_sets: Sequence[set[str]]
    ) -> Indices[T_Index]:
        not_subset_player_set_ind = np.where(
            [self._not_subset_set_in_team(game[0], game[1], not_player_sets) for game in games]
        )
        return np.array(not_subset_player_set_ind[0], dtype=self.index_dtype)

    def info(self) -> TeamSetterInfoDict[T_Rating, T_Index]:
        info = super().info()
        game_combs = info["game_combinations"]
        all_ind = np.arange(len(game_combs), dtype=self.index_dtype)
        indices = [all_ind]

        if len(self.settings.subset_player_sets):
            info["subset_player_sets_ind"] = self._subset_player_set_constraint(
                game_combs, self.settings.subset_player_sets
            )
            indices.append(info["subset_player_sets_ind"])
        else:
            info["subset_player_sets_ind"] = None
        if len(self.settings.not_subset_player_sets):
            info["not_subset_player_sets_ind"] = self._not_subset_player_set_constraint(
                game_combs, self.settings.not_subset_player_sets
            )
            indices.append(info["not_subset_player_sets_ind"])
        else:
            info["not_subset_player_sets_ind"] = None
        info["all_constraints_ind"] = self.intersection(*indices)
        return info

    @deprecated("Redundant method. Use create_game_stats instead")
    def get_elo_game_statistics(self, game: Game[T_Rating]) -> GameStats[T_Rating]:
        return self.create_game_stats(game)
