from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Final, Hashable, Literal, TypedDict

import numpy as np

from team_system import TeamSystem, TeamSystemSettings


__all__ = ['TeamBanParams',
           'TeamBan']


# default of platform
default_float = np.float64
default_int = np.int64
default_uint = np.uint64

# non-numpy types
type Player = tuple[Hashable, int | float]
type Players = list[tuple[Hashable, int | float]]
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
class TeamBanParams(TeamSystemSettings): ...


class TeamBanInfoDict[T_Rating: Rating = default_float, T_Index: Index = default_uint](TypedDict):
    players_elo_sorted: PlayersStruct[T_Rating]
    game_statistics: GameStats[T_Rating]
    game_combinations: Games[T_Rating]
    game_combinations_ind: Indices[T_Index]
    game_ban_ind: Indices[T_Index]
    all_constraints_ind: Indices[T_Index]


class TeamBan[T_Rating: Rating = default_float, T_Index: Index = default_uint](TeamSystem[T_Rating, T_Index]):
    def __init__(
        self,
        settings: TeamBanParams,
        players: Players | PlayersStruct[T_Rating] | None = None,
        bans: defaultdict[Player, set[Player]] = defaultdict(set),
    ) -> None:
        super().__init__(settings, players)
        self.bans = bans

    @staticmethod
    def _ban_player_constraint(
        team0: Team[T_Rating], team1: Team[T_Rating], bans: defaultdict[Player, set[Player]] = defaultdict(set)
    ) -> bool:
        for player1, player2 in combinations(team0["player"], 2):
            if player2 in bans[player1]:
                return False
        for player1, player2 in combinations(team1["player"], 2):
            if player2 in bans[player1]:
                return False
        return True

    def _games_ban_ind(
        self, games: Games[T_Rating], bans: defaultdict[Player, set[Player]] = defaultdict(set)
    ) -> Indices[T_Index]:
        games_played_with_ind = np.where([TeamBan._ban_player_constraint(game[0], game[1], bans) for game in games])
        return np.array(games_played_with_ind[0], dtype=self.index_dtype)

    def info(self) -> TeamBanInfoDict[T_Rating, T_Index]:
        info = super().info()
        game_combs = info["game_combinations"]
        all_ind = np.arange(len(game_combs), dtype=self.index_dtype)
        indices = [all_ind]

        info["game_ban_ind"] = self._games_ban_ind(game_combs, self.bans)
        indices.append(info["game_ban_ind"])
        info["all_constraints_ind"] = self.intersection(*indices)
        return info
