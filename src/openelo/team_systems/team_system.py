import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property, partial, reduce
from itertools import combinations
from math import comb
from typing import Final, Hashable, Protocol, Sequence, TypedDict

import numpy as np
import numpy.typing as npt

__all__ = ['TeamSystem']

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
type PlayersStruct[T: Rating = default_float] = np.ndarray[tuple[int, ...], PlayerStructDType[T]]
type Team[T: Rating = default_float] = np.ndarray[tuple[int, ...], PlayerStructDType[T]]
type Teams[T: Rating = default_float] = np.ndarray[tuple[int, ...], PlayerStructDType[T]]
type Game[T: Rating = default_float] = np.ndarray[tuple[int, ...], PlayerStructDType[T]]
type Games[T: Rating = default_float] = np.ndarray[tuple[int, ...], PlayerStructDType[T]]
type GameStats[T: Rating = default_float] = np.ndarray[tuple[int, ...], GameStatsDType[T]]
type Indices[T: np.integer = default_uint] = np.ndarray[tuple[int, ...], np.dtype[T]]
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
class TeamSystemSettings(Protocol): ...


class AbstractTeamSystem[T_Rating: Rating = default_float, T_Index: Index = default_uint](ABC):

    @property
    @abstractmethod
    def players(self) -> PlayersStruct[T_Rating] | None:
        raise NotImplementedError

    @players.setter
    @abstractmethod
    def players(self, players: npt.ArrayLike | None) -> None:
        raise NotImplementedError

    @abstractmethod
    def info(self) -> GameStats[T_Rating]: ...


class TeamSystemInfoDict[T_Rating: Rating = default_float, T_Index: Index = default_uint](TypedDict):
    players_elo_sorted: PlayersStruct[T_Rating]
    game_statistics: GameStats[T_Rating]
    game_combinations: Games[T_Rating]
    game_combinations_ind: Indices[T_Index]


class TeamSystem[T_Rating: Rating = default_float, T_Index: Index = default_uint](AbstractTeamSystem):

    def __init__(
        self,
        settings: TeamSystemSettings | None = None,
        players: npt.ArrayLike | None = None,
    ) -> None:
        # public attributes
        self.settings = settings
        self.players = players

    @property
    def players(self) -> PlayersStruct[T_Rating] | None:
        return self._players

    def __verify_players(self, players: npt.ArrayLike | PlayersStruct[T_Rating] | None) -> bool:
        if isinstance(players, (list, tuple)):
            if not all(
                isinstance(player, tuple)
                and len(player) == 2
                and isinstance(player[0], Hashable)
                and isinstance(player[1], (int, float))
                for player in players
            ):
                return False
            return True
        elif isinstance(players, np.ndarray):
            dtype: npt.DTypeLike = players.dtype
            if dtype == self.dtype:
                return False
            player_dtype = players["player"].dtype
            rating_dtype = players["rating"].dtype
            if not np.issubdtype(player_dtype, np.object_) or not np.issubdtype(rating_dtype, np.number):
                return False
            return True
        return False

    @players.setter
    def players(self, players: npt.ArrayLike | None) -> None:
        if players is None:
            self._players = None
            return
        if not self.__verify_players(players):
            raise InvalidPlayers(
                f"Input player data is {players}. The player list must be a list of tuples or a numpy array with the correct dtype!"
            )
        n: int = len(players)
        if n % 2 == 1 or n < 3:
            raise InvalidPlayerCount(f"Input player count is {n}. The player list must be even and greater than 3!")

        self._players = np.array(players, self.struct_dtype)

    @cached_property
    def dtype(self) -> np.dtype[T_Rating]:
        if hasattr(self, "__orig_class__"):
            generic_type = self.__orig_class__.__args__[0]
        else:
            generic_type = T_Rating.__default__
        return np.dtype(generic_type)

    @cached_property
    def index_dtype(self) -> np.dtype[T_Index]:
        if hasattr(self, "__orig_class__"):
            generic_type = self.__orig_class__.__args__[1]
        else:
            generic_type = T_Index.__default__
        return np.dtype(generic_type)

    @cached_property
    def struct_dtype(self) -> PlayerStructDType[T_Rating]:
        if hasattr(self, "__orig_class__"):
            generic_type = self.__orig_class__.__args__[0]
        else:
            generic_type = T_Rating.__default__
        return np.dtype([("player", np.object_), ("rating", generic_type)])

    @cached_property
    def game_stats_dtype(self) -> GameStatsDType[T_Rating]:
        if hasattr(self, "__orig_class__"):
            generic_type = self.__orig_class__.__args__[0]
        else:
            generic_type = T_Rating.__default__
        # team 0 sum, team 1 sum, team0 avg, team1 avg, elo diff, elo abs
        return np.dtype(
            [
                ("team0_sum", generic_type),
                ("team1_sum", generic_type),
                ("team0_avg", generic_type),
                ("team1_avg", generic_type),
                ("rating_diff", generic_type),
                ("rating_abs", generic_type),
            ]
        )

    def create_games_combs(self) -> Games[T_Rating]:
        if self.players is None:
            raise InvalidPlayers("Players are not set!")
        total_players: int = len(self.players)
        if total_players % 2 == 1 or total_players < 2:
            raise InvalidPlayerCount(
                f"Input player count is {total_players}. The player list must be even and greater than 3!"
            )
        players_per_team: int = total_players // 2
        half_comb: int = comb(total_players, players_per_team) // 2
        array_copy = np.array(self.players)
        combs = list(combinations(array_copy, players_per_team))
        all_comb = np.array(combs)
        teams0_comb_list = all_comb[:half_comb]
        teams1_not0_list = all_comb[half_comb:][::-1]
        stacked = np.stack((teams0_comb_list, teams1_not0_list), axis=1)
        return stacked

    def create_games_combs_ind(self) -> Indices[T_Index]:
        player_n = len(self.players)
        players_ind = np.arange(player_n)
        half_ind = player_n // 2
        half_comb = comb(player_n, half_ind) // 2
        all_comb = np.array(list(combinations(np.array(players_ind), half_ind)), dtype=self.index_dtype)
        teams0_comb_list = all_comb[0:half_comb]
        teams1_not0_list = all_comb[half_comb:][::-1]
        stacked = np.stack((teams0_comb_list, teams1_not0_list), axis=1)
        return stacked

    def sort_players_by_rating(self) -> PlayersStruct[T_Rating]:
        players_array = np.array(self.players, dtype=self.struct_dtype)
        players_array.sort(order=("rating", "player"))
        return players_array

    @staticmethod
    def intersection(*indices: Indices[T_Index]) -> Indices[T_Index]:
        intersect_rating_unique = partial(np.intersect1d, assume_unique=True)
        return reduce(intersect_rating_unique, indices)

    # creates a single structure
    def create_game_stats(self, game: Game[T_Rating]) -> GameStats[T_Rating]:
        with warnings.catch_warnings(record=True) as w:
            team0, team1 = game[0], game[1]
            team0_elo_sum, team0_elo_avg = self.create_team_stats(team0)
            team1_elo_sum, team1_elo_avg = self.create_team_stats(team1)
            elo_diff = team0_elo_sum - team1_elo_sum
            elo_abs = np.abs(elo_diff, dtype=self.dtype)
            # below must be a tuple, not a list
            game_stats = np.array(
                (
                    team0_elo_sum,
                    team1_elo_sum,
                    team0_elo_avg,
                    team1_elo_avg,
                    elo_diff,
                    elo_abs,
                ),
                dtype=self.game_stats_dtype,
            )
            # runtime warning handling, particularly for summations that might overflow
            for warning in w:
                if issubclass(warning.category, RuntimeWarning):
                    print("Overflow warning detected:", warning.message)
        return game_stats

    def create_team_stats(self, team: Team[T_Rating]) -> tuple[T_Rating, T_Rating]:
        with warnings.catch_warnings(record=True) as w:
            ratings = team["rating"]
            elo_sum = np.sum(ratings, dtype=self.dtype)
            elo_avg = np.mean(ratings, dtype=self.dtype)
            # note: without rounding, truncates to lower integer
            # if np.issubdtype(self.dtype, np.integer):
            #     elo_avg = np.round(elo_avg)
            for warning in w:
                if issubclass(warning.category, RuntimeError):
                    print("Overflow warning detected:", warning.message)
        return elo_sum, elo_avg

    def create_games_stats(self, games: Games[T_Rating]) -> GameStats[T_Rating]:
        return np.array(
            [self.create_game_stats(game) for game in games],
            dtype=self.game_stats_dtype,
        )

    def info(self) -> TeamSystemInfoDict[T_Rating, T_Index]:
        info = {}
        info["players_elo_sorted"] = self.sort_players_by_rating()
        games_comb = self.create_games_combs()
        info["game_combinations"] = games_comb
        info["game_statistics"] = self.create_games_stats(games_comb)
        info["game_combinations_ind"] = self.create_games_combs_ind()
        return info


class InvalidPlayerCount(Exception):
    """
    An exception class for error handling odd teams.
    """

    ...


class InvalidPlayers(Exception):
    """
    An exception class for error handling invalid teams.
    """

    ...

