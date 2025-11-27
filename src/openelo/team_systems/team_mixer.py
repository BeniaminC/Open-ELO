from collections import defaultdict
from dataclasses import dataclass
from itertools import chain, combinations
from typing import Annotated, Final, Hashable, Literal, TypedDict

import numpy as np
from pandas.api.typing import DataFrameGroupBy
from pandas import DataFrame
from team_system import TeamSystem, TeamSystemInfoDict, TeamSystemSettings


__all__ = ['TeamMixerParams',
           'TeamMixer']


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
class TeamMixerParams(TeamSystemSettings):
    games_played_with_ratio: float = 0.5  # percentage played with a player
    players_last_played_with_ratio: float = 2.0 / 3.0  # don't play with last players


class TeamMixerInfoDict[T_Rating: Rating = default_float, T_Index: Index = default_uint](TeamSystemInfoDict, TypedDict):
    game_ratios_ind: Indices[T_Index]
    last_teammates_ind: Indices[T_Index]
    all_constraints_ind: Indices[T_Index]


class TeamMixer[T_Rating: Rating = default_float, T_Index: Index = default_uint](TeamSystem[T_Rating, T_Index]):
    def __init__(
        self,
        settings: TeamMixerParams,
        players: Players | PlayersStruct[T_Rating] | None = None,
        games: Games[T_Rating] | Annotated[DataFrame, DataFrameGroupBy] | None = None,
    ) -> None:
        super().__init__(settings, players)
        self.games = games

    @property
    def games(self) -> Games[T_Rating] | Annotated[DataFrame, DataFrameGroupBy] | None:
        return self._games

    @games.setter
    def games(self, games: Games[T_Rating] | Annotated[DataFrame, DataFrameGroupBy]) -> None:
        self._games = games

    @staticmethod
    def _games_played_with_ratio_constraint(
        team0: Team[T_Rating],
        team1: Team[T_Rating],
        ratios: defaultdict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float)),
        ratio_constraint: float = 0.5,
    ) -> bool:
        # the ratio of player1 to player2 IS NOT THE SAME AS the ratio of player2 to player1
        for player1, player2 in chain(combinations(team0["player"], 2), combinations(team1["player"], 2)):
            if ratios[player1][player2] > ratio_constraint or ratios[player2][player1] > ratio_constraint:
                return False
        return True

    def _games_played_with_ratio_ind(
        self,
        games: Games[T_Rating],
        ratios: defaultdict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float)),
        ratio_constraint: float = 0.5,
    ) -> Indices[T_Index]:
        games_played_with_ind = np.where(
            [
                TeamMixer._games_played_with_ratio_constraint(game[0], game[1], ratios, ratio_constraint)
                for game in games
            ]
        )
        return np.array(games_played_with_ind[0], dtype=self.index_dtype)

    @staticmethod
    def _players_last_played_with_constraint(
        team0: Team[T_Rating],
        team1: Team[T_Rating],
        last_played: defaultdict[str, list[str]] = defaultdict(list),
        ratio_constraint: float = 0.5,
    ) -> bool:
        players0 = team0["player"]
        for player in players0:
            player_last_played = last_played[player]
            intersection = np.intersect1d(players0, player_last_played, assume_unique=True)
            if len(intersection) / len(team0) > ratio_constraint:
                return False
        players1 = team1["player"]
        for player in players1:
            player_last_played = last_played[player]
            intersection = np.intersect1d(players1, player_last_played, assume_unique=True)
            if len(intersection) / len(team1) > ratio_constraint:
                return False
        return True

    def _players_last_played_with_ind(
        self,
        games: Games[T_Rating],
        last_played: defaultdict[str, list[str]] = defaultdict(list),
        ratio_constraint: float = 0.5,
    ) -> Indices[T_Index]:
        games_played_with_ind = np.where(
            [
                TeamMixer._players_last_played_with_constraint(game[0], game[1], last_played, ratio_constraint)
                for game in games
            ]
        )
        return np.array(games_played_with_ind[0], dtype=self.index_dtype)

    # helper function to create a defaultdict of game ratios
    # the data input
    # can make these dispatches
    @staticmethod
    def _compute_team_game_ratios(
        games: Games[T_Rating] | Annotated[DataFrame, DataFrameGroupBy],
        player_key: str = "player",
    ) -> defaultdict[str, defaultdict[str, float]]:
        # total games must start with 1, otherwise player ratio gets stuck at 100%, otherwise
        # each player would need to play with completely new players for the second game.
        # For example, same players in a new season (0 games) for two games.

        total_games: defaultdict[str, float] = defaultdict(lambda: 1.0)
        ratios: defaultdict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
        if isinstance(games, np.ndarray):
            for game in games:
                # each game has two separate arrays for each team
                team0, team1 = game[0][player_key], game[1][player_key]
                # get the total games for each player
                for player in chain(team0, team1):
                    total_games[player] += 1
                # get the combitions of teammates
                for player1, player2 in combinations(team0, 2):
                    ratios[player1][player2] += 1.0
                    ratios[player2][player1] += 1.0
                for player1, player2 in combinations(team1, 2):
                    ratios[player1][player2] += 1.0
                    ratios[player2][player1] += 1.0
            for player1, player1dict in ratios.items():
                for player2, num_games in player1dict.items():
                    g = total_games[player1]
                    ratio = num_games / g
                    player1dict[player2] = ratio
            return ratios
        elif isinstance(games, DataFrame):
            for _, sub_data in games:
                team0, team1 = (
                    sub_data[sub_data["team"] == 0][player_key],
                    sub_data[sub_data["team"] == 1][player_key],
                )
                # get the total games for each player
                for player in chain(team0, team1):
                    total_games[player] += 1
                # get the combitions of teammates
                for player1, player2 in combinations(team0, 2):
                    ratios[player1][player2] += 1.0
                    ratios[player2][player1] += 1.0
                for player1, player2 in combinations(team1, 2):
                    ratios[player1][player2] += 1.0
                    ratios[player2][player1] += 1.0
            for player1, player1dict in ratios.items():
                for player2, num_games in player1dict.items():
                    g = total_games[player1]
                    ratio = num_games / g
                    player1dict[player2] = ratio
            return ratios
        else:
            raise TypeError("Expected a numpy array of structs or dataframe of players")

    # NOTE: value returned must be a container, not an iterator
    @staticmethod
    def _compute_last_teammates(
        games: Games[T_Rating] | Annotated[DataFrame, DataFrameGroupBy],
        player_key: str = "player",
    ) -> defaultdict[str, list[str]]:
        last_teammates: defaultdict[str, list[str]] = defaultdict(list)
        if isinstance(games, np.ndarray):
            for game in games:
                team0, team1 = game[0]["player"], game[1]["player"]
                for player in team0:
                    last_teammates[player] = list(team0)
                for player in team1:
                    last_teammates[player] = list(team1)
            return last_teammates
        elif isinstance(games, DataFrame):
            # iterate through the games history and find the latest teammates
            for _, sub_data in games:
                team0, team1 = (
                    sub_data[sub_data["team"] == 0][player_key],
                    sub_data[sub_data["team"] == 1][player_key],
                )
                # simply overwrite
                for player in team0:
                    last_teammates[player] = list(team0)
                for player in team1:
                    last_teammates[player] = list(team1)
            return last_teammates
        else:
            raise TypeError("Expected a numpy array of structs or dataframe of players")

    def info(self) -> TeamMixerInfoDict[T_Rating, T_Index]:
        info = super().info()
        game_combs = info["game_combinations"]
        all_ind = np.arange(len(game_combs), dtype=self.index_dtype)
        indices = [all_ind]

        team_game_ratios = self._compute_team_game_ratios(self.games)
        last_teammates = self._compute_last_teammates(self.games)
        info["game_ratios_ind"] = self._games_played_with_ratio_ind(
            game_combs, team_game_ratios, self.settings.games_played_with_ratio
        )
        indices.append(info["game_ratios_ind"])
        info["last_teammates_ind"] = self._players_last_played_with_ind(
            game_combs, last_teammates, self.settings.players_last_played_with_ratio
        )
        indices.append(info["last_teammates_ind"])
        info["all_constraints_ind"] = self.intersection(*indices)
        return info
