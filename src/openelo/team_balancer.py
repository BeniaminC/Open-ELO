'''
Elo Team Balancer
Author: Beniamin#9574 (Beniamin Condrea)
Written 3/2/2023
'''
from dataclasses import dataclass
from functools import partial, reduce
from itertools import batched, combinations
from math import comb
from typing import Any, Hashable

import numpy as np
import numpy.typing as npt
from nptyping import Float32, Int32, NDArray, Shape, Structure


__all__ = ['EloTeamBalancerParams',
           'EloTeamBalancer']


player_np_dt = np.dtype([('player', np.object_), ('rating', np.float32)])
Player = tuple[Hashable, float]
Players = list[Player] | NDArray[Shape['*'], Structure['player: Object, rating: Float32']]
Team = NDArray[Shape['*'], Structure['player: Object, rating: Float32']]
Game = NDArray[Shape['2, *'], Structure['player: Object, rating: Float32']]
Games = NDArray[Shape['*, 2, *'], Structure['player: Object, rating: Float32']]
GameStats = NDArray[Shape['*, 6'], Float32]
Indices = NDArray[Shape['*, ...'], Int32]


class InvalidPlayerCount(Exception):
    '''
    An exception class for error handling odd teams.
    '''
    pass


@dataclass
class EloTeamBalancerParams:
    top_k: int = 50
    elo_diff: float = 200.
    player_balance: bool = True
    new_player_offset: float = 0.


class EloTeamBalancer:
    '''
    From a list of tuples or lists of Players (username, elo), create an instance
    for creating a dictionary of team balancing information.

        Args:
            players (:obj:`Iterable[Tuple(str, float)]`): A list of players and their respective elo rating.

            k_constraint: (:obj:`Optional[int]`): take only :obj:`k` best combinations of teams based on
                team elo ratings. If omitted, take all the combinations of both teams.

            elo_diff_constraint (:obj:`Optional[float]`): Filter the team combinations that have an difference
            of their respective elo averages less than the constraint. If omitted, take all teams regardless of
            their elo differences.

            player_balance_constraint (:obj:`Optional[bool]`): Filter the team combinations that have a "good
            mix" of players, where, given the set of two players ordered by their elo ratings, the players
            cannot be on the same team (e.g., the two best players cannot be on the same team, the next two
            best players cannot be on the same team, etc.)

        Returns:
            :obj:`None`

        Example::

            player_names = list('ABCDEFGHIJ')
            random_elos = np.random.normal(25, 25/3, 10)
            players_set = list(map(lambda x,y : (x,y), player_names, random_elos))


            elo_balancer = EloTeamBalancer(players_set, 10, 50, True)
            best_elo_balanced_teams = elo_balancer.create_elo_info()


            print(best_elo_balanced_teams['game_combinations'][best_elo_balanced_teams['all_constraints_ind']])

            print(best_elo_balanced_teams['game_statistics'][best_elo_balanced_teams['all_constraints_ind']])
    '''
    def __init__(self, settings: EloTeamBalancerParams) -> None:
        self.settings = settings

    @staticmethod
    def __sort_players_by_elo(players: Players) -> npt.NDArray:
        players_array = np.array(players, dtype=player_np_dt)
        players_array.sort(order=('rating', 'player'))
        return players_array

    @staticmethod
    def __create_team_combinations(players: Players) -> Games:
        total_players = len(players)
        if total_players % 2 == 1 or total_players < 2:
            raise InvalidPlayerCount(
                f'Input player count is {total_players}. The player list must be even!')
        players_per_team = total_players // 2
        half_comb = comb(total_players, players_per_team) // 2
        players_np = np.array(players, dtype=player_np_dt)
        all_comb = np.array(list(combinations(np.array(players_np), players_per_team)))
        teams0_comb_list = all_comb[:half_comb]
        teams1_not0_list = all_comb[half_comb:][::-1]
        stacked = np.stack((teams0_comb_list, teams1_not0_list), axis=1)

        return stacked

    @staticmethod
    def __create_team_combinations_ind(player_n: int) -> Indices:
        if player_n % 2 == 1 or player_n < 2:
            raise InvalidPlayerCount(
                f'Input player count is {player_n}. The player list must be even!')
        players_ind = np.arange(player_n)
        half_ind = player_n // 2
        half_comb = comb(player_n, half_ind) // 2
        all_comb = np.array(list(combinations(np.array(players_ind), half_ind)))
        teams0_comb_list = all_comb[0:half_comb]
        teams1_not0_list = all_comb[half_comb:][::-1]
        stacked = np.stack((teams0_comb_list, teams1_not0_list), axis=1)
        return stacked

    @staticmethod
    def __sets_of_two_balance_constraint(team0: Team, team1: Team) -> bool:
        all_players = np.concatenate((team0, team1))
        all_players.sort(order=('rating', 'player'))
        u, c = np.unique(all_players['rating'], return_counts=True)
        dups = u[c > 1]
        for p1, p2 in batched(all_players, 2):
            if (((p1 in team0) and (p2 in team0)) or ((p1 in team1) and (p2 in team1))) and not (p1['rating'] in dups or p2['rating'] in dups):
                return False
        return True

    @staticmethod
    def __get_elo_team_statistics(team: Team) -> tuple[Any, np.floating[Any]]:
        elos = team['rating']
        elo_sum = np.sum(elos)
        elo_avg = np.mean(elos)
        return elo_sum, elo_avg

    @staticmethod
    def __get_elo_game_statistics(game: Game) -> GameStats:
        team0, team1 = game[0], game[1]
        team0_elo_sum, team0_elo_avg = EloTeamBalancer.__get_elo_team_statistics(
            team0)
        team1_elo_sum, team1_elo_avg = EloTeamBalancer.__get_elo_team_statistics(
            team1)
        elo_diff = team0_elo_sum - team1_elo_sum
        elo_abs = np.abs(elo_diff)
        game_stats = np.array([team0_elo_sum, team1_elo_sum, team0_elo_avg, team1_elo_avg, elo_diff, elo_abs], dtype=np.float32)
        return game_stats

    @staticmethod
    def __get_elo_games_statistics(games: Games) -> GameStats:
        return np.array([EloTeamBalancer.__get_elo_game_statistics(game) for game in games], dtype=np.float32)

    @staticmethod
    def __partition_k_teams_elo_ind(games: Games, k: int) -> npt.NDArray:
        k = k if k < len(games) else len(games)
        elos_abs = EloTeamBalancer.__get_elo_games_statistics(games)[:, 5]
        best_k_elos_partition = np.argpartition(elos_abs, k-1)[:k]
        return best_k_elos_partition

    @staticmethod
    def __players_balance_partition_constraint(games: Games) -> Indices:
        balanced_ind = np.where([EloTeamBalancer.__sets_of_two_balance_constraint(game[0], game[1]) for game in games])
        return balanced_ind[0]

    @staticmethod
    def __teams_elo_difference_constraint(games: Games, diff: float) -> Indices:
        elos_abs = EloTeamBalancer.__get_elo_games_statistics(games)[:, 5]
        diff_ind = np.where(elos_abs <= diff)
        return diff_ind[0]

    def set_players(self, players: Players):
        n = len(players)
        if n % 2 == 1 or n < 2:
            raise InvalidPlayerCount(
                f'Input player count is {n}. The player list must be even!')
        self.players = players

    def create_elo_info(self):
        '''
            Creates a dictionary of information regarding the teams. Using the combination lists
            and the indices supplied in the dictionary, obtain the players.

            Keys:
                :obj:`player_elo_sorted`: Get all the players sorted by their elo.

                :obj:`games_combinations`: Get all the game combinations. Not sorted.

                :obj:`games_combinations_ind`: Get all the game combination of play indices. Not sorted.

                :obj:`best_games_partition_ind`: Get the indices of the top :obj:`k`
                constraint. Sorted from lowest average elo different to highest.

                :obj:`player_balance_ind`: Get the indices of the teams with the "good
                mix" constraint.

                :obj:`elo_diff_ind`: Get the indices of the teams with average elo
                differences less than the :obj:`elo_diff` constraint.

                :obj:`game_statistics`: Get the game statistics of all the games with
                the same indices as the game combinations. The listing returns a list
                of tuples which contain the elo sum of each team, elo average of each
                team, the elo difference of each team (team0 - team1) and the absolute
                value of the elo difference:
                :obj:`((team0_elo_sum, team1_elo_sum), (team0_elo_avg, team1_elo_avg), elo_diff, elo_abs)`


                :obj:`all_constraints_ind`: Get the indices of the intersection of all the
                teams which follow all the constraints. This can return empty!

        '''
        team_balancer_info = {}
        team_balancer_info['players_elo_sorted'] = EloTeamBalancer.__sort_players_by_elo(self.players)
        game_comb = EloTeamBalancer.__create_team_combinations(self.players)
        team_balancer_info['game_statistics'] = EloTeamBalancer.__get_elo_games_statistics(game_comb)
        n = len(self.players)
        team_balancer_info['game_combinations'] = game_comb
        team_balancer_info['games_combinations_ind'] = EloTeamBalancer.__create_team_combinations_ind(n)
        team_balancer_info['best_games_partition_ind'] = EloTeamBalancer.__partition_k_teams_elo_ind(
            game_comb, self.settings.top_k) if self.settings.top_k is not None else None
        team_balancer_info['player_balance_ind'] = EloTeamBalancer.__players_balance_partition_constraint(
            game_comb) if self.settings.player_balance is not False else None
        team_balancer_info['elo_diff_ind'] = EloTeamBalancer.__teams_elo_difference_constraint(
            game_comb, self.settings.elo_diff) if self.settings.elo_diff is not None else None
        all_ind = np.arange(len(game_comb))
        intersect1d_unique = partial(np.intersect1d, assume_unique=True)
        team_balancer_info['all_constraints_ind'] = reduce(intersect1d_unique, (team_balancer_info['best_games_partition_ind'] if self.settings.top_k is not None else all_ind,
                                                                                team_balancer_info['player_balance_ind'] if self.settings.player_balance is not False else all_ind,
                                                                                team_balancer_info['elo_diff_ind'] if self.settings.elo_diff is not None else all_ind))
        return team_balancer_info

    @staticmethod
    def get_elo_game_statistics(game: Game) -> GameStats:
        return EloTeamBalancer.__get_elo_game_statistics(game)

    def get_best_game(self) -> tuple[Game, GameStats]:
        elo_info = self.create_elo_info()
        best_ind = np.argmin(elo_info['game_statistics'][elo_info['all_constraints_ind']][:, 5])
        best_game = elo_info['game_combinations'][elo_info['all_constraints_ind']][best_ind]
        best_stats = elo_info['game_statistics'][elo_info['all_constraints_ind']][best_ind]
        return best_game, best_stats

    def get_new_player_offset(self):
        return self.settings.new_player_offset


def pretty_print_teams_string(games: list[Game], stats: list[GameStats]):
    string_list = []
    string_list.append('**Suggested Teams:**\n')
    for i, game in enumerate(games):
        string_list.append(f'**Game #{i+1} (Elo Difference: {stats[i][5].astype(int)})\n------------------**')
        for j, team in enumerate(game):
            string_list.append(f'**Team {j+1} (Elo sum {stats[i][j].astype(int)} | Elo average {stats[i][2 + j].astype(int)})**')
            players = '\n'.join([str(player) for player in team])
            string_list.append(players)
        string_list.append('\n')
    return '\n'.join(string_list)


def main():
    player_names = list('ABCDEFGHIJ')
    random_elos = np.random.normal(1500., 500., 10)
    players_set = list(map(lambda x, y: (x, y), player_names, random_elos))
    params = EloTeamBalancerParams(20, 200., True, 0.)
    elo_balancer = EloTeamBalancer(params)
    elo_balancer.set_players(players_set)
    best_elo_balanced_teams = elo_balancer.create_elo_info()
    sorted_stats_ind = np.argsort(best_elo_balanced_teams['game_statistics'][best_elo_balanced_teams['all_constraints_ind']][:, 5])
    print(best_elo_balanced_teams['game_combinations'][best_elo_balanced_teams['all_constraints_ind']][sorted_stats_ind])
    print(best_elo_balanced_teams['game_statistics'][best_elo_balanced_teams['all_constraints_ind']][sorted_stats_ind])


if __name__ == '__main__':
    main()