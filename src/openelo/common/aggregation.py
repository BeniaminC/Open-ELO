
from abc import ABC, abstractmethod
from typing import Any

from .player import Player
from .term import Rating


__all__ = ['TeamSumAggregation',
           'TeamAverageAggregation',
           'TeamMaxAggregation',
           'TeamMinAggregation',
           'TeamAverageAggregationN',
           'TeamSumAggregationN']


class TeamRatingAggregation(ABC):
    '''
    Abstract base class for team rating aggregation.
    '''
    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Rating:
        '''
        Dunder method required to be defined for `TeamRatingAggegation`.
        '''
        pass


class TeamSumAggregation(TeamRatingAggregation):
    def __call__(self, players: list[Player]) -> Rating:
        '''
        Given a list of player objects, aggregate the team rating by summing
        the ratings and variance of each player, respectively. Note that this
        is the sum of the variances of each player, not the standard deviation.

        Args:
            players (:obj:`list[Player]`): List of players to aggregate.

        Returns:
            :obj:`Rating`: Aggregated rating object containing `mu` and `sigma`.
        '''
        return Rating(sum(player.approx_posterior.mu for player in players),
                      sum(player.approx_posterior.sig ** 2 for player in players))


class TeamAverageAggregation(TeamRatingAggregation):
    def __call__(self, players: list[Player]) -> Rating:
        '''
        Given a list of player objects, aggregate the team rating by averaging
        the ratings and variance of each player, respectively. Note that this
        is the average of the variances of each player, not the standard deviation.

        Args:
            players (:obj:`list[Player]`): List of players to aggregate.

        Returns:
            :obj:`Rating`: Aggregated rating object containing `mu` and `sigma`.
        '''
        return Rating(sum(player.approx_posterior.mu for player in players) / len(players),
                      sum(player.approx_posterior.sig ** 2 for player in players) / len(players))


class TeamMaxAggregation(TeamRatingAggregation):
    def __call__(self, players: list[Player]) -> Rating:
        '''
        Given a list of player objects, aggregate the team rating by the max of
        the ratings and variance of each player, respectively. Note that this
        is the max of the variances of each player, not the standard deviation.

        Args:
            players (:obj:`list[Player]`): List of players to aggregate.

        Returns:
            :obj:`Rating`: Aggregated rating object containing `mu` and `sigma`.
        '''
        max_player = max(players, key=lambda player: player.approx_posterior.mu)
        return Rating(max_player.approx_posterior.mu,
                      max_player.approx_posterior.sig ** 2)


class TeamMinAggregation(TeamRatingAggregation):
    def __call__(self, players: list[Player]) -> Rating:
        '''
        Given a list of player objects, aggregate the team rating by the min of
        the ratings and variance of each player, respectively. Note that this
        is the min of the variances of each player, not the standard deviation.

        Args:
            players (:obj:`list[Player]`): List of players to aggregate.

        Returns:
            :obj:`Rating`: Aggregated rating object containing `mu` and `sigma`.
        '''
        min_player = min(players, key=lambda player: player.approx_posterior.mu)
        return Rating(min_player.approx_posterior.mu,
                      min_player.approx_posterior.sig ** 2)


class TeamAverageAggregationN(TeamRatingAggregation):
    def __init__(self, n: int = 3, best: bool = True) -> None:
        '''
        Initialize the aggregation object with of `n` players.  Setting `n`
        to be less than or equal to the number of players in each team. Either
        set `best` to either be the best `n` players or the worst `n` players by
        rating.

        Args:
            n (:obj:`int`): The number of players per team to aggregate.

            best (:obj:`bool`): Compute the best `n` players (default `True`).

        Returns:
            :obj:`None`
        '''
        self.n = n
        self.best = best

    def __call__(self, players: list[Player]) -> Rating:
        '''
        Given a list of player objects, aggregate the team rating by the average of
        the ratings and variance of `n` players, respectively. Note that this
        is the average of the variances of each player, not the standard deviation.

        Args:
            players (:obj:`list[Player]`): List of players to aggregate.

        Returns:
            :obj:`Rating`: Aggregated rating object containing `mu` and `sigma`.
        '''
        sorted_players = sorted(players, key=lambda player: player.approx_posterior.mu)
        n_players = sorted_players[-self.n:] if self.best else sorted_players[:self.n]
        return Rating(sum(player.approx_posterior.mu for player in n_players) / len(n_players),
                      sum(player.approx_posterior.sig ** 2 for player in n_players) / len(n_players))


class TeamSumAggregationN(TeamRatingAggregation):
    def __init__(self, n: int = 3, best: bool = True) -> None:
        '''
        Initialize the aggregation object with of `n` players.  Setting `n`
        to be less than or equal to the number of players in each team. Either
        set `best` to either be the best `n` players or the worst `n` players by
        rating.

        Args:
            n (:obj:`int`): The number of players per team to aggregate.

            best (:obj:`bool`): Compute the best players (default `True`).
        
        Returns:
            :obj:`None`
        '''
        self.n = n
        self.best = best

    def __call__(self, players: list[Player]) -> Rating:
        '''
        Given a list of player objects, aggregate the team rating by the sum of
        the ratings and variance of `n` players, respectively. Note that this
        is the sum of the variances of each player, not the standard deviation.

        Args:
            players (:obj:`list[Player]`): List of players to aggregate.

        Returns:
            :obj:`Rating`: Aggregated rating object containing `mu` and `sigma`.
        '''
        sorted_players = sorted(players, key=lambda player: player.approx_posterior.mu)
        n_players = sorted_players[-self.n:] if self.best else sorted_players[:self.n]
        return Rating(sum(player.approx_posterior.mu for player in n_players),
                      sum(player.approx_posterior.sig ** 2 for player in n_players))
