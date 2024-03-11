from abc import ABC, abstractmethod
from typing import Any, NamedTuple, TypedDict

from .aggregation import TeamRatingAggregation
from .common import ContestRatingParams, Standings
from .player import Player
from .term import Rating

TeamInfo = TypedDict('TeamInfo', {'players': list, 'rank': int})


class TeamRating(NamedTuple):
    team: int
    rank: int
    rating: Rating


class TeamRatingSystem(ABC):
    '''
    Abstract base class for team rating systems. Must define `team_round_update`.
    '''
    @abstractmethod
    def team_round_update(self,
                          params: ContestRatingParams,
                          standings: Standings,
                          agg: TeamRatingAggregation):
        pass

    @staticmethod
    def convert_to_teams(standings: Standings) -> dict[int, TeamInfo]:
        '''
        Given a standard list of standings, create a dictionary of team keys
        containing a dictionary of players list of the team and the rank of
        the team.
        
        Args:
            standings (:obj:`Standings`): A standard standings
            of each player with their respective team and rank.
        
        Returns:
            :obj:`dict[int, TeamInfo]`: A dictionary of arbitrary key of team
            integer and `TeamInfo`, which contains a dictionary of players list
            and the rank of the team.
        '''
        teams: dict[int, Any] = {}
        for player, team, rank in standings:
            if team not in teams:
                teams[team] = {}
                teams[team]['players'] = []
                teams[team]['rank'] = rank
            teams[team]['players'].append(player)
        return teams

    @staticmethod
    def team_individual_update(player: Player, team_sig_sq: float, omega: float, delta_i: float, kappa: float) -> None:
        '''
        If a team rating system includes team variance, omega, delta(i) and kappa,
        update the rating of the player object.

        Args:
            player (:obj:`Player`): Player object to modify.

            team_sig_sq (:obj:`float`): Team variance (not deviation).

            omega (:obj:`float`): The computed sum of delta to compute new rating.

            delta_i (:obj:`float`): The computed sum of eta to compute the new sigma.

            kappa (:obj:`float`): Value to prevent new sigma computation from being 
            negative. Should be close to zero.
        
        Returns:
            :obj:`None`
        '''
        old_mu = player.approx_posterior.mu
        old_sig_sq = player.approx_posterior.sig ** 2
        sig_ratio = (old_sig_sq / team_sig_sq)
        new_mu = old_mu + (sig_ratio) * omega
        new_sig = ((old_sig_sq) * max(1 - sig_ratio * delta_i, kappa)) ** 0.5
        player.update_rating(Rating(new_mu, new_sig), 0.)