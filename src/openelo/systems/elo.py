import concurrent.futures
from dataclasses import dataclass, field
from math import comb
from typing import NamedTuple

from ..common.aggregation import TeamRatingAggregation
from ..common.common import ContestRatingParams, Standings
from ..common.rating_system import RatingSystem
from ..common.team_rating_system import TeamRating, TeamRatingSystem
from ..common.term import Rating


__all__ = ['KFactor',
           'Elo']


class KFactor(NamedTuple):
    k: float
    games: int
    rating: float


@dataclass
class Elo(RatingSystem, TeamRatingSystem):
    '''
    Classic Elo rating system.
    '''
    beta: float = 400
    k_factors: list[KFactor] = field(default_factory=list)

    def round_update(self,
                     params: ContestRatingParams,
                     standings: Standings) -> None:
        raise NotImplementedError()

    @staticmethod
    def _standard_performance_rating(opp_ratings: list[float], wins: int, loses: int, s: float):
        return (sum(opp_rating for opp_rating in opp_ratings) + s * (wins - loses)) / len(opp_ratings)

    @staticmethod
    def _win_probability(rating_i: float, rating_j: float, s: float) -> float:
        return 1. / (1 + 10 ** ((rating_j - rating_i) / s))

    @staticmethod
    def _r(N: int, rank_i: int):
        return (N - rank_i) / comb(N, 2)

    def k_factor(self, games: int, rating: float, default: int = 40) -> float:
        for k_factor in self.k_factors:
            if k_factor.games and k_factor.rating and k_factor.games > games and k_factor.rating > rating:
                return k_factor.k
            elif k_factor.games and k_factor.games > games:
                return k_factor.k
            elif k_factor.rating and k_factor.rating > rating:
                return k_factor.k
        return default

    def team_round_update(self,
                          params: ContestRatingParams,
                          standings: Standings,
                          agg: TeamRatingAggregation) -> None:
        '''
        Update the player ratings in teams according to their team and rank.

        Args:
            params (:obj:`ContestRatingParams`): Parameters of a particular contest.

            standings (:obj:`Standings`): Standings of each player
            according to their `team` and `rank`. Must be in order.
        '''
        s = self.beta / (params.weight ** 0.5)
        self.init_players_event(standings)
        team_standings = self.convert_to_teams(standings)
        team_ratings = list(TeamRating(team, team_info['rank'], agg(team_info['players'])) for team, team_info in team_standings.items())
        N = len(team_ratings)
        prob_denom = comb(N, 2)
        k = 40 * params.weight  # TODO: create a variable k-factor based on teams (default 40)

        def _update_player_rating(relative_rank: int, team_i: TeamRating):
            team_i_mu = team_i.rating.mu
            r_i = (N - relative_rank) / prob_denom
            total_probabilty = 0.
            for team_q in team_ratings:
                if team_i is team_q:
                    continue
                total_probabilty += Elo._win_probability(team_i.rating.mu, team_q.rating.mu, s)
            total_probabilty /= prob_denom
            team_new_mu = team_i.rating.mu + k * (r_i - total_probabilty)
            team_i_rating_sum = sum(player.approx_posterior.mu for player in team_standings[team_i.team]['players'])
            for player in team_standings[team_i.team]['players']:
                old_mu = player.approx_posterior.mu
                w = 1.  # TODO: adjust weight based on teammate (default 1.)
                new_mu = old_mu + w * (team_new_mu - team_i_mu)
                player.update_rating(Rating(new_mu, 0), 0)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(_update_player_rating, *zip(*((i+1, team_i) for i, team_i in enumerate(team_ratings))))