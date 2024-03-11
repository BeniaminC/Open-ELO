import concurrent.futures
from collections import Counter
from dataclasses import dataclass, field
from math import exp

from ..common.aggregation import TeamRatingAggregation
from ..common.constants import (DEFAULT_BETA, DEFAULT_WEIGHT_LIMIT, 
                                DEFAULT_SIG_LIMIT, DEFAULT_DRIFTS_PER_DAY)
from ..common.common import (ContestRatingParams, ranks_ge, Standings)
from ..common.ordering import Ordering
from ..common.player import Player
from ..common.rating_system import RatingSystem
from ..common.team_rating_system import TeamRating, TeamRatingSystem


__all__ = ['PlackettLuce']


@dataclass
class PlackettLuce(RatingSystem, TeamRatingSystem):
    '''
    Plackett-Luce rating system.
    '''
    beta: float = DEFAULT_BETA
    kappa: float = 1e-4

    weight_limit: float = DEFAULT_WEIGHT_LIMIT
    noob_delay: list[float] = field(default_factory=list)
    sig_limit: float = DEFAULT_SIG_LIMIT
    drift_per_day: float = DEFAULT_DRIFTS_PER_DAY

    def round_update(self,
                     params: ContestRatingParams,
                     standings: Standings) -> None:
        raise NotImplementedError()

    def team_round_update(self,
                          params: ContestRatingParams,
                          standings: Standings,
                          agg: TeamRatingAggregation,
                          contest_time: int = 0) -> None:
        '''
        Update the player ratings in teams according to their team and rank.

        Args:
            params (:obj:`ContestRatingParams`): Parameters of a particular contest.

            standings (:obj:`Standings`): Standings of each player
            according to their `team` and `rank`. Must be in order.
        '''
        self.init_players_event(standings, contest_time=contest_time)

        def _update_player(player: Player):
            weight = self.compute_weight(params.weight, self.weight_limit, self.noob_delay, player.times_played_excl())
            sig_drift = self.compute_sig_drift(weight, self.sig_limit, self.drift_per_day, float(player.delta_time))
            player.add_noise_and_collapse(sig_drift)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for player, _, _ in standings:
                executor.submit(_update_player, player)
        team_standings = self.convert_to_teams(standings)
        team_ratings = list(TeamRating(team, team_info['rank'], agg(team_info['players'])) for team, team_info in team_standings.items())
        team_ratings.sort(key=lambda team: team.rank)
        sig_perf_sq = self.beta ** 2 / (self.weight_limit * params.weight)
        c = float(sum(team_i.rating.sig + sig_perf_sq for team_i in team_ratings) ** 0.5)
        A_q = Counter(team.rank for team in team_ratings)
        def _update_player_rating(team_i: TeamRating):
            team_i_sig_sq = team_i.rating.sig
            gamma_q = (team_i_sig_sq ** 0.5) / c
            delta_q = 0.
            eta_q = 0.
            for team_q in team_ratings:
                C_q = ranks_ge(team_ratings, team_q.rank, key=lambda team: team.rank)
                summation = sum(exp(team.rating.mu / c) for team in C_q)
                p_i_C_q = (exp(team_i.rating.mu / c)) / summation
                delta_quot = team_i_sig_sq / (c * A_q[team_q.rank])
                eta_quot = (gamma_q * team_i_sig_sq) / (c ** 2 * A_q[team_q.rank])
                match Ordering.cmp(team_q.rank, team_i.rank):
                    case Ordering.LESS:
                        outcome_delta = -p_i_C_q
                        outcome_eta = p_i_C_q * (1 - p_i_C_q)
                    case Ordering.EQUAL:
                        if team_q is team_i:
                            outcome_delta = 1. - p_i_C_q
                            outcome_eta = p_i_C_q * (1 - p_i_C_q)
                        else:
                            outcome_delta = -p_i_C_q
                            outcome_eta = p_i_C_q * (1 - p_i_C_q)
                    case Ordering.GREATER:
                        outcome_delta = 0.
                        outcome_eta = 0.
                delta_q += (delta_quot * outcome_delta)
                eta_q += (eta_quot * outcome_eta)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for player in team_standings[team_i.team]['players']:
                    executor.submit(self.team_individual_update, player, team_i_sig_sq, delta_q, eta_q, self.kappa)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(_update_player_rating, (team_i for team_i in team_ratings))

