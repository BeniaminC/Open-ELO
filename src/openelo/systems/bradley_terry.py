'''
Bradley-Terry Rating System
Source: https://jmlr.csail.mit.edu/papers/volume12/weng11a/weng11a.pdf
'''

import concurrent.futures
from dataclasses import dataclass, field
from math import exp
from operator import itemgetter

from ..common.aggregation import TeamRatingAggregation
from ..common.constants import (DEFAULT_BETA, DEFAULT_DRIFTS_PER_DAY,
                                DEFAULT_SIG_LIMIT, DEFAULT_WEIGHT_LIMIT)
from ..common.common import ContestRatingParams, Standings, total_partial
from ..common.numericals import (standard_logistic_cdf)
from ..common.ordering import Ordering
from ..common.player import Player
from ..common.rating_system import RatingSystem
from ..common.team_rating_system import TeamRating, TeamRatingSystem
from ..common.term import Rating


__all__ = ['BradleyTerry',
           'BradleyTerryPartial']


@dataclass
class BradleyTerry(RatingSystem, TeamRatingSystem):
    '''
    Bradley-Terry Rating System.
    '''
    beta: float = DEFAULT_BETA
    kappa: float = 1e-4
    weight_limit: float = DEFAULT_WEIGHT_LIMIT
    noob_delay: list[float] = field(default_factory=list)
    sig_limit: float = DEFAULT_SIG_LIMIT
    drift_per_day: float = DEFAULT_DRIFTS_PER_DAY

    @staticmethod
    def _win_probability(c: float, player_i: Rating, Player_q: Rating) -> float:
        z = (player_i.mu - Player_q.mu) / c
        return standard_logistic_cdf(z)

    @staticmethod
    def _team_win_probability(c: float, team_i: Rating, team_q: Rating) -> float:
        exp1 = exp(team_i.mu / c)
        exp2 = exp(team_q.mu / c)
        return exp1 / (exp1 + exp2)

    def round_update(self,
                     params: ContestRatingParams,
                     standings: Standings) -> None:
        '''
        Update the player ratings according to the standings.

        Args:
            params (:obj:`ContestRatingParams`): Parameters of a particular contest.

            standings (:obj:`Standings`): Standings of each player
            according to `team` and `rank`, respectively. Must be in order.
        '''
        self.init_players_event(standings)

        def _update_player(player: Player, lo: int):
            weight = self.compute_weight(params.weight, self.weight_limit, self.noob_delay, player.times_played_excl())
            sig_drift = self.compute_sig_drift(weight, self.sig_limit, self.drift_per_day, float(player.delta_time))
            player.add_noise_and_collapse(sig_drift)
            return (player.approx_posterior, lo)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            all_ratings = list(executor.map(_update_player, *zip(*((player, lo) for player, lo, _ in standings))))
        sig_perf_sq = self.beta ** 2 / (self.weight_limit * params.weight)

        def _update_player_rating(player: Player, my_lo: int):
            my_rating = player.approx_posterior
            old_sig_sq = my_rating.sig ** 2
            info = 0.
            update = 0.
            for rating, lo in all_ratings:
                match Ordering.cmp(my_lo, lo):
                    case Ordering.LESS:
                        outcome = 1.
                    case Ordering.EQUAL:
                        outcome = 0.5
                    case Ordering.GREATER:
                        outcome = 0.
                c_sq = old_sig_sq + rating.sig ** 2 + 2. * sig_perf_sq
                c = c_sq ** 0.5
                probability = self._win_probability(c, my_rating, rating)
                info += probability * (1. - probability) / c_sq
                update += (outcome - probability) / c
            info = 0.25 / (old_sig_sq + 2. * sig_perf_sq)
            update /= float(len(all_ratings))
            info *= old_sig_sq
            sig = my_rating.sig * max(self.kappa, 1. - info) ** 0.5
            update *= old_sig_sq
            mu = my_rating.mu + update
            player.update_rating(Rating(mu, sig), 0.)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(_update_player_rating, *zip(*((player, my_lo) for player, my_lo, _ in standings)))

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
            executor.map(_update_player, (player for player, _, _ in standings))
        team_standings = self.convert_to_teams(standings)
        team_ratings = list(TeamRating(team, team_info['rank'], agg(team_info['players'])) for team, team_info in team_standings.items())
        sig_perf_sq = self.beta ** 2 / (self.weight_limit * params.weight)

        def _update_player_rating(team_i: TeamRating):
            team_i_sig_sq = team_i.rating.sig
            delta_q = 0.
            eta_q = 0.
            for team_q in team_ratings:
                if team_i is team_q:
                    continue
                team_q_sig_sq = team_q.rating.sig
                match Ordering.cmp(team_i.rank, team_q.rank):
                    case Ordering.LESS:
                        outcome = 1.
                    case Ordering.EQUAL:
                        outcome = 0.5
                    case Ordering.GREATER:
                        outcome = 0.
                c_sq = (team_i_sig_sq + team_q_sig_sq + 2 * sig_perf_sq)
                c = c_sq ** 0.5
                gamma_q = (team_i_sig_sq ** 0.5) / c
                probability_iq = self._team_win_probability(c, team_i.rating, team_q.rating)
                probability_qi = 1 - probability_iq
                delta_q += (team_i_sig_sq / c) * (outcome - probability_iq)
                eta_q += gamma_q * (team_i_sig_sq / c_sq) * probability_iq * probability_qi

            with concurrent.futures.ThreadPoolExecutor() as executor:
                for player in team_standings[team_i.team]['players']:
                    executor.submit(self.team_individual_update, player, team_i_sig_sq, delta_q, eta_q, self.kappa)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(_update_player_rating, (team_i for team_i in team_ratings))


@dataclass
class BradleyTerryPartial(RatingSystem, TeamRatingSystem):
    beta: float = DEFAULT_BETA
    kappa: float = 1e-4
    weight_limit: float = DEFAULT_WEIGHT_LIMIT
    noob_delay: list[float] = field(default_factory=list)
    sig_limit: float = DEFAULT_SIG_LIMIT
    drift_per_day: float = DEFAULT_DRIFTS_PER_DAY

    @staticmethod
    def _win_probability(c: float, player: Rating, foe: Rating) -> float:
        z = (player.mu - foe.mu) / c
        return standard_logistic_cdf(z)

    @staticmethod
    def _team_win_probability(c: float, team_i: Rating, team_q: Rating) -> float:
        exp1 = exp(team_i.mu / c)
        exp2 = exp(team_q.mu / c)
        return exp1 / (exp1 + exp2)

    def round_update(self,
                     params: ContestRatingParams,
                     standings: Standings) -> None:
        self.init_players_event(standings)

        def _update_player(player: Player, lo: int):
            weight = self.compute_weight(params.weight, self.weight_limit, self.noob_delay, player.times_played_excl())
            sig_drift = self.compute_sig_drift(weight, self.sig_limit, self.drift_per_day, float(player.delta_time))
            player.add_noise_and_collapse(sig_drift)
            return (player.approx_posterior, lo)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            all_ratings = list(executor.map(_update_player, *zip(*((player, lo) for player, lo, _ in standings))))
        all_ratings.sort(key=itemgetter(1))
        sig_perf_sq = self.beta ** 2 / params.weight

        def _update_player_rating(player: Player, my_lo: int):
            my_rating = player.approx_posterior
            old_sig_sq = my_rating.sig ** 2
            info = 0.
            update = 0.
            partial_team_ratings = total_partial(all_ratings, my_lo, key=itemgetter(1))
            for rating, lo in partial_team_ratings:
                match Ordering.cmp(my_lo, lo):
                    case Ordering.LESS:
                        outcome = 1.
                    case Ordering.EQUAL:
                        outcome = 0.5
                    case Ordering.GREATER:
                        outcome = 0.
                c_sq = old_sig_sq + rating.sig ** 2 + 2. * sig_perf_sq
                c = c_sq ** 0.5
                probability = self._win_probability(c, my_rating, rating)

                info += probability * (1. - probability) / c_sq
                update += (outcome - probability) / c
            info = 0.25 / (old_sig_sq + 2. * sig_perf_sq)
            update /= float(len(all_ratings))
            info *= old_sig_sq
            sig = my_rating.sig * max(self.kappa, 1. - info) ** 0.5
            update *= old_sig_sq
            mu = my_rating.mu + update
            player.update_rating(Rating(mu, sig), 0.)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(_update_player_rating, *zip(*((player, lo) for player, lo, _, in standings)))

    def team_round_update(self,
                          params: ContestRatingParams,
                          standings: Standings,
                          agg: TeamRatingAggregation,
                          contest_time: int = 0) -> None:
        self.init_players_event(standings, contest_time=contest_time)

        def _update_player(player: Player):
            weight = self.compute_weight(params.weight, self.weight_limit, self.noob_delay, player.times_played_excl())
            sig_drift = self.compute_sig_drift(weight, self.sig_limit, self.drift_per_day, float(player.delta_time))
            player.add_noise_and_collapse(sig_drift)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(_update_player, (player for player, _, _ in standings))
        team_standings = self.convert_to_teams(standings)
        team_ratings = list(TeamRating(team, team_info['rank'], agg(team_info['players'])) for team, team_info in team_standings.items())
        team_ratings.sort(key=lambda team: team.rank)
        sig_perf_sq = self.beta ** 2 / (self.weight_limit * params.weight)

        def _update_player_rating(team_i: TeamRating):
            team_i_sig_sq = team_i.rating.sig
            delta_q = 0.
            eta_q = 0.

            partial_team_ratings = total_partial(team_ratings, team_i.rank, key=lambda team_q: team_q.rank)
            for team_q in partial_team_ratings:
                team_q_sig_sq = team_q.rating.sig
                match Ordering.cmp(team_i.rank, team_q.rank):
                    case Ordering.LESS:
                        outcome = 1.
                    case Ordering.EQUAL:
                        outcome = 0.5
                    case Ordering.GREATER:
                        outcome = 0.
                c_sq = (team_i_sig_sq + team_q_sig_sq + 2 * sig_perf_sq)
                c = c_sq ** 0.5
                gamma_q = (team_i_sig_sq ** 0.5) / c
                probability_iq = self._team_win_probability(c, team_i.rating, team_q.rating)
                probability_qi = 1 - probability_iq
                delta_q += (team_i_sig_sq / c) * (outcome - probability_iq)
                eta_q += gamma_q * (team_i_sig_sq / c_sq) * probability_iq * probability_qi
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for player in team_standings[team_i.team]['players']:
                    executor.submit(self.team_individual_update, player, team_i_sig_sq, delta_q, eta_q, self.kappa)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(_update_player_rating, (team_i for team_i in team_ratings))
