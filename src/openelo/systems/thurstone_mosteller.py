import concurrent.futures
from dataclasses import dataclass, field
from operator import itemgetter

from ..common.aggregation import TeamRatingAggregation
from ..common.constants import (DEFAULT_BETA, DEFAULT_WEIGHT_LIMIT, 
                                DEFAULT_SIG_LIMIT, DEFAULT_DRIFTS_PER_DAY, 
                                DRAW_PROBABILITY)
from ..common.common import (ContestRatingParams, total_partial)
from ..common.numericals import (standard_normal_cdf, standard_normal_pdf)
from ..common.ordering import Ordering
from ..common.player import Player
from ..common.rating_system import RatingSystem
from ..common.team_rating_system import TeamRating, TeamRatingSystem


__all__ = ['ThurstoneMosteller', 
           'ThurstoneMostellerPartial']


def _V(x: float, t: float) -> float:
    return standard_normal_pdf(x - t) / standard_normal_cdf(x - t)


def _W(x: float, t: float) -> float:
    return _V(x, t) * (_V(x, t) + (x - t))


def _tilde_V(x: float, t: float) -> float:
    return -((standard_normal_pdf(t - x) - standard_normal_pdf(-t - x)) / (standard_normal_cdf(t - x) - standard_normal_cdf(-t - x)))


def _tilde_W(x: float, t: float) -> float:
    return ((t - x) * standard_normal_pdf(t - x) - (-(t + x)) * standard_normal_pdf(-(t + x))) / (standard_normal_cdf(t - x) - standard_normal_cdf(-t - x)) + _tilde_V(x, t) ** 2


@dataclass
class ThurstoneMosteller(RatingSystem, TeamRatingSystem):
    '''
    Thurstone-Mosteller rating system.
    '''
    beta: float = DEFAULT_BETA
    kappa: float = 1e-4
    weight_limit: float = DEFAULT_WEIGHT_LIMIT
    noob_delay: list[float] = field(default_factory=list)
    sig_limit: float = DEFAULT_SIG_LIMIT
    drift_per_day: float = DEFAULT_DRIFTS_PER_DAY

    def round_update(self,
                     params: ContestRatingParams,
                     standings: list[tuple[Player, int, int]]) -> None:
        raise NotImplementedError()

    @staticmethod
    def _V(x: float, t: float) -> float:
        return standard_normal_pdf(x - t) / standard_normal_cdf(x - t)

    @staticmethod
    def _W(x: float, t: float) -> float:
        return _V(x, t) * (_V(x, t) + (x - t))

    @staticmethod
    def _tilde_V(x: float, t: float) -> float:
        return -((standard_normal_pdf(t - x) - standard_normal_pdf(-t - x)) / (standard_normal_cdf(t - x) - standard_normal_cdf(-t - x)))

    @staticmethod
    def _tilde_W(x: float, t: float) -> float:
        return ((t - x) * standard_normal_pdf(t - x) - (-(t + x)) * standard_normal_pdf(-(t + x))) / (standard_normal_cdf(t - x) - standard_normal_cdf(-t - x)) + _tilde_V(x, t) ** 2

    def team_round_update(self,
                          params: ContestRatingParams,
                          standings: list[tuple[Player, int, int]],
                          agg: TeamRatingAggregation,
                          contest_time: int = 0) -> None:
        '''
        Update the player ratings in teams according to their team and rank.

        Args:
            params (:obj:`ContestRatingParams`): Parameters of a particular contest.

            standings (:obj:`list[tuple[Player, int, int]]): Standings of each player
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
        sig_perf_sq = self.beta ** 2 / (self.weight_limit * params.weight)

        def _update_player_rating(team_i: TeamRating):
            team_i_mu, team_i_sig_sq = team_i.rating.mu, team_i.rating.sig
            delta_q = 0.
            eta_q = 0.
            for team_q in team_ratings:
                if team_i is team_q:
                    continue
                team_q_mu, team_q_sig_sq = team_q.rating.mu, team_q.rating.sig
                c_sq = (team_i_sig_sq + team_q_sig_sq + 2 * sig_perf_sq)
                c = c_sq ** 0.5
                t = DRAW_PROBABILITY / c
                match Ordering.cmp(team_i.rank, team_q.rank):
                    case Ordering.LESS:
                        x = (team_i_mu - team_q_mu) / c
                        outcome_v = _V(x, t)
                        outcome_w = _W(x, t)
                    case Ordering.EQUAL:
                        x = (team_i_mu - team_q_mu) / c
                        outcome_v = _tilde_V(x, t)
                        outcome_w = _tilde_W(x, t)
                    case Ordering.GREATER:
                        x = (team_q_mu - team_i_mu) / c
                        outcome_v = -_V(x, t)
                        outcome_w = _tilde_W(x, t)
                delta_q += (team_i_sig_sq / c) * outcome_v
                eta_q += (team_i_sig_sq / c_sq) * outcome_w
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for player in team_standings[team_i.team]['players']:
                    executor.submit(self.team_individual_update, player, team_i_sig_sq, delta_q, eta_q, self.kappa)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(_update_player_rating, (team_i for team_i in team_ratings))


@dataclass
class ThurstoneMostellerPartial(RatingSystem, TeamRatingSystem):
    '''
    Thurstone-Mosteller partial rating system.
    '''
    beta: float = DEFAULT_BETA
    kappa: float = 1e-4
    weight_limit: float = DEFAULT_WEIGHT_LIMIT
    noob_delay: list[float] = field(default_factory=list)
    sig_limit: float = DEFAULT_SIG_LIMIT
    drift_per_day: float = DEFAULT_DRIFTS_PER_DAY

    def round_update(self,
                     params: ContestRatingParams,
                     standings: list[tuple[Player, int, int]]) -> None:
        raise NotImplementedError()

    @staticmethod
    def _V(x: float, t: float) -> float:
        return standard_normal_pdf(x - t) / standard_normal_cdf(x - t)

    @staticmethod
    def _W(x: float, t: float) -> float:
        return _V(x, t) * (_V(x, t) + (x - t))

    @staticmethod
    def _tilde_V(x: float, t: float) -> float:
        return -((standard_normal_pdf(t - x) - standard_normal_pdf(-t - x)) / (standard_normal_cdf(t - x) - standard_normal_cdf(-t - x)))

    @staticmethod
    def _tilde_W(x: float, t: float) -> float:
        return ((t - x) * standard_normal_pdf(t - x) - (-(t + x)) * standard_normal_pdf(-(t + x))) / (standard_normal_cdf(t - x) - standard_normal_cdf(-t - x)) + _tilde_V(x, t) ** 2

    def team_round_update(self,
                          params: ContestRatingParams,
                          standings: list[tuple[Player, int, int]],
                          agg: TeamRatingAggregation) -> None:
        '''
        Update the player ratings in teams according to their team and rank.

        Args:
            params (:obj:`ContestRatingParams`): Parameters of a particular contest.

            standings (:obj:`list[tuple[Player, int, int]]): Standings of each player
            according to their `team` and `rank`. Must be in order.
        '''
        self.init_players_event(standings)

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
            team_i_mu, team_i_sig_sq = team_i.rating.mu, team_i.rating.sig
            delta_q = 0.
            eta_q = 0.
            partial_team_ratings = total_partial(team_ratings, team_i.rank, key=itemgetter(1))
            for team_q in partial_team_ratings:
                team_q_mu, team_q_sig_sq = team_q.rating.mu, team_q.rating.sig
                c_sq = (team_i_sig_sq + team_q_sig_sq + 2 * sig_perf_sq)
                c = c_sq ** 0.5
                t = DRAW_PROBABILITY / c
                match Ordering.cmp(team_i.rank, team_q.rank):
                    case Ordering.LESS:
                        x = (team_i_mu - team_q_mu) / c
                        outcome_v = ThurstoneMostellerPartial._V(x, t)
                        outcome_w = ThurstoneMostellerPartial._W(x, t)
                    case Ordering.EQUAL:
                        x = (team_i_mu - team_q_mu) / c
                        outcome_v = ThurstoneMostellerPartial._tilde_V(x, t)
                        outcome_w = ThurstoneMostellerPartial._tilde_W(x, t)
                    case Ordering.GREATER:
                        x = (team_q_mu - team_i_mu) / c
                        outcome_v = -ThurstoneMostellerPartial._V(x, t)
                        outcome_w = ThurstoneMostellerPartial._tilde_W(x, t)
                delta_q += (team_i_sig_sq / c) * outcome_v
                eta_q += (team_i_sig_sq / c_sq) * outcome_w
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for player in team_standings[team_i.team]['players']:
                    executor.submit(self.team_individual_update, player, team_i_sig_sq, delta_q, eta_q, self.kappa)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(_update_player_rating, (team_i for team_i in team_ratings))
