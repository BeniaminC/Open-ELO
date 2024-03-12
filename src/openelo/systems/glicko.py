import concurrent.futures
from dataclasses import dataclass, field
from math import comb, pi

from ..common.aggregation import TeamRatingAggregation
from ..common.constants import (DEFAULT_BETA, DEFAULT_WEIGHT_LIMIT, 
                                DEFAULT_SIG_LIMIT, DEFAULT_DRIFTS_PER_DAY, 
                                GLICKO_Q, TANH_MULTIPLIER)
from ..common.common import (ContestRatingParams, Standings)
from ..common.numericals import standard_logistic_cdf
from ..common.ordering import Ordering
from ..common.player import Player
from ..common.rating_system import RatingSystem
from ..common.team_rating_system import TeamRating, TeamRatingSystem
from ..common.term import Rating


__all__ = ['Glicko']


@dataclass
class Glicko(RatingSystem, TeamRatingSystem):
    '''
    Glicko rating system.
    '''
    beta: float = DEFAULT_BETA
    weight_limit: float = DEFAULT_WEIGHT_LIMIT
    noob_delay: list[float] = field(default_factory=list)
    sig_limit: float = DEFAULT_SIG_LIMIT
    drift_per_day: float = DEFAULT_DRIFTS_PER_DAY

    @staticmethod
    def _win_probability(sig_perf: float, player: Rating, foe: Rating) -> float:
        z = (player.mu - foe.mu) / ((foe.sig ** 2 + sig_perf ** 2) ** 0.5)
        return standard_logistic_cdf(z)

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
        sig_perf = self.beta / (params.weight ** 0.5)
        self.init_players_event(standings)

        def _update_player(player: Player, lo: int):
            weight = self.compute_weight(params.weight, self.weight_limit, self.noob_delay, player.times_played_excl())
            sig_drift = self.compute_sig_drift(weight, self.sig_limit, self.drift_per_day, float(player.delta_time))
            player.add_noise_and_collapse(sig_drift)
            g = (1 / (1 + (player.approx_posterior.sig / sig_perf) ** 2) ** 0.5)
            return (player.approx_posterior, lo, g)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            all_ratings = list(executor.map(_update_player, *zip(*((player, lo) for player, lo, _ in standings))))
        gli_q = TANH_MULTIPLIER / sig_perf

        def _update_player_rating(player: Player, my_lo: int):
            my_rating = player.approx_posterior
            info = 0.
            update = 0.
            for rating, lo, g in all_ratings:
                match Ordering.cmp(my_lo, lo):
                    case Ordering.LESS:
                        outcome = 1.
                    case Ordering.EQUAL:
                        outcome = 0.5
                    case Ordering.GREATER:
                        outcome = 0.
                probability = Glicko._win_probability(sig_perf, my_rating, rating)
                info += g * g * probability * (1. - probability)
                update += g * (outcome - probability)
            info = 0.25
            update /= float(len(all_ratings))
            info *= gli_q * gli_q
            sig = (1 / (my_rating.sig ** -2 + info)) ** 0.5
            update *= gli_q * sig * sig
            mu = my_rating.mu + update
            player.update_rating(Rating(mu, sig), 0.)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for rating, my_lo, _ in standings:
                executor.submit(_update_player_rating, rating, my_lo)

    @staticmethod
    def _g(sig_sq: float, sig_perf: float):
        return 1. / (1 + (sig_sq / sig_perf ** 2)) ** 0.5

    # @staticmethod
    # def _pr_i(mu_i: float, mu_j: float, sig_sq_i: float, sig_sq_j: float, sig_perf: float, scale: float):
    #     inner_g_sq = (sig_sq_i + sig_sq_j + sig_perf ** 2)
    #     diff_mu = (mu_i - mu_j)
    #     denom = (1 + 10 ** ((-Glicko._g(inner_g_sq, sig_perf) * diff_mu) / scale))
    #     return 1. / denom

    @staticmethod
    def _r(N: int, rank_i: int):
        return (N - rank_i) / comb(N, 2)

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
        sig_perf = self.beta / ((self.weight_limit * params.weight) ** 0.5)
        gli_q = TANH_MULTIPLIER / sig_perf
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
        N = len(team_ratings)
        prob_denom = comb(N, 2)

        def _update_player_rating(relative_rank: int, team_i: TeamRating):
            team_i_mu = team_i.rating.mu
            team_i_sig_sq = team_i.rating.sig
            pr_i = 0.
            info = 0.
            update = 0.
            team_i_players = team_standings[team_i.team]['players']
            num_players_in_team = len(team_i_players)
            r_i = (N - relative_rank) / prob_denom
            for team_j in team_ratings:
                if team_i is team_j:
                    continue
                team_j_sig_sq = team_j.rating.sig
                pr_i += Glicko._win_probability(sig_perf, team_i.rating, team_j.rating)  # TODO: make parameter for scale factor
            pr_i /= prob_denom
            for team_j in team_ratings:
                if team_i is team_j:
                    continue
                team_j_sig_sq = team_j.rating.sig
                g = Glicko._g(team_j_sig_sq, sig_perf)
                info += g * g * pr_i * (1 - pr_i)
                update += g * (r_i - pr_i)
            info *= gli_q * gli_q
            team_new_sig_sq = 1. / ((1. / team_i_sig_sq) + info)
            update *= gli_q * team_new_sig_sq
            team_new_sig = team_new_sig_sq ** 0.5
            team_new_mu = team_i_mu + update

            def _update_individual(player: Player):
                old_mu = player.approx_posterior.mu
                old_sig = player.approx_posterior.sig
                w_mu = 1. / num_players_in_team
                w_sig = 1. / num_players_in_team
                new_mu = old_mu + w_mu * (team_new_mu - team_i_mu)
                new_sig = max(1e-4, old_sig + w_sig * (team_new_sig - team_i_sig_sq ** 0.5))
                player.update_rating(Rating(new_mu, new_sig), 0)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(_update_individual,  team_i_players)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(_update_player_rating, *zip(*((i+1, team_i) for i, team_i in enumerate(team_ratings))))
