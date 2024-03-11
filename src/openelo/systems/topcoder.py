import concurrent.futures
from dataclasses import dataclass

from ..common.common import (ContestRatingParams, Standings)
from ..common.numericals import (clamp, standard_normal_cdf,
                                 standard_normal_cdf_inv)
from ..common.player import Player
from ..common.rating_system import RatingSystem
from ..common.term import Rating


__all__ = ['Topcoder']


@dataclass
class Topcoder(RatingSystem):
    '''
    Topcoder rating system.
    '''
    weight_noob: float = 0.6
    weight_limit: float = 0.18

    @staticmethod
    def _win_probability(sqrt_weight: float, player: Rating, foe: Rating) -> float:
        z = sqrt_weight * (player.mu - foe.mu) / ((player.sig ** 2 + foe.sig ** 2) ** 0.5)
        return standard_normal_cdf(z)

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
        num_coders = float(len(standings))
        avg_rating = sum(player.approx_posterior.mu for player, _, _ in standings) / num_coders
        mean_vol_sq = sum(player.approx_posterior.sig ** 2 for player, _, _ in standings) / num_coders
        if num_coders > 1:
            mean_vol_sq += sum((player.approx_posterior.mu - avg_rating) ** 2 for player, _, _ in standings) / (num_coders - 1)
        c_factor = mean_vol_sq ** 0.5
        sqrt_contest_weight = params.weight ** 0.5
        weight_extra = self.weight_noob - self.weight_limit

        def _update_player(player: Player, lo: int, hi: int):
            old_rating = player.approx_posterior.mu
            vol_sq = player.approx_posterior.sig ** 2
            ex_rank = sum(self._win_probability(sqrt_contest_weight, foe.approx_posterior, player.approx_posterior) for foe, _, _ in standings)
            ac_rank = float(0.5 * (1 + lo + hi))
            ex_perf = -standard_normal_cdf_inv(ex_rank / num_coders)
            ac_perf = -standard_normal_cdf_inv(ac_rank / num_coders)
            perf_as = old_rating + c_factor * (ac_perf - ex_perf)
            perf_as = min(perf_as, params.perf_ceiling)
            num_contests = float(player.times_played())
            weight = self.weight_limit + weight_extra / num_contests
            cap = 150. + 1500. / (num_contests + 1.)
            cap *= sqrt_contest_weight * weight / (0.18 + 0.42 / num_contests)
            weight *= params.weight / (1. - weight)
            if old_rating >= 2500:
                weight *= 0.8
            elif old_rating >= 2000:
                weight *= 0.9
            try_rating = (old_rating + weight * perf_as) / (1. + weight)
            new_rating = clamp(try_rating, old_rating - cap, old_rating + cap)
            new_vol = ((try_rating - old_rating) ** 2 / weight + vol_sq / (1. + weight)) ** 0.5

            return Rating(new_rating, new_vol), perf_as
        with concurrent.futures.ThreadPoolExecutor() as executor:
            new_ratings = executor.map(_update_player, *zip(*standings))

        def _update_player_rating(player: Player, new_rating: Rating, new_perf: float):
            player.update_rating(new_rating, new_perf)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(_update_player_rating, *zip(*((player, new_rating, new_perf)
                                                       for ((player, _, _), (new_rating, new_perf))
                                                       in zip(standings, new_ratings))))

