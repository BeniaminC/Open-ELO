import concurrent.futures
from dataclasses import dataclass


from ..common.aggregation import TeamRatingAggregation
from ..common.common import ContestRatingParams, Standings
from ..common.constants import DEFAULT_BETA, TANH_MULTIPLIER
from ..common.numericals import standard_logistic_cdf
from ..common.player import Player
from ..common.rating_system import RatingSystem
from ..common.term import Rating, TanhTerm, robust_average


__all__ = ['Codeforce']


@dataclass
class Codeforce(RatingSystem):
    '''
    Code Force rating system.
    '''
    beta: float = DEFAULT_BETA
    weight: float = 1.

    @staticmethod
    def _win_probability(sig_perf: float,
                         player: Rating,
                         foe: Rating) -> float:
        z = (player.mu - foe.mu) / sig_perf
        return standard_logistic_cdf(z)

    def compute_performance(
        self,
        sig_perf: float,
        better: list[Rating],
        worse: list[Rating],
        all: list[Rating],
        my_rating: Rating
    ) -> float:
        pos_offset: float = sum(map(lambda x: 1./x.sig, better))
        neg_offset: float = sum(map(lambda x: 1./x.sig, worse))
        all_offset: float = sum(map(lambda x: 1./x.sig, all))
        ac_rank = 0.5 * (pos_offset - neg_offset + all_offset + (1. / my_rating.sig))
        ex_rank = 0.5 / my_rating.sig + sum(map(lambda rating: Codeforce._win_probability(sig_perf, rating, my_rating) / rating.sig, all))
        geo_rank = (ac_rank * ex_rank) ** 0.5
        geo_offset = 2. * geo_rank - (1. / my_rating.sig) - all_offset
        geo_rating = robust_average(list(map(TanhTerm.from_rating, all)), TANH_MULTIPLIER * geo_offset, 0.)
        return geo_rating

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
        sig_perf = self.beta / params.weight ** 0.5

        def _update_player(player):
            return Rating(player.approx_posterior.mu, sig_perf)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            all_ratings: list[Rating] = list(executor.map(_update_player, (player for player, _, _ in standings)))

        def _update_player_rating(player: Player, lo: int, hi: int, my_rating: Rating):
            geo_perf = min(self.compute_performance(sig_perf,
                                                    all_ratings[:lo],
                                                    all_ratings[(hi+1):],
                                                    all_ratings,
                                                    my_rating),
                           params.perf_ceiling)
            wt = params.weight * self.weight
            mu = (my_rating.mu + wt * geo_perf) / (1. + wt)
            sig = player.approx_posterior.sig
            player.update_rating(Rating(mu, sig), geo_perf)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for (player, lo, hi), my_rating in zip(standings, all_ratings):
                executor.submit(_update_player_rating, player, lo, hi, my_rating)

    def team_round_update(self,
                          params: ContestRatingParams,
                          standings: Standings,
                          agg: TeamRatingAggregation) -> None:
        raise NotImplementedError()
