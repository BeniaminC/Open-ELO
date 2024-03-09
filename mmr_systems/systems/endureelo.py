import concurrent.futures
from dataclasses import dataclass

from mmr_systems.common.common import (ContestRatingParams, Standings)
from mmr_systems.common.constants import (DEFAULT_BETA)
from mmr_systems.common.numericals import standard_logistic_cdf
from mmr_systems.common.player import Player
from mmr_systems.common.rating_system import RatingSystem
from mmr_systems.common.term import Rating


@dataclass
class EndureElo(RatingSystem):
    beta: float = DEFAULT_BETA
    sig_drift: float = 35.

    @staticmethod
    def _win_probability(sig_perf: float, player: Rating, foe: Rating) -> float:
        z = (player.mu, foe.mu) / ((foe.sig ** 2. + sig_perf ** 2.) ** 0.5)
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
        self.init_players_event(standings)

        def _update_player(player: Player):
            player.add_noise_and_collapse(self.sig_drift)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(_update_player, (player for player, _, _ in standings))

        def _update_player_rating(player: Player):
            my_rating = player.approx_posterior
            probability = 0.5
            info = probability * (1.0 - probability)
            sig = (1. / (my_rating.sig ** -2 + info)) ** 0.5
            mu = my_rating.mu
            player.update_rating(Rating(mu, sig), 0.)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(_update_player_rating, (player for player, _, _ in standings))
