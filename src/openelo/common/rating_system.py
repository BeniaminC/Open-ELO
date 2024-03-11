from abc import ABC, abstractmethod
import concurrent.futures

from .common import ContestRatingParams, Standings
from .constants import SECS_PER_DAY


class RatingSystem(ABC):
    '''
    Abstract base class for rating systems.
    '''
    @abstractmethod
    def round_update(self, params: ContestRatingParams, standings: Standings) -> None:
        '''
        Abstract method required to be defined for `RatingSystem`.

        Args:
            params (:obj:`ContestRatingParams`): Contest rating parameters for individual
            contest.

            standings (:obj:`Standings`): List of `Player` objects with
            their standings (start, end).
        '''
        pass

    @staticmethod
    def init_players_event(standings: Standings, contest_time: int = 0) -> None:
        '''
        Initialize the `PlayerEvent` for each player in the list `standings`.
        If the rating system does not utilize contest time `contest_time`, then
        do not set it.

        Args:
            standings (:obj:`Standings`): List of `Player` objects with
            their standings (start, end).

            contest_time (:obj:`int`): Contest time (in seconds).
        Returns:
            :obj:`None`
        '''
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for player, lo, _ in standings:
                executor.submit(player.init_player_event, lo, contest_time)

    def compute_weight(self, contest_weight: float, weight_limit: float, noob_delay: list[float], n: int) -> float:
        '''
        Compute the new weight based on `contest_weight`, `weight_limit`, and `noob_delay`. 
        This simply computes the product of all three.

        Args:
            contest_weight (:obj:`float`): Weight of individual contest.

            weight_limit (:obj:`float`): Weight limit of the rating system.

            noob_delay (:obj:`list[float]`): List of weights for noob delay for individual
            player.  If player played less than or equal to the length of list, then the 
            weight at the i'th game of player is applied.

            n (:obj:`int`): The i'th game of the player.
        Returns:
            :obj:`float`: The computed weight.
        
        '''
        computed_weight = contest_weight * weight_limit
        if n < len(noob_delay):
            computed_weight *= noob_delay[n]
        return computed_weight

    def compute_sig_perf(self, weight: float, sig_limit: float, drift_per_day: float) -> float:
        '''
        Computes the performance deviation given the total weight, sigma limit,
        and drifts-per-day. This computes the discrete performance (weight and
        sigma limit) and the continuous performance (drifts-per-day and weight).

        Args:
            weight (:obj:`float`): total weight of the game.

            sig_limit (:obj:`float`): The sigma limit.

            drift_per_day (:obj:`float`): The rating drifts per day (e.g., 10./7.
            will increase deviation by 10 every week).
        Returns:
            :obj:`float`: the performance deviation (sigma).
        '''
        discrete_perf = (1. + 1. / weight) * sig_limit * sig_limit
        continuous_perf = drift_per_day / weight
        return (discrete_perf + continuous_perf) ** 0.5

    def compute_sig_drift(self, weight: float, sig_limit: float, drift_per_day: float, delta_secs: float, ) -> float:
        '''
        Computes the deviation given the weight, sigma limit, drifts-per-day, and
        the change in seconds between player contest. This is typically used to 
        add noise to the skill posterior of a player.

        Args:
            weight (:obj:`float`): total weight of the game.

            sig_limit (:obj:`float`): The sigma limit.

            drift_per_day (:obj:`float`): The rating drifts per day (e.g., 10./7.
            will increase deviation by 10 every week).

            delt_secs (:obj:`float`): Seconds between games of particular player.
        Returns:
            :obj:`float`: Deviation drift (sigma).
        '''
        discrete_drift = weight * sig_limit * sig_limit
        continuous_drift = drift_per_day * delta_secs / SECS_PER_DAY
        return (discrete_drift + continuous_drift) ** 0.5