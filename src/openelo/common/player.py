from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Self
from .constants import DEFAULT_MU, DEFAULT_SIG, DEFAULT_SIG_LIMIT
from .term import Rating, TanhTerm, robust_average


__all__ = ['Player',
           'PlayersByName']


@dataclass
class PlayerEvent:
    contest_index: Optional[int]
    rating_mu: float
    rating_sig: float
    perf_score: int
    place: int

    def get_display_rating(self) -> int:
        '''
        Get the display rating according to the DEFAULT_SIG_LIMIT.
        '''
        return round(self.rating_mu - 3. * (self.rating_sig - DEFAULT_SIG_LIMIT))

    def display_rating(self, stdevs: float = 2, sig_limit: float = DEFAULT_SIG_LIMIT) -> str:
        return f"{self.rating_mu} ± {stdevs * (self.rating_sig - sig_limit)}"


@dataclass
class Player:
    '''
    Player class to track normal factor, logistic factors, event history, approximate posterior,
    update time, and the change of time (delta time).  Defaults to the global variables DEFAULT_*.

    Example::
    
        players = {'Ben': Player(), 'John': Player()}
    '''
    normal_factor: Rating = field(default_factory=lambda: Rating(mu=DEFAULT_MU, sig=DEFAULT_SIG))
    logistic_factors: deque[TanhTerm] = field(default_factory=deque)
    event_history: list[PlayerEvent] = field(default_factory=list)
    approx_posterior: Rating = field(default_factory=lambda: Rating(mu=DEFAULT_MU, sig=DEFAULT_SIG))
    update_time: int = 0
    delta_time: int = 0

    @classmethod
    def with_rating(cls, mu: float, sig: float, update_time: int) -> Self:
        '''
        Class method to create a new player object from rating and update time.  Sets the normal
        factor and approximate posterior ratings to `mu` and `sig`.

        Args:
            mu (:obj;`float`): The rating of the player.

            sig (:obj;`float`): The deviation of the player's rating

            update_time(:obj;`int`): The starting time (in seconds).
        
        Returns:
            :obj:`Self`: Return a new player object given the arguments.
        '''

        return cls(normal_factor=Rating(mu, sig),
                   logistic_factors=deque(),
                   event_history=[],
                   approx_posterior=Rating(mu, sig),
                   update_time=update_time,
                   delta_time=0)

    def times_played(self) -> int:
        '''
        Return an the number of times played.

        Returns:
            :obj:`int`: The number of times played.
        '''
        return len(self.event_history)

    def times_played_excl(self) -> int:
        '''
        Return the `length - 1` of the player's played history. This is to get the number of games
        played if a history event has already been appended.

        Returns:
            :obj:`int`: The number of games excluding the unrecorded initialized game.
        '''
        return len(self.event_history) - 1

    def is_newcomer(self) -> bool:
        '''
        Returns a boolean of if a player has zero games or not.

        Returns:
            :obj:`bool`: If a player has zero history events.
        '''
        return self.times_played_excl() == 0

    def update_rating(self, rating: Rating, performance_score: float) -> None:
        '''
        Updates the last event of the player, given the rating and performance score.
        The values are rounded. Requires the last event to be empty (all zeroes).

        Args:
            rating (:obj:`Rating`): The rating object containing the rating and deviation.

            performance score (:obj:`float`): The performance score of the event.
        
        Returns:
            :obj:`None`
        '''
        # assume that a placeholder history item has been pushed containing contest id and time
        last_event = self.event_history[-1]
        assert last_event.rating_mu == 0
        assert last_event.rating_sig == 0
        assert last_event.perf_score == 0
        self.approx_posterior = rating
        last_event.rating_mu = round(rating.mu)
        last_event.rating_sig = round(rating.sig)
        last_event.perf_score = round(performance_score)

    def update_rating_with_normal(self, performance: Rating) -> None:
        '''
        Updates the rating under normal distribution. This updates the `normal_factor` of the
        player.  If there are logistical factors, set the new rating with the robust average
        with the logistical factors.

        Args:
            performance (:obj:`Rating`): The rating of the performance to update the player's rating.

        Returns:
            :obj:`None`
        '''
        wn = self.normal_factor.sig ** -2
        wp = performance.sig ** -2
        self.normal_factor.mu = (wn * self.normal_factor.mu + wp * performance.mu) / (wn + wp)
        self.normal_factor.sig = (wn + wp) ** -0.5
        new_rating = self.normal_factor if len(self.logistic_factors) == 0 else self.approximate_posterior(performance.sig)
        self.update_rating(new_rating, performance.mu)

    def update_rating_with_logistic(self, performance: Rating, max_history: int) -> None:
        '''
        Updates the player rating under logistic distribution. This updates the `normal_factor` of
        the player if the length of logistical factors equal max_history and create and appends a 
        `TanhTerm` to the `logistic_factors`.  The new rating is the robust average of the logistic
        factors. 

        Args:
            performance (:obj:`Rating`): The rating of the performance to update the player's rating.

            max_history (:obj:`int`): The number of logistical factors to include.

        Returns:
            :obj:`None`     
        '''
        if len(self.logistic_factors) >= max_history:
            logistic = self.logistic_factors.popleft()
            wn = self.normal_factor.sig ** -2
            wl = logistic.get_weight()
            self.normal_factor.mu = (wn * self.normal_factor.mu + wl * logistic.mu) / (wn + wl)
            self.normal_factor.sig = (wn + wl) ** -0.5
        self.logistic_factors.append(TanhTerm.from_rating(performance))
        new_rating = self.approximate_posterior(performance.sig)
        self.update_rating(new_rating, performance.mu)

    def approximate_posterior(self, perf_sig: float) -> Rating:
        '''
        Create a `Rating` object from the robust average of the `logistic_factors`, given the
        performance deviation.

        Args:
            perf_sig (:obj:`float`): The performance deviation.
        
        Returns:
            :obj:`Rating`: The approximate posterior rating.
        '''
        normal_weight = self.normal_factor.sig ** -2
        mu = robust_average(
            self.logistic_factors.copy(),
            -self.normal_factor.mu * normal_weight,
            normal_weight,
        )
        sig = (self.approx_posterior.sig ** -2 + perf_sig ** -2) ** -0.5
        return Rating(mu, sig)

    def add_noise_and_collapse(self, sig_noise: float) -> None:
        '''
        Adds noise to the approximate posterior and set the normal factor to the approximate posterior.
        Clears all the logistic factors.

        Args:
            sig_noise (:obj:`float`): the noise added to the approximate posterior.
        
        Return::
            :obj:`None`
        
        '''
        self.approx_posterior = self.approx_posterior.with_noise(sig_noise)
        self.normal_factor = self.approx_posterior
        self.logistic_factors.clear()

    def add_noise_in_front(self, sig_noise: float) -> None:
        '''
        Adds noise to the normal factor from the logstic factors.

        Args:
            sig_noise (:obj:`float`): the noise added to the approximate posterior and changes the weight
            of each logistic factor.

        Returns:
            :obj:`None`
        '''
        decay = 1.0
        decay *= sig_noise / self.approx_posterior.sig
        self.approx_posterior.sig *= decay
        self.normal_factor.sig *= decay
        for rating in self.logistic_factors:
            rating.w_out /= decay * decay

    def add_noise_best(self, sig_noise: float, transfer_speed: float) -> None:
        '''
        Adds noise to the approximate posterior and transfers weights from the logistic
        factors to the normal (Gaussian) factors.

        Args:
            sig_noise (:obj:`float`): the noise added to the approximate posterior and changes the weight
            of each logistic factor.

            transfer_speed (:obj:`float`): Transfer speed of weights from the logistic factors to the normal
            (Gassian) factor.
        
        Returns:
            :obj:`None`
        '''
        # sig noise is beta, transfer speed is rho
        new_posterior = self.approx_posterior.with_noise(sig_noise)  # u_i, sqrt(sig, beta)
        decay = (self.approx_posterior.sig / new_posterior.sig) ** 2  # gamma is new posterior
        transfer = decay ** transfer_speed  # K^(rho)
        self.approx_posterior = new_posterior  # set u^pi_i, d_i
        wt_norm_old = self.normal_factor.sig ** -2  # w_(i,0)
        wt_from_norm_old = transfer * wt_norm_old  # W_G
        wt_from_transfers = (1. - transfer) * (wt_norm_old + sum(r.get_weight() for r in self.logistic_factors))  # W_L
        wt_total = wt_from_norm_old + wt_from_transfers  # (W_G + W_L)
        self.normal_factor.mu = (wt_from_norm_old * self.normal_factor.mu + wt_from_transfers * self.approx_posterior.mu) / wt_total  # p_(i,0)
        self.normal_factor.sig = (decay * wt_total) ** -0.5  # w_(i,0)
        for r in self.logistic_factors:  # for loop
            r.w_out *= transfer * decay

    def init_player_event(self, lo: int, contest_time: int = 0) -> None:
        '''
        Initializes a player event with the `lo` placement and the contest time.  The values initialized are placeholders (zeroes).

        Args:
            lo (:obj:`int`): The placement of the player.

            contest_time (:obj:`int`): The contest time of the event.
        
        Returns:
            :obj:`None`
        '''
        self.delta_time = contest_time - self.update_time
        self.update_time = contest_time
        self.event_history.append(
            PlayerEvent(contest_index=None,
                        rating_mu=0,
                        rating_sig=0,
                        perf_score=0,
                        place=lo)
        )


PlayersByName = dict[str, Player]
