import concurrent.futures
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from functools import reduce
from itertools import chain, groupby
from operator import itemgetter
from typing import Self

from ..common.aggregation import TeamRatingAggregation
from ..common.bucket import bucket, cmp_by_bucket, same_bucket
from ..common.constants import (BOUNDS, DEFAULT_WEIGHT_LIMIT, DEFAULT_SIG_LIMIT,
                                DEFAULT_DRIFTS_PER_DAY, DEFAULT_SPLIT_TIES,
                                INT_MAX, DEFAULT_TRANSFER_SPEED, FLOAT_MAX)
from ..common.common import (ContestRatingParams, EloMMRVariant, TanhTerm,
                             eval_equal, eval_grea, eval_less, Standings)
from ..common.numericals import (clamp, solve_newton)
from ..common.player import Player
from ..common.rating_system import RatingSystem
from ..common.team_rating_system import TeamRating, TeamRatingSystem
from ..common.term import Rating


__all__ = ['SimpleEloMMR',
           'EloMMR']


@dataclass
class SimpleEloMMR(RatingSystem, TeamRatingSystem):
    '''
    Simple Elo-MMR rating system.
    '''
    weight_limit: float = DEFAULT_WEIGHT_LIMIT
    noob_delay: list[float] = field(default_factory=list)
    sig_limit: float = DEFAULT_SIG_LIMIT
    drift_per_day: float = DEFAULT_DRIFTS_PER_DAY

    split_ties: bool = DEFAULT_SPLIT_TIES
    history_len: int = INT_MAX
    transfer_speed: float = DEFAULT_TRANSFER_SPEED

    def individual_update(self, params: ContestRatingParams, player: Player, mu_perf: float):
        weight = self.compute_weight(params.weight, self.weight_limit, self.noob_delay, player.times_played_excl())
        sig_perf = self.compute_sig_perf(weight, self.sig_limit, self.drift_per_day)
        sig_drift = self.compute_sig_drift(weight, self.sig_limit, self.drift_per_day, float(player.delta_time))
        player.add_noise_best(sig_drift, self.transfer_speed)
        player.update_rating_with_logistic(
            Rating(
                mu=mu_perf,
                sig=sig_perf,
            ),
            self.history_len,
        )

    def round_update(self, params: ContestRatingParams, standings: Standings, contest_time: int = 0) -> None:
        '''
        Update the player ratings according to the standings.

        Args:
            params (:obj:`ContestRatingParams`): Parameters of a particular contest.

            standings (:obj:`Standings): Standings of each player
            according to `team` and `rank`, respectively. Must be in order.
        '''
        self.init_players_event(standings, contest_time)

        def _update_player(player: Player):
            weight = self.compute_weight(params.weight, self.weight_limit, self.noob_delay, player.times_played_excl())
            sig_perf = self.compute_sig_perf(weight, self.sig_limit, self.drift_per_day)
            sig_drift = self.compute_sig_drift(weight, self.sig_limit, self.drift_per_day, float(player.delta_time))
            player.add_noise_best(sig_drift, self.transfer_speed)

            return TanhTerm.from_rating(player.approx_posterior.with_noise(sig_perf))
        with concurrent.futures.ThreadPoolExecutor() as executor:
            tanh_terms = list(executor.map(_update_player, (player for player, _, _ in standings)))
        mul = 1. if self.split_ties else 2.

        def _update_player_rating(player: Player, lo: int, hi: int):
            bounds = BOUNDS

            def f(x):
                itr1 = (eval_less(term, x) for term in tanh_terms[:lo])
                itr2 = (eval_equal(term, x, mul) for term in tanh_terms[lo:hi+1])
                itr3 = (eval_grea(term, x) for term in tanh_terms[hi+1:])
                chained = chain(
                    itr1,
                    itr2,
                    itr3)
                acc = reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]), chained, (0., 0.))
                return acc
            mu_perf = clamp(solve_newton(bounds, f), params.perf_floor, params.perf_ceiling)
            weight = self.compute_weight(params.weight, self.weight_limit, self.noob_delay, player.times_played_excl())
            sig_perf = self.compute_sig_perf(weight, self.sig_limit, self.drift_per_day)
            player.update_rating_with_logistic(Rating(mu_perf, sig_perf), self.history_len)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(_update_player_rating, *zip(*((player, lo, hi) for player, lo, hi in standings)))

    def team_round_update(self,
                          params: ContestRatingParams,
                          standings: Standings,
                          agg: TeamRatingAggregation,
                          contest_time: int = 0) -> None:
        '''
        Update the player ratings in teams according to their team and rank.

        Args:
            params (:obj:`ContestRatingParams`): Parameters of a particular contest.

            standings (:obj:`Standings): Standings of each player
            according to their `team` and `rank`. Must be in order.
        '''
        self.init_players_event(standings, contest_time)

        def _update_player(player: Player, team: int):

            weight = self.compute_weight(params.weight, self.weight_limit, self.noob_delay, player.times_played_excl())
            sig_perf = self.compute_sig_perf(weight, self.sig_limit, self.drift_per_day)
            sig_drift = self.compute_sig_drift(weight, self.sig_limit, self.drift_per_day, float(player.delta_time))
            player.add_noise_best(sig_drift, self.transfer_speed)
            return (sig_perf, team)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            sig_perfs = list(executor.map(_update_player, *zip(*((player, team) for player, team, _ in standings))))
        agg_sig_perfs = {k: sum(v[0] ** 2 for v in g) for k, g in groupby(sorted(sig_perfs, key=itemgetter(1)), key=itemgetter(1))}
        team_standings = self.convert_to_teams(standings)
        team_ratings = list(TeamRating(team, team_info['rank'], agg(team_info['players'])) for team, team_info in team_standings.items())
        team_ratings.sort(key=lambda team: team.rank)

        def _team_tanh_terms(team_i: TeamRating):
            return TanhTerm.from_rating(Rating(team_i.rating.mu, (agg_sig_perfs[team_i.team] + team_i.rating.sig) ** 0.5))
        with concurrent.futures.ThreadPoolExecutor() as executor:
            team_tanh_terms = list(executor.map(_team_tanh_terms, team_ratings))
        mul = 1. if self.split_ties else 2.

        def _update_player_rating(team_i: TeamRating):
            team_i_players = team_standings[team_i.team]['players']
            team_size = len(team_i_players)
            bounds = (1500 - (7500 * team_size), 1500 + (7500 * team_size))
            lo = bisect_left(team_ratings, team_i.rank, key=lambda x: x.rank)
            hi = bisect_right(team_ratings, team_i.rank, key=lambda x: x.rank)

            def f(x):
                itr1 = (eval_less(term, x) for term in team_tanh_terms[:lo])
                itr2 = (eval_equal(term, x, mul) for term in team_tanh_terms[lo:hi])
                itr3 = (eval_grea(term, x) for term in team_tanh_terms[hi:])
                chained = chain(
                    itr1,
                    itr2,
                    itr3)
                acc = reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]), chained, (0., 0.))
                return acc
            # TODO: make team performance clamp coincide with aggregation
            team_mu_perf = clamp(solve_newton(bounds, f), params.perf_floor * team_size, params.perf_ceiling * team_size)
            team_i_mu = team_i.rating.mu

            def _update_individual(player: Player):
                teammates_rating = team_i_mu - player.approx_posterior.mu
                player_mu_perf = clamp(team_mu_perf - teammates_rating, params.perf_floor, params.perf_ceiling)
                weight = self.compute_weight(params.weight, self.weight_limit, self.noob_delay, player.times_played_excl())
                sig_perf = self.compute_sig_perf(weight, self.sig_limit, self.drift_per_day)
                player.update_rating_with_logistic(Rating(player_mu_perf, sig_perf), self.history_len)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(_update_individual,  team_i_players)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(_update_player_rating, team_ratings)


@dataclass
class EloMMR(RatingSystem, TeamRatingSystem):
    '''
    Elo-MMR rating system.
    '''
    weight_limit: float = DEFAULT_WEIGHT_LIMIT
    noob_delay: list[float] = field(default_factory=list)
    sig_limit: float = DEFAULT_SIG_LIMIT
    drift_per_day: float = DEFAULT_DRIFTS_PER_DAY
    split_ties: bool = DEFAULT_SPLIT_TIES
    subsample_size: int = INT_MAX
    subsample_bucket: float = 1e-5
    variant: EloMMRVariant = field(default_factory=lambda: EloMMRVariant.logistic(1.))

    def __post_init__(self):
        assert self.weight_limit > 0
        assert self.sig_limit > 0

    @classmethod
    def from_limit(cls,
                   weight_limit: float,
                   sig_limit: float,
                   split_ties: bool,
                   fast: bool,
                   variant: EloMMRVariant) -> Self:
        noob_delay: list[float] = []
        subsample_size = 100 if fast else INT_MAX
        subsample_bucket = 2. if fast else 1e-5
        return cls(
            weight_limit,
            noob_delay,
            sig_limit,
            0.,
            split_ties,
            subsample_size,
            subsample_bucket,
            variant)

    @classmethod
    def default_fast(cls) -> Self:
        return cls.from_limit(DEFAULT_WEIGHT_LIMIT, DEFAULT_SIG_LIMIT, DEFAULT_SPLIT_TIES, True, EloMMRVariant.logistic(1.))

    @classmethod
    def default_gaussian(cls) -> Self:
        return cls.from_limit(DEFAULT_WEIGHT_LIMIT, DEFAULT_SIG_LIMIT, DEFAULT_SPLIT_TIES, False, EloMMRVariant.gaussian())

    @classmethod
    def default_gaussian_fast(cls) -> Self:
        return cls.from_limit(DEFAULT_WEIGHT_LIMIT, DEFAULT_SIG_LIMIT, DEFAULT_SPLIT_TIES, True, EloMMRVariant.gaussian())

    @staticmethod
    def subsample(terms: list[tuple[Rating, list[int]]], rating: float, num_samples: int, subsample_bucket: float) -> range:
        beg = bisect_left(terms, rating, key=lambda term: cmp_by_bucket(term[0].mu, rating, subsample_bucket) > 0)
        end = beg + 1
        expand = (num_samples - (end - beg) + 1) // 2
        beg = max(0, beg - expand)
        end = min(len(terms), end + expand)
        expand = num_samples - (end - beg)
        beg = max(0, beg - expand)
        end = min(len(terms), end + expand)
        return range(beg, end)

    def round_update(self, params: ContestRatingParams, standings: Standings, contest_time: int = 0):
        '''
        Update the player ratings according to the standings.

        Args:
            params (:obj:`ContestRatingParams`): Parameters of a particular contest.

            standings (:obj:`Standings): Standings of each player
            according to `team` and `rank`, respectively. Must be in order.
        '''
        self.init_players_event(standings, contest_time)

        def _update_player(player: Player, lo: int):
            weight = self.compute_weight(params.weight, self.weight_limit, self.noob_delay, player.times_played_excl())
            sig_perf = self.compute_sig_perf(weight, self.sig_limit, self.drift_per_day)
            sig_drift = self.compute_sig_drift(weight, self.sig_limit, self.drift_per_day, float(player.delta_time))
            transfer_speed = self.variant.value
            match self.variant:
                case EloMMRVariant('Logistic') if transfer_speed is not None and transfer_speed < FLOAT_MAX:
                    player.add_noise_best(sig_drift, transfer_speed)
                case _:
                    player.add_noise_and_collapse(sig_drift)
            return (player.approx_posterior.with_noise(sig_perf), lo)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            base_terms = list(executor.map(_update_player, *zip(*((player, lo) for player, lo, _ in standings))))
        base_terms.sort(key=lambda x: (
            bucket(x[0].mu, self.subsample_bucket),
            bucket(x[0].sig, self.subsample_bucket),
            x[1])
        )
        normal_terms: list[tuple[Rating, list[int]]] = []
        for term, lo in base_terms:
            if normal_terms:
                last_term, ranks = normal_terms[-1]
                if same_bucket(last_term.mu, term.mu, self.subsample_bucket) and same_bucket(last_term.sig, term.sig, self.subsample_bucket):
                    length = float(len(ranks))
                    last_term.mu = (length * last_term.mu + term.mu) / (length + 1)
                    last_term.sig = (length * last_term.sig + term.sig) / (length + 1)
                    ranks.append(lo)
                    continue
            normal_terms.append((term, [lo]))
        tanh_terms: list[tuple[TanhTerm, list[int]]] = []
        for rating, ranks in normal_terms:
            tanh_terms.append((TanhTerm.from_rating(rating), ranks.copy()))

        idx_len_max = 9999

        def _update_player_rating(player: Player, my_rank: int):
            player_mu = player.approx_posterior.mu
            idx_subsample = self.subsample(normal_terms, player_mu, self.subsample_size, self.subsample_bucket)
            idx_len_upper_bound = len(idx_subsample)
            if idx_len_max < idx_len_upper_bound:
                print(f'Subsampling {idx_len_upper_bound} opponents might be slow; consider decreasing subsample_size.')
            bounds = BOUNDS
            weight = self.compute_weight(params.weight, self.weight_limit, self.noob_delay, player.times_played_excl())
            sig_perf = self.compute_sig_perf(weight, self.sig_limit, self.drift_per_day)
            if self.variant == EloMMRVariant.gaussian():
                normal_subsample: list[tuple[Rating, list[int]]] = []
                for i in idx_subsample:
                    normal_subsample.append(normal_terms[i])

                def f(x):
                    res: list[tuple[float, float]] = []
                    for rating, ranks in normal_subsample:
                        res.append(rating.evals(x, ranks, my_rank, self.split_ties))
                    acc = reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]), res, (0., 0.))
                    return acc
                mu_perf = solve_newton(bounds, f)
                player.update_rating_with_normal(Rating(mu_perf, sig_perf))
            elif self.variant == EloMMRVariant.logistic():
                tanh_subsample: list[tuple[TanhTerm, list[int]]] = []
                for i in idx_subsample:
                    tanh_subsample.append(tanh_terms[i])

                def f(x):
                    res: list[tuple[float, float]] = []
                    for rating, ranks in tanh_subsample:
                        res.append(rating.evals(x, ranks, my_rank, self.split_ties))
                    acc = reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]), res, (0., 0.))
                    return acc
                mu_perf = clamp(solve_newton(bounds, f), params.perf_floor, params.perf_ceiling)
                player.update_rating_with_logistic(Rating(mu_perf, sig_perf), self.subsample_size)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(_update_player_rating, *zip(*((player, lo) for player, lo, _ in standings)))

    def team_round_update(self, params: ContestRatingParams, standings: Standings, agg: TeamRatingAggregation, contest_time: int = 0):
        '''
        Update the player ratings in teams according to their team and rank.

        Args:
            params (:obj:`ContestRatingParams`): Parameters of a particular contest.

            standings (:obj:`Standings): Standings of each player
            according to their `team` and `rank`. Must be in order.
        '''
        self.init_players_event(standings, contest_time)

        def _update_player(player: Player, team: int):
            weight = self.compute_weight(params.weight, self.weight_limit, self.noob_delay, player.times_played_excl())
            sig_perf = self.compute_sig_perf(weight, self.sig_limit, self.drift_per_day)
            sig_drift = self.compute_sig_drift(weight, self.sig_limit, self.drift_per_day, float(player.delta_time))
            transfer_speed = self.variant.value
            match self.variant:
                case EloMMRVariant('Logistic') if transfer_speed is not None and transfer_speed < FLOAT_MAX:
                    player.add_noise_best(sig_drift, transfer_speed)
                case _:
                    player.add_noise_and_collapse(sig_drift)
            return (sig_perf, team)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            sig_perfs = list(executor.map(_update_player, *zip(*((player, team) for player, team, _ in standings))))

        agg_sig_perfs = {k: sum(v[0] ** 2 for v in g) for k, g in groupby(sorted(sig_perfs, key=itemgetter(1)), key=itemgetter(1))}
        team_standings = self.convert_to_teams(standings)
        team_ratings = list(TeamRating(team, team_info['rank'], agg(team_info['players'])) for team, team_info in team_standings.items())
        team_ratings.sort(key=lambda team: team.rank)

        def _team_base_terms(team_i: TeamRating):
            return Rating(team_i.rating.mu, (team_i.rating.sig + agg_sig_perfs[team_i.team]) ** 0.5), team_i.rank
        with concurrent.futures.ThreadPoolExecutor() as executor:
            team_base_terms = list(executor.map(_team_base_terms, team_ratings))
        team_base_terms.sort(key=lambda x: (
            bucket(x[0].mu, self.subsample_bucket),
            bucket(x[0].sig, self.subsample_bucket),
            x[1])
        )
        team_normal_terms: list[tuple[Rating, list[int]]] = []
        for term, rank in team_base_terms:
            if team_normal_terms:
                last_term, ranks = team_normal_terms[-1]
                if same_bucket(last_term.mu, term.mu, self.subsample_bucket) and same_bucket(last_term.sig, term.sig, self.subsample_bucket):
                    length = float(len(ranks))
                    last_term.mu = (length * last_term.mu + term.mu) / (length + 1)
                    last_term.sig = (length * last_term.sig + term.sig) / (length + 1)
                    ranks.append(rank)
                    continue
            team_normal_terms.append((term, [rank]))

        def _team_tanh_terms(team_normal_term: Rating, ranks: list[int]):
            return TanhTerm.from_rating(team_normal_term), ranks
        with concurrent.futures.ThreadPoolExecutor() as executor:
            team_tanh_terms = list(executor.map(_team_tanh_terms, *zip(*((team_normal_term, ranks) for team_normal_term, ranks in team_normal_terms))))
        idx_len_max = 9999

        def _update_player_rating(team_i: TeamRating):
            team_i_mu = team_i.rating.mu
            idx_subsample = self.subsample(team_normal_terms, team_i_mu, self.subsample_size, self.subsample_bucket)
            idx_len_upper_bound = len(idx_subsample)
            if idx_len_max < idx_len_upper_bound:
                print(f'Subsampling {idx_len_upper_bound} opponents might be slow; consider decreasing subsample_size.')
            team_i_players = team_standings[team_i.team]['players']
            team_size = len(team_i_players)
            bounds = (1500 - (7500 * team_size), 1500 + (7500 * team_size))
            if self.variant == EloMMRVariant.gaussian():
                normal_subsample: list[tuple[Rating, list[int]]] = []
                for i in idx_subsample:
                    normal_subsample.append(team_normal_terms[i])

                def f(x):
                    res: list[tuple[float, float]] = []
                    for rating, ranks in normal_subsample:
                        res.append(rating.evals(x, ranks, team_i.rank, self.split_ties))
                    acc = reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]), res, (0., 0.))
                    return acc
                team_mu_perf = solve_newton(bounds, f)
            elif self.variant == EloMMRVariant.logistic():
                tanh_subsample: list[tuple[TanhTerm, list[int]]] = []
                for i in idx_subsample:
                    tanh_subsample.append(team_tanh_terms[i])

                def f(x):
                    res: list[tuple[float, float]] = []
                    for rating, ranks in tanh_subsample:
                        res.append(rating.evals(x, ranks, team_i.rank, self.split_ties))
                    acc = reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]), res, (0., 0.))
                    return acc
                # TODO: make team performance clamp coincide with aggregation
                team_mu_perf = clamp(solve_newton(bounds, f), params.perf_floor * team_size, params.perf_ceiling * team_size)

            def _update_individual(player: Player):
                teammates_rating = team_i_mu - player.approx_posterior.mu
                player_mu_perf = clamp(team_mu_perf - teammates_rating, params.perf_floor, params.perf_ceiling)
                weight = self.compute_weight(params.weight, self.weight_limit, self.noob_delay, player.times_played_excl())
                sig_perf = self.compute_sig_perf(weight, self.sig_limit, self.drift_per_day)
                if self.variant == EloMMRVariant.gaussian():
                    player.update_rating_with_normal(Rating(player_mu_perf, sig_perf))
                elif self.variant == EloMMRVariant.logistic():
                    player.update_rating_with_logistic(Rating(player_mu_perf, sig_perf), self.subsample_size)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(_update_individual,  team_i_players)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(_update_player_rating, team_ratings)
