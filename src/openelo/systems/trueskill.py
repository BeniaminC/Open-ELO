import concurrent.futures
from collections import namedtuple
from dataclasses import dataclass, field
from itertools import chain, groupby
from operator import itemgetter

import trueskill as ts

from ..common.aggregation import TeamRatingAggregation
from ..common.common import (ContestRatingParams, Standings)
from ..common.constants import (TS_MU, TS_SIG, TS_BETA, TS_TAU,
                                TS_DRAW_PROB, TS_BACKEND, DEFAULT_WEIGHT_LIMIT,
                                DEFAULT_SIG_LIMIT, DEFAULT_DRIFTS_PER_DAY)
from ..common.player import Player
from ..common.rating_system import RatingSystem
from ..common.team_rating_system import TeamRatingSystem
from ..common.term import Rating


__all__ = ['TrueSkill']


TeamRating = namedtuple('TeamRating', ['team', 'rank'])

@dataclass
class TrueSkill(RatingSystem, TeamRatingSystem):
    '''
    Trueskill rating system (an extension of Trueskill.py)
    '''
    ts_env: ts.TrueSkill = field(default_factory=lambda: ts.TrueSkill(TS_MU,
                                                                      TS_SIG,
                                                                      TS_BETA,
                                                                      TS_TAU,
                                                                      TS_DRAW_PROB,
                                                                      TS_BACKEND))
    weight_limit: float = DEFAULT_WEIGHT_LIMIT
    noob_delay: list[float] = field(default_factory=list)
    sig_limit: float = DEFAULT_SIG_LIMIT
    drift_per_day: float = DEFAULT_DRIFTS_PER_DAY

    def _convert_player_to_ts_rating(self, rating: Rating) -> ts.Rating:
        return self.ts_env.create_rating(rating.mu, rating.sig)

    def _convert_standings_to_ts_ranks(self, ratings: list[tuple[ts.Rating, int]]) -> list[tuple[ts.Rating, ...]]:
        return list(tuple(r[0] for r in g) for _, g in groupby(ratings, key=itemgetter(1)))

    def round_update(self,
                    params: ContestRatingParams,
                    standings: Standings) -> None:
        raise NotImplementedError()

    def team_round_update(self,
                          params: ContestRatingParams,
                          standings: Standings,
                          agg: TeamRatingAggregation | None = None,
                          contest_time: int = 0) -> None:
        self.init_players_event(standings, contest_time=contest_time)

        def _update_player(player: Player, lo: int):
            weight = self.compute_weight(params.weight, self.weight_limit, self.noob_delay, player.times_played_excl())
            sig_drift = self.compute_sig_drift(weight, self.sig_limit, self.drift_per_day, float(player.delta_time))
            player.add_noise_and_collapse(sig_drift)
            ts_rating = self._convert_player_to_ts_rating(player.approx_posterior)
            return (ts_rating, lo)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            all_ratings = list(executor.map(_update_player, *zip(*((player, lo) for player, lo, _ in standings))))
        rating_groups = self._convert_standings_to_ts_ranks(all_ratings)
        weight = params.weight * self.weight_limit
        weights = [tuple(weight for _ in range(len(t))) for t in rating_groups]
        new_ratings = self.ts_env.rate(rating_groups, ranks=[0, 1], weights=weights)
        new_ratings = chain.from_iterable(new_ratings)

        def _update_player_rating(player_ts: ts.Rating, player: Player):
            mu = player_ts.mu
            sig = player_ts.sigma
            player.update_rating(Rating(mu, sig), 0.0)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(_update_player_rating, *zip(*zip(new_ratings, (player for player, _, _ in standings))))

