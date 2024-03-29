import math
from sys import maxsize

#  math constants
INT_MAX = maxsize
TANH_MULTIPLIER: float = math.pi / 1.7320508075688772
FRAC_2_SQRT_PI = 1.12837916709551257389615890312154517
SQRT_2 = 1.41421356237309504880168872420969808
LN_10: float = 2.30258509299404568401799145468436421

# default ratings
DEFAULT_MU = 1500.
DEFAULT_SIG = 500.

# rating constants
BOUNDS = (-6000., 9000.)
DEFAULT_BETA = 400. * TANH_MULTIPLIER / LN_10
DEFAULT_DRIFTS_PER_DAY = 0.
DEFAULT_SIG_LIMIT = 40.
DEFAULT_SPLIT_TIES = False
DEFAULT_TRANSFER_SPEED = 1.
DEFAULT_WEIGHT_LIMIT = 0.2
FLOAT_MAX: float = float('inf')

# drifts constants
SECS_PER_DAY = 86400

# Trueskill constants
TS_MU = 1500.
TS_SIG = 500.
TS_BETA = 250.
TS_TAU = 5.
TS_DRAW_PROB = 0.001
TS_BACKEND = 'scipy'
DRAW_PROBABILITY = 0.0001

# Glicko constants
GLICKO_Q = math.log(10) / 400.
