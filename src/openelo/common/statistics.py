import numpy as np
from scipy.stats import logistic, norm, rv_continuous, skewnorm


__all__ = ['generate_normal_ratings_perc', 
           'generate_logistic_ratings_perc', 
           'generate_skewnormal_ratings_perc', 
           'generate_normal_ratings', 
           'generate_logistic_ratings',
           'generate_skewnormal_ratings',
           'MixtureModel']


def generate_normal_ratings_perc(n: int,low_perc: float=0.01, hi_perc: float=0.99, mu: float=0., sig: float=1.):
    '''
    Generate a normal distribution based on percent range.

    Args:
        n (:obj:`int`): The number of data points.

        low_perc (:obj:`float`): The lower bound percent.

        hi_perc (:obj:`float`): The upper bound percent.
    
    Returns:
        obj:`np.ndarray`: A normal distribution of the number of data points.
    '''
    return norm.ppf(np.linspace(low_perc, hi_perc, n), mu, sig)


def generate_logistic_ratings_perc(n: int,low_perc: float=0.01, hi_perc: float=0.99, mu: float=0., sig: float=1.):
    '''
    Generate a logistic distribution based on percent range.

    Args:
        n (:obj:`int`): The number of data points.

        low_perc (:obj:`float`): The lower bound percent.

        hi_perc (:obj:`float`): The upper bound percent.

        mu (:obj:`float`): The expected value (mu) of the normal distribution.

        sig (:obj:`float`): The deviation (sigma) of the normal distribution.

    Returns:
        obj:`np.ndarray`: A logistic distribution of the number of data points.
    '''
    return logistic.ppf(np.linspace(low_perc, hi_perc, n), mu, sig)


def generate_skewnormal_ratings_perc(n: int,low_perc: float=0.01, hi_perc: float=0.99, mu: float=0., sig: float=1., a: float=0.):
    '''
    Generate a logistic distribution based on percent range.

    Args:
        n (:obj:`int`): The number of data points.

        low_perc (:obj:`float`): The lower bound percent.

        hi_perc (:obj:`float`): The upper bound percent.

        mu (:obj:`float`): The expected value (mu) of the normal distribution.

        sig (:obj:`float`): The deviation (sigma) of the normal distribution.

        a (:obj:`float`): Skewness parameter. When a = 0, it is identical to a normal distribution.

    Returns:
        obj:`np.ndarray`: A logistic distribution of the number of data points.
    '''
    return skewnorm.ppf(np.linspace(low_perc, hi_perc, n), loc=mu, scale=sig, a=a)


def generate_normal_ratings(n: int, low_rating: float=-3, hi_rating: float=3., mu: float=0., sig: float=1.):
    '''
    Generate a logistic distribution based on percent range.

    Args:
        n (:obj:`int`): The number of data points.

        low_rating (:obj:`float`): The lower bound rating.

        hi_rating (:obj:`float`): The upper bound rating.

        mu (:obj:`float`): The expected value (mu) of the normal distribution.

        sig (:obj:`float`): The deviation (sigma) of the normal distribution.

        a (:obj:`float`): Skewness parameter. When a = 0, it is identical to a normal distribution.

    Returns:
        obj:`np.ndarray`: A logistic distribution of the number of data points.
    '''
    norm_dist = norm(mu, sig)
    low_perc, high_perc = norm_dist.cdf(low_rating), norm_dist.cdf(hi_rating)
    return norm_dist.ppf(np.linspace(low_perc, high_perc, n))


def generate_logistic_ratings(n: int, low_rating: float=-3., hi_rating: float=3., mu: float=0., sig: float=1.):
    '''
    Generate a logistic distribution based on percent range.

    Args:
        n (:obj:`int`): The number of data points.

        low_rating (:obj:`float`): The lower bound rating.

        hi_rating (:obj:`float`): The upper bound rating.

        mu (:obj:`float`): The expected value (mu) of the normal distribution.

        sig (:obj:`float`): The deviation (sigma) of the normal distribution.

        a (:obj:`float`): Skewness parameter. When a = 0, it is identical to a normal distribution.

    Returns:
        obj:`np.ndarray`: A logistic distribution of the number of data points.
    '''
    logistic_dist = logistic(mu, sig)
    low_perc, high_perc = logistic_dist.cdf(low_rating), logistic_dist.cdf(hi_rating)
    return logistic_dist.ppf(np.linspace( low_perc, high_perc, n))


def generate_skewnormal_ratings(n: int, low_rating: float=-3., hi_rating: float=3., mu: float=0., sig: float=1., a: float=0.):
    '''
    Generate a logistic distribution based on percent range.

    Args:
        n (:obj:`int`): The number of data points.

        low_rating (:obj:`float`): The lower bound rating.

        hi_ratign (:obj:`float`): The upper bound rating.

        mu (:obj:`float`): The expected value (mu) of the normal distribution.

        sig (:obj:`float`): The deviation (sigma) of the normal distribution.

        a (:obj:`float`): Skewness parameter. When a = 0, it is identical to a normal distribution.

    Returns:
        obj:`np.ndarray`: A logistic distribution of the number of data points.
    '''
    norm_dist = skewnorm(loc=mu, scale=sig, a=a)
    low_perc, high_perc = norm_dist.cdf(low_rating), norm_dist.cdf(hi_rating)
    return norm_dist.ppf(np.linspace(low_perc, high_perc, n))


class MixtureModel(rv_continuous):
    def __init__(self, submodels, *args, weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.submodels = submodels
        if weights is None:
            weights = [1 for _ in submodels]
        if len(weights) != len(submodels):
            raise (ValueError(f'There are {len(submodels)} submodels and {len(weights)} weights, but they must be equal.'))
        self.weights = [w / sum(weights) for w in weights]

    def _pdf(self, x):
        pdf = self.submodels[0].pdf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            pdf += submodel.pdf(x) * weight
        return pdf

    def _sf(self, x):
        sf = self.submodels[0].sf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            sf += submodel.sf(x) * weight
        return sf

    def _cdf(self, x):
        cdf = self.submodels[0].cdf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            cdf += submodel.cdf(x) * weight
        return cdf

    def rvs(self, size):
        submodel_choices = np.random.choice(len(self.submodels), size=size, p = self.weights)
        submodel_samples = [submodel.rvs(size=size) for submodel in self.submodels]
        rvs = np.choose(submodel_choices, submodel_samples)
        return rvs

    def _ppf(self, x):
        ppf = self.submodels[0].ppf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            ppf += submodel.ppf(x) * weight
        return ppf

    def _isf(self, x):
        isf = self.submodels[0].isf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            isf += submodel.isf(x) * weight
        return isf

    def _logsf(self, x):
        logsf = self.submodels[0].logsf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            logsf += submodel.logsf(x) * weight
        return logsf
