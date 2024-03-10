import numpy as np
from scipy.stats import logistic, norm, rv_continuous, skewnorm


def generate_normal_ratings_perc(low_perc, hi_perc, n, mu, sig):
    return norm.ppf(np.linspace(low_perc, hi_perc, n), mu, sig)


def generate_logistic_ratings_perc(low_perc, hi_perc, n, mu, sig):
    return logistic.ppf(np.linspace(low_perc, hi_perc, n), mu, sig)


def generate_skewnormal_ratings_perc(low_perc, hi_perc, n, mu, sig, a):
    return skewnorm.ppf(np.linspace(low_perc, hi_perc, n), loc=mu, scale=sig, a=a)


def generate_normal_ratings(low_rating, hi_rating, n, mu, sig):
    norm_dist = norm(mu, sig)
    low_perc, hi_perc = norm_dist.cdf(low_rating), norm_dist.cdf(hi_rating)
    return norm_dist.ppf(np.linspace(low_perc, hi_perc, n))


def generate_logistic_ratings(low_rating, hi_rating, n, mu, sig):
    logistic_dist = logistic(mu, sig)
    low_perc, hi_perc = logistic_dist.cdf(low_rating), logistic_dist.cdf(hi_rating)
    return logistic_dist.ppf(np.linspace(low_perc, hi_perc, n))


def generate_skewnormal_ratings(low_rating, hi_rating, n, mu, sig, a):
    norm_dist = skewnorm(loc=mu, scale=sig, a=a)
    low_perc, hi_perc = norm_dist.cdf(low_rating), norm_dist.cdf(hi_rating)
    return norm_dist.ppf(np.linspace(low_perc, hi_perc, n))


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
