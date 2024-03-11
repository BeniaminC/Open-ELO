from abc import ABC, abstractmethod
from math import pi, sqrt

import numpy as np


__all__ = ['NormalSkillAdjuster',
           'LogisticSkillAdjuster',
           'WeightedSkillAdjuster']


class SkillAdjuster(ABC):
    @abstractmethod
    def __init__(self, player_ratings: np.ndarray, mu: float, sig: float) -> None:
        pass

    @abstractmethod
    def identity(self, player_ratings: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def linear(self, player_ratings: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def manual_weights(self, player_ratings: np.ndarray, weights: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def binary_step(self, player_ratings: np.ndarray, step_loc: float, offset: float) -> np.ndarray:
        pass

    @abstractmethod
    def sigmoid(self, player_ratings: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def tanh(self, player_ratings: np.ndarray, a: float = 1, b: float = 1, c: float = 1, d: float = 1) -> np.ndarray:
        pass

    @abstractmethod
    def relu(self, player_ratings: np.ndarray, standard_deviation: float):
        pass

    @abstractmethod
    def elu(self, player_ratings: np.ndarray, base: float = np.e, sd_dev: float = 0):
        pass

    @abstractmethod
    def softplus(self, player_ratings: np.ndarray):
        pass

    @abstractmethod
    def mod_softplus(self, player_ratings: np.ndarray, base: float = np.e):
        pass


class NormalSkillAdjuster(SkillAdjuster):
    def __init__(self, mu: float, sig: float) -> None:
        self.mu = mu
        self.sig = sig

    def identity(self, player_ratings: np.ndarray) -> np.ndarray:
        return player_ratings

    def linear(self, player_ratings: np.ndarray) -> np.ndarray:
        standardized = player_ratings / (self.mu + self.sig * 3.)  # assume the highest rank is 3 standard deviations away
        return player_ratings * standardized

    def manual_weights(self, player_ratings: np.ndarray, weights: np.ndarray) -> np.ndarray:
        assert len(player_ratings) == len(weights)
        return player_ratings * weights

    def binary_step(self, player_ratings: np.ndarray, step_loc: float, offset: float, reversed=False) -> np.ndarray:
        if reversed:
            return np.where(player_ratings > step_loc, player_ratings + offset, player_ratings)
        return np.where(player_ratings < step_loc, player_ratings + offset, player_ratings)

    def sigmoid(self, player_ratings: np.ndarray, reversed=False) -> np.ndarray:
        standard_normal = (player_ratings - self.mu) / self.sig
        if reversed:
            return player_ratings * (1. / (1. + np.exp(standard_normal)))
        return player_ratings * (1. / (1. + np.exp(-standard_normal)))

    def tanh(self, player_ratings: np.ndarray, a: float = 1, b: float = 1, c: float = 1, d: float = 1, reversed=False) -> np.ndarray:
        standard_normal = (player_ratings - self.mu) / self.sig
        if reversed:
            return player_ratings * ((((np.exp(-a * standard_normal) - np.exp(b * standard_normal)) / (np.exp(-c * standard_normal) + np.exp(d * standard_normal))) + 1) / 2)
        return player_ratings * ((((np.exp(a * standard_normal) - np.exp(-b * standard_normal)) / (np.exp(c * standard_normal) + np.exp(-d * standard_normal))) + 1) / 2)

    def relu(self, player_ratings: np.ndarray, standard_deviation: float, reversed=False):
        standard_normal = (player_ratings - self.mu) / self.sig
        if reversed:
            return player_ratings * np.where(standard_normal > standard_deviation, 0, 1)
        return player_ratings * np.where(standard_normal < standard_deviation, 0, 1)

    def elu(self, player_ratings: np.ndarray, base: float = np.e, sd_dev: float = 0, reversed=False):
        standard_normal = (player_ratings - self.mu) / self.sig
        if reversed:
            return player_ratings * np.where(standard_normal > sd_dev, np.power(base, -standard_normal - sd_dev), 1)
        return player_ratings * np.where(standard_normal < sd_dev, np.power(base, standard_normal - sd_dev), 1)

    def softplus(self, player_ratings: np.ndarray, reversed=False):
        standard_normal = (player_ratings - self.mu) / self.sig
        if reversed:
            return player_ratings * (np.log(1 + np.exp(-standard_normal)) / 3.)
        return player_ratings * (np.log(1 + np.exp(standard_normal)) / 3.)

    def mod_softplus(self, player_ratings: np.ndarray, base: float = np.e, reversed=False):
        standard_normal = (player_ratings - self.mu) / self.sig
        if reversed:
            return player_ratings * (np.emath.logn(base, (1 + np.power(base, -standard_normal))) / 3.)
        return player_ratings * (np.emath.logn(base, (1 + np.power(base, standard_normal))) / 3.)


# f(x; mu, s) = (1 / 4s) sech^2((x - mu) / 2 * s), where sig = sqrt(3) / pi * s
class LogisticSkillAdjuster(SkillAdjuster):
    log_factor = (sqrt(3) / pi)

    def __init__(self, mu: float, sig: float) -> None:
        self.mu = mu
        self.sig = sig * LogisticSkillAdjuster.log_factor  # logistic has lower scaled standard deviation of sigma

    def identity(self, player_ratings: np.ndarray) -> np.ndarray:
        return player_ratings

    def linear(self, player_ratings: np.ndarray) -> np.ndarray:
        standardized = player_ratings / (self.mu + self.sig * (3. / LogisticSkillAdjuster.log_factor))  # assume the highest rank is 3 standard deviations away
        return player_ratings * standardized

    def manual_weights(self, player_ratings: np.ndarray, weights: np.ndarray) -> np.ndarray:
        assert len(player_ratings) == len(weights)
        return player_ratings * weights

    def binary_step(self, player_ratings: np.ndarray, step_loc: float, offset: float, reversed=False) -> np.ndarray:
        if reversed:
            return np.where(player_ratings > step_loc, player_ratings + offset, player_ratings)
        return np.where(player_ratings < step_loc, player_ratings + offset, player_ratings)

    def sigmoid(self, player_ratings: np.ndarray, reversed=False) -> np.ndarray:
        standard_normal = (player_ratings - self.mu) / self.sig
        if reversed:
            return player_ratings * (1. / (1. + np.exp(standard_normal)))
        return player_ratings * (1. / (1. + np.exp(-standard_normal)))

    def tanh(self, player_ratings: np.ndarray, a: float = 1, b: float = 1, c: float = 1, d: float = 1, reversed=False) -> np.ndarray:
        standard_normal = (player_ratings - self.mu) / self.sig
        if reversed:
            return player_ratings * ((((np.exp(-a * standard_normal) - np.exp(b * standard_normal)) / (np.exp(-c * standard_normal) + np.exp(d * standard_normal))) + 1) / 2)
        return player_ratings * ((((np.exp(a * standard_normal) - np.exp(-b * standard_normal)) / (np.exp(c * standard_normal) + np.exp(-d * standard_normal))) + 1) / 2)

    def relu(self, player_ratings: np.ndarray, standard_deviation: float, reversed=False):
        standard_normal = (player_ratings - self.mu) / self.sig
        if reversed:
            return player_ratings * np.where(standard_normal > standard_deviation, 0, 1)
        return player_ratings * np.where(standard_normal < standard_deviation, 0, 1)

    def elu(self, player_ratings: np.ndarray, base: float = np.e, sd_dev: float = 0, reversed=False):
        standard_normal = (player_ratings - self.mu) / self.sig
        if reversed:
            return player_ratings * np.where(standard_normal > sd_dev, np.power(base, -standard_normal - sd_dev), 1)
        return player_ratings * np.where(standard_normal < sd_dev, np.power(base, standard_normal - sd_dev), 1)

    def softplus(self, player_ratings: np.ndarray, reversed=False):
        standard_normal = (player_ratings - self.mu) / self.sig
        if reversed:
            return player_ratings * (np.log(1 + np.exp(-standard_normal)) / 3.)
        return player_ratings * (np.log(1 + np.exp(standard_normal)) / 3.)

    def mod_softplus(self, player_ratings: np.ndarray, base: float = np.e, reversed=False):
        standard_normal = (player_ratings - self.mu) / self.sig
        if reversed:
            return player_ratings * (np.emath.logn(base, (1 + np.power(base, -standard_normal))) / 3.)
        return player_ratings * (np.emath.logn(base, (1 + np.power(base, standard_normal))) / 3.)


class WeightedSkillAdjuster(SkillAdjuster):

    def __init__(self, mu: float, sig: float, sig_weight: float = 1) -> None:
        self.mu = mu
        self.sig = sig * sig_weight  # logistic has lower scaled standard deviation of sigma

    def identity(self, player_ratings: np.ndarray) -> np.ndarray:
        return player_ratings

    def linear(self, player_ratings: np.ndarray) -> np.ndarray:
        standardized = player_ratings / (self.mu + self.sig)  # assume the highest rank is 3 standard deviations away
        return player_ratings * standardized

    def manual_weights(self, player_ratings: np.ndarray, weights: np.ndarray) -> np.ndarray:
        assert len(player_ratings) == len(weights)
        return player_ratings * weights

    def binary_step(self, player_ratings: np.ndarray, step_loc: float, offset: float, reversed=False) -> np.ndarray:
        if reversed:
            return np.where(player_ratings > step_loc, player_ratings + offset, player_ratings)
        return np.where(player_ratings < step_loc, player_ratings + offset, player_ratings)

    def sigmoid(self, player_ratings: np.ndarray, reversed=False) -> np.ndarray:
        standard_normal = (player_ratings - self.mu) / self.sig
        if reversed:
            return player_ratings * (1. / (1. + np.exp(standard_normal)))
        return player_ratings * (1. / (1. + np.exp(-standard_normal)))

    def tanh(self, player_ratings: np.ndarray, a: float = 1, b: float = 1, c: float = 1, d: float = 1, reversed=False) -> np.ndarray:
        standard_normal = (player_ratings - self.mu) / self.sig
        if reversed:
            return player_ratings * ((((np.exp(-a * standard_normal) - np.exp(b * standard_normal)) / (np.exp(-c * standard_normal) + np.exp(d * standard_normal))) + 1) / 2)
        return player_ratings * ((((np.exp(a * standard_normal) - np.exp(-b * standard_normal)) / (np.exp(c * standard_normal) + np.exp(-d * standard_normal))) + 1) / 2)

    def relu(self, player_ratings: np.ndarray, standard_deviation: float, reversed=False):
        standard_normal = (player_ratings - self.mu) / self.sig
        if reversed:
            return player_ratings * np.where(standard_normal > standard_deviation, 0, 1)
        return player_ratings * np.where(standard_normal < standard_deviation, 0, 1)

    def elu(self, player_ratings: np.ndarray, base: float = np.e, sd_dev: float = 0, reversed=False):
        standard_normal = (player_ratings - self.mu) / self.sig
        if reversed:
            return player_ratings * np.where(standard_normal > sd_dev, np.power(base, -standard_normal - sd_dev), 1)
        return player_ratings * np.where(standard_normal < sd_dev, np.power(base, standard_normal - sd_dev), 1)

    def softplus(self, player_ratings: np.ndarray, reversed=False):
        standard_normal = (player_ratings - self.mu) / self.sig
        if reversed:
            return player_ratings * (np.log(1 + np.exp(-standard_normal)) / 3.)
        return player_ratings * (np.log(1 + np.exp(standard_normal)) / 3.)

    def mod_softplus(self, player_ratings: np.ndarray, base: float = np.e, reversed=False):
        standard_normal = (player_ratings - self.mu) / self.sig
        if reversed:
            return player_ratings * (np.emath.logn(base, (1 + np.power(base, -standard_normal))) / 3.)
        return player_ratings * (np.emath.logn(base, (1 + np.power(base, standard_normal))) / 3.)
