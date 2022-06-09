import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.stats import chi2, poisson, t
from scipy.special import factorial


def get_arr(file_name: str, type: type = None) -> list:
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        arr = list(reader)[0]
    if type is None:
        return arr
    for i in range(len(arr)):
        arr[i] = type(arr[i])
    return arr


def get_expectation(arr: list) -> float:
    sum = 0
    for i in arr:
        sum += i
    return sum / len(arr)


def get_dispersion(arr: list) -> float:
    sum = 0
    m = get_expectation(arr)
    for i in arr:
        sum += (i - m)**2
    return sum / (len(arr) - 1)


def f_normal(x: float, mean: float, sd: float) -> float:
    power = -((x - mean) ** 2) / (2 * sd ** 2)
    return 1 / (sd * np.sqrt(2 * np.pi)) * np.exp(power)


def f_exp_distrib(x: float, lmbd: float) -> float:
    # if x < 0:
    #     raise ValueError("x < 0 in exponential distribution")
    return lmbd * np.exp(-lmbd * x)


def F_exp(x: float, lmbd: float) -> float:
    # if x < 0:
    #     raise ValueError("x < 0 in exponential distribution")
    return 1 - np.exp(-lmbd * x)


def f_geom(n: int, p: float) -> float:
    if p < 0 or p > 1:
        raise ValueError("p < 0 or p > 1 in geom distribution")
    return p * ((1 - p)**(n-1))


def F_poisson(x: float, mean: float) -> float:
    return poisson.cdf(x, mean)


def f_poisson(x: float, mean: float) -> float:
    return np.exp(-mean) * (mean ** x) / factorial(x)


def get_mean_confidence_interval(data: list, confidence_level: float) -> tuple:
    mean = get_expectation(data)
    dispersion = get_dispersion((data))
    n = len(data)
    tmp = t.ppf(confidence_level, n - 1)
    eps = np.sqrt(dispersion) * tmp / np.sqrt(n - 1)
    return (mean - eps, mean + eps)


def get_standard_deviation_conf_interval(data: list, confidence_level: float) -> tuple:
    mean = get_expectation(data)
    dispersion = get_dispersion(data)
    standard_deviation = np.sqrt(dispersion)
    n = len(data)
    a1 = (1 - confidence_level) / 2.0
    a2 = (1 + confidence_level) / 2.0
    standard_deviation_left = np.sqrt((n - 1) / chi2.ppf(a2, n - 1)) * standard_deviation
    standard_deviation_right = np.sqrt((n - 1) / chi2.ppf(a1, n - 1)) * standard_deviation
    return (standard_deviation_left, standard_deviation_right)

