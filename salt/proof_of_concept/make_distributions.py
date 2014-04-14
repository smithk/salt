import matplotlib.pyplot as pl
import numpy as np
from scipy import stats


def generate_uniform():
    pl.cla()
    pl.plot([1, 1])
    #pl.show(block=True)


def generate_normal():
    pl.cla()
    pl.plot(stats.norm.pdf(np.linspace(-10, 10, 100), 0, 1))
    pl.show(block=True)


def generate_lognormal():
    pass
    pl.show(block=True)


def generate_loguniform():
    pass
    pl.show(block=True)


def generate_gmm():
    pass
    pl.show(block=True)

if __name__ == '__main__':
    generate_uniform()
    generate_normal()
    generate_lognormal()
    generate_loguniform()
    generate_gmm()
