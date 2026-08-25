import numpy as np
import matplotlib.pyplot as plt


def logtanh(x):
    return -np.log(np.tanh(x / 2))


def plot_large_x():
    import matplotlib.pyplot as plt
    x_series = np.arange(0, 35, 0.0001)
    exact_vals = logtanh(x_series)
    t = np.exp(-x_series)

    approx_vals = 2 * t
    rel_error = np.abs(exact_vals - approx_vals) / exact_vals
    plt.semilogy(x_series, rel_error)

    approx_vals = 2 * (t + t ** 3 / 3)
    rel_error = np.abs(exact_vals - approx_vals) / exact_vals
    plt.semilogy(x_series, rel_error)

    approx_vals = 2 * (t + t ** 3 / 3 + t ** 5 / 5)
    rel_error = np.abs(exact_vals - approx_vals) / exact_vals
    plt.semilogy(x_series, rel_error)
    plt.legend(['First-order', 'Third-order', 'Fifth-order'])
    plt.grid()
    plt.show()


def plot_small_x():
    logx = np.arange(-13, 0, 0.001)
    x_series = np.exp(logx)
    exact_vals = logtanh(x_series)

    approx_vals = -np.log(x_series / 2)
    rel_error = np.abs(exact_vals - approx_vals) / exact_vals
    plt.semilogy(x_series, rel_error)

    approx_vals = -np.log(x_series / 2) + x_series ** 2 / 12
    rel_error = np.abs(exact_vals - approx_vals) / exact_vals
    plt.semilogy(x_series, rel_error)
    plt.xscale('log')
    plt.legend(['First-order', 'Second-order'])
    plt.grid()
    plt.show()


if __name__ == '__main__':
    plot_small_x()
    plot_large_x()

