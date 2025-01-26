"""This module implements the Modified Sinc Kernel from [1].

[1]: https://pubs.acs.org/doi/pdf/10.1021/acsmeasuresciau.1c00054
"""

import numpy
from matplotlib import pyplot
from scipy import ndimage


def filter_modified_sinc(
    signal: numpy.ndarray, n: int, edge_frequency: int, sample_rate: int
):
    """Filter a signal with an optimal lowpass filter, a modified sinc kernel.

    Args:
        signal: Input signal.
        n: Desired order. Must be an even number between 2 and 10.
        edge_frequency: Desired edge frequency of the lowpass filter in Hz.
        sample_rate: Corresponding signal sample rate.

    Returns:
        The smoothed signal
    """
    if (n < 2) or (n > 10) or numpy.mod(n, 2):
        raise ValueError("The value 'n' must be an even number between 2 and 10")

    kernel = _get_ms_kernel(n=n, edge_frequency=edge_frequency, sample_rate=sample_rate)
    return ndimage.convolve1d(signal, kernel)


def _get_ms_kernel(n: int, edge_frequency: int, sample_rate: int):
    """Compose the modified sinc kernel.

    Args:
        n: Desired order. Must be an even number between 2 and 10.
        edge_frequency: Desired edge frequency of the lowpass filter in Hz.
        sample_rate: Corresponding signal sample rate.

    Returns:
        The modified kernel
    """
    m = _compute_m(n, edge_frequency=edge_frequency, sample_rate=sample_rate)

    x = numpy.arange(-m, m + 1) / (m + 1)
    gaussian = _get_gaussian(x)

    argument = numpy.where(x == 0, 1e-20, x)
    sinc = numpy.sin((n + 4) / 2 * numpy.pi * argument) / (
        (n + 4) / 2 * numpy.pi * argument
    )

    if n >= 6:
        a, b, c = _get_correction_coefficients(n, num_terms=1)
        kappa = a + b / (c - m)
        nu = _get_nu(n)
        sinc += kappa * x * numpy.sin(nu * numpy.pi * x)

    modified_sinc = gaussian * sinc
    return modified_sinc / numpy.sum(modified_sinc)


def _compute_m(n, edge_frequency: int, sample_rate: int):
    b = 2 * edge_frequency / sample_rate
    m = (0.745 + 0.249 * n) / b - 1
    return int(numpy.ceil(m))


def _get_gaussian(x: numpy.ndarray, alpha=4):
    return (
        numpy.exp(-alpha * x**2)
        + numpy.exp(-alpha * (x + 2) ** 2)
        + numpy.exp(-alpha * (x - 2) ** 2)
        - 2 * numpy.exp(-alpha)
        - numpy.exp(-9 * alpha)
    )


def _get_correction_coefficients(n, num_terms=1):
    if n == 6:
        return 0.00172, 0.02437, 1.64375
    elif n == 8:
        if num_terms == 1:
            return 0.00440, 0.08821, 2.35938
        else:
            return 0.00615, 0.02472, 3.63594
    elif n == 10:
        if num_terms == 1:
            return 0.00118, 0.04219, 2.74688
        else:
            return 0.00367, 0.12780, 2.77031
    else:
        raise ValueError(
            "The desired value for 'n' is invalid. Choose between 6 and 10"
        )


def _get_nu(n: int):
    return 1 if numpy.mod(n / 2, 2) else 2


def _get_noisy_sinusoid(len_signal_sec: int, sample_rate: int, frequency: int = 100):
    len_sec = round(len_signal_sec * sample_rate)
    time = numpy.arange(len_sec) / sample_rate
    sinusoid = numpy.cos(2 * numpy.pi * frequency * time)

    noise_std = 0.2
    noisy_sinusoid = sinusoid + noise_std * numpy.random.randn(len_sec)
    return noisy_sinusoid, sinusoid, time


if __name__ == "__main__":
    sample_rate = 8_000
    len_signal_sec = 2
    frequency = 100

    noisy_sinusoid, clean_sinusoid, time = _get_noisy_sinusoid(
        len_signal_sec=len_signal_sec, sample_rate=sample_rate, frequency=frequency
    )

    n = 4
    edge_frequency = 80
    smoothed_sinusoid = filter_modified_sinc(
        noisy_sinusoid, n=n, edge_frequency=edge_frequency, sample_rate=sample_rate
    )

    fig, ax = pyplot.subplots()
    ax.plot(time, clean_sinusoid, color="k", label="Clean")
    ax.plot(time, noisy_sinusoid, label="Noisy")
    ax.plot(time, smoothed_sinusoid, linewidth=2.5, label="Smoothed")
    ax.legend()
    ax.set_xlim(0, 1e-1)

    pyplot.show()
