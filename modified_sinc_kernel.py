"""This module implements the Modified Sinc Kernel from [1].

[1]: https://pubs.acs.org/doi/pdf/10.1021/acsmeasuresciau.1c00054
"""

import numpy
from matplotlib import pyplot
from scipy import ndimage, signal


def filter_modified_sinc(
    signal_vector: numpy.ndarray, n: int, edge_frequency: int, sample_rate: int
):
    """Filter a signal with an optimal lowpass filter, a modified sinc kernel.

    Args:
        signal_vector: Input signal.
        n: Desired order. Must be an even number between 2 and 10.
        edge_frequency: Desired edge frequency of the lowpass filter in Hz.
        sample_rate: Corresponding signal sample rate.

    Returns:
        The smoothed signal
    """
    if (n < 2) or (n > 10) or numpy.mod(n, 2):
        raise ValueError("The value 'n' must be an even number between 2 and 10")

    kernel = _get_ms_kernel(n=n, edge_frequency=edge_frequency, sample_rate=sample_rate)
    return ndimage.convolve1d(signal_vector, kernel)


def _get_ms_kernel(n: int, edge_frequency: int, sample_rate: int):
    """Compose the modified sinc kernel.

    Args:
        n: Desired order. Must be an even number between 2 and 10.
        edge_frequency: Desired edge frequency of the lowpass filter in Hz.
        sample_rate: Corresponding signal sample rate.

    Returns:
        The modified kernel
    """
    b = 2 * edge_frequency / sample_rate
    m = int(numpy.ceil((0.745 + 0.249 * n) / b - 1))

    x = numpy.arange(-m, m + 1) / (m + 1)
    gaussian = _get_gaussian(x)

    argument = numpy.where(x == 0, 1e-20, x)
    sinc = numpy.sin((n + 4) / 2 * numpy.pi * argument) / (
        (n + 4) / 2 * numpy.pi * argument
    )

    if n >= 6:
        a, b, c = _get_correction_coefficients(n, num_terms=1)
        kappa = a + b / (c - m) ** 3
        nu = 1 if numpy.mod(n / 2, 2) else 2
        sinc += kappa * x * numpy.sin(nu * numpy.pi * x)

    modified_sinc = gaussian * sinc
    return modified_sinc / numpy.sum(modified_sinc)


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


def _get_noisy_sinusoid(len_signal_sec: int, sample_rate: int, frequency: int = 100):
    len_sec = round(len_signal_sec * sample_rate)
    time = numpy.arange(len_sec) / sample_rate
    sinusoid = numpy.cos(2 * numpy.pi * frequency * time)

    noise_std = 0.2
    noisy_sinusoid = sinusoid + noise_std * numpy.random.randn(len_sec)
    return noisy_sinusoid, sinusoid, time


def _create_tone_complex(
    signal_length_sec, fundamental_frequency, sample_rate, num_harmonics=10
):
    """Create a harmonic tone complex.

    Parameters:
        signal_length_sec: Duration of the signal in seconds.
        fundamental_frequency: Base frequency in Hz.
        sample_rate: Sampling rate in Hz.
        num_harmonics: Number of harmonics to include. Defaults to 10.

    Returns:
        tuple of
            - Generated noisy harmonic tone signal.
            - Generated noisy harmonic tone signal.
            - Time vector.
    """
    signal_length = round(signal_length_sec * sample_rate)
    time = numpy.arange(signal_length) / sample_rate
    tone_complex = numpy.zeros_like(time)

    for n in range(1, num_harmonics + 1):
        tone_complex += numpy.sin(2 * numpy.pi * fundamental_frequency * n * time)

    return tone_complex, time


def _create_noisy_spectrum(fundamental_frequency: int, sample_rate: int):
    len_signal_sec = 2
    tone_complex, time = _create_tone_complex(
        signal_length_sec=len_signal_sec,
        fundamental_frequency=fundamental_frequency,
        sample_rate=sample_rate,
        num_harmonics=10,
    )

    block_size_sec = 100e-3
    block_size = round(block_size_sec * sample_rate)
    overlap = round(block_size * 0.5)
    window = "hann"
    fft_size = int(2 ** (numpy.ceil(numpy.log2(block_size))))
    frequency, spectrum = signal.welch(
        tone_complex,
        fs=sample_rate,
        window=window,
        nperseg=block_size,
        noverlap=overlap,
        nfft=fft_size,
    )

    log_spectrum = 10 * numpy.log10(spectrum)

    noise_std = 5
    noisy_spectrum = noise_std * numpy.random.randn(len(log_spectrum)) + log_spectrum
    return noisy_spectrum, log_spectrum, frequency


if __name__ == "__main__":
    sample_rate = 8_000
    len_signal_sec = 2
    frequency = 100

    noisy_sinusoid, clean_sinusoid, time = _get_noisy_sinusoid(
        len_signal_sec=len_signal_sec, sample_rate=sample_rate, frequency=frequency
    )
    noisy_spectrum, clean_spectrum, frequency = _create_noisy_spectrum(
        fundamental_frequency=300, sample_rate=sample_rate
    )

    n_sinusoid = 8
    edge_frequency_sinusoid = 80
    smoothed_sinusoid = filter_modified_sinc(
        noisy_sinusoid,
        n=n_sinusoid,
        edge_frequency=edge_frequency_sinusoid,
        sample_rate=sample_rate,
    )

    n_spectrum = 8
    edge_frequency_spectrum = 500
    smoothed_spectrum = filter_modified_sinc(
        noisy_spectrum,
        n=n_spectrum,
        edge_frequency=edge_frequency_spectrum,
        sample_rate=sample_rate
    )

    fig, ax = pyplot.subplots()
    ax.plot(time, clean_sinusoid, color="k", label="Clean")
    ax.plot(time, noisy_sinusoid, label="Noisy")
    ax.plot(time, smoothed_sinusoid, linewidth=2.5, label="Smoothed")
    ax.legend()
    ax.set_xlim(0, 1e-1)

    fig, ax = pyplot.subplots()
    ax.plot(frequency, clean_spectrum, color="k", label="Clean")
    ax.plot(frequency, noisy_spectrum, label="Noisy")
    ax.plot(frequency, smoothed_spectrum, linewidth=2.5, label="Smoothed")
    ax.legend()

    pyplot.show()
