import numpy as np
import scipy.signal


def make_windows(acc, size):
    acc = acc[:int(np.trunc(acc.shape[0] / size)) * size, :]
    acc = np.stack(np.split(acc, acc.shape[0] / size), axis=-1)
    return acc


def convolve_signal(acc):
    conv = np.real(np.fft.ifft(np.multiply(np.fft.fft(acc[:, 0, :], axis=0), np.fft.fft(acc[:, 1, :], axis=0)), axis=0))
    conv = conv.T
    return conv


def normalize_convolution(conv, dx=0.174):
    conv -= np.mean(conv, axis=1, keepdims=True)
    conv /= np.trapz(conv ** 2, dx=dx, axis=1).reshape(-1, 1) ** 0.5
    return conv


def calculate_features(signal, fs=1 / 0.174):
    acc = signal[['accX', 'accY', 'accZ']].to_numpy()
    acc = make_windows(acc, 512)
    # calculate the convolution accX*accY
    conv = convolve_signal(acc)
    conv_rank = np.log10(np.amax(conv, axis=1) - np.amin(conv, axis=1))
    conv = normalize_convolution(conv)
    # normalizes conv so its mean is 0 and its power is 1
    f, dsp = scipy.signal.periodogram(conv, fs=fs, scaling='density', axis=1)
    dsp[:, 0] = conv_rank
    return dsp


