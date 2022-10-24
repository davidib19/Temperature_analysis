import numpy as np
import scipy.signal

import movimiento as mov
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def fourier_ventana(signal, window, normalize=False):
    aux = int(signal.size/window)*window
    signal = signal[:aux].reshape((int(aux/window), window))
    if normalize:
        squaredsignal = signal**2
        signal = signal/np.sqrt(squaredsignal.sum(axis=1, keepdims=True))
    transformada = np.fft.fft(signal)
    return transformada.reshape(aux)


def convolucion_ventana(df, window):
    x = df['accX'].to_numpy()
    y = df['accY'].to_numpy()
    x = fourier_ventana(x, window)
    y = fourier_ventana(y, window, True)
    conv = x*y
    aux = int(x.size/window)*window
    conv = conv[:aux].reshape((int(aux/window), window))
    conv = np.real(np.fft.ifft(conv))
    return conv.reshape((aux))


def max_freq_dsp(signal):

    signal = signal-np.mean(signal)
    f, s = scipy.signal.periodogram(signal)
    if np.max(signal)-np.min(signal)> 100:
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(f,s,'.-')
        axs[1].plot(signal)
        plt.show()
    return f[np.argmax(s)]


df = mov.ReadIMUData(os.path.join(mov.tortugometro_path, '01_2021', 'T30.csv'))
convo = convolucion_ventana(df, 2048)
frec = np.apply_along_axis(max_freq_dsp, 1, convo.reshape((-1, 2048)))
plt.hist(frec, bins=20)
plt.show()

#fig, axs = plt.subplots(2,1, sharex=True)
#fig.autofmt_xdate()
#y = convolucion_ventana(df, 2048)
#x = mdates.date2num(df['datetime'][:y.size].to_numpy())
#axs[0].plot_date(x, y,'-')
#axs[1].plot_date(x, df['accX'][:y.size],'-')
#plt.show()

