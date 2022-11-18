import numpy as np
import scipy.signal
import movimiento as mov
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def fourier_ventana(signal, normalize=False):
    #aux = int(signal.size/window)*window
    #signal = signal[:aux].reshape((int(aux/window), window))
    if normalize:  
        squaredsignal = signal**2
        signal = signal/np.sqrt(squaredsignal.sum(axis=1, keepdims=True))
    transformada = np.fft.fft(signal)
    return transformada


def convolucion_ventana(df, window):
    x = df['accX'].to_numpy()
    y = df['accY'].to_numpy()
    z = df['accZ'].to_numpy()
    aux = int(x.size / window) * window
    x = x[:aux].reshape((int(aux / window), window))
    y = y[:aux].reshape((int(aux / window), window))
    z = z[:aux].reshape((int(aux / window), window))
    xmean = x.mean(axis=1)
    ymean = y.mean(axis=1)
    zmean = z.mean(axis=1)
    r = (xmean**2 + ymean**2 + zmean**2)**0.5
    tanx = xmean/zmean
    tany = ymean/zmean
    x = fourier_ventana(x)
    y = fourier_ventana(y)
    conv = x*y
    #aux = int(x.size/window)*window
    #conv = conv[:aux].reshape((int(aux/window), window))
    conv = np.real(np.fft.ifft(conv))
    rangos = np.amax(conv, axis=1)-np.amin(conv, axis=1)
    s = np.apply_along_axis(dsp, 1, conv)
    r = r.reshape((r.size, 1))
    tanx = tanx.reshape((r.size, 1))
    tany = tany.reshape((r.size, 1))
    rangos = rangos.reshape((r.size,1))
    #print(r.shape, tanx.shape, s.shape)
    return np.concatenate((rangos, r, tanx, tany, s), axis=1)


def dsp(signal):
    signal = signal-np.mean(signal)
    f, s = scipy.signal.periodogram(signal)
    return s

"""
df = mov.ReadIMUData(os.path.join(mov.tortugometro_path, '01_2021', 'T30.csv'))
datos = convolucion_ventana(df, 2048)
print(datos.shape)
"""
"""

df = mov.ReadIMUData(os.path.join(mov.tortugometro_path, '01_2021', 'T30.csv'))
convo = convolucion_ventana(df, 2048)
frec = np.apply_along_axis(max_freq_dsp, 1, convo.reshape((-1, 2048)))
plt.hist(frec, bins=20)
plt.show()
"""
#fig, axs = plt.subplots(2,1, sharex=True)
#fig.autofmt_xdate()
#y = convolucion_ventana(df, 2048)
#x = mdates.date2num(df['datetime'][:y.size].to_numpy())
#axs[0].plot_date(x, y,'-')
#axs[1].plot_date(x, df['accX'][:y.size],'-')
#plt.show()

