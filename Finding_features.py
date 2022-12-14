"""En este archivo estan las funciones empleadas para calcular las features de los datos"""

import numpy as np
import scipy.signal
import os
import LeerDatosExcel as lee
import datetime
import pickle as pkl


def database_etiquetada():
    """Genera la base de datos etiquetada asignando un segmento de senial a cada etiqueta
    Devuelve un array de nx513x3 donde cada fila es un segmento, la primera columna corresponde a la etiqueta y
    la ultima dimension corresponde a cada eje x y z (parecido a como usualmente se trabaja con imagenes rgb). Si se
    obtienen nuevos datos y se colocan de manera adecuada en la carpeta Datos al correr este codigo se obtiene una base
    de datos actualizada"""
    db = np.zeros((1, 515, 3))
    for campa in os.listdir(lee.etiqueta_path):
        for document in os.listdir(os.path.join(lee.etiqueta_path, campa)):
            if document[-3:] == 'csv' and os.path.exists(os.path.join(mov.tortugometro_path, campa, document)):
                dftags = lee.ReadTags(os.path.join(lee.etiqueta_path, campa, document))
                dfacc = mov.ReadIMUData(os.path.join(mov.tortugometro_path, campa, document))
                for i in dftags.index:
                    f = dftags.loc[i, 'date']
                    h = dftags.loc[i, 'time']
                    t = datetime.datetime(year=f.year, month=f.month, day=f.day, hour=h.hour, minute=h.minute,
                                          second=h.second)
                    coincidence = dfacc[
                        (dfacc['datetime'] > t) & (dfacc['datetime'] < t + datetime.timedelta(minutes=4))]
                    if len(coincidence.index) >= 514:
                        j = coincidence.index[0]
                        aux = np.zeros((1, 515, 3))
                        aux[0, 0, :] = dftags.loc[i, 'tag']
                        aux[0, 1:, 0] = dfacc.loc[j:j + 513, 'accX']
                        aux[0, 1:, 1] = dfacc.loc[j:j + 513, 'accY']
                        aux[0, 1:, 2] = dfacc.loc[j:j + 513, 'accZ']
                        db = np.concatenate((db, aux), axis=0)
    db = np.delete(db, 0, 0)
    with open(os.path.join(os.getcwd(), 'aceleraciones_etiquetadas', 'database_cruda_etiquetada.pickle'),
              'wb') as handle:
        pkl.dump(db, handle)

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


