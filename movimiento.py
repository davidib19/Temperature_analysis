import pandas as pd
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal


tortugometro_path = os.path.join(os.getcwd(), "Datos_tortugometro_crudo")

def ReadIMUData(path):
    """Lee los datos del tortugómetro con el formato que vienen y los guarda en un pandas dataframe con una columna
    en tipo datetime.datetime transformada a hora de Argentina."""
    df = pd.read_csv(path, parse_dates=['date'], converters={'timeGMT': formatIMU}, sep=';', usecols=['date', 'timeGMT',
                                                                                                      'accX', 'accY', 'accZ',
                                                                                                      'tempIMU_C'])
    df['datetime'] = df.apply(
        lambda r: datetime.datetime.combine(r['date'], r['timeGMT']) - datetime.timedelta(hours=3), 1)
    return df.drop(['date', 'timeGMT'], axis=1)


def formatIMU(stringhour):
    microsecond = int(stringhour[-3:]) * 1000
    second = int(stringhour[-6:-4])
    minute = int(stringhour[-8:-6])
    hour = int(stringhour[:-8])
    if second > 59:
        second -= 60
        minute += 1
    if minute > 59:
        minute -= 60
        hour += 1
    return datetime.time(microsecond=microsecond, second=second, minute=minute, hour=hour)


def rango_convolucion(vector):
    #convolucion = scipy.signal.convolve(vector[:128], np.concatenate((vector[128:], vector[128:])), mode='full', method='direct')
    convolucion =  np.real(np.fft.ifft( np.fft.fft(vector[:128])*np.fft.fft(vector[128:])))
    return np.amax(convolucion)-np.amin(convolucion)


def movimiento_por_convolucion(df):
    x = df['accX'].to_numpy()
    y = df['accY'].to_numpy()
    aux = int(x.shape[0]/128)*128
    x = x[:aux].reshape((int(aux/128), 128))
    y = y[:aux].reshape((int(aux/128), 128))
    acc = np.concatenate((x, y), axis=1)
    #print(acc.shape)
    y = np.apply_along_axis(rango_convolucion, 1, acc)
    bins = 10 ** (np.arange(3.5, 10, 0.25))
    plt.xscale('log')
    plt.hist(y, bins=bins)
    plt.xlabel('Rango de la convolución')
    plt.title('Distribución rango de convolución x-y')
    plt.savefig(os.path.join(os.getcwd(), "Histograma_convolucion", document+'.png'))
    plt.show()


for document in os.listdir(tortugometro_path):
    movimiento_por_convolucion(ReadIMUData(os.path.join(tortugometro_path, document)))

