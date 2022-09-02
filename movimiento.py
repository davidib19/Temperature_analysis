import pandas as pd
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.dates as mdates
import matplotlib as mpl
import LeerDatosExcel as lee
plt.style.use("seaborn")
mpl.rcParams.update(
    {
        "axes.titlesize": 24,
        "font.size": 20,
        "axes.labelsize": 20,
        "legend.fontsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "figure.figsize": [9, 6.5],
        "figure.autolayout": True,
        "font.family": "serif",
        "font.sans-serif": ["Helvetica"],
        "savefig.format": 'pdf',
        "savefig.bbox": 'tight'
    }
)

tortugometro_path = os.path.join(os.path.split(os.getcwd())[0], "Datos")


def toTimestamp(d):
  return datetime.datetime.timestamp(d)


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
    convolucion = np.real(np.fft.ifft(np.fft.fft(vector[:128])*np.fft.fft(vector[128:])))
    return np.amax(convolucion)-np.amin(convolucion)


def movimiento_por_convolucion(df):
    x = df['accX'].to_numpy()
    y = df['accY'].to_numpy()
    aux = int(x.shape[0]/128)*128
    x = x[:aux].reshape((int(aux/128), 128))
    y = y[:aux].reshape((int(aux/128), 128))
    acc = np.concatenate((x, y), axis=1)
    rangos = np.apply_along_axis(rango_convolucion, 1, acc)
    return rangos


def histograma_rangos(y):
    bins = 10 ** (np.arange(3.5, 10, 0.25))
    plt.xscale('log')
    plt.hist(y, bins=bins)
    plt.xlabel('Rango de la convolución')
    plt.title('Distribución rango de convolución x-y')
    plt.savefig(os.path.join(os.getcwd(), "Histograma_convolucion", document[:-4]+'.png'))
    plt.show()


def color_curve(df, rangos):
    x = mdates.date2num(df['datetime'])
    y = df['accX'].to_numpy()
    aux = int(y.shape[0] / 128) * 128
    points = np.array([x[:aux], y[:aux]]).T.reshape(int(aux/128), 128, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    cmap = ListedColormap(['r', 'y', 'g'])
    norm = BoundaryNorm([0, 10**5, 10**7, 10**10], cmap.N)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(rangos)
    lc.set_linewidth(2)
    fig, ax = plt.subplots()
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_minor_locator(mdates.HourLocator())
    timeFmt = mdates.DateFormatter("%d %H")
    ax.xaxis.set_major_formatter(timeFmt)
    ax.autoscale_view()
    plt.show()


def movimiento_vs_T(df, rangos):
    first_date = df['datetime'][0]
    last_date = df['datetime'][len(df)-1]
    if datetime.datetime(year=2019, month=9, day=26) <= first_date and last_date <= datetime.datetime(year=2020, month=11, day=30):
        ibutton = lee.ReadData(os.path.join(lee.data_path, "iButton_Campo", "09_2019-11_2020.xls"))
        temp = 'refugio Oeste'
    else:
        if datetime.datetime(year=2021, month=9, day=22) <= first_date and last_date <= datetime.datetime(year=2022, month=4, day=18):
            ibutton = lee.ReadData(os.path.join(lee.data_path, "iButton_Campo", "09_2021-04_2022.xls"))
            temp = 'refugio'
        else:
            if datetime.datetime(year=2020, month=12, day=2) <= first_date and last_date <= datetime.datetime(year=2021, month=8, day=29):
                ibutton = lee.ReadData(os.path.join(lee.data_path, "iButton_Campo", "12_2020-08_2021.xls"))
                temp = 'enterrado'
            else:
                return False, pd.DataFrame(columns=['rangos', 'Temperatura'])

    dict = {'datetime': df['datetime'][:rangos.size*128:128], 'rangos': rangos,
            'Temperatura': np.interp(df['datetime'][:rangos.size*128:128].apply(toTimestamp), ibutton['datetime'].apply(toTimestamp), ibutton[temp])}
    df2 = pd.DataFrame(data=dict)
    #fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    #df2.loc[df2['rangos'] > 10**6].hist(column='Temperatura', ax=axs[0])
    #df2.loc[df2['rangos'] < 10 ** 5].hist(column='Temperatura', ax=axs[1])
    #axs[0].title.set_text('Movimiento')
    #axs[1].title.set_text('Quietud')
    #plt.savefig(os.path.join(os.getcwd(), 'Histograma_movimiento_vs_temperatura_ibutton', document[:-4]+'-{}_{}.png'.format(first_date.month, first_date.year)))
    return True, df2


todos = pd.DataFrame({'rangos': pd.Series(dtype=np.float64), 'Temperatura': pd.Series(dtype=np.float64)})
for campa in os.listdir(tortugometro_path):
    for document in os.listdir(os.path.join(tortugometro_path, campa)):
        if document[-3:] == 'csv':
            df = ReadIMUData(os.path.join(tortugometro_path, campa, document))
            rangos = movimiento_por_convolucion(df)
            boo, df2 = movimiento_vs_T(df, rangos)
            if boo:
                todos = pd.concat([todos, df2[['rangos', 'Temperatura']]], ignore_index=True)
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
todos['rangos'] = todos['rangos'].astype(np.float64)
todos['Temperatura'] = todos['Temperatura'].astype(np.float64)
todos.loc[(10 ** 3 < todos['rangos']) & (todos['rangos'] < 10 ** 5)].hist(column='Temperatura', ax=axs[1])
todos.loc[todos['rangos'] > 10 ** 6].hist(column='Temperatura', ax=axs[0])
axs[0].title.set_text('Movimiento')
axs[1].title.set_text('Quietud')
plt.savefig(os.path.join(os.getcwd(), 'Histograma_movimiento_vs_temperatura_ibutton', 'todos.png'))
