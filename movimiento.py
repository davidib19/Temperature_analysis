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
import pickle as pkl

#plt.style.use("seaborn")
mpl.rcParams.update(
    {
        "axes.titlesize": 14,
        "font.size": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
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


def ReadIMUData(path, cols=['date', 'timeGMT', 'accX', 'accY', 'accZ', 'tempIMU_C', 'lat',  'lon']):
    """Lee los datos del tortugómetro con el formato que vienen y los guarda en un pandas dataframe con una columna
    en tipo datetime.datetime transformada a hora de Argentina."""
    df = pd.read_csv(path, parse_dates=['date'], converters={'timeGMT': formatIMU}, sep=';', usecols=cols)
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
    #plt.savefig(os.path.join(os.getcwd(), "Histograma_convolucion", document[:-4]+'.png'))
    plt.show()


def color_curve(df, rangos, window):

    cmap = ListedColormap(['r', 'y', 'g','k'])
    norm = BoundaryNorm([-0.01, 0.99, 1.99, 2.99, 3.99], cmap.N)
    df['dia'] = df['datetime'].dt.date
    groups = df.groupby('dia')
    fig, ax = plt.subplots(groups.ngroups, 1, sharex=True, sharey=True)
    lim1 = mdates.date2num(df['datetime'].iloc[0].replace(hour=8, minute=0, second=0))
    lim2 = mdates.date2num(df['datetime'].iloc[0].replace(hour=20, minute=59, second=59))
    i = 0
    for name, group in groups:
        x = mdates.date2num(group['datetime'] - (name - next(iter(groups.groups.keys()))))
        y = group['accX']
        num_lines = int(np.trunc(y.shape[0] / window))
        x = np.stack(np.split(x[:window * num_lines], num_lines), axis=-1)
        y = np.stack(np.split(y[:window * num_lines], num_lines), axis=-1)
        points = np.stack((x.T, y.T), axis=-1)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(rangos[i])
        ax[i].add_collection(lc)
        ax[i].set_ylabel(str(group['datetime'].iloc[0].day) + '/' + str(group['datetime'].iloc[0].month),
                             rotation=45)
        i += 1
    i -= 1
    ax[i].xaxis.set_major_locator(mdates.HourLocator())
    ax[i].xaxis.set_minor_locator(mdates.MinuteLocator())
    timeFmt = mdates.DateFormatter("%H:%M")
    ax[i].xaxis.set_major_formatter(timeFmt)
    ax[i].set_xlim(lim1, lim2)
    ax[i].set_ylim(-8000,10000)
    ax[i].tick_params(axis='x', labelrotation=45)
    ax[i].set_xlabel("Hora del día")

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

    dict = {'datetime': df['datetime'][:rangos.size*512:512], 'rangos': rangos,
            'Temperatura_campo': np.interp(df['datetime'][:rangos.size*512:512].apply(toTimestamp), ibutton['datetime'].apply(toTimestamp), ibutton[temp]),
            'tempIMU_C':df['tempIMU_C'][:rangos.size*512:512]}
    df2 = pd.DataFrame(data=dict)
    #fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    #df2.loc[df2['rangos'] > 10**6].hist(column='Temperatura', ax=axs[0])
    #df2.loc[df2['rangos'] < 10 ** 5].hist(column='Temperatura', ax=axs[1])
    #axs[0].title.set_text('Movimiento')
    #axs[1].title.set_text('Quietud')
    #plt.savefig(os.path.join(os.getcwd(), 'Histograma_movimiento_vs_temperatura_ibutton', document[:-4]+'-{}_{}.png'.format(first_date.month, first_date.year)))
    return True, df2

#histograma_temperatura_vs_movimiento()
"""fig, axs = plt.subplots(2, 1, sharex=True, sharey=False)
todos = pd.read_pickle('todos.pkl')
print(todos)
q = todos.loc[(10 ** 3 < todos['rangos']) & (todos['rangos'] < 10 ** 5)]
m = todos.loc[todos['rangos'] > 10 ** 6]
quieto, bquieto = np.histogram(q['Temperatura'], bins=np.arange(10, 38, 2.6), density=False)
mov, bmov = np.histogram(m['Temperatura'], bins=np.arange(10, 38, 2.6), density=False)
mov = mov * 0.3712 / 60
quieto = quieto * 0.3712 / 60
condicional = mov/(mov+quieto)
axs[0].bar(np.arange(10, 35, 2.6), mov, width=2.6, align='edge')
axs[1].bar(np.arange(10, 35, 2.6), quieto, width=2.6, align='edge')
axs[0].set_ylabel('Hours moving')
axs[1].set_ylabel('Hours still')
axs[1].set_xlabel('Temperature (C)')
plt.show()
plt.savefig(os.path.join(os.getcwd(), 'Histograma_movimiento_vs_temperatura_ibutton', 'todos.png'))
plt.bar(np.arange(10, 35, 2.6), condicional, width=2.6, align='edge')
plt.ylabel("Hours moving / Hours the air had each temperature")
plt.xlabel('Temperature (C)')
plt.show()
"""
"""
#color curve temperature vs time vs motion
for campa in os.listdir(tortugometro_path):
    for document in os.listdir(os.path.join(tortugometro_path, campa)):
        if document[-3:] == 'csv':
            df = ReadIMUData(os.path.join(tortugometro_path, campa, document))
            rangos = movimiento_por_convolucion(df)
            boo, df2 = movimiento_vs_T(df, rangos)
            if boo:
                df2['dia']=df2['datetime'].dt.day
                groups = df2.groupby('dia')
                fig, ax = plt.subplots(groups.ngroups, 1, sharex=True, sharey=False)
                i = 0
                cmap = ListedColormap(['r', 'y', 'g'])
                norm = BoundaryNorm([0, 10 ** 5, 10 ** 7, 10 ** 10], cmap.N)
                lim1=mdates.date2num(df2['datetime'].iloc[0].replace(hour=8,minute=0,second=0))
                lim2=mdates.date2num(df2['datetime'].iloc[0].replace(hour=20,minute=59,second=59))
                if groups.ngroups > 1:
                    for name, group in groups:
                        x = mdates.date2num(group['datetime'] - datetime.timedelta(days=name-next(iter(groups.groups.keys()))))
                        y = group['tempIMU_C']
                        ax[i].scatter(x,y, c=group['rangos'], cmap=cmap, norm=norm, s=0.75)
                        ax[i].plot(x, group['Temperatura_campo'], color='b')
                        ax[i].set_ylabel(str(group['datetime'].iloc[0].date()))
                        i += 1
                    i-=1
                    ax[i].xaxis.set_major_locator(mdates.HourLocator())
                    ax[i].xaxis.set_minor_locator(mdates.MinuteLocator())
                    timeFmt = mdates.DateFormatter("%H:%M")
                    ax[i].xaxis.set_major_formatter(timeFmt)
                    ax[i].set_xlim(lim1, lim2)
                    ax[i].tick_params(axis='x', labelrotation=45)
                    ax[i].set_xlabel("Time of the day")
                    fig.supylabel("Temperature")
                    plt.subplots_adjust(wspace=0, hspace=0)
                #plt.tight_layout()
                    #plt.savefig(os.path.join(os.getcwd(), 'Color_curves', document[:-4]+' '+campa+'.pdf'))
                    plt.show()

#histograma_temperatura_vs_movimiento()

#plt.savefig(os.path.join(os.getcwd(), 'Histograma_movimiento_vs_temperatura_ibutton', 'todos.png'))
"""

