import LeerDatosExcel as lee
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import matplotlib as mpl
import datetime
import numpy as np
import movimiento as mov
import pickle as pkl
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

plt.style.use("seaborn")
mpl.rcParams.update(
    {
        "axes.titlesize": 14,
        "font.size": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.figsize": [6, 10],
        "figure.autolayout": True,
        "font.family": "serif",
        "font.sans-serif": ["Computer modern"],
        "savefig.format": 'pdf',
        "savefig.bbox": 'tight',
        "text.usetex": True
    }
)
FONTSIZE = 10
LEGEND_FONTSIZE = 10


def plotacc(campa, document):
    df = mov.ReadIMUData(os.path.join(mov.tortugometro_path, campa, document))
    df['dia'] = df['datetime'].dt.date
    groups = df.groupby('dia')
    fig, ax = plt.subplots(groups.ngroups, 1, sharex=True, sharey=False)
    lim1 = mdates.date2num(df['datetime'].iloc[0].replace(hour=8, minute=0, second=0))
    lim2 = mdates.date2num(df['datetime'].iloc[0].replace(hour=20, minute=59, second=59))
    i = 0
    if groups.ngroups > 1:
        for name, group in groups:
            x = mdates.date2num(group['datetime'] - (name - next(iter(groups.groups.keys()))))
            y = group['accX']
            ax[i].plot(x, y, label='accX')
            ax[i].plot(x, group['accY'], label='accY')
            ax[i].plot(x, group['accZ'], label='accZ')
            ax[i].set_ylabel(str(group['datetime'].iloc[0].day)+'/'+str(group['datetime'].iloc[0].month),rotation=45)
            i += 1
        i -= 1
        ax[i].xaxis.set_major_locator(mdates.HourLocator())
        ax[i].xaxis.set_minor_locator(mdates.MinuteLocator())
        timeFmt = mdates.DateFormatter("%H:%M")
        ax[i].xaxis.set_major_formatter(timeFmt)
        ax[i].set_xlim(lim1, lim2)
        ax[0].legend()
        ax[i].tick_params(axis='x', labelrotation=45)
        ax[i].set_xlabel("Hora del día")

    else:
        for name, group in groups:
            x = mdates.date2num(group['datetime'])
            y = group['accX']
            ax.plot(x, y)
            ax.plot(x, group['accY'])
            ax.plot(x, group['accZ'])
            ax.set_ylabel(str(group['datetime'].iloc[0].date()))

        ax.xaxis.set_major_locator(mdates.HourLocator())
        ax.xaxis.set_minor_locator(mdates.MinuteLocator())
        timeFmt = mdates.DateFormatter("%H:%M")
        ax.xaxis.set_major_formatter(timeFmt)
        ax.set_xlim(lim1, lim2)
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_xlabel("Hora del día")
    fig.suptitle('Aceleración del espécimen ' + document[:-4] + ' ' + campa[0:2]+'/'+campa[3:])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def plotacc_con_etiqueta(campa, document):
    df = mov.ReadIMUData(os.path.join(mov.tortugometro_path, campa, document))
    df['dia'] = df['datetime'].dt.date
    etiquetas = os.path.exists(os.path.join(os.getcwd(),"Etiquetas", campa, document))
    if etiquetas:
        tags = lee.ReadTags(os.path.join(os.getcwd(),"Etiquetas", campa, document))
        print(tags)
    groups = df.groupby('dia')
    fig, ax = plt.subplots(groups.ngroups, 1, sharex=True, sharey=False)
    lim1 = mdates.date2num(df['datetime'].iloc[0].replace(hour=8, minute=0, second=0))
    lim2 = mdates.date2num(df['datetime'].iloc[0].replace(hour=20, minute=59, second=59))
    i = 0
    if groups.ngroups > 1:
        for name, group in groups:
            x = mdates.date2num(group['datetime'] - (name - next(iter(groups.groups.keys()))))
            y = group['accX']
            ax[i].plot(x, y)
            ax[i].plot(x, group['accY'])
            ax[i].plot(x, group['accZ'])
            ax[i].set_ylabel(str(group['datetime'].iloc[0].date()))

            if etiquetas:
                for j in range(len(tags.index)):
                    if tags['date'][j] == name:
                        print(tags['date'][j])
                        ax[i].text(mdates.date2num(df['datetime'].iloc[0].replace(hour=tags['time'][j].hour,minute=tags['time'][j].minute,
                                                                          second = tags['time'][j].second)),0,s=str(tags['tag'][j]),fontsize=20)
            i += 1
        i -= 1

        ax[i].xaxis.set_major_locator(mdates.HourLocator())
        ax[i].xaxis.set_minor_locator(mdates.MinuteLocator())
        timeFmt = mdates.DateFormatter("%H:%M")
        ax[i].xaxis.set_major_formatter(timeFmt)
        ax[i].set_xlim(lim1, lim2)
        ax[i].tick_params(axis='x', labelrotation=45)
        ax[i].set_xlabel("Time of the day")

    else:
        for name, group in groups:
            x = mdates.date2num(group['datetime'])
            y = group['accX']
            ax.plot(x, y)
            ax.plot(x, group['accY'])
            ax.plot(x, group['accZ'])
            ax.set_ylabel(str(group['datetime'].iloc[0].date()))
            if etiquetas:
                for j in range(len(tags.index)):
                    if tags['date'][j] == name:
                        print(tags['date'][j])
                        ax.text(mdates.date2num(
                            df['datetime'].iloc[0].replace(hour=tags['time'][j].hour, minute=tags['time'][j].minute,
                                                           second=tags['time'][j].second)), 0, s=str(tags['tag'][j]),
                                   fontsize=20)
        ax.xaxis.set_major_locator(mdates.HourLocator())
        ax.xaxis.set_minor_locator(mdates.MinuteLocator())
        timeFmt = mdates.DateFormatter("%H:%M")
        ax.xaxis.set_major_formatter(timeFmt)
        ax.set_xlim(lim1, lim2)
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_xlabel("Time of the day")
    fig.suptitle(document[:-4] + ' ' + campa)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def plot_two_tortoises(campa, dia, t1, t2):
    df1 = mov.ReadIMUData(os.path.join(mov.tortugometro_path, campa, t1))
    df1 = df1[(df1['datetime'] > dia.replace(hour=6, minute=0, second=0))&(df1['datetime'] < dia.replace(hour=23, minute=59, second=59))]
    df2 = mov.ReadIMUData(os.path.join(mov.tortugometro_path, campa, t2))
    df2 = df2[(df2['datetime'] > dia.replace(hour=6, minute=0, second=0))&(df2['datetime'] < dia.replace(hour=23, minute=59, second=59))]
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=False)
    lim1 = mdates.date2num(dia.replace(hour=8, minute=0, second=0))
    lim2 = mdates.date2num(dia.replace(hour=21, minute=59, second=59))
    x1 = mdates.date2num(df1['datetime'])
    y1 = df1['accX']
    ax[0].plot(x1, y1)
    ax[0].plot(x1, df1['accY'])
    ax[0].plot(x1, df1['accZ'])
    ax[0].set_ylabel(t1[:-4])
    x2 = mdates.date2num(df2['datetime'])
    y2 = df2['accX']
    ax[1].plot(x2, y2)
    ax[1].plot(x2, df2['accY'])
    ax[1].plot(x2, df2['accZ'])
    ax[1].set_ylabel(t2[:-4])
    ax[1].xaxis.set_major_locator(mdates.HourLocator())
    ax[1].xaxis.set_minor_locator(mdates.MinuteLocator())
    timeFmt = mdates.DateFormatter("%H:%M")
    ax[1].xaxis.set_major_formatter(timeFmt)
    ax[1].set_xlim(lim1, lim2)
    ax[1].tick_params(axis='x', labelrotation=45)
    ax[1].set_xlabel("Time of the day")
    fig.suptitle(dia)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def histograma_movimiento_por_horadeldia(campanias):
    movimientos = []
    todos = []
    for campa in campanias:
        for document in os.listdir(os.path.join(mov.tortugometro_path, campa)):
            if document[-3:] == 'csv':
                df = mov.ReadIMUData(os.path.join(mov.tortugometro_path, campa, document))
                rangos = mov.movimiento_por_convolucion(df)
                aux = df['datetime'][:rangos.size * 128:128].dt.time.to_numpy()
                todos.append(aux)
                for i in np.arange(rangos.shape[0]):
                    if rangos[i] > 10 ** 7:
                        movimientos.append(aux[i])
    # pickle
    with open('movimientos.pkl', 'wb') as f:
        pkl.dump(movimientos, f)
    with open('todoshora.pkl', 'wb') as f:
        pkl.dump(todos, f)

    # unpickle
    with open('movimientos.pkl', 'rb') as f:
        movimientos = pkl.load(f)
    with open('todoshora.pkl', 'rb') as f:
        todos = pkl.load(f)
    todos = [item for sublist in todos for item in sublist]

    f8t10 = 0
    f10to12 = 0
    f12to14 = 0
    f14to16 = 0
    f16to18 = 0
    f18to20 = 0
    for i in np.arange(len(todos)):
        if todos[i] >= datetime.time(8, 0, 0):
            if todos[i] < datetime.time(10, 0, 0):
                f8t10 += 1
            else:
                if todos[i] < datetime.time(12, 0, 0):
                    f10to12 += 1
                else:
                    if todos[i] < datetime.time(14, 0, 0):
                        f12to14 += 1
                    else:
                        if todos[i] < datetime.time(16, 0, 0):
                            f14to16 += 1
                        else:
                            if todos[i] < datetime.time(18, 0, 0):
                                f16to18 += 1
                            else:
                                if todos[i] < datetime.time(20, 0, 0):
                                    f18to20 += 1
    x = np.arange(8, 20, 2)
    y = np.array([f8t10, f10to12, f12to14, f14to16, f16to18, f18to20]) / (len(todos))
    plt.bar(x, y, width=2, align='edge')

    # repeat the same for movimientos
    f8t10 = 0
    f10to12 = 0
    f12to14 = 0
    f14to16 = 0
    f16to18 = 0
    f18to20 = 0
    for i in np.arange(len(movimientos)):
        if movimientos[i] >= datetime.time(8, 0, 0):
            if movimientos[i] < datetime.time(10, 0, 0):
                f8t10 += 1
            else:
                if movimientos[i] < datetime.time(12, 0, 0):
                    f10to12 += 1
                else:
                    if movimientos[i] < datetime.time(14, 0, 0):
                        f12to14 += 1
                    else:
                        if movimientos[i] < datetime.time(16, 0, 0):
                            f14to16 += 1
                        else:
                            if movimientos[i] < datetime.time(18, 0, 0):
                                f16to18 += 1
                            else:
                                if movimientos[i] < datetime.time(20, 0, 0):
                                    f18to20 += 1
    y = np.array([f8t10, f10to12, f12to14, f14to16, f16to18, f18to20]) / (len(todos))
    plt.bar(x, y, width=2, align='edge')
    plt.legend(['Total de mediciones', 'Mediciones donde se detectó movimiento'])
    plt.xlabel('Hora del día')
    plt.title('Enero')
    plt.show()


def histograma_temperatura_vs_movimiento():
    todos = pd.DataFrame({'rangos': pd.Series(dtype=np.float64), 'tempIMU_C': pd.Series(dtype=np.float64)})
    for campa in os.listdir(mov.tortugometro_path):
        for document in os.listdir(os.path.join(mov.tortugometro_path, campa)):
            if document[-3:] == 'csv':
                df = mov.ReadIMUData(os.path.join(mov.tortugometro_path, campa, document))
                rangos = mov.movimiento_por_convolucion(df)
                boo, df2 = mov.movimiento_vs_T(df, rangos)
                if boo:
                    aux = pd.DataFrame(
                        {'rangos': pd.Series(dtype=np.float64), 'tempIMU_C': pd.Series(dtype=np.float64)})
                    aux['tempIMU_C'] = df['tempIMU_C'].iloc[:rangos.size * 128:128]
                    aux['rangos'] = rangos
                    todos = pd.concat([todos, aux], ignore_index=True)
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=False)
    todos['rangos'] = todos['rangos'].astype(np.float64)
    todos['tempIMU_C'] = todos['tempIMU_C'].astype(np.float64)
    todos.to_pickle('todosimu.pkl')

    q = todos.loc[(10 ** 3 < todos['rangos']) & (todos['rangos'] < 10 ** 5)]
    m = todos.loc[todos['rangos'] > 10 ** 6]
    quieto, bquieto = np.histogram(q['Temperatura'], bins=np.arange(15, 38, 2.5), density=False)
    movs, bmov = np.histogram(m['Temperatura'], bins=np.arange(15, 38, 2.5), density=False)
    movs = movs * 0.3712 / 60
    quieto = quieto * 0.3712 / 60
    axs[0].bar(np.arange(15, 36, 2.5), movs/(movs+quieto), width=2.5, align='edge')
    axs[1].bar(np.arange(15, 36, 2.5), quieto/(movs+quieto), width=2.5, align='edge')
    axs[0].title.set_text('Movimiento')
    axs[1].title.set_text('Quietud')
    #plt.savefig(os.path.join(os.getcwd(), 'Histograma_movimiento_vs_temperatura_ibutton', 'todosimmu.png'))
    plt.show()


def colorcurve(campa, document):
    if document[-3:] == 'csv':
        df = mov.ReadIMUData(os.path.join(mov.tortugometro_path, campa, document))
        rangos = mov.movimiento_por_convolucion(df)
        boo, df2 = mov.movimiento_vs_T(df, rangos)
        if boo:
            df2['dia'] = df2['datetime'].dt.day
            groups = df2.groupby('dia')
            fig, ax = plt.subplots(groups.ngroups, 1, sharex=True, sharey=False)
            i = 0
            cmap = ListedColormap(['r', 'y', 'g'])
            norm = BoundaryNorm([0, 10 ** 5, 10 ** 7, 10 ** 10], cmap.N)
            lim1 = mdates.date2num(df2['datetime'].iloc[0].replace(hour=8, minute=0, second=0))
            lim2 = mdates.date2num(df2['datetime'].iloc[0].replace(hour=20, minute=59, second=59))
            if groups.ngroups > 1:
                for name, group in groups:
                    x = mdates.date2num(
                        group['datetime'] - datetime.timedelta(days=name - next(iter(groups.groups.keys()))))
                    y = group['tempIMU_C']
                    ax[i].scatter(x, y, c=group['rangos'], cmap=cmap, norm=norm, s=1, label='Temperatura sobre caparazón')
                    ax[i].plot(x, group['Temperatura_campo'], color='b',label='Temperatura ambiente')
                    ax[i].set_ylabel(str(group['datetime'].iloc[0].day)+'/'+str(group['datetime'].iloc[0].month))
                    i += 1
                i -= 1
                ax[i].xaxis.set_major_locator(mdates.HourLocator())
                ax[i].xaxis.set_minor_locator(mdates.MinuteLocator())
                timeFmt = mdates.DateFormatter("%H:%M")
                ax[i].xaxis.set_major_formatter(timeFmt)
                ax[i].set_xlim(lim1, lim2)
                ax[i].tick_params(axis='x', labelrotation=45)
                ax[i].set_xlabel("Hora del dia")
                ax[0].legend()
                #fig.suptitle('Temperatura ambiente y sobre el caparazón del espécimen ' + document[:-4]+ ' ' + campa[:2]+'/'+campa[3:])
                fig.supylabel("Temperatura")
                plt.subplots_adjust(wspace=0, hspace=0)
                # plt.tight_layout()
                plt.savefig(os.path.join(os.getcwd(), 'Color_curves','tesis.pdf'))
                plt.show()

def histograma_etiquetas():
    todos = pd.DataFrame({'date', 'time','tag','observation'})
    for campa in os.listdir(os.path.join(os.getcwd(), 'Etiquetas')):
        for document in os.listdir(os.path.join(os.getcwd(), 'Etiquetas', campa)):
            if document[-3:] == 'csv':
                print(campa+document)
                df = lee.ReadTags(os.path.join(os.getcwd(), 'Etiquetas', campa, document))
                todos = pd.concat([todos, df], ignore_index=True)
    return todos
