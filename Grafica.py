"""
En este archivo se encuentra el codigo para generar todas las figuras presentadas en la tesis.
"""
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
from matplotlib.colors import ListedColormap, BoundaryNorm
import scipy.signal
import Finding_features as ft

# En la siguiente linea defino los parametros para los graficos. El tamanio de la figura y de la fuente estan elegidos
# de tal manera que encajen en la tesis (formato A4, ancho 6.1 pulgadas) sin tener que reescalear la figura, y asi queda
# el texto de la figura del mismo tamanio que el texto de la tesis
plt.style.use("seaborn")
mpl.rcParams.update(
    {
        "text.usetex": True,
        "axes.titlesize": 10,
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "figure.figsize": [6.1, 4],
        "figure.autolayout": True,
        "savefig.format": 'pdf',
        "savefig.bbox": 'tight',
        "axes.linewidth": 1,
        "lines.linewidth": 1,
        "lines.markersize": 1,
    }
)


def plotsegment(df):
    """Grafica un segmento de la aceleracion de la tortuga en los 3 ejes
    toma como entrada el dataframe"""
    df['dia'] = df['datetime'].dt.date
    groups = df.groupby('dia')
    fig, ax = plt.subplots()
    for name, group in groups:
        x = mdates.date2num(group['datetime'])
        ax.plot(x, group['accX'], label='accX')
        ax.plot(x, group['accY'], label='accY')
        ax.plot(x, group['accZ'], label='accZ')
        # descomentar la siguiente linea si tambien quiere graficar la aceleracion total
        #ax.plot(x,(y**2+group['accY']**2+group['accZ']**2)**0.5, label='accTotal', color='black')
        ax.set_ylabel(str(group['datetime'].iloc[0].date()))
        ax.xaxis.set_major_locator(mdates.HourLocator())
        ax.xaxis.set_minor_locator(mdates.MinuteLocator())
        timeFmt = mdates.DateFormatter("%H:%M")
        ax.xaxis.set_major_formatter(timeFmt)
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_xlabel("Hora")
        ax.set_ylabel("Aceleración (u)")
        #ax.legend(loc=1, bbox_to_anchor=(0.3, 0.75))
        '''las siguientes lineas son para trazar lineas verticales donde se cortan los segmentos de 512
         puntos para la inferencia'''
        #for i in np.arange(int(np.trunc(x.shape[0]/512))+1):
        #    ax.axvline(x=x[512*i])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def plotacc(campa, document):
    """Grafica todos los datos de acelerometro disponibles para una tortuga en una campania dada
    Insertar la campania como string en el mismo formato que las carpetas de datos ('11_2020' por ejemplo) y
    el documento como string T seguido del numero de tortuga y .csv ('T30.csv' por ejemplo)"""
    df = mov.ReadIMUData(os.path.join(mov.tortugometro_path, campa, document))
    df['dia'] = df['datetime'].dt.date
    groups = df.groupby('dia')
    fig, ax = plt.subplots(groups.ngroups, 1, sharex='all',figsize=(6.1, 1.5*groups.ngroups))
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
            ax[i].set_ylabel(str(group['datetime'].iloc[0].day) + '/' + str(group['datetime'].iloc[0].month))
            ax[i].yaxis.set_label_position("right")
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
    fig.suptitle('Tortuga' + document[:-4] + ' ' + campa[0:2] + '/' + campa[3:])
    fig.supylabel('Aceleración (u)')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def plotacc_con_etiqueta(campa, document):
    """Hace lo mismo que plot_acc pero ademas encuentra las etiquetas disponibles para la tortuga y escribe el numero
    correspondiente a la actividad en el momento correspondiente. Las actividades son 0 quieta 1 camina 2 hace nido
    3 come 4 macho_copulando 5 hembra_copulando 6 pelea 7 otro """
    df = mov.ReadIMUData(os.path.join(mov.tortugometro_path, campa, document))
    df['dia'] = df['datetime'].dt.date
    etiquetas = os.path.exists(os.path.join(os.getcwd(), "Etiquetas", campa, document))
    if etiquetas:
        tags = lee.ReadTags(os.path.join(os.getcwd(), "Etiquetas", campa, document))
        print(tags)
    groups = df.groupby('dia')
    fig, ax = plt.subplots(groups.ngroups, 1, sharex='all')
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
                        ax[i].text(mdates.date2num(
                            df['datetime'].iloc[0].replace(hour=tags['time'][j].hour, minute=tags['time'][j].minute,
                                                           second=tags['time'][j].second)), 0, s=str(tags['tag'][j]),
                            fontsize=20)
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
    """Grafica la aceleracion de dos tortugas en el mismo dia, esto es util para observar que hacen las tortugas cuando
    se encuentran para ver si estan peleando, copulando, o algo mas. Insertar campa t1 y t2 con el mismo formato
    que en las otras funciones y el dia en formato datetime.datetime con año, mes, dia"""
    df1 = mov.ReadIMUData(os.path.join(mov.tortugometro_path, campa, t1))
    df1 = df1[(df1['datetime'] > dia.replace(hour=6, minute=0, second=0)) & (
            df1['datetime'] < dia.replace(hour=23, minute=59, second=59))]
    df2 = mov.ReadIMUData(os.path.join(mov.tortugometro_path, campa, t2))
    df2 = df2[(df2['datetime'] > dia.replace(hour=6, minute=0, second=0)) & (
            df2['datetime'] < dia.replace(hour=23, minute=59, second=59))]
    fig, ax = plt.subplots(2, 1, sharex='all')
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
    """Grafica el histograma de cantidad de movimiento en funcion de la hora del dia
    Ingresar las campanias como una lista por ejemplo ['11_2020', '11_2021'], lo hice de esta manera para poder separar
    por temporada y ver si el comportamiento entre verano y primavera es distinto"""
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
    # Cambiar el titulo segun las campanias en la siguiente linea
    plt.title('Enero')
    plt.show()


def histograma_temperatura_vs_movimiento():
    """Grafica el histograma de tiempo en movimiento dada la temperatura, en funcion de la temperatura
    Esta tod comentado porque el calculo tarda bastante asi que lo guarde en un pickle, para reahcer el calculo
    descomentar y correrlo (puede servir para actualizar si se tienen datos nuevos) """
    #todos = pd.DataFrame({'rangos': pd.Series(dtype=np.float64), 'tempIMU_C': pd.Series(dtype=np.float64)})
    #for campa in os.listdir(mov.tortugometro_path):
    #    for document in os.listdir(os.path.join(mov.tortugometro_path, campa)):
    #        if document[-3:] == 'csv':
    #            df = mov.ReadIMUData(os.path.join(mov.tortugometro_path, campa, document))
    #            rangos = ft.calculate_features(df)
    #            rangos = rangos[:,0]
    #            boo, df2 = mov.movimiento_vs_T(df, rangos)
    #            if boo:
    #                aux = pd.DataFrame(
    #                    {'rangos': pd.Series(dtype=np.float64), 'tempIMU_C': pd.Series(dtype=np.float64)})
    #                aux['tempIMU_C'] = df['tempIMU_C'].iloc[:rangos.size * 128:128]
    #                aux['rangos'] = rangos
    #                todos = pd.concat([todos, aux], ignore_index=True)
    fig, axs = plt.subplots(2, 1, sharex='all', figsize=[4, 6])
    #todos['rangos'] = todos['rangos'].astype(np.float64)
    #todos['tempIMU_C'] = todos['tempIMU_C'].astype(np.float64)
    #todos.to_pickle('todosimu.pkl')
    todos = pd.read_pickle('todos.pkl')

    q = todos.loc[(10 ** 3 < todos['rangos']) & (todos['rangos'] < 10 ** 5)]
    m = todos.loc[todos['rangos'] > 10 ** 6]
    quieto, bquieto = np.histogram(q['Temperatura'], bins=np.arange(15, 38, 2.5), density=False)
    movs, bmov = np.histogram(m['Temperatura'], bins=np.arange(15, 38, 2.5), density=False)
    movs = movs * 0.3712 / 60
    quieto = quieto * 0.3712 / 60
    axs[0].bar(np.arange(15, 36, 2.5), movs / (movs + quieto), width=2.5, align='edge')
    axs[1].bar(np.arange(15, 36, 2.5),(movs + quieto)/np.sum(movs+quieto), width=2.5, align='edge')

    axs[0].set_ylabel(r'P(movimiento $\vert$ T)')
    axs[1].set_ylabel('P(T)')
    axs[1].set_xlabel('Temperatura (°C)')
    # descomentar la siguiente linea para gguardar la figura
    # plt.savefig(os.path.join(os.getcwd(), 'Histograma_movimiento_vs_temperatura_ibutton', 'mov_dado_Tambiente.pdf'))
    plt.show()


def colorcurve(campa, document):
    """Realiza un grafico de la temperatura del ambiente y la temperatura sobre la tortuga, en funcion del tiempo y
    colorea la temperatura sobre la tortuga segun el estado de la tortuga (quieta o en movimiento) las variables campa
    y document siguen la misma convencion que en las otras funciones"""
    if document[-3:] == 'csv':
        df = mov.ReadIMUData(os.path.join(mov.tortugometro_path, campa, document))
        rangos = mov.movimiento_por_convolucion(df)
        boo, df2 = mov.movimiento_vs_T(df, rangos)
        if boo:
            df2['dia'] = df2['datetime'].dt.day
            groups = df2.groupby('dia')
            fig, ax = plt.subplots(groups.ngroups, 1, sharex='all')
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
                    ax[i].scatter(x, y, c=group['rangos'], cmap=cmap, norm=norm, s=1,
                                  label='Temperatura sobre caparazón')
                    ax[i].plot(x, group['Temperatura_campo'], color='b', label='Temperatura ambiente')
                    ax[i].set_ylabel(str(group['datetime'].iloc[0].day) + '/' + str(group['datetime'].iloc[0].month))
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
                # fig.suptitle('Temperatura ambiente y sobre el caparazón del espécimen ' + document[:-4]+ ' ' + campa[:2]+'/'+campa[3:])
                fig.supylabel("Temperatura")
                plt.subplots_adjust(wspace=0, hspace=0)
                # plt.tight_layout()
                plt.savefig(os.path.join(os.getcwd(), 'Color_curves', 'tesis.pdf'))
                plt.show()


def histograma_etiquetas():
    with open(os.path.join(os.getcwd(), 'aceleraciones_etiquetadas', 'database_cruda_etiquetada.pickle'),
              'rb') as handle:
        db = pkl.load(handle)

    h, b = np.histogram(db[:, 0, 0], bins=np.arange(-0.5, 8.5, 1.), density=False)
    fig, ax = plt.subplots(figsize=(6.1, 4))
    ax.bar(['quieto', 'camina', 'hace nido', 'come', 'copulando (m)', 'copulando (h)', 'pelea', 'otros'], h)
    fig.set_size_inches(6.1, 4)
    plt.title("Cantidad de etiquetas")
    plt.xticks(rotation=30)
    plt.savefig('C:/Users/bicho/OneDrive/Documentos/Balseiro/maestria/Tesis/figuras/maestria/cantidad_etiquetas.pdf')
    plt.show()


def database_etiquetada():
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


def acc_conv_dsp(tag=1):
    with open(os.path.join(os.getcwd(), 'aceleraciones_etiquetadas', 'database_cruda_etiquetada.pickle'),
              'rb') as handle:
        db = pkl.load(handle)
    fig, axs = plt.subplots(2, 3, sharex='col', sharey='col', figsize=[6.1, 8])
    actividad = db[db[:, 0, 0] == tag]
    acc = actividad[:, 1:513, :]
    acc = np.swapaxes(acc, 0, 1)
    t = 0.174 * np.arange(512)
    acc = np.swapaxes(acc, 1, 2)
    conv = ft.convolve_signal(acc)
    conv = ft.normalize_convolution(conv)
    f, dsp = scipy.signal.periodogram(conv, fs=1 / 0.174, axis=1)
    for i in np.arange(2):
        axs[i, 0].plot(t, acc[:, 0, i], label='accX')
        axs[i, 0].plot(t, acc[:, 1, i], label='accY')
        axs[i, 0].plot(t, acc[:, 2, i], label='accZ')
        axs[i, 1].plot(t, conv[i, :])
        axs[i, 2].plot(f[:15], dsp[i, :15], '-o')
    axs[0, 0].legend()
    axs[0, 0].set_title('Señal cruda')
    axs[1, 0].ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
    axs[0, 1].set_title('Convolución x*y')
    axs[0, 2].set_title('Densidad espectral')
    axs[1, 0].set_xlabel('Tiempo (s)')
    axs[1, 1].set_xlabel('Tiempo (s)')
    axs[1, 2].set_xlabel('Frecuencia (Hz)')
    plt.savefig(
        'C:/Users/bicho/OneDrive/Documentos/Balseiro/maestria/Tesis/figuras/maestria/accconvdsp_{}.pdf'.format(tag))
    plt.show()


def acc_conv_dsp_s(acc):
    fig, axs = plt.subplots(2, 3, sharex='col', sharey='col', figsize=[6.1, 8])
    acc = ft.make_windows(acc, 512)
    conv = ft.convolve_signal(acc)
    conv = ft.normalize_convolution(conv, 0.174)
    t = 0.174 * np.arange(512)
    f, dsp = scipy.signal.periodogram(conv, fs=1 / 0.174, axis=1)
    for i in np.arange(2):
        axs[i, 0].plot(t, acc[:, 0, i], label='accX')
        axs[i, 0].plot(t, acc[:, 1, i], label='accY')
        axs[i, 0].plot(t, acc[:, 2, i], label='accZ')
        axs[i, 1].plot(t, conv[i, :])
        axs[i, 2].plot(f[:15], dsp[i, :15], '-o')
    axs[0, 0].legend()
    axs[0, 0].set_title('Señal cruda')
    axs[1, 0].ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
    axs[0, 1].set_title('Convolución x*y')
    axs[0, 2].set_title('Densidad espectral')
    axs[1, 0].set_xlabel('Tiempo (s)')
    axs[1, 1].set_xlabel('Tiempo (s)')
    axs[1, 2].set_xlabel('Frecuencia (Hz)')
    plt.savefig(
        'C:/Users/bicho/OneDrive/Documentos/Balseiro/maestria/Tesis/figuras/maestria/accconvcopula.pdf')
    plt.show()

def dsp_accvsconv(data):
    t = np.arange(512)*.174
    fig, axs = plt.subplots(2, 2, sharex='col', figsize=[6.1, 4])
    axs[0,0].plot(t,data[:, 0], label='accX')
    axs[0,0].plot(t,data[:, 1], label='accY')
    axs[0,0].plot(t,data[:, 2], label='accZ')
    accx_normalized = data[:, 0]-np.mean(data[:, 0])
    accx_normalized = accx_normalized/np.trapz(accx_normalized ** 2, dx=0.174) ** 0.5
    accy_normalized = data[:, 1]-np.mean(data[:, 1])
    accy_normalized = accy_normalized/np.trapz(accy_normalized ** 2, dx=0.174) ** 0.5
    accz_normalized = data[:, 2]-np.mean(data[:, 2])
    accz_normalized = accz_normalized/np.trapz(accz_normalized ** 2, dx=0.174) ** 0.5
    f,dspx=scipy.signal.periodogram(accx_normalized, fs=1 / 0.174)
    dspy=scipy.signal.periodogram(accy_normalized, fs=1 / 0.174)[1]
    dspz=scipy.signal.periodogram(accz_normalized, fs=1 / 0.174)[1][2:]
    axs[0,1].plot(f[:25],dspx[:25], label='X')
    axs[0,1].plot(f[:25],dspy[:25], label='Y')

    conv = np.real(np.fft.ifft(np.multiply(np.fft.fft(data[:, 0], axis=0), np.fft.fft(data[:, 1], axis=0)), axis=0))
    axs[1,0].plot(t,conv)
    conv = conv - np.mean(conv)
    conv = conv / np.trapz(conv ** 2, dx=0.174) ** 0.5
    axs[1,1].plot(f[:25], scipy.signal.periodogram(conv, fs=1 / 0.174)[1][:25])
    axs[0,0].legend()
    axs[0,1].legend()
    axs[0,0].set_title('Señal cruda')
    axs[0,1].set_title('Densidad espectral')
    axs[1,0].set_title('Convolución x*y')
    axs[1,1].set_title('Densidad espectral de la convolucion')
    axs[1,0].set_xlabel('Tiempo (s)')
    axs[1,1].set_xlabel('Frecuencia (Hz)')
    #plt.savefig('C:/Users/bicho/OneDrive/Documentos/Balseiro/maestria/Tesis/figuras/maestria/convolucion.pdf')
    plt.show()