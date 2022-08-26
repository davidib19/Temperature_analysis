import LeerDatosExcel as lee
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import datetime



def grafica_temp(df, label):
    x = mdates.date2num(df['datetime'])
    fig, ax = plt.subplots()
    fig.autofmt_xdate()
    ax.plot_date(x, df.iloc[label], '-')
    plt.show()


ibuttons = lee.ReadData(os.path.join(lee.iButton_path, 'Nov2021PosiblesDatosDeibuttons.xls'))
x1 = mdates.date2num(ibuttons['datetime'])
for col in ibuttons.columns:
    if os.path.exists(os.path.join(lee.tortugometro_path, col+'.csv')):
        imu = lee.ReadIMUData(os.path.join(lee.tortugometro_path, col+'.csv'))
        fig, ax = plt.subplots()
        fig.autofmt_xdate()
        ax.plot_date(x1, ibuttons[col], '-')
        ax.plot_date(mdates.date2num(imu['datetime']), imu['tempIMU_C'])
        plt.show()


