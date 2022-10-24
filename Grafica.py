import LeerDatosExcel as lee
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import matplotlib as mpl
import datetime
import numpy as np


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

FONTSIZE = 14
LEGEND_FONTSIZE = 12


"""
for campa in os.listdir(lee.data_path):
    campa_path = os.path.join(lee.data_path, campa)
    if os.path.exists(os.path.join(campa_path, "ibutton.xls")):
        ibuttons = lee.ReadData(os.path.join(campa_path, "ibutton.xls"))
        i = -1
        for col in ibuttons.columns:
            if col[:4] == 'date':
                i += 1
            if os.path.exists(os.path.join(campa_path, col + '.csv')):
                imu = lee.ReadIMUData(os.path.join(campa_path, col + '.csv'))
                fig, ax = plt.subplots()
                fig.autofmt_xdate()
                if i == 0:
                    x1 = mdates.date2num(ibuttons['datetime'])
                else:
                    x1 = mdates.date2num(ibuttons['datetime.' + str(i)])
                ax.plot_date(x1, ibuttons[col], '-')
                ax.plot_date(mdates.date2num(imu['datetime']), imu['tempIMU_C'], '-')

                print('Campaña: {} tortuga: {}'.format(campa, col))
                plt.legend(['iButton', 'Tortugómetro'])
                plt.title(col)
                plt.show()
                #plt.savefig(os.path.join(os.getcwd(),'Imu_vs_ibutton', '{}-{}'.format(campa, col)+'.png'))
"""
"""
for campa in os.listdir(lee.data_path):
    campa_path = os.path.join(lee.data_path, campa)
    if os.path.exists(os.path.join(campa_path, "ibutton.xls")):
        ibuttons = lee.ReadData(os.path.join(campa_path, "ibutton.xls"))
        i = -1
        for col in ibuttons.columns:
            if col[:4] == 'date':
                i += 1
            if os.path.exists(os.path.join(campa_path, col + '.csv')):
                imu = lee.ReadIMUData(os.path.join(campa_path, col + '.csv'))
                fig, ax = plt.subplots()
                fig.autofmt_xdate()
                if i == 0:
                    T = np.interp(df['datetime'][:rangos.size*128:128].apply(toTimestamp), ibutton['datetime'].apply(toTimestamp), ibutton[temp])
                else:
                    x1 = mdates.date2num(ibuttons['datetime.' + str(i)])
                ax.plot_date(x1, ibuttons[col], '-')
                ax.plot_date(mdates.date2num(imu['datetime']), imu['tempIMU_C'], '-')

                print('Campaña: {} tortuga: {}'.format(campa, col))
                plt.legend(['iButton', 'Tortugómetro'])
                plt.title(col)
                plt.show()
                #plt.sav
"""

