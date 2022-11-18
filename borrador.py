import Grafica
import datetime
import Finding_features as ft
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import matplotlib as mpl
import os

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
'''
with open('todos.pkl', 'rb') as f:
    todos = pkl.load(f)
q = todos.loc[(10 ** 3 < todos['rangos']) & (todos['rangos'] < 10 ** 5)]
m = todos.loc[todos['rangos'] > 10 ** 6]
quieto, bquieto = np.histogram(q['Temperatura'], bins=np.arange(15, 38, 2.5), density=False)
movs, bmov = np.histogram(m['Temperatura'], bins=np.arange(15, 38, 2.5), density=False)
total = quieto+movs
fig, axs = plt.subplots(2, 1, sharex=True, sharey=False)
axs[0].bar(np.arange(15, 36, 2.5), movs/total, width=2.5, align='edge')
axs[1].bar(np.arange(15, 36, 2.5), quieto/total, width=2.5, align='edge')
axs[0].title.set_text('Movimiento')
axs[1].title.set_text('Quietud')
axs[1].set_xlabel('Temperatura')
plt.show()
'''
'''
todos = Grafica.histograma_etiquetas()
h, b = np.histogram(todos['tag'], bins=np.arange(-0.5, 8.5, 1.), density=False)

plt.bar(['quieto', 'camina','hace nido','come','macho copulando','hembra copulando', 'pelea', 'otros'],h)
plt.title("Cantidad de etiquetas")
plt.xticks(rotation=30)
plt.savefig(os.getcwd()+'/test.pdf')
plt.show()
'''
Grafica.colorcurve('11_2021','T54.csv')
