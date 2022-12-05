from copy import copy
import time
import Finding_features as ft
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import movimiento as mov
import datetime
import os
import matplotlib as mpl
import scipy.signal
plt.style.use("seaborn")
mpl.rcParams.update(
    {
        "text.usetex": True,
        "axes.titlesize": 10,
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "figure.figsize": [6.1, 4],
        "savefig.format": 'pdf',
        "savefig.bbox": 'tight',
        "axes.linewidth":1,
        "lines.linewidth":1,
        "lines.markersize":1,
    }
)

fig, axes = plt.subplots(nrows=2, figsize=(6.1, 5), constrained_layout=True,sharex='all')


df = mov.ReadIMUData(os.path.join(mov.tortugometro_path,'11_2021','T11.csv'))
df=df[(df['datetime']>datetime.datetime(year=2021, month=11, day=21, hour=16, minute=18,second=0))&(df['datetime']<datetime.datetime(year=2021, month=11, day=21, hour=16, minute=21,second=0))]
#df = mov.ReadIMUData(os.path.join(mov.tortugometro_path,'01_2021','T30.csv'))
#df = df[(df['datetime']>datetime.datetime(year=2021, month=1, day=12, hour=9, minute=39,second=0))&(df['datetime']<datetime.datetime(year=2021, month=1, day=12, hour=11, minute=52,second=0))]

#df = mov.ReadIMUData(os.path.join(mov.tortugometro_path,'11_2021','T10.csv'))
#df = df[(df['datetime']>datetime.datetime(year=2021, month=11, day=26, hour=12, minute=53, second=0))&(df['datetime']<datetime.datetime(year=2021, month=11, day=26, hour=13, minute=4,second=0))]

acc = df[['accX', 'accY', 'accZ']].to_numpy()
del df
acc = ft.make_windows(acc, 512)
conv = ft.convolve_signal(acc)
conv = ft.normalize_convolution(conv)
f, dsp = scipy.signal.periodogram(conv, fs=1 / 0.174, scaling='density',axis=1)

x = f[:15]
Y = dsp[:, :15]
# Plot series using `plot` and a small value of `alpha`. With this view it is
# very difficult to observe the sinusoidal behavior because of how many
# overlapping series there are. It also takes a bit of time to run because so
# many individual artists need to be generated.
tic = time.time()
axes[0].plot(x, Y.T, color="C0", alpha=0.1)
toc = time.time()
axes[0].set_title("Densidad espectral")
print(f"{toc-tic:.3f} sec. elapsed")

num_series = dsp.shape[0]
num_points = dsp.shape[1]
# Now we will convert the multiple time series into a histogram. Not only will
# the hidden signal be more visible, but it is also a much quicker procedure.
tic = time.time()
# Linearly interpolate between the points in each time series
num_fine = 800
x_fine = np.linspace(x.min(), x.max(), num_fine)
y_fine = np.empty((num_series, num_fine), dtype=float)
for i in range(num_series):
    y_fine[i, :] = np.interp(x_fine, x, Y[i, :])
y_fine = y_fine.flatten()
x_fine = np.tile(x_fine, num_series).flatten()


# Plot (x, y) points in 2d histogram with log colorscale
# It is pretty evident that there is some kind of structure under the noise
# You can tune vmax to make signal more visible
cmap = copy(plt.cm.plasma)
cmap.set_bad(cmap(0))
h, xedges, yedges = np.histogram2d(x_fine, y_fine, bins=[400, 100])
pcm = axes[1].pcolormesh(xedges, yedges, h.T, cmap=cmap,
                         norm=LogNorm(vmax=1.5e2), rasterized=True)
fig.colorbar(pcm, ax=axes[1], pad=0)
axes[1].set_title("Histograma 2D de la densidad")
axes[1].set_xlabel('Frecuencia (Hz)')
plt.savefig('C:/Users/bicho/OneDrive/Documentos/Balseiro/maestria/Tesis/figuras/maestria/histdensidadnidi.pdf')
# Same data but on linear color scale
plt.show()
