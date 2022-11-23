from copy import copy
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import movimiento as mov
import datetime
import os
import scipy.signal
fig, axes = plt.subplots(nrows=3, figsize=(6, 8), constrained_layout=True,sharex=True)


df = mov.ReadIMUData(os.path.join(mov.tortugometro_path,'01_2021','T30.csv'))
df=df[(df['datetime']>datetime.datetime(year=2021, month=1, day=12, hour=10, minute=0,second=0))&(df['datetime']<datetime.datetime(year=2021, month=1, day=12, hour=13, minute=0,second=0))]
acc = df[['accX', 'accY', 'accZ']].to_numpy()
del df
acc=acc[:int(np.trunc(acc.shape[0]/2048))*2048,:]
acc=np.stack(np.split(acc,acc.shape[0]/2048),axis=-1)
conv = np.real(np.fft.ifft(np.multiply(np.fft.fft(acc[:,0,:],axis=0), np.fft.fft(acc[:,1,:],axis=0)), axis=0))
conv=conv.T
conv -= np.mean(conv,axis=1,keepdims=True)
conv /= np.trapz(conv ** 2, dx=0.174,axis=1).reshape(-1,1) ** 0.5

f, dsp = scipy.signal.periodogram(conv, fs=1 / 0.174, scaling='density',axis=1)

x=f[:20]
Y=dsp[:,:20]
# Plot series using `plot` and a small value of `alpha`. With this view it is
# very difficult to observe the sinusoidal behavior because of how many
# overlapping series there are. It also takes a bit of time to run because so
# many individual artists need to be generated.
tic = time.time()
axes[0].plot(x, Y.T, color="C0", alpha=0.1)
toc = time.time()
axes[0].set_title("Line plot with alpha")
print(f"{toc-tic:.3f} sec. elapsed")

num_series=dsp.shape[0]
num_points=dsp.shape[1]
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
                         norm=LogNorm(vmax=5e2), rasterized=True)
fig.colorbar(pcm, ax=axes[1], label="# points", pad=0)
axes[1].set_title("2d histogram and log color scale")

# Same data but on linear color scale
pcm = axes[2].pcolormesh(xedges, yedges, h.T, cmap=cmap,
                         vmax=1.5e2, rasterized=True)
fig.colorbar(pcm, ax=axes[2], label="# points", pad=0)
axes[2].set_title("2d histogram and linear color scale")

toc = time.time()
print(f"{toc-tic:.3f} sec. elapsed")
plt.show()
