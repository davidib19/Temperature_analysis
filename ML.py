import numpy as np
import Finding_features as ft
import os
import movimiento as mov
import pickle as pkl
import sklearn.cluster
import matplotlib.pyplot as plt
from copy import copy
from matplotlib.colors import LogNorm
import numpy.matlib
"""
# Generar la base de datos
database = np.zeros((1,1029))
for campa in os.listdir(mov.tortugometro_path):
    for document in os.listdir(os.path.join(mov.tortugometro_path, campa)):
        if document[-3:] == 'csv':
            df = mov.ReadIMUData(os.path.join(mov.tortugometro_path, campa, document))
            df['date']=df['datetime'].dt.date
            for name, group in df.groupby('date'):
                database = np.concatenate((database, ft.convolucion_ventana(group, 2048)))
print(database.shape)
# pickle the data base
with open('database.pkl', 'wb') as f:
    pkl.dump(database, f)
"""


#unpickle database22
with open('database.pkl', 'rb') as f:
    database= pkl.load(f)
database = np.delete(database, 0, axis=0)

def normalizar(database):
    for i in range(database.shape[0]):
        # for j in np.arange(4,35):
        #    if database[i,j]<1000:
        #        database[i,j]=0
        if np.max(database[i, 4:50] > 0):
            database[i, 4:50] = database[i, 4:50] / np.sum(database[i, 4:50])
        if database[i, 0] < 1:
            database[i, 0] = 1
    database[:, 0] = np.log10(database[:, 0])
    database = np.nan_to_num(database)
    return database[:, :50]


def artesanal_classifier(database):
    label = np.zeros(database.shape[0])
    for i in range(database.shape[0]):
        if database[i, 0] < 2:
            label[i] = 0
        else:
            if 1 > np.sum(database[i, 4:9]) > 0.8:
                label[i] = 1
            else:
                if np.sum(database[i, 4:14]) > 0.5:
                    label[i] = 2
                else:
                    label[i] = 3
    return label

def series_histogram(database):
    x = np.arange(4,50)
    Y = database[:,4:50]
    num_fine = 800
    x_fine = np.linspace(x.min(), x.max(), num_fine)
    y_fine = np.empty((Y.shape[1], num_fine), dtype=float)
    for i in range(Y.shape[1]):
        y_fine[i, :] = np.interp(x_fine, x, Y[i, :])
    y_fine = y_fine.flatten()
    x_fine = np.matlib.repmat(x_fine, Y.shape[1], 1).flatten()
    fig, ax = plt.subplots()
    cmap = copy(plt.cm.plasma)
    cmap.set_bad(cmap(0))
    h, xedges, yedges = np.histogram2d(x_fine, y_fine, bins=[50, 100])
    pcm = ax.pcolormesh(xedges, yedges, h.T, cmap=cmap,
                         norm=LogNorm(vmax=1.5e2), rasterized=True)
    fig.colorbar(pcm, ax=ax, label="# points", pad=0)
    ax.set_title("2d histogram and linear color scale")
    plt.show()
database=normalizar(database)
db=np.zeros((database.shape[0],4))
db[:,0]=database[:,0]
db[:,1]=database[:,4:10].sum(axis=1)
db[:,2]=database[:,10:15].sum(axis=1)
db[:,3]=database[:,15:25].sum(axis=1)
#db = db[np.where(db[0,:]>2),:]
# kmeans
minibatch = sklearn.cluster.MiniBatchKMeans(n_clusters=4)
labels = minibatch.fit_predict(db)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(db[:,0],db[:,1],db[:,2],c=labels)
plt.show()

df = mov.ReadIMUData(os.path.join(mov.tortugometro_path, '01_2021', 'T30.csv'))
procesado = ft.convolucion_ventana(df, 2048)
procesado = normalizar(procesado)
pr = np.zeros((procesado.shape[0],4))
pr[:,0]=procesado[:,0]
pr[:,1]=procesado[:,4:10].sum(axis=1)
pr[:,2]=procesado[:,10:15].sum(axis=1)
pr[:,3]=procesado[:,15:50].sum(axis=1)
labels1 = artesanal_classifier(procesado)
mov.color_curve(df, labels1, 2048)


