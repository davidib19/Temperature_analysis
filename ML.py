import numpy as np
import scipy.signal
import datetime
import Finding_features as ft
import os
import movimiento as mov
import pickle as pkl
import sklearn.cluster
import LeerDatosExcel as Lee
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


def generate_database(save=False):
    database = np.zeros((1, 257))
    for campa in os.listdir(mov.tortugometro_path):
        for document in os.listdir(os.path.join(mov.tortugometro_path, campa)):
            if document[-3:] == 'csv':
                df = mov.ReadIMUData(os.path.join(mov.tortugometro_path, campa, document),
                                     cols=['date', 'timeGMT', 'accX', 'accY', 'accZ'])
                df['date'] = df['datetime'].dt.date
                for name, group in df.groupby('date'):
                    database = np.concatenate((database, ft.calculate_features(group)))
    print(database.shape)
    # pickle the data base
    if save:
        with open('database.pkl', 'wb') as f:
            pkl.dump(database, f)
    return database

def mi_database():
    tags = Lee.ReadMyTags()
    print(tags)
    db = np.zeros((1, 258))
    for j in tags.index:
        df = mov.ReadIMUData(os.path.join(mov.tortugometro_path, tags['campa'][j], tags['tortuga'][j] + '.csv'))
        df = df[(df['datetime'] > tags['inicio'][j]) & (df['datetime'] < tags['fin'][j])]
        data = ft.calculate_features(df)
        etiqueta = np.ones_like(data[:, 0]) * tags['etiqueta'][j].reshape((-1, 1))
        data = np.concatenate((etiqueta.T, data), axis=1)
        print(data.shape)
        db = np.concatenate((db, data), axis=0)
    db = db[1:, :]
    with open(os.path.join(os.getcwd(), 'aceleraciones_etiquetadas', 'mi_database.pkl'),
              'wb') as handle:
        pkl.dump(db, handle)
    print('database saved at mi_database.pkl')
# unpickle database
with open('database.pkl', 'rb') as f:
    database = pkl.load(f)
database = np.delete(database, 0, axis=0)


def artesanal_classifier(data):
    labels =np.zeros_like(data[:,0])
    labels[data[:, 0] > 5] = 5
    labels[((data[:, 3] > 0.6) |(data[:, 4] > 0.6)) & (data[:, 0] > 5)] = 2
    labels[((data[:, 1] > 0.6) & (data[:, 0] > 5))] = 1
    labels[((np.argmax(data[:, 1:], axis=1) == 0) & (data[:, 6] > 0.08) & (data[:, 0] > 5) & (np.amax(data[:, 1:],
                                                                                                    axis=1) > 0.2))]=3
    labels[((data[:, 0]) > 5) & (np.sum(data[:, 10:],axis=1) > 0.4) & (np.amax(data[:, 1:],axis=1)<0.5)] = 4
    return labels


def clustering(n):
    db = np.delete(database, database[:, 0] > 5, 0)
    db[:, 14] = np.sum(db[:, 14:], axis=1)
    db = db[:, 1:15]
    db =db[~np.isnan(db).any(axis=1),:]
    minibatch = sklearn.cluster.MiniBatchKMeans(n_clusters=n)
    labels = minibatch.fit_predict(db)
    with open('minibatch.pkl', 'wb') as f:
        pkl.dump(minibatch, f)
    return labels, minibatch


def Hopfield(examples, beta, e):
    return np.round(examples @ (np.exp(beta * examples.T @ e) / sum(np.exp(beta * examples.T @ e))),decimals=2)


def make_examples():
    with open(os.path.join(os.getcwd(), 'aceleraciones_etiquetadas', 'database_cruda_etiquetada.pickle'),
              'rb') as handle:
        db = pkl.load(handle)
    examples = np.zeros((14, 3))
    for etiqueta in [1]:
        aux = db[db[:, 0, 0] == etiqueta]
        conv0 = scipy.signal.fftconvolve(aux[1, 1:513, 0], np.concatenate((aux[1, 1:513, 1], aux[1, 1:513, 1])),
                                         mode='same')
        conv0 -= np.mean(conv0)
        conv0 /= np.trapz(conv0 ** 2, dx=0.174) ** 0.5
        f, dsp = scipy.signal.periodogram(conv0, fs=1 / 0.174, scaling='density')
        examples[:, etiqueta - 1] = dsp[1:15]
        examples[-1,etiqueta -1] =np.sum(dsp[14:])
    examples[2, 1] = 1.
    df = mov.ReadIMUData(os.path.join(mov.tortugometro_path, '11_2021', 'T11.csv'))
    df = df[(df['datetime'] > datetime.datetime(year=2021, month=11, day=21, hour=16, minute=20, second=0)) & (
                df['datetime'] < datetime.datetime(year=2021, month=11, day=21, hour=16, minute=29, second=0))]
    acc = df[['accX', 'accY', 'accZ']].to_numpy()
    del df
    acc = ft.make_windows(acc, 512)
    conv = ft.convolve_signal(acc)
    conv = ft.normalize_convolution(conv)
    f, dsp = scipy.signal.periodogram(conv, fs=1 / 0.174, scaling='density', axis=1)
    examples[:, 2] = dsp[0, 1:15]
    examples[-1, 2] = np.sum(dsp[0,14:])

    examples=np.round(examples,decimals=2)
    # pickle examples
    with open('examples.pkl', 'wb') as f:
        pkl.dump(examples, f)
    return examples


def predict_Hopfield(dsp, beta, examples, precision):
    labels=np.zeros_like(dsp[:,0])
    for j in np.arange(labels.shape[0]):
        if dsp[j, 0] > 5:
            labels[j] = 5
            dsp[j,14]=np.sum(dsp[j,14:])
            out = Hopfield(examples, beta, np.round(dsp[j, 1:15],decimals=2))
            for i in np.arange(examples.shape[1]):
                if np.sum(np.abs(out - examples[:, i])) <= precision:
                    labels[j] = i+1
    labels[labels == 3] = 4
    return labels


def test_methods_unlabeled(df):
    dsp = ft.calculate_features(df)
    labels_artesanal = artesanal_classifier(dsp)
    print(labels_artesanal)

    examples = make_examples()
    # examples = examples[:,np.array([0,1,3])]
    # examplest= examples.T
    # print(examplest)
    # aux = np.ones((3,1))*6
    # examplest=np.c_[aux,examplest]
    labels_hopfield = predict_Hopfield(dsp, 10, examples, 0.2)
    print(labels_hopfield)
    dummy, minibatch = clustering(5)
    labels_cluster=np.zeros_like(labels_hopfield)
    dsp[:, 14] = np.sum(dsp[:, 14:],axis=1)
    labels_cluster[dsp[:,0]>5] = minibatch.predict(dsp[dsp[:,0]>5, 1:15])+1
    print(labels_cluster)
    return labels_artesanal, labels_hopfield, labels_cluster


def test_methods_labeled(data):
    true_labels = data[:, 0]
    #acc = data[:, 1:513, :]
    #acc = np.swapaxes(acc, 0, 1)
    #acc = np.swapaxes(acc, 1, 2)
    #conv = ft.convolve_signal(acc)
    #conv_rank = np.log10(np.amax(conv, axis=1) - np.amin(conv, axis=1))
    #conv = ft.normalize_convolution(conv)
    # normalizes conv so its mean is 0 and its power is 1
    #f, dsp = scipy.signal.periodogram(conv, fs=1/0.174, scaling='density', axis=1)
    #dsp[:, 0] = conv_rank
    dsp=data[:, 1:]
    labels_artesanal = artesanal_classifier(dsp)
    #print(labels_artesanal)
    # examples = make_examples()
    with open('examples.pkl', 'rb') as f:
        examples = pkl.load(f)
    # examples = examples[:,np.array([0,1,3])]
    # examplest= examples.T
    # print(examplest)
    # aux = np.ones((3,1))*6
    # examplest=np.c_[aux,examplest]
    labels_hopfield = predict_Hopfield(dsp, 10, examples, 0.2)
    #print(labels_hopfield)
    with open('minibatch.pkl', 'rb') as f:
        minibatch = pkl.load(f)
    #dummy, minibatch = clustering(5)
    labels_cluster = np.zeros_like(labels_hopfield)
    labels_cluster[dsp[:,0]>5] = minibatch.predict(dsp[dsp[:, 0] > 5, 1:15])+1
    #print(labels_cluster)
    return true_labels, labels_artesanal, labels_hopfield, labels_cluster


'''
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
'''
