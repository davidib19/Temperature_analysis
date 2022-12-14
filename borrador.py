import Grafica
import datetime
import Finding_features as ft
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import matplotlib as mpl
import os
import ML
import scipy.signal
import LeerDatosExcel as Lee
import movimiento as mov
from sklearn import metrics
#plt.style.use("seaborn")

'''

mpl.rcParams.update(
    {
        "axes.titlesize": 12,
        "font.size": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.figsize": [6.1, 10],
        "figure.autolayout": True,
        "font.family": "serif",
        "font.sans-serif": ["Computer modern"],
        "savefig.format": 'pdf',
        "savefig.bbox": 'tight',
        "text.usetex": True
    }
)

'''

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
examples=np.zeros((14,4))
for i in np.arange(4):
    examples[i+1,i]=1.0
print(examples[:,1])
print(ML.Hopfield(examples,6,examples[:,1]))

df = mov.ReadIMUData(os.path.join(mov.tortugometro_path,'01_2021','T30.csv'))
df['dia'] = df['datetime'].dt.date
groups = df.groupby('dia')
predicciones=[]
for name, group in groups:
    acc = group[['accX', 'accY', 'accZ']].to_numpy()
    prediccion = []
    for i in np.arange(int(np.trunc(acc.shape[0]/512))):
        prediccion.append(ML.predict_Hopfield(acc[i*512:(i+1)*512,:],3,examples,0.1))
    predicciones.append(prediccion)
#print(predicciones)
mov.color_curve(df,predicciones,512)

'''
'''
#Grafica.plotacc_con_etiqueta('11_2021','T10.csv')
with open(os.path.join(os.getcwd(), 'aceleraciones_etiquetadas', 'mi_database.pkl'),
          'rb') as handle:
    db = pkl.load(handle)
tr,art,hop,clust=ML.test_methods_labeled(db)
matrix=np.stack((tr,art,hop,clust)).T
matrix =matrix.astype(int)
confusion_tree=metrics.confusion_matrix(matrix[:,0],matrix[:,1])
confusion_tree=np.flip(confusion_tree,0)
confusion_tree[:,:-1]=np.flip(confusion_tree[:,:-1],1)
#confusion_tree=np.flip(confusion_tree,1)
confusion_tree=confusion_tree[1:,:]
confusion_tree=confusion_tree/confusion_tree.astype(np.float).sum(axis=1)[:,None]
cm_tree_display = metrics.ConfusionMatrixDisplay(confusion_tree,display_labels=['copula','come','hace nido', 'camina'])
cm_tree_display.plot()
plt.grid(False)
plt.xlabel('Predicción')
plt.ylabel('Verdadero')
plt.savefig('C:/Users/bicho/OneDrive/Documentos/Balseiro/maestria/Tesis/figuras/maestria/confusion_tree.pdf')
plt.show()
confusion_tree=metrics.confusion_matrix(matrix[:,0],matrix[:,2])
confusion_tree=np.flip(confusion_tree,0)
confusion_tree[:,:-1]=np.flip(confusion_tree[:,:-1],1)
#confusion_tree=np.flip(confusion_tree,1)
confusion_tree=confusion_tree[1:,:]
confusion_tree=confusion_tree/confusion_tree.astype(np.float).sum(axis=1)[:,None]
cm_tree_display = metrics.ConfusionMatrixDisplay(confusion_tree,display_labels=['copula','come','hace nido', 'camina'])
cm_tree_display.plot()
plt.grid(False)
plt.xlabel('Predicción')
plt.ylabel('Verdadero')
plt.savefig('C:/Users/bicho/OneDrive/Documentos/Balseiro/maestria/Tesis/figuras/maestria/confusion_hop.pdf')
plt.show()
fig,axs=plt.subplots(4,sharex='all',figsize=(6.1,6.1))
actividades = ['copula','come','hace nido', 'camina']
for i in np.arange(4)+1:
    axs[i-1].hist(matrix[matrix[:,0]==i,3],bins=np.arange(1,7)-0.5,rwidth=0.25)
    axs[i-1].set_ylabel(actividades[-1*i])
    #axs[i-1].legend()
axs[3].set_xlabel('Cluster')
plt.savefig('C:/Users/bicho/OneDrive/Documentos/Balseiro/maestria/Tesis/figuras/maestria/confusion_kmeans.pdf')
plt.show()
print(matrix)

'''
'''
Grafica.plotacc('01_2021','T30.csv')
df = mov.ReadIMUData(os.path.join(mov.tortugometro_path,'01_2021','T30.csv'))
df = df[(df['datetime']>datetime.datetime(year=2021, month=1, day=12, hour=11, minute=27,second=0))&(df['datetime']<datetime.datetime(year=2021, month=1, day=12, hour=11, minute=52,second=0))]
acc = df[['accX', 'accY', 'accZ']].to_numpy()
acc = acc[512*2:3*512,:]
def convolucion_circular(acc):
    convolucion = np.zeros((acc.shape[0]))
    for t in np.arange(acc.shape[0]):
        convolucion[t] = np.sum(acc[:,0] * np.roll(acc[:,1], t))
    return convolucion
Grafica.dsp_accvsconv(acc)
'''
'''
fig, axs = plt.subplots(2, 2, sharex='col', sharey=False)
circular = convolucion_circular(acc)
convfft = np.real(np.fft.ifft(np.multiply(np.fft.fft(acc[:, 0], axis=0), np.fft.fft(acc[:, 1], axis=0)), axis=0))
axs[0,0].plot(circular)
axs[1,0].plot(convfft)
f, dspcirc = scipy.signal.periodogram(circular, fs =1/0.174,axis=0)
axs[0,1].plot(f, dspcirc)
f, dspfft = scipy.signal.periodogram(convfft,fs =1/0.174,axis=0)
axs[1,1].plot(f, dspfft)
print(dspfft[:15])
print(dspcirc[:15])
plt.show()
'''
df= mov.ReadIMUData(os.path.join(mov.tortugometro_path,'01_2021','T30.csv'))
df = df[(df['datetime']>datetime.datetime(year=2021, month=1, day=13, hour=12, minute=40,second=0))&(df['datetime']<datetime.datetime(year=2021, month=1, day=13, hour=13, minute=10,second=0))]
Grafica.plotsegment(df)