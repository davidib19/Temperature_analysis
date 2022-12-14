import pandas as pd
import datetime
import os

data_path = os.path.join(os.getcwd(), "Datos")
etiqueta_path = os.path.join(os.getcwd(), "Etiquetas")
tortugometro_path = os.path.join(os.path.split(os.getcwd())[0], "Datos")


def toDatetime(string_datetime):
    """Transforma la fecha y hora del formato que viene en el archivo original
     a un datetime.datetime. es una funcion auxiliar"""
    if isinstance(string_datetime, str):
        if string_datetime != "":
            lista = string_datetime.split(" ")
            if len(lista[0].split('/')[2]) == 4:
                string_datetime = datetime.datetime.strptime(string_datetime, '%m/%d/%Y %I:%M:%S %p')
            else:
                string_datetime = datetime.datetime.strptime(string_datetime, '%m/%d/%y %I:%M:%S %p')
    else:
        string_datetime = string_datetime.replace(month=string_datetime.day, day=string_datetime.month)
    return string_datetime


def ReadData(path):
    """Lee los datos del iButton del excel como viene e interpreta correctamente la fecha y la hora
    y la guarda en formato datetime.datetime"""
    df = pd.read_excel(path, converters={'Date Time, GMT-03:00': toDatetime, 'Date Time, GMT-03:00.1': toDatetime,
                                         'Date Time, GMT-03:00.2': toDatetime, 'Date Time, GMT-03:00.3': toDatetime,
                                         'Date Time, GMT-03:00.4': toDatetime, 'Date Time, GMT-03:00.5': toDatetime,
                                         'Date Time, GMT-03:00.6': toDatetime, 'Date/Time': toDatetime,
                                         'Date/Time.1': toDatetime, 'Date/Time.2': toDatetime, 'Date/Time.3': toDatetime,
                                         'Date/Time.4': toDatetime, 'Date/Time.5': toDatetime, 'Date/Time.6': toDatetime})
    df.rename(columns={'Date Time, GMT-03:00': 'datetime', 'Date Time, GMT-03:00.1': 'datetime.1',
                       'Date Time, GMT-03:00.2': 'datetime.2', 'Date Time, GMT-03:00.3': 'datetime.3',
                       'Date Time, GMT-03:00.4': 'datetime.4', 'Date Time, GMT-03:00.5': 'datetime.5',
                       'Date Time, GMT-03:00.6': 'datetime.6'}, inplace=True)
    df.rename(columns={'Date/Time': 'datetime',
                       'Date/Time.1': 'datetime.1', 'Date/Time.2': 'datetime.2', 'Date/Time.3': 'datetime.3',
                       'Date/Time.4': 'datetime.4', 'Date/Time.5': 'datetime.5', 'Date/Time.6': 'datetime.6'}, inplace=True)
    return df


def ReadIMUData(path):
    """Lee los datos del tortugómetro con el formato que vienen y los guarda en un pandas dataframe con una columna
    en tipo datetime.datetime transformada a hora de Argentina."""
    df = pd.read_csv(path, parse_dates=['date'], converters={'timeGMT': formatIMU}, sep=',', usecols=['date', 'timeGMT',
                                                                                                      'lat', 'lon',
                                                                                                      'tempIMU_C'])
    df['datetime'] = df.apply(
        lambda r: datetime.datetime.combine(r['date'], r['timeGMT']) - datetime.timedelta(hours=3), 1)
    return df.drop(['date', 'timeGMT'], axis=1)


def ReadIMUDataO(path, cols=['date', 'timeGMT', 'accX', 'accY', 'accZ', 'tempIMU_C', 'lat',  'lon']):
    """Lee los datos del tortugómetro con el formato que vienen y los guarda en un pandas dataframe con una columna
    en tipo datetime.datetime transformada a hora de Argentina."""
    df = pd.read_csv(path, parse_dates=['date'], converters={'timeGMT': formatIMU}, sep=';', usecols=cols)
    df['datetime'] = df.apply(
        lambda r: datetime.datetime.combine(r['date'], r['timeGMT']) - datetime.timedelta(hours=3), 1)
    return df.drop(['date', 'timeGMT'], axis=1)



def ReadTags(string):
    """Lee el archivo de tags y lo guarda en un dataframe con las columnas datetime y tag"""
    df = pd.read_csv(string, sep=';', header=None, names=['date', 'time','tag','observation'])
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time
    return df


def ReadMyTags():
    """Lee el archivo de my tags que contienen hora de inicio y final
    y lo guarda en un dataframe con las columnas datetime y tag"""
    df = pd.read_csv(os.path.join(os.getcwd(), "aceleraciones_etiquetadas", "mis_etiquetas.csv"), sep=';')
    df['inicio']=pd.to_datetime(df['fecha']+df['hora_inicio'], format='%d/%m/%Y%H:%M')
    df['fin']=pd.to_datetime(df['fecha']+df['hora_final'], format='%d/%m/%Y%H:%M')
    df=df.drop(['fecha','hora_inicio','hora_final'],axis=1)
    return df


def formatIMU(stringhour):
    """Transforma la hora del formato que viene en el archivo de tortugometro a un datetime.time es una funcion auxiliar
    usada en las otras"""
    microsecond = int(stringhour[-3:]) * 1000
    second = int(stringhour[-6:-4])
    minute = int(stringhour[-8:-6])
    hour = int(stringhour[:-8])
    if second > 59:
        second -= 60
        minute += 1
    if minute > 59:
        minute -= 60
        hour += 1
    return datetime.time(microsecond=microsecond, second=second, minute=minute, hour=hour)


def toTimestamp(d):
  return datetime.datetime.timestamp(d)

