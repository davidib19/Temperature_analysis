import pandas as pd
import datetime
import os

iButton_path = os.path.join(os.getcwd(), "Datos_iButton")
tortugometro_path = os.path.join(os.getcwd(), "Datos_tortugometro")


def toDatetime(string_datetime):
    if isinstance(string_datetime, str):
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
    df = pd.read_excel(path, converters={'Date Time, GMT-03:00': toDatetime})
    df.rename(columns={'Date Time, GMT-03:00': 'datetime'}, inplace=True)
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


def formatIMU(stringhour):
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


#for document in os.listdir(tortugometro_path):
#    print(ReadIMUData(os.path.join(tortugometro_path, document)))
#    grafica_temp(ReadData(os.path.join(DATA_PATH, document)))
