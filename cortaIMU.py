import pandas as pd
import os


tortugometro_path = os.path.join(os.getcwd(), "Datos_tortugometro_crudo")


for document in os.listdir(tortugometro_path):
    df = pd.read_csv(os.path.join(tortugometro_path, document), dtype={'timeGMT':str}, sep=';', usecols=['date', 'timeGMT',
                                                                                                      'lat', 'lon',
                                                                                                      'tempIMU_C'])
    df[df['lat'].notna()].to_csv(os.path.join(os.getcwd(), "Datos_tortugometro", document))
