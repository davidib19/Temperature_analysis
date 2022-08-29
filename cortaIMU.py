import pandas as pd
import os

Crudos_path = os.path.join(os.path.split(os.getcwd())[0], "Datos")
Cortado_path = os.path.join(os.getcwd(), "Datos")

for campa in os.listdir(Crudos_path):
    for document in os.listdir(os.path.join(Crudos_path, campa)):
        if document[-3:] == "csv":
            docu = os.path.join(Crudos_path, campa, document)
            with open(docu, 'r') as f:
                lines = f.readlines()
                lines[0] = "date;timeGMT;accX;accY;accZ;tempIMU;girX;girY;girZ;tempIMU_C;Tsample;lat;lon\n"
            with open(docu, 'w') as f:
                f.writelines(lines)
            print(docu)
            df = pd.read_csv(os.path.join(docu), dtype={'timeGMT':str}, sep=';', usecols=['date', 'timeGMT',
                                                                                          'lat', 'lon',
                                                                                          'tempIMU_C'], skipinitialspace=True)
            df[df['lat'].notna()].to_csv(os.path.join(Cortado_path, campa,document))
