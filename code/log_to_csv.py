## Imports
import os
import pandas as pd
import numpy as np
import re

## Paths
data_log_path = "../../data"

M1_data = "ML_EMA_Scalper_V_0_01_00 EURUSD M1 20140101 20240101.log"
M5_data = "ML_EMA_Scalper_V_0_01_00 EURUSD M5 20140101 20240101.log"

## Abrimos el archivo conteniendo los datos
#Para ver el encoding, abrimos el archivo con el bloc de notas y abajo a la derecha aparece
log_file = M1_data
f1 = open(os.path.join(data_log_path, log_file), "r", encoding="utf_16_le")

## Recorremos el archivo guardando los datos en listas

# Variables intermedias
seq = np.zeros([100, 4])
idx = len(seq)
target = []

# Listas finales
x_data = []
y_data = []
order_data=[]
# Bucle rellenando las listas finales
for row in f1:
  if "based on order" in row:
    entry_order = re.search("#[0-9]*", row).group()
  if "Bar " in row:
    hour = float(re.search("[0-2][0-9]:[0-5][0-9]:[0-5][0-9] ", row).group()[0:2])
    date = re.search("[0-9]{4}.[0-1][0-9].[0-3][0-9]", row).group()
    date = float(re.sub("\.", "", date))
    price = float(re.search("Close Price: [0-9].[0-9]*", row).group()[12:])
    volume = int(re.search("Volume: [0-9]*", row).group()[7:])
    seq[idx-1] = [date, hour, price, volume]
    idx=idx-1
  if idx == 0:
    x_data.append(seq.copy()) # Si le pasas directamente secuencia, x_data se desordena
    y_data.append([entry_order, 0])
    idx= len(seq)
  if "take profit" in row:
    exit_order = re.search("triggered #[0-9]*", row).group()[10:]
    order_data.append([exit_order, 1])
print(len(x_data))
df = pd.DataFrame(y_data, columns=["order", "target"])
print(df)
df1 = pd.DataFrame(order_data, columns=["order", "target"])
print(df1)
f1.close()

df.loc[df["order"].isin(df1["order"]), "target"] = 1

#Calculate the total number of rows needed in the DataFrame
total_rows = len(x_data) * 100

# Create a list to hold the data
data_list = []

# Iterate over x_data_list and df_y to construct the data
for idx, (seq, row) in enumerate(zip(x_data, df.itertuples())):
    order = row.order
    target = row.target
    for seq_idx, el in enumerate(seq):
      date = re.sub(r"(\d{4})(\d{2})(\d{2})", r"\1-\2-\3", str(el[0]))[0:-2]
      data_list.append([date, el[1], el[2], el[3], order, target])

# Convert the list to a DataFrame
data = pd.DataFrame(data_list, columns=["date","hour", "price", "volume", "order", "target"])

# Save DataFrame to a CSV file
data_csv_path="../../data/"
if "M5" in log_file:
  data.to_csv(data_csv_path + 'M5_data_TFT.csv', index=True)
elif "M1" in log_file:
  data.to_csv(data_csv_path + 'M1_data_TFT.csv', index=True)