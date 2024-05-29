import MetaTrader5 as mt5
from MetaTrader5 import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

from keras import Model, saving
from keras.utils import to_categorical
from utils import *
import tensorflow as tf

data_path = "C:/Users/Jesús/Desktop/Proyectos/Brokptions/data/"
model_path= "../model/"

M1_data = "ML_EMA_Scalper_V_0_01_00 EURUSD M1 20140101 20240101.log"
M5_data = "ML_EMA_Scalper_V_0_01_00 EURUSD M5 20140101 20240101.log"

## Abrimos el archivo conteniendo los datos
#Para ver el encoding, abrimos el archivo con el bloc de notas y abajo a la derecha aparece
log_file = M5_data

if "M5" in log_file:
  data = pd.read_csv(data_path + 'M5_data_TFT.csv', index_col=0)
elif "M1" in log_file:
  data = pd.read_csv(data_path + 'M1_data_TFT.csv', index_col=0)

data["hour"] = hour_to_sinus(data)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.loc[:, "hour":"volume"])
data.loc[:, "hour":"volume"] = data_scaled

valid_boundary = "2023-01-01"
index = data['date']
train = data.loc[index < valid_boundary]
val = data.loc[(index >= valid_boundary)]


# We get arrays os shape (num_seq, 100, num_features) for x and (num_seq, 1) for targets
X_train = train[['hour', 'price', 'volume']].values.reshape(-1, 100, 3)
y_train = to_categorical(train["target"].values.reshape(-1, 100, 1)[:, 0, :])

X_val = val[['hour', 'price', 'volume']].values.reshape(-1, 100, 3)
y_val = to_categorical(val["target"].values.reshape(-1, 100, 1)[:, 0, :])

print("Train classes:", np.sum(y_train, axis=0), "Val classes:", np.sum(y_val, axis=0))

model_name = "best_model_no_normalization.keras"

LSTM = saving.load_model(model_path + model_name)

# Test the model
y_pred = LSTM.predict(X_val)

acc = accuracy_score(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1))
print("Accuracy on validation set: ", acc)

if mt5.initialize():
    print("Platform MT5 is on")
else:
    print("initialize() failed, error code =", mt5.last_error())

while mt5.terminal_info().connected:

    orders = mt5.orders_get(symbol="EURUSD") # Los símbolos activos se podrían leer de un file también
    if len(orders) > 0:
        print(orders.state)

