### IMPORTS ###

import tensorflow as tf
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import numpy as np

from keras import Model, saving
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, root_mean_squared_error

from utils import *


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

## Data loading ##
## Paths
data_path = "../../data/"
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

# We load scaler and price_scaler
with open(model_path + 'scaler.pkl', 'rb') as file:
  scaler = pickle.load(file)

with open(model_path + 'price_scaler.pkl', 'rb') as file:
  price_scaler = pickle.load(file)

orig_price = data[["date","price"]]
data_scaled = scaler.transform(data.loc[:, "hour":"volume"])
data.loc[:, "hour":"volume"] = data_scaled

valid_boundary = "2023-01-01"
index = data['date']
val = data.loc[(index >= valid_boundary)]
orig_price_val = orig_price.loc[(index >= valid_boundary)]

n_steps_ahead = 1
n_input_steps = 100 - n_steps_ahead

# Validation data definition
X_val = val[['hour', 'price', 'volume']].values.reshape(-1, 100, 3)[:, :n_input_steps, :]
y_val = orig_price_val['price'].values.reshape(-1, 100, 1)[:, n_input_steps:]

model_name = "best_model_99_attention_reg.keras"

LSTM = saving.load_model(model_path + model_name)

# Test the model
y_pred = LSTM.predict(X_val)

y_pred = price_scaler.inverse_transform(y_pred)

# Flatten the arrays
y_pred_flat = y_pred.flatten()
y_val_flat = y_val.flatten()

# Plot the predicted and true values
plt.figure(figsize=(15, 6))
plt.plot(y_val_flat, label='True Values')
plt.plot(y_pred_flat, label='Predicted Values', alpha=0.7)
plt.xlabel('Timesteps')
plt.ylabel('Values')
plt.title('Predicted vs True Values' + model_name)
plt.legend()
plt.show()