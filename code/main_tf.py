### IMPORTS ###

import tensorflow as tf
import keras
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam

from keras import backend as K
from build_tf_model import transformer_encoder, classification_build_model

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from utils import *

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

## Data loading ##
## Paths
data_path = "../data/"
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
price_scaler = MinMaxScaler()
scaler = MinMaxScaler()

price_scaled = price_scaler.fit_transform(data[["price"]])
data_scaled = scaler.fit_transform(data.loc[:, "hour":"volume"])
data.loc[:, "hour":"volume"] = data_scaled

# # Save data_scaler and price_scaler to a file
# with open(model_path + 'scaler.pkl', 'wb') as file:
#     pickle.dump(scaler, file)

# with open(model_path + 'price_scaler.pkl', 'wb') as file:
#     pickle.dump(price_scaler, file)

valid_boundary = "2023-01-01"
index = data['date']
train = data.loc[index < valid_boundary]
val = data.loc[(index >= valid_boundary)]


# We get arrays of shape (num_seq, 100, num_features) for x and (num_seq, 1) for targets
X_train = train[['hour', 'price', 'volume']].values.reshape(-1, 100, 3)
#X_train = train.loc[:, "hour":"volume"].values.reshape(-1, 100, 3)
y_train = to_categorical(train["target"].values.reshape(-1, 100, 1)[:, 0, :])

# We balance the labels so the model doesn't lear to just predict the majority class
class_weights = compute_class_weight('balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))

# Convert the class weights to a dictionary
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

# Validation data definition
X_val = val[['hour', 'price', 'volume']].values.reshape(-1, 100, 3)
#X_val = val['price'].values.reshape(-1, 100, 1)
y_val = to_categorical(val["target"].values.reshape(-1, 100, 1)[:, 0, :])

model = classification_build_model(
    input_shape = X_train.shape[1:],
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25,
)

print(model.summary())

#Compile the model
model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy')

# Call back to stop training upon learning block
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
# Callback to save the model periodically
model_checkpoint = ModelCheckpoint(model_path + 'tf_class_best_model.keras', save_best_only=True, monitor='val_loss')

# Callback to reduce learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

# Train the model
history = model.fit(X_train, y_train, validation_data= (X_val, y_val), epochs=50, batch_size=64, class_weight=class_weights_dict, callbacks=[early_stopping, model_checkpoint, reduce_lr])

# Test the model
y_pred = model.predict(X_val)

acc = accuracy_score(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1))
print("Accuracy on validation set: ", acc)