import torch
import os
device = torch.device("cuda:0")
print(f"Using device: {device}")

epochs = 100
window_size = 15

import pandas as pd
import numpy as np
from keras import layers, models
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import random
import time
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_log_error

seed = 1
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

def performance(y_true, y_pred, time_s):
    # Calculate metrics
    rmse_v = rmse(y_true, y_pred)
    mae_v = mean_absolute_error(y_true, y_pred)
    mape_v = mean_absolute_percentage_error(y_true, y_pred)
    msle_v = mean_squared_log_error(y_true, abs(y_pred))
    # RMSE & MAE & MAPE & MSLE & time
    result = (f'{rmse_v:.2E} & {mae_v:.2E} & {mape_v:.2E} & {msle_v:.2E} & {time_s:.2E} \\\\')
    return result

CNN_layers = 3
num_heads = 4
filters = 238
kernel_size = 4

inputs = layers.Input(shape=(window_size, features))
x = layers.Conv1D(filters, kernel_size, activation='relu', padding='causal')(inputs)
for _ in range(CNN_layers):
  x = layers.Conv1D(filters, kernel_size, activation='relu', padding='causal')(x)
attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=32)(x, x)
x = layers.Concatenate()([x, attention])
x = layers.GlobalAveragePooling1D()(x)
outputs = layers.Dense(1)(x)
model = models.Model(inputs, outputs)
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
start_time = time.time()
history = model.fit(X_train, y_train, epochs=epochs, batch_size=32,
                    validation_split=0.2, verbose=1)
end_time = time.time()
time_s = end_time - start_time
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Loss: {test_loss[1]:.4f}")
test_part = X_test[0].reshape(1, window_size, features)
test_predictions = model.predict(test_part)

# add test_predictions to the test_part
test_part = np.append(test_part, test_predictions)

# Adding window size to the predictions
for i in range(1, window_size):
    test_predictions = model.predict(test_part[-window_size:].reshape(1, window_size, features))
    test_part = np.append(test_part, test_predictions)
test_part = test_part[-window_size:].flatten()

# Calculate performance metrics
test_part = scaler.inverse_transform(test_part.reshape(-1, 1)).flatten()
y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
X_test, y_test = X[split:], y[split:]
y_true = y_test[:window_size]
y_pred = test_part

rmse_v.append(rmse(y_true, y_pred))
mae_v.append(mean_absolute_error(y_true, y_pred))
mape_v.append(mean_absolute_percentage_error(y_true, y_pred))
msle_v.append(mean_squared_log_error(y_true, abs(y_pred)))

# Calculate performance metrics
performance_metrics = performance(y_test[:window_size], test_part, time_s)
print(f"Performance metrics: {performance_metrics}")
