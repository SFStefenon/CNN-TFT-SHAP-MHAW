import os
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error

# Reproducibility
seed = 1
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Check whether TensorFlow detects a GPU
gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow GPUs:", gpus)

window_size = 15
epochs = 100
CNN_layers = 3
num_heads = 4
filters = 238
kernel_size = 4

def performance(y_true, y_pred, time_s):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    rmse_v = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_v = mean_absolute_error(y_true, y_pred)
    mape_v = mean_absolute_percentage_error(y_true, y_pred)

    # MSLE requires non-negative values
    y_true_msle = np.maximum(y_true, 0)
    y_pred_msle = np.maximum(y_pred, 0)
    msle_v = mean_squared_log_error(y_true_msle, y_pred_msle)

    return (
        f"{rmse_v:.2E} & "
        f"{mae_v:.2E} & "
        f"{mape_v:.2E} & "
        f"{msle_v:.2E} & "
        f"{time_s:.2E} \\\\"
    )

def create_dataset(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size, 0])
    return np.asarray(X), np.asarray(y)

# Load data
df = pd.read_csv("tucurui.csv", sep=";")
df.columns = [column.strip() for column in df.columns]
df["Data"] = pd.to_datetime(df["Data"], dayfirst=True)
df["UPH610010000"] = (
    df["UPH610010000"]
    .astype(str)
    .str.replace(",", ".", regex=False)
    .astype(float)
)
df["Natural Flow"] = (
    df["Natural Flow"]
    .astype(str)
    .str.replace(",", ".", regex=False)
    .astype(float)
)

df = df.sort_values("Data").reset_index(drop=True)
df = df.rename(
    columns={
        "Natural Flow": "y",
        "UPH610010000": "precipitation",
    }
)

# Univariate forecasting
data = df[["y"]].to_numpy(dtype=np.float32)
features = data.shape[1]

# Time-ordered split performed before window creation
raw_split = int(0.8 * len(data))

train_data = data[:raw_split]
test_data = data[raw_split - window_size:]

# Fit scaler using training data only
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

X_train, y_train = create_dataset(train_scaled, window_size)
X_test, y_test = create_dataset(test_scaled, window_size)

# Build model
inputs = layers.Input(shape=(window_size, features))
x = inputs

# CNN_layers now represents the actual number of convolutional layers
for _ in range(CNN_layers):
    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        activation="relu",
        padding="causal",
    )(x)

attention = layers.MultiHeadAttention(
    num_heads=num_heads,
    key_dim=32,
)(x, x)

x = layers.Concatenate()([x, attention])
x = layers.GlobalAveragePooling1D()(x)
outputs = layers.Dense(1)(x)

model = models.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="adam", loss="mse", metrics=["mse"])

# Training time
training_start = time.perf_counter()

history = model.fit(
    X_train,
    y_train,
    epochs=epochs,
    batch_size=32,
    validation_split=0.2,
    shuffle=False,
    verbose=1,
)

training_time = time.perf_counter() - training_start

# Evaluate the complete test set
test_loss, test_mse = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MSE: {test_mse:.4f}")

# Direct one-step predictions for every test window
prediction_start = time.perf_counter()
y_pred_scaled = model.predict(X_test, verbose=0).flatten()
prediction_time = time.perf_counter() - prediction_start

# Convert predictions and targets back to the original scale
y_pred = scaler.inverse_transform(
    y_pred_scaled.reshape(-1, 1)
).flatten()

y_true = scaler.inverse_transform(
    y_test.reshape(-1, 1)
).flatten()

total_time = training_time + prediction_time

performance_metrics = performance(
    y_true=y_true,
    y_pred=y_pred,
    time_s=total_time,
)

print(f"Performance metrics: {performance_metrics}")
