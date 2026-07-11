pip -q install bayesian-optimization keras-tuner scikit-learn

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import gc
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from keras import layers, models
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization
from scipy.interpolate import griddata

window_size = 15
epochs = 50
batch_size = 32
seed = 1

os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus[0]}")
    except RuntimeError as error:
        print(f"GPU configuration error: {error}")
else:
    print("Using CPU")

def create_dataset(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size, 0])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

def build_model(CNN_layers, num_heads, filters, kernel_size):
    inputs = layers.Input(shape=(window_size, features))
    x = inputs
    for _ in range(CNN_layers):
        x = layers.Conv1D(filters, kernel_size, activation="relu", padding="causal")(x)
    attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=32)(x, x)
    x = layers.Concatenate()([x, attention])
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mse"])
    return model

df = pd.read_csv("tucurui.csv", sep=";")
df.columns = [col.strip() for col in df.columns]
df["Data"] = pd.to_datetime(df["Data"], dayfirst=True)
df["UPH610010000"] = df["UPH610010000"].astype(str).str.replace(",", ".", regex=False).astype(float)
df["Natural Flow"] = df["Natural Flow"].astype(str).str.replace(",", ".", regex=False).astype(float)
df = df.sort_values("Data").reset_index(drop=True)
df["time_idx"] = df.index
df["group"] = "tucurui"
df = df.rename(columns={"Natural Flow": "y", "UPH610010000": "precipitation"})

data = df[["y"]].to_numpy(dtype=np.float32)
# data = df[["y", "precipitation"]].to_numpy(dtype=np.float32)
features = data.shape[1]

raw_split = int(0.8 * len(data))
scaler = StandardScaler()
scaler.fit(data[:raw_split])
scaled_data = scaler.transform(data)

X, y = create_dataset(scaled_data, window_size)
split = raw_split - window_size
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

print(f"Training windows: {len(X_train)}")
print(f"Testing windows: {len(X_test)}")
print(f"Input shape: {X_train.shape}")

evaluated_configs = {}
trial_number = 0

def objective_function(CNN_layers, num_heads, filters, kernel_size):
    global trial_number
    CNN_layers = int(round(CNN_layers))
    num_heads = int(round(num_heads))
    filters = int(round(filters))
    kernel_size = int(round(kernel_size))
    config = (CNN_layers, num_heads, filters, kernel_size)

    if config in evaluated_configs:
        return evaluated_configs[config]
    trial_number += 1
    print(f"\nTrial {trial_number}: CNN_layers={CNN_layers}, num_heads={num_heads}, filters={filters}, kernel_size={kernel_size}")

    tf.keras.backend.clear_session()
    gc.collect()
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    model = build_model(CNN_layers, num_heads, filters, kernel_size)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_split=0.2, shuffle=False, verbose=0)
    val_rmse = np.sqrt(min(history.history["val_mse"]))
    target = -val_rmse
    evaluated_configs[config] = target
    print(f"Validation RMSE: {val_rmse:.6f}")

    del model
    tf.keras.backend.clear_session()
    gc.collect()
    return target

pbounds = {
    "CNN_layers": (1, 12),
    "num_heads": (2, 5),
    "filters": (16, 256),
    "kernel_size": (2, 5)}

optimizer = BayesianOptimization(f=objective_function, pbounds=pbounds, random_state=seed, verbose=2)
optimizer.maximize(init_points=5, n_iter=50)

best_params = {
    "CNN_layers": int(round(optimizer.max["params"]["CNN_layers"])),
    "num_heads": int(round(optimizer.max["params"]["num_heads"])),
    "filters": int(round(optimizer.max["params"]["filters"])),
    "kernel_size": int(round(optimizer.max["params"]["kernel_size"]))}

print(f"\nBest validation RMSE: {-optimizer.max['target']:.6f}")
print("Best parameters:", best_params)

pd.DataFrame.from_dict(best_params, orient="index", columns=["Value"]).to_csv(
    "best_params.csv", index_label="Parameter")

results = []
for res in optimizer.res:
    results.append({
        "CNN_layers": int(round(res["params"]["CNN_layers"])),
        "num_heads": int(round(res["params"]["num_heads"])),
        "filters": int(round(res["params"]["filters"])),
        "kernel_size": int(round(res["params"]["kernel_size"])),
        "rmse": -res["target"]
    })

results_df = pd.DataFrame(results)
results_df.to_csv("bayesian_optimization_results.csv", index=False)

# Convergence plot
results_df["best_rmse"] = results_df["rmse"].cummin()
plt.figure(figsize=(5, 3))
plt.plot(results_df.index + 1, results_df["best_rmse"], label="Best RMSE")
plt.scatter(results_df.index + 1, results_df["rmse"], alpha=0.3, label="Trials")
plt.xlabel("Iteration")
plt.ylabel("Validation RMSE")
plt.legend(loc="upper right")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("Bayesian-Optimization-Convergence.pdf", bbox_inches="tight")
plt.show()

def contour_plot(results_df, x_param, y_param, filename):
    grouped = results_df.groupby([x_param, y_param], as_index=False)["rmse"].min()
    x = grouped[x_param].to_numpy(dtype=float)
    y = grouped[y_param].to_numpy(dtype=float)
    z = grouped["rmse"].to_numpy(dtype=float)

    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    method = "cubic" if len(grouped) >= 4 else "linear"
    zi = griddata((x, y), z, (xi, yi), method=method)

    if zi is None or np.isnan(zi).all():
        zi = griddata((x, y), z, (xi, yi), method="nearest")
    else:
        nearest = griddata((x, y), z, (xi, yi), method="nearest")
        zi = np.where(np.isnan(zi), nearest, zi)

    plt.figure(figsize=(5, 3.5))
    contour = plt.contour(xi, yi, zi, levels=20, colors="black", linewidths=0.8)
    plt.clabel(contour, inline=True, fontsize=8, fmt="%.3f")
    plt.scatter(x, y, c="black", s=50, edgecolor="black", linewidth=0.5, label="Sampled points")
    plt.xlabel(x_param.replace("_", " ").title())
    plt.ylabel(y_param.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.show()

contour_plot(results_df, "CNN_layers", "num_heads", "Hyperparameter-Contour-Plot1.pdf")
contour_plot(results_df, "filters", "kernel_size", "Hyperparameter-Contour-Plot2.pdf")
