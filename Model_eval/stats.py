import os
import gc
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_log_error
from scipy.stats import skew, kurtosis
from statsmodels.tools.eval_measures import rmse

# ============================================================
# Configuration
window_size = 15
epochs = 1
number_of_runs = 3

CNN_layers = 3
num_heads = 4
filters = 238
kernel_size = 4
batch_size = 32

# ============================================================
# TensorFlow device configuration
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

# ============================================================
# Functions
def create_dataset(data, window_size):
    X, y = [], []
    for index in range(len(data) - window_size):
        X.append(data[index:index + window_size])
        y.append(data[index + window_size, 0])
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32))

def performance(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    rmse_value = rmse(y_true, y_pred)
    mae_value = mean_absolute_error(y_true, y_pred)
    mape_value = mean_absolute_percentage_error(y_true, y_pred)

    # MSLE requires non-negative values
    y_true_msle = np.clip(y_true, a_min=0, a_max=None)
    y_pred_msle = np.clip(y_pred, a_min=0, a_max=None)
    msle_value = mean_squared_log_error(y_true_msle, y_pred_msle)

    result = (
        f"{rmse_value:.2E} & "
        f"{mae_value:.2E} & "
        f"{mape_value:.2E} & "
        f"{msle_value:.2E} \\\\"
    )
    return result

def build_model(window_size, features, CNN_layers, filters, kernel_size, num_heads):
    inputs = layers.Input(shape=(window_size, features))
    x = inputs

    # CNN_layers now represents exactly the number of CNN layers
    for _ in range(CNN_layers):
        x = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation="relu",
            padding="causal",
        )(x)
    attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=32)(x, x)
    x = layers.Concatenate()([x, attention])
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mse"])
    return model

# ============================================================
# Load and prepare data
df = pd.read_csv("tucurui.csv", sep=";")
df.columns = [column.strip() for column in df.columns]
df["Data"] = pd.to_datetime(df["Data"], dayfirst=True)
df["UPH610010000"] = (df["UPH610010000"]
    .astype(str)
    .str.replace(",", ".", regex=False)
    .astype(float))
df["Natural Flow"] = (df["Natural Flow"]
    .astype(str)
    .str.replace(",", ".", regex=False)
    .astype(float))
df = (df.sort_values("Data")
    .reset_index(drop=True))
df["time_idx"] = df.index
df["group"] = "tucurui"
df = df.rename(columns={"Natural Flow": "y",
        "UPH610010000": "precipitation"})
data = df[["y"]].to_numpy(dtype=np.float32)
features = data.shape[1]

# ============================================================
# Chronological division and normalization
# This index represents the first original test target
raw_split = int(0.8 * len(data))

# Fit the scaler only with training observations
scaler = StandardScaler()
scaler.fit(data[:raw_split])

# Transform the entire dataset using training statistics
scaled_data = scaler.transform(data)

# Create windows after normalization
X, y = create_dataset(scaled_data, window_size)

# Convert original-data split into windowed-data split
split = raw_split - window_size
X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]
print(f"Training windows: {len(X_train)}")
print(f"Testing windows: {len(X_test)}")
print(f"Input shape: {X_train.shape}")

# ============================================================
# Store results from all runs
rmse_values = []
mae_values = []
mape_values = []
msle_values = []

# ============================================================
# Multiple experiments
for run in range(number_of_runs):
    print(f"\nRun {run + 1}/{number_of_runs}: " 
          f"seed = {run}")
    seed = run
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    # Clear the previous model from memory
    tf.keras.backend.clear_session()
    gc.collect()
    model = build_model(
        window_size=window_size,
        features=features,
        CNN_layers=CNN_layers,
        filters=filters,
        kernel_size=kernel_size,
        num_heads=num_heads)

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        shuffle=False,
        verbose=1)

    test_loss, test_mse = model.evaluate(
        X_test,
        y_test,
        verbose=0)

    print(f"Test MSE: {test_mse:.4f}")

    # ========================================================
    # Recursive forecasting
    # First test input window
    recursive_window = X_test[0].copy()
    recursive_predictions = []
    for forecast_step in range(window_size):
        model_input = recursive_window.reshape(1, window_size, features)
        prediction = model.predict(model_input, verbose=0)[0, 0]
        recursive_predictions.append(prediction)

        # Update the target variable with the prediction
        new_observation = recursive_window[-1].copy()
        new_observation[0] = prediction

        # In a multivariate case, the remaining features retain
        # their most recent observed values
        recursive_window = np.vstack(
            [recursive_window[1:],
             new_observation])
    recursive_predictions = np.asarray(
        recursive_predictions,
        dtype=np.float32)

    # ========================================================
    # Return predictions to the original scale
    if features == 1:
        y_pred = scaler.inverse_transform(recursive_predictions.reshape(-1, 1)).flatten()
    else:
        # Create an auxiliary array for inverse transformation
        inverse_array = np.zeros((window_size, features), dtype=np.float32)
        inverse_array[:, 0] = recursive_predictions
        y_pred = scaler.inverse_transform(
            inverse_array)[:, 0]

    # True values corresponding to the recursive horizon
    y_true = data[raw_split:raw_split + window_size, 0]
    if len(y_true) != window_size:
        raise ValueError(
            "The test set does not contain enough observations "
            "for the requested forecasting horizon.")

    # ========================================================
    # Calculate metrics
    current_rmse = rmse(y_true, y_pred)
    current_mae = mean_absolute_error(y_true, y_pred)
    current_mape = mean_absolute_percentage_error(y_true, y_pred)
    current_msle = mean_squared_log_error(
        np.clip(y_true, a_min=0, a_max=None),
        np.clip(y_pred, a_min=0, a_max=None))
    rmse_values.append(current_rmse)
    mae_values.append(current_mae)
    mape_values.append(current_mape)
    msle_values.append(current_msle)
    performance_metrics = performance(y_true, y_pred)
    print(f"Performance metrics: {performance_metrics}")

# ============================================================
# Save metrics

pd.DataFrame({"RMSE": rmse_values}).to_csv("rmse_v1.csv", index=False)
pd.DataFrame({"MAE": mae_values}).to_csv("mae_v1.csv", index=False)
pd.DataFrame({"MAPE": mape_values}).to_csv("mape_v1.csv", index=False)
pd.DataFrame({"MSLE": msle_values}).to_csv("msle_v1.csv", index=False)

# ============================================================
# Read saved results

rmse_array = pd.read_csv("rmse_v1.csv")["RMSE"].to_numpy()
mae_array = pd.read_csv("mae_v1.csv")["MAE"].to_numpy()
mape_array = pd.read_csv("mape_v1.csv")["MAPE"].to_numpy()
msle_array = pd.read_csv("msle_v1.csv")["MSLE"].to_numpy()

# ============================================================
# Statistical analysis
def compute_stats(values):
    values = np.asarray(values)
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    return {
        "Mean": np.mean(values),
        "Std": np.std(values, ddof=1),
        "Min": np.min(values),
        "Max": np.max(values),
        "Median": np.median(values),
        "Q1 (25\\%)": q1,
        "Q3 (75\\%)": q3,
        "Range": np.max(values) - np.min(values),
        "IQR": q3 - q1,
        "Skewness": skew(values),
        "Kurtosis": kurtosis(values),
    }

rmse_stats = compute_stats(rmse_array)
mae_stats = compute_stats(mae_array)
mape_stats = compute_stats(mape_array)
msle_stats = compute_stats(msle_array)

# ============================================================
# LaTeX table
latex_table = r"""
\begin{table}[!ht]
\centering
\caption{Statistical results over 100 runs.}
\begin{tabular}{lcccc}
\toprule
Statistic & RMSE & MAE & MAPE & MSLE \\
\midrule
"""

for statistic in rmse_stats.keys():
    latex_table += (
        f"{statistic} & "
        f"{rmse_stats[statistic]:.4e} & "
        f"{mae_stats[statistic]:.4e} & "
        f"{mape_stats[statistic]:.4e} & "
        f"{msle_stats[statistic]:.4e} \\\\\n"
    )

latex_table += r"""\bottomrule
\end{tabular}
\label{tab:extended_stats_eng}
\end{table}
"""
print(latex_table)

# ============================================================
# DataFrame containing all metrics
results_df = pd.DataFrame(
    {
        "RMSE": rmse_array,
        "MAE": mae_array,
        "MAPE": mape_array,
        "MSLE": msle_array,
    }
)

# ============================================================
# Boxplot
plt.figure(figsize=(5, 3), facecolor="white")
ax = plt.gca()
ax.set_facecolor("white")
plt.boxplot(
    [results_df[col] for col in results_df.columns],
    labels=results_df.columns,
    patch_artist=True,
    boxprops=dict(facecolor="white", color="black", linewidth=1),
    whiskerprops=dict(color="black", linestyle="-", linewidth=1),
    capprops=dict(color="black", linewidth=1),
    medianprops=dict(color="red", linewidth=1),
    flierprops=dict(marker="o", markersize=3, markerfacecolor="black", markeredgecolor="none")
)
plt.ylabel("Value", fontsize=10, labelpad=8)
plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.5)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig("sta1a.pdf", dpi=300, bbox_inches="tight", facecolor="white")
plt.show()

# Line plot
plt.figure(figsize=(5, 3))
for col in results_df.columns:
    plt.plot(results_df.index + 1, results_df[col], marker="o", markersize=3, label=col)
plt.xlabel("Run")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("sta2a.pdf", bbox_inches="tight")
plt.show()
