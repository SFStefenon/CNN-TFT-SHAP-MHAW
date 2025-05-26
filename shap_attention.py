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
from keras.metrics import MeanAbsoluteError
import torch
from scipy.ndimage import gaussian_filter1d



seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def performance(y_true, y_pred):
    # Calculate metrics
    rmse_v = rmse(y_true, y_pred)
    mae_v = mean_absolute_error(y_true, y_pred)
    mape_v = mean_absolute_percentage_error(y_true, y_pred)
    msle_v = mean_squared_log_error(y_true, abs(y_pred))
    # r2_v = abs(r2_score(y_true, y_pred))
    # RMSE & MAE & MAPE & MSLE & R2 & time
    result = (f'{rmse_v:.2E} & {mae_v:.2E} & {mape_v:.2E} & {msle_v:.2E} \\\\')  # & {r2_v:.2E}
    return result

df = pd.read_csv("tucurui.csv", sep=";")
df.columns = [col.strip() for col in df.columns]
df["Data"] = pd.to_datetime(df["Data"], dayfirst=True)
df["UPH610010000"] = df["UPH610010000"].str.replace(",", ".").astype(float)
df["Natural Flow"] = df["Natural Flow"].str.replace(",", ".").astype(float)

df = df.sort_values("Data").reset_index(drop=True)
df["time_idx"] = df.index
df["group"] = "tucurui"
df = df.rename(columns={"Natural Flow": "y", "UPH610010000": "precipitation"})
data = df[["y", "precipitation"]]


data = df[["y"]]
# convert y and precipitation to numpy arrays
data = data.to_numpy()
features = data.shape[1]


# create_dataset function
def create_dataset(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size, 0])
    return np.array(X), np.array(y)


window_size = 30  # Use 30 previous steps to predict next step
X, y = create_dataset(data, window_size)

# Split into train/test sets
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, features)).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, features)).reshape(X_test.shape)
y_train = scaler.transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
y_test = scaler.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

# Reshape inputs for Conv1D (samples, timesteps, features)
X_train = X_train.reshape((-1, window_size, features))
X_test = X_test.reshape((-1, window_size, features))

def build_model_with_attention(window_size, features=1):
    inputs = layers.Input(shape=(window_size, features), name='input')

    x = layers.Conv1D(64, 3, activation='relu', padding='causal')(inputs)
    x = layers.Conv1D(64, 3, activation='relu', padding='causal')(x)

    mha = layers.MultiHeadAttention(num_heads=2, key_dim=32, name='mha')
    att_output, att_scores = mha(x, x, return_attention_scores=True)

    x = layers.Concatenate()([x, att_output])
    x = layers.GlobalAveragePooling1D()(x)
    forecast = layers.Dense(1, name='forecast')(x)

    # Treinas com forecast
    model = models.Model(inputs=inputs, outputs=forecast)
    # Guardas att_scores para usar depois
    return model, inputs, att_scores

# Train the model
model, inputs, att_scores = build_model_with_attention(window_size=30)
model.load_weights("cnn_tft_model.h5")  # carrega apenas pesos
attention_model = models.Model(inputs=inputs, outputs=att_scores)

model.compile(optimizer='adam', loss='mae', metrics=[MeanAbsoluteError()])

sample = X_test[0:1]
attention_scores = attention_model.predict(sample)


mean_attention = np.mean(attention_scores, axis=1).squeeze()


temporal_weights = mean_attention.sum(axis=0)

lags = [f"t-{i}" for i in range(window_size, 0, -1)]  # ['t-30', 't-29', ..., 't-1']
plt.figure(figsize=(10, 4))
plt.bar(lags, temporal_weights)
plt.title("Mean Attention Weights")
plt.xlabel("Time step (lag)")
plt.ylabel("Attention weight")
plt.xticks(ticks=np.array(list(np.arange(0, window_size, 3)) + [window_size-1]), rotation=0, ha='center')
plt.grid(True)
plt.show()

import matplotlib.cm as cm

colors = cm.viridis(np.linspace(0, 1, window_size))

plt.figure(figsize=(12, 4))
bars = plt.bar(lags, temporal_weights, color=colors)
plt.title("Mean Attention Weights", fontsize=14)
plt.xlabel("Time Step (Lag)", fontsize=12)
plt.ylabel("Attention Weight", fontsize=12)
plt.xticks(
    ticks=np.array(list(np.arange(0, window_size, 3)) + [window_size - 1]),
    rotation=0,
    ha='center',
    fontsize=9
)
plt.yticks(fontsize=9)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

colors = cm.plasma(np.linspace(0, 1, window_size))
# plt.figure(figsize=(12, 4))
plt.figure(figsize=(5, 3))
bars = plt.bar(lags, temporal_weights, color=colors)


# plt.title("Mean Attention Weights", fontsize=14)
plt.xlabel("Time Step (Lag)", fontsize=12)
plt.ylabel("Attention Weight", fontsize=12)
plt.xticks(
    ticks=np.array(list(np.arange(0, window_size, 3)) + [window_size - 1]),
    rotation=0,
    ha='center',
    fontsize=9
)
plt.yticks(fontsize=9)
#plt.axhline(np.mean(temporal_weights), color='gray', linestyle='--', linewidth=1, label='Mean Attention')
#plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
# plt.show()
plt.savefig("mean_attention_weights.png", dpi=300)

import shap


background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
explainer = shap.Explainer(model, background)


test_samples = X_test[:4]


explainer = shap.GradientExplainer(model, background)
shap_values = explainer.shap_values(test_samples)



for i in range(len(test_samples)):
    plt.figure(figsize=(10, 3))
    plt.bar(np.arange(window_size), shap_values[i].flatten())
    plt.title(f"SHAP values for sample {i}")
    plt.xlabel("Time Step (Lag)")
    plt.ylabel("SHAP Value")
    plt.grid(True)
    plt.show()

import matplotlib.cm as cm
import matplotlib.colors as mcolors

for i in range(len(test_samples)):
    shap_vec = shap_values[i].flatten()


    cmap = cm.seismic  # ou 'coolwarm'
    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=shap_vec.min(), vmax=shap_vec.max())
    colors = cmap(norm(shap_vec))

    # plt.figure(figsize=(12, 4))
    plt.figure(figsize=(5, 3))
    bars = plt.bar(np.arange(window_size), shap_vec, color=colors)

    # plt.title(f"SHAP Values for Sample {i}", fontsize=14)
    plt.xlabel("Time Step (Lag)", fontsize=12)
    plt.ylabel("SHAP Value", fontsize=12)


    lag_labels = [f"t-{i}" for i in range(window_size, 0, -1)]
    plt.xticks(
        ticks=np.array(list(np.arange(0, window_size, 3)) + [window_size - 1]),
        labels=[lag_labels[j] for j in np.array(list(np.arange(0, window_size, 3)) + [window_size - 1])],
        rotation=0,
        ha='center',
        fontsize=9
    )
    plt.yticks(fontsize=9)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)

    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"shap_sample_{i}.png", dpi=300)

    break

shap_vector = shap_values[3].flatten()


third = window_size // 3
initial_segment = shap_vector[:third]
middle_segment = shap_vector[third:2*third]
final_segment = shap_vector[2*third:]


impact_initial = np.sum(np.abs(initial_segment))
impact_middle = np.sum(np.abs(middle_segment))
impact_final = np.sum(np.abs(final_segment))


print(f"Initial Impact  (lags 0–{third-1}):   {impact_initial:.4f}")
print(f"Middle Impact  (lags {third}–{2*third-1}): {impact_middle:.4f}")
print(f"Final Impact    (lags {2*third}–{window_size-1}): {impact_final:.4f}")

total_impact = impact_initial + impact_middle + impact_final
print(f"Initial: {impact_initial / total_impact:.2%}")
print(f"Central: {impact_middle / total_impact:.2%}")
print(f"Final:   {impact_final / total_impact:.2%}")

plt.imshow(shap_values[0], aspect='auto', cmap='seismic')
plt.colorbar()
plt.title("SHAP value heatmap (samples vs lags)")
plt.xlabel("Time step")
plt.ylabel("Sample")
plt.show()


shap_vector = shap_values[0].flatten()
attention_vector = temporal_weights
combined_map = shap_vector * attention_vector
smoothed_map = gaussian_filter1d(combined_map, sigma=2)

plt.figure(figsize=(10, 4))
plt.plot(np.arange(window_size), smoothed_map, label='SHAP × Attention (smoothed)', color='purple')
plt.plot(np.arange(window_size), combined_map, label='Original (raw)', alpha=0.3, linestyle='--')
plt.title("(SHAP × Attention) Combined Map")
plt.xlabel("Time Step (Lag)")
plt.ylabel("Combined Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

lags = [f"t-{i}" for i in range(window_size, 0, -1)]  # Lags: t-30 → t-1

# plt.figure(figsize=(12, 4))
plt.figure(figsize=(5, 3))


plt.plot(lags, smoothed_map, label='SHAP × Attention (Smoothed)',
         color='#9467bd', linewidth=2.5)


plt.plot(lags, combined_map, label='Original (Raw)',
         color='gray', linestyle='--', linewidth=1.5, alpha=0.5)


#plt.title("Combined Importance Map: SHAP × Attention", fontsize=14)
plt.xlabel("Time Step (Lag)", fontsize=12)
plt.ylabel("Combined Value", fontsize=12)
plt.xticks(ticks=np.array(list(np.arange(0, window_size, 3)) + [window_size - 1]))
plt.yticks(fontsize=9)
plt.grid(True, linestyle='--', alpha=0.6)


plt.axhline(0, color='black', linestyle=':', linewidth=1)


plt.legend(fontsize=10, loc='lower left', frameon=True, framealpha=0.9, edgecolor='gray')

plt.tight_layout()
# plt.show()
plt.savefig("combined_importance_map.png", dpi=300)

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

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
performance_metrics = performance(y_test[:window_size], test_part)
print(f"Performance metrics: {performance_metrics}")

last_points_train = X_test[0].reshape(-1, 1).flatten()

# plotting
index_x = np.arange(len(last_points_train)) + 1
index_x_2 = index_x + window_size - 1
plt.figure(figsize=(5, 3))
plt.plot(index_x_2, y_test[:window_size], label='Observed', color='blue', alpha=0.6)
plt.plot(index_x_2, test_part, label='Predicted', color='red', alpha=0.6, linestyle='--')
# plot the X_train[-1] values
plt.plot(index_x, last_points_train, label='Data', color='green', alpha=0.6, linestyle='--')
# plt.title('Time Series Prediction: Observed vs Predicted')
plt.xlabel('Time Steps')
plt.ylabel('Normalized Value')
plt.legend()
plt.grid(True)
# plt.show()
plt.tight_layout()
plt.savefig("observed_vs_predicted.png", dpi=300)
