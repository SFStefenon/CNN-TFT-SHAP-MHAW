# CNN-TFT-SHAP-MHAW

This repository presents a convolutional neural network with temporal fusion transformer (CNN-TFT) hypertunded by Baesyana optimization for time series forecasting. The optimized CNN-TFT model is available [here](https://github.com/SFStefenon/CNN-TFT/blob/main/CNN-TFT.py). 

To compute the experiments and validate the model's performance, the natural inflow of the Tucurui hydroelectric power plant was considered (available [here](https://github.com/SFStefenon/CNN-TFT/blob/main/tucurui.csv)).

To have these hyperparameters set up, a hyperparameter tuning was computed using Bayesian optimization as presented [here](https://github.com/SFStefenon/CNN-TFT/blob/main/Model_eval/hypertuning.py). To prove that the proposed TFT-CNN model is stable, a statistical analysis was computed, as presented [here](http://github.com/SFStefenon/CNN-TFT/blob/main/Model_eval/stats.py).

The number of attention heads, CNN layers, filters, and kernel size are tuned for creating the following architecture:

![image](https://github.com/user-attachments/assets/8a329886-42b7-465d-b53b-9dc2453848e6)


