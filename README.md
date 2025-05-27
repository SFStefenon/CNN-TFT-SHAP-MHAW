# CNN-TFT

This repository presents a convolutional neural network with temporal fusion transformer (CNN-TFT) hypertunded by Baesyana optimization for time series forecasting. The optimized CNN-TFT model is available [here](https://github.com/SFStefenon/CNN-TFT/blob/main/CNN-TFT.py). 

To compute the experiments and validate the model's performance, the natural inflow of the Tucurui hydroelectric power plant was considered (available [here](https://github.com/SFStefenon/CNN-TFT/blob/main/tucurui.csv)).

To have these hyperparameters set up, a hyperparameter tuning was computed using Bayesian optimization as presented [here](https://github.com/SFStefenon/CNN-TFT/blob/main/Model_eval/hypertuning.py). To prove that the proposed TFT-CNN model is stable, a statistical analysis was computed, as presented [here](http://github.com/SFStefenon/CNN-TFT/blob/main/Model_eval/stats.py).

The number of attention heads, CNN layers, filters, and kernel size are tuned for creating the following arquitecture:
![image](https://github.com/user-attachments/assets/3c36a812-b13e-4000-acbe-9491b69157c5)


---

Thank you

Dr. **Stefano Frizzo Stefenon**

Postdoctoral fellow at the University of Regina

Faculty of Engineering and Applied Sciences

Regina, SK, Canada, 2025.
