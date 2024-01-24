import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
import matplotlib.pyplot as plt


# Load the data
data = pd.read_csv("input/data/Tesla_Nasdaq_Prediction.csv")

# Prepare the dataset
features_df = data.iloc[:, 2:]
print(features_df.head())
target_df = data['Close/Last']
print(target_df.head())

features = data.iloc[:, 2:].values
target = data['Close/Last'].values
