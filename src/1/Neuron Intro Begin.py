import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.optimize import curve_fit
from tabulate import tabulate

class DataAnalysis:
    def __init__(self, data):
        self.data = data

    def display_data_as_table(self):
        

    def create_scatter_plot(self):
        

    def create_linear_fit_plot(self):
       

    def sigmoid(self, x, L, x0, k):
        

    def create_sigmoid_fit_plot(self):
       

def main():
    # Original data
    original_data = pd.DataFrame({
        'Age': [23, 25, 48, 52, 47, 56, 54, 60, 33, 60, 18, 29, 27, 29, 50],
        'Guess': [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1]
    })

    data_analysis = DataAnalysis(original_data)
    data_analysis.display_data_as_table()
    data_analysis.create_scatter_plot()
    data_analysis.create_linear_fit_plot()
    data_analysis.create_sigmoid_fit_plot()

if __name__ == "__main__":
    main()
