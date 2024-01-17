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
        table = tabulate(self.data, headers='keys', tablefmt='fancy_grid')
        print("Original Data:")
        print(table)

    def create_scatter_plot(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.data['Age'], self.data['Guess'], color='blue', label='Original Data')
        plt.xlabel('Age')
        plt.ylabel('Probability')
        plt.title('Age vs Guess - Scatter Plot')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        scatter_plot_path = 'output/neuron/age_vs_guess_scatter_plot.png'
        plt.savefig(scatter_plot_path)
        plt.close()
        print("Scatter plot created and saved in output/neuron directory")

    def create_linear_fit_plot(self):
        slope, intercept, _, _, _ = linregress(self.data['Age'], self.data['Guess'])
        # Print the slope and intercept
        print("Slope:", slope)
        print("Intercept:", intercept)
        
        x_values = np.linspace(10, 65, 300)
        linear_fit_values = slope * x_values + intercept
        plt.figure(figsize=(10, 6))
        plt.scatter(self.data['Age'], self.data['Guess'], color='blue', label='Original Data')
        plt.plot(x_values, linear_fit_values, color='green', linestyle='-', linewidth=2, label='Linear Fit')
        plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xlabel('Age')
        plt.ylabel('Probability')
        plt.title('Age vs Guess with Linear Fit')
        plt.legend()
        linear_fit_plot_path = 'output/neuron/age_vs_guess_linear_plot_with_lines.png'
        plt.savefig(linear_fit_plot_path)
        plt.close()
        print("Linear fit plot created and saved in output/neuron directory")

    def sigmoid(self, x, L, x0, k):
        y = L / (1 + np.exp(-k * (x - x0)))
        return y

    def create_sigmoid_fit_plot(self):
        L = 1
        x0 = np.median(self.data['Age'])
        k = 1
        x_values = np.linspace(10, 65, 300)
        y_values = self.sigmoid(x_values, L, x0, k)
        plt.figure(figsize=(10, 6))
        plt.scatter(self.data['Age'], self.data['Guess'], color='blue', label='Original Data')
        plt.plot(x_values, y_values, color='red', label='Sigmoid Fit')
        plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)
        plt.axvline(x=x0, color='gray', linestyle='--', linewidth=1)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xlabel('Age')
        plt.ylabel('Guess / Sigmoid Probability')
        plt.title('Age vs Guess with Sigmoid Fit')
        plt.legend()
        plot_path_with_lines = 'output/neuron/age_vs_guess_sigmoid_plot_with_lines.png'
        plt.savefig(plot_path_with_lines)
        plt.close()
        print("Sigmoid fit plot created and saved in output/neuron directory")

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
