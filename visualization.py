# visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class HousingVisualizer:
    @staticmethod
    def plot_correlation_matrix(df):
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
        plt.title('Feature Correlation Matrix')
        plt.show()

    @staticmethod
    def plot_price_distribution(df):
        plt.figure(figsize=(10, 6))
        sns.histplot(df['price'], kde=True)
        plt.title('Price Distribution')
        plt.xlabel('Price')
        plt.ylabel('Count')
        plt.show()

    @staticmethod
    def plot_scatter_matrix(df, features):
        sns.pairplot(df[features])
        plt.show()

    @staticmethod
    def plot_residuals(y_true, y_pred):
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.show()
    

    @staticmethod
    def plot_predictions(predictions, model_name):
        pred_df = predictions[model_name]
        plt.figure(figsize=(10, 6))
        plt.scatter(pred_df['Actual_Price'], pred_df['Predicted_Price'], alpha=0.5)
        plt.plot([pred_df['Actual_Price'].min(), pred_df['Actual_Price'].max()],
                [pred_df['Actual_Price'].min(), pred_df['Actual_Price'].max()],
                'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title(f'{model_name} - Actual vs Predicted Prices')
        plt.show()