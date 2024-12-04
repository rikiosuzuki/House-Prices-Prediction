# model_training.py
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class HousePriceModel:
    def __init__(self):
        self.models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.best_model = None
        self.best_score = float('-inf')
        
    def train_evaluate_models(self, X_train, X_test, y_train, y_test):
        results = {}
        
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            r2 = r2_score(y_test, test_pred)
            
            cv_scores = cross_val_score(model, X_train, y_train, 
                                      cv=5, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            results[name] = {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'r2_score': r2,
                'cv_rmse': cv_rmse
            }
            
            if r2 > self.best_score:
                self.best_model = model
                self.best_score = r2
        
        return results
    
    def plot_feature_importance(self, X, model_name='random_forest'):
        if model_name not in ['random_forest', 'gradient_boosting']:
            raise ValueError("Feature importance only available for tree-based models")
            
        model = self.models[model_name]
        importance = model.feature_importances_
        
        feat_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        })
        feat_importance = feat_importance.sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feat_importance)
        plt.title(f'Feature Importance ({model_name})')
        plt.show()
        
        return feat_importance

    
    def get_predictions(self, X_test, y_test):
        predictions = {}
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            pred_df = pd.DataFrame({
                'Actual_Price': y_test,
                'Predicted_Price': y_pred,
                'Difference': y_test - y_pred
            })
            predictions[name] = pred_df
        return predictions