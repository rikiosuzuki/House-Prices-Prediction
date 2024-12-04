# data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

class HousingDataPreprocessor:
    def __init__(self):
        self.num_imputer = SimpleImputer(strategy='mean')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def preprocess_data(self, df):
        binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                      'airconditioning', 'prefarea']
        for col in binary_cols:
            df[col] = df[col].map({'yes': 1, 'no': 0})
        
        df['furnishingstatus'] = df['furnishingstatus'].map({
            'furnished': 2, 
            'semi-furnished': 1, 
            'unfurnished': 0
        })
        
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        df_scaled = pd.DataFrame(self.scaler.fit_transform(df[num_cols]), 
                               columns=num_cols)
        
        return df_scaled