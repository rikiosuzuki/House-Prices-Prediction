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
        # Convert categorical yes/no to 1/0
        binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                      'airconditioning', 'prefarea']
        for col in binary_cols:
            df[col] = df[col].map({'yes': 1, 'no': 0})
        
        # Handle furnishing status
        df['furnishingstatus'] = df['furnishingstatus'].map({
            'furnished': 2, 
            'semi-furnished': 1, 
            'unfurnished': 0
        })
        
        # Separate numerical and categorical columns
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        # Scale numerical features
        df_scaled = pd.DataFrame(self.scaler.fit_transform(df[num_cols]), 
                               columns=num_cols)
        
        return df_scaled