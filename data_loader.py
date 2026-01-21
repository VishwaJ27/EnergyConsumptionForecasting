"""
Data loading module for Energy Consumption Forecasting
"""

import pandas as pd
import numpy as np
import yaml
import os
from pathlib import Path


class DataLoader:
    """Load and parse energy consumption data"""
    
    def __init__(self, config_path='config/config.yaml'):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def load_raw_data(self):
        """Load raw household power consumption data"""
        data_path = self.config['data']['raw_path']
        separator = self.config['data']['separator']
        
        print(f"Loading data from {data_path}...")
        
        # Read the data
        df = pd.read_csv(
            data_path,
            sep=separator,
            low_memory=False,
            na_values=['?', '']
        )
        
        # Combine Date and Time into datetime
        df['datetime'] = pd.to_datetime(
            df['Date'] + ' ' + df['Time'],
            format=f"{self.config['data']['date_format']} {self.config['data']['time_format']}"
        )
        
        # Set datetime as index
        df.set_index('datetime', inplace=True)
        
        # Drop original Date and Time columns
        df.drop(['Date', 'Time'], axis=1, inplace=True)
        
        # Convert columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"\nMissing values:\n{df.isnull().sum()}")
        
        return df
    
    def load_processed_data(self, filename='hourly_consumption.csv'):
        """Load preprocessed data"""
        processed_path = os.path.join(
            self.config['data']['processed_path'],
            filename
        )
        
        if not os.path.exists(processed_path):
            raise FileNotFoundError(f"Processed data not found at {processed_path}")
        
        df = pd.read_csv(processed_path, index_col=0, parse_dates=True)
        print(f"Processed data loaded. Shape: {df.shape}")
        
        return df
    
    def save_processed_data(self, df, filename='hourly_consumption.csv'):
        """Save processed data"""
        processed_path = self.config['data']['processed_path']
        
        # Create directory if it doesn't exist
        Path(processed_path).mkdir(parents=True, exist_ok=True)
        
        filepath = os.path.join(processed_path, filename)
        df.to_csv(filepath)
        print(f"Processed data saved to {filepath}")


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    df = loader.load_raw_data()
    print("\nFirst few rows:")
    print(df.head())
    print("\nData info:")
    print(df.info())
