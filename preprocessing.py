"""
Data preprocessing module for Energy Consumption Forecasting
"""

import pandas as pd
import numpy as np
import yaml
from scipy import stats


class DataPreprocessor:
    """Preprocess energy consumption data"""
    
    def __init__(self, config_path='config/config.yaml'):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        print("\nHandling missing values...")
        print(f"Missing values before: {df.isnull().sum().sum()}")
        
        fill_method = self.config['preprocessing']['fill_method']
        
        # Forward fill for small gaps
        df = df.fillna(method=fill_method, limit=6)  # Fill up to 6 consecutive NaNs
        
        # Interpolate for remaining gaps
        df = df.interpolate(method='linear', limit_direction='both')
        
        # Drop any remaining NaN rows
        df = df.dropna()
        
        print(f"Missing values after: {df.isnull().sum().sum()}")
        
        return df
    
    def remove_outliers(self, df, columns=None):
        """Remove outliers using Z-score method"""
        print("\nRemoving outliers...")
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        threshold = self.config['preprocessing']['outlier_threshold']
        original_len = len(df)
        
        for col in columns:
            z_scores = np.abs(stats.zscore(df[col]))
            df = df[z_scores < threshold]
        
        removed = original_len - len(df)
        print(f"Removed {removed} outlier rows ({removed/original_len*100:.2f}%)")
        
        return df
    
    def aggregate_data(self, df):
        """Aggregate data to specified time level"""
        aggregation_level = self.config['preprocessing']['aggregation_level']
        
        print(f"\nAggregating data to {aggregation_level} level...")
        
        # Aggregate numerical columns
        df_agg = df.resample(aggregation_level).agg({
            'Global_active_power': 'mean',
            'Global_reactive_power': 'mean',
            'Voltage': 'mean',
            'Global_intensity': 'mean',
            'Sub_metering_1': 'sum',
            'Sub_metering_2': 'sum',
            'Sub_metering_3': 'sum'
        })
        
        print(f"Aggregated shape: {df_agg.shape}")
        
        return df_agg
    
    def preprocess_pipeline(self, df):
        """Complete preprocessing pipeline"""
        print("=" * 50)
        print("Starting preprocessing pipeline...")
        print("=" * 50)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Aggregate data
        df = self.aggregate_data(df)
        
        # Remove outliers
        df = self.remove_outliers(df)
        
        print("\n" + "=" * 50)
        print("Preprocessing complete!")
        print("=" * 50)
        
        return df


if __name__ == "__main__":
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    df = loader.load_raw_data()
    
    # Preprocess
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess_pipeline(df)
    
    # Save processed data
    loader.save_processed_data(df_processed)
    
    print("\nProcessed data summary:")
    print(df_processed.describe())
