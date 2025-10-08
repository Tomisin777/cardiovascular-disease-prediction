"""
Data loading and preprocessing functions for cardiovascular disease dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os

# Add parent directory to path to import config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


def load_data(url=config.DATA_URL, column_names=config.COLUMN_NAMES):
    """
    Load the UCI Heart Disease dataset from URL.
    
    Args:
        url (str): URL to the dataset
        column_names (list): List of column names for the dataset
    
    Returns:
        pd.DataFrame: Raw dataset
    """
    data = pd.read_csv(url, header=None, names=column_names)
    return data


def clean_data(data):
    """
    Clean the dataset by handling missing values and converting data types.
    
    Args:
        data (pd.DataFrame): Raw dataset
    
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    # Clean 'ca' column
    data['ca'] = data['ca'].replace('?', np.nan)
    data['ca'] = pd.to_numeric(data['ca'])
    data['ca'] = data['ca'].fillna(data['ca'].mean())
    
    # Clean 'thal' column
    data['thal'] = data['thal'].replace('?', np.nan)
    data['thal'] = pd.to_numeric(data['thal'])
    data['thal'] = data['thal'].fillna(data['thal'].mean())
    
    # Convert target to binary (0 = no disease, 1 = disease)
    data['target'] = (data['target'] > 0).astype(int)
    
    return data


def split_data(data, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE):
    """
    Split data into training and testing sets.
    
    Args:
        data (pd.DataFrame): Cleaned dataset
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X = data.drop('target', axis=1)
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test


def prepare_data():
    """
    Complete data pipeline: load, clean, and split data.
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print("Loading data...")
    data = load_data()
    
    print("Cleaning data...")
    data = clean_data(data)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(data)
    
    return X_train, X_test, y_train, y_test