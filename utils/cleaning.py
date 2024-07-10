import os
import numpy as np
import pandas as pd

def cleaning_pipeline(data, save_folder, save_filename):
    """
    Performs a series of data cleaning operations on the input DataFrame "data" and saves the cleaned data to a CSV file

    Parameters:
    - data (DataFrame): Input DataFrame containing movie data
    - save_folder (str): Folder path where the cleaned CSV file will be saved
    - save_filename (str): Filename for the cleaned CSV file

    Returns:
    - None
    """
    data["class"] = data["class"].map({"neg": 0, "pos": 1})
    data.replace("na", np.nan, inplace=True)

    threshold = len(data) * 0.5
    data.dropna(thresh=threshold, axis=1, inplace=True)
      
    for column in data.columns:
        if data[column].isna().any():
            data[column] = pd.to_numeric(data[column], errors="coerce")
            median = data[column].median()
            data[column].fillna(median, inplace=True)
            try:
                data[column] = data[column].astype("int64")
            except ValueError:
                print("Couldn't convert to int64")
    
    save_path = os.path.join(save_folder, f"{save_filename}.csv")
    data.to_csv(save_path, index=False)

class DataFrameAnalyzer:
    """
    A class for analyzing a DataFrame to find columns with "na" values and check for non-digit and non-"na" values.

    Attributes:
    - data (DataFrame): The dataset to be analyzed

    Methods:
    - __init__: Initialize the DataFrameAnalyzer object with a DataFrame
    - find_na_columns: Identify columns containing the string "na" and return their counts and data types
    - is_digit_or_na: Check if a given value is a digit (integer or float) or the string "na"
    - check_non_digit_and_non_na: Check if a column contains any values that are neither digits nor "na"
    """
    def __init__(self, data):
        """
        Initialize the DataFrameAnalyzer with a DataFrame
        
        Parameters:
        data (pd.DataFrame): The DataFrame to analyze
        """
        self.data = data

    def find_na_columns(self):
        """
        Identify columns containing the string "na" and return their counts and data types

        Returns:
        dict: A dictionary with column names as keys and a dict with "na" values and dtypes of the column as values
        """
        na_columns = {}
        for column in self.data.columns:
            # Count "na" values in the column, case-insensitive
            na_count = self.data[column].astype(str).str.lower().eq("na").sum()
            if na_count > 0:
                na_columns[column] = {"na_count": na_count, "dtype": self.data[column].dtype}
        return na_columns

    def is_digit_or_na(self, value):
        """
        Check if a given value is a digit (integer or float) or the string "na"

        Parameters:
        - value (str): The value to check

        Returns:
        - bool: True if the value is a digit or "na", False otherwise
        """
        try:
            float(value)
            return True
        except:
            return value.lower() == "na"

    def check_non_digit_and_non_na(self, column):
        """
        Check if a column contains any values that are neither digits nor "na"

        Parameters:
        - column (pd.Series): The column to check

        Returns:
        - bool: True if the column contains values that are neither digits nor "na", False otherwise
        """
        # Convert the column to string and apply the check function
        return self.data[column].astype(str).apply(lambda x: not self.is_digit_or_na(x)).any()