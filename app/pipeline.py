import pandas as pd
import numpy as np
import pickle

class PredictPipeline:
    """
    A class for predicting truck maintenance needs using a pre-trained model and preprocessing pipeline

    Methods:
    - __init__: Initializes the PredictPipeline object by loading the mapping, preprocessor, and model from .pkl files
    - preprocess_dataset: Processes the input dataset, ensuring it contains the required columns
    - preprocess_data: Preprocesses the input data, including feature engineering and transformation
    - predict: Predicts maintenance needs based on the input data
    """
    def __init__(self):
        """
        Initializes the PredictPipeline object by loading the mapping, preprocessor and model from .pkl files
        """
        # Load mapping
        mapping_path = "app/artifacts/transformation_anova_columns.pkl"
        with open(mapping_path, "rb") as f:
            self.mapping = pickle.load(f)

        # Load preprocessor
        preprocessor_path = "app/artifacts/preprocessor_anova_columns.pkl"
        with open(preprocessor_path, "rb") as f:
            self.preprocessor = pickle.load(f)

        # Load model
        model_path = "app/artifacts/model.pkl"
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def preprocess_dataset(self, input_data):
        """
        Processes the input dataset, ensuring it contains the required columns

        Parameters:
        - input_data: The input data to be processed

        Returns:
        - pandas.DataFrame: The processed input data
        """
        # Convert column names to lowercase, remove text within brackets, strip trailing spaces, and replace spaces with underscores
        input_data.columns = input_data.columns.str.lower().str.replace(r"\[.*\]", "", regex=True).str.rstrip().str.replace(" ", "_")

        # Drop the "class" column if it exists
        if "class" in input_data.columns:
            input_data = input_data.drop(columns=["class"])
        
        # Replace "na" string values with np.nan to mark missing values and iterate through each column in the DataFrame to handle missing values
        if "na" in input_data.values:
            input_data.replace("na", np.nan, inplace=True)
        for column in input_data.columns:
            if input_data[column].isna().any():
                # Convert column to numeric, coercing errors to NaN
                input_data[column] = pd.to_numeric(input_data[column], errors="coerce")
                # Calculate and fill missing values with the median
                median = input_data[column].median()
                input_data[column].fillna(median, inplace=True)
                try:
                    input_data[column] = input_data[column].astype("int64")
                except ValueError as e:
                    raise ValueError(f"Couldn't preprocess this dataset {e}")

        return input_data

    def preprocess_data(self, data):
        """
        Preprocesses the input data, including feature engineering and transformation

        Parameters:
        - data: The input data to be preprocessed

        Returns:
        - pandas.DataFrame: The preprocessed data
        """
        # Create the gross_per_vote column and select the onehot encoded columns
        log_cols = self.mapping["log_columns"]
        cbrt_cols = self.mapping["cubic_columns"]

        data = self.preprocessor.transform(data)

        # Create a DataFrame with the transformed data and feature names
        new_data = pd.DataFrame(data, columns=log_cols + cbrt_cols)
        return new_data

    def predict(self, data):
        """
        Predicts meaintenance needs based on the input data

        Parameters:
        - data: The input data for prediction

        Returns:
        - list: The predicted maintenance needs (Healthy or Needs Maintenance)
        """
        preds_proba = self.model.predict_proba(self.preprocess_data(data))[:,1]
        preds = (preds_proba >= 0.6).astype(int)
        prediction_classes = ["Healthy", "Needs Maintenance"]

        # If predicting from a dataset
        predicted_classes = [prediction_classes[pred] for pred in preds]
        return predicted_classes