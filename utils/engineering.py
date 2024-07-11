import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.preprocessing import RobustScaler
import pickle
import os

class Transformations:
    """
    A class for performing data transformations and outlier analysis

    Methods:
    - __init__: Initialize the Transformations object
    - count_outliers: Counts the number of outliers in a given column using the IQR method
    - top_n_outliers: Finds and returns the top columns with the most outliers
    - columns_transformations: Applies logarithmic and cubic transformations to numeric columns
    - sort_results: Sorts columns based on the reduction in outliers after transformation
    """
    def __init__(self):
        """
        Initialize the Engineering object
        """
        pass

    def count_outliers(self, data, column):
        """
        Counts the number of outliers in a given column of a DataFrame using the IQR method

        Parameters:
        - data (pd.DataFrame): The input DataFrame
        - column (str): The column name for which to count outliers

        Returns:
        - int: The number of outliers in the specified column
        """
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)

        # Calculate IQR
        IQR = Q3 - Q1

        # Define outlier thresholds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Count outliers
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]

        return outliers.shape[0]

    def top_n_outliers(self, data, target):
        """
        Finds and returns the top columns with the most outliers

        Parameters:
        - data (pd.DataFrame): The input DataFrame
        - target (str): The target column name

        Returns:
        - dict: A dictionary where keys are column names and values are the number of outliers
        """
        outlier_counts = {}

        for column in data.drop(columns=[target]).select_dtypes(include="number").columns:
            outlier_counts[column] = self.count_outliers(data, column)
        
        return outlier_counts

    def columns_transformations(self, data, target):
        """
        Applies logarithmic and cubic transformations to numeric columns

        Parameters:
        - data (pd.DataFrame): The input DataFrame
        - target (str): The target column name

        Returns:
        - data (pd.DataFrame): DataFrame with columns transformed
        - log_columns (list): Columns transformed with log function
        - cubic_columns (list): Columns transformed with cubic function
        """
        log_columns = []
        cubic_columns = []
        for col in data.drop(columns=[target]):
            # Calculate skewness after applying log1p and cbrt transformations
            log_transformation = np.log1p(data[col]).skew()
            cubic_transformation = np.cbrt(data[col]).skew()
            skewness_values = [log_transformation, cubic_transformation]

            # Apply the transformation that reduces skewness more effectively
            if np.argmin(np.abs(skewness_values)) == 0:
                log_columns.append(col)
                data[col] = np.log1p(data[col])
            else:
                cubic_columns.append(col)
                data[col] = np.cbrt(data[col])

        # Return the transformed DataFrame and lists of transformed columns
        return data, log_columns, cubic_columns

    def sort_results(self, outliers_before, outliers_after):
        """
        Sorts columns based on the reduction in outliers after transformation

        Parameters:
        - outliers_before (dict): Dictionary with column names as keys and initial outlier counts as values
        - outliers_after (dict): Dictionary with column names as keys and final outlier counts as values

        Returns:
        - list: Sorted list of tuples where each tuple contains a column name and the reduction in outliers
        """
        results = []
        for column in outliers_before.keys():
            initial = outliers_before[column]
            final = outliers_after[column]
            reduced = initial - final
            if reduced != 0:
                results.append((column, reduced))

        # Sort the results by the number of outliers reduced in descending order
        return sorted(results, key=lambda x: x[1], reverse=True)

class FeaturesSelection:
    """
    A class to perform feature selection using PCA and ANOVA methods, and save the mappings
    
    Attributes:
    - data (DataFrame): The input data for feature selection
    - target (str): The target column name
    - pca (PCA or None): The PCA model instance
    - transformation_cols_anova (dict or None): Mappings of selected features after ANOVA
    - transformation_cols_all (dict or None): All provided columns for potential transformation
    
    Methods:
    - __init__: Initialize the FeaturesSelection object with input data and target column
    - fit_pca: Perform PCA on the data to reduce dimensions while retaining variance
    - select_features_anova: Select features using ANOVA and update transformation mappings
    - save_mappings: Save mappings of selected features post-ANOVA and all transformation columns to files
    """
    def __init__(self, data, target):
        """
        Initializes the FeaturesSelection object with input data and target column
        
        Parameters:
        - data (DataFrame): Input data containing features and target
        - target (str): Name of the target column
        """
        self.data = data
        self.target = target
        self.pca = None
        self.transformation_cols_anova = None
        self.transformation_cols_all = None
    
    def fit_pca(self, n_components):
        """
        Perform PCA on the data to reduce dimensions while retaining variance
        
        Parameters:
        - n_components (int): Number of principal components to retain
        """
        # Initialize PCA to retain 99% variance
        self.pca = PCA(n_components=n_components)
        
        # Fit PCA on scaled data
        scaler = RobustScaler()
        X_transformed_scaled = pd.DataFrame(scaler.fit_transform(self.data.drop(columns=self.target)),
                                            columns=self.data.drop(columns=self.target).columns)
        self.pca.fit(X_transformed_scaled)
        
        # Plot eigenvalues and inflection point
        sorted_eigenvalues = sorted(self.pca.explained_variance_, reverse=True)
        plt.figure(figsize=(10, 4))
        plt.plot(range(1, len(sorted_eigenvalues)+1), sorted_eigenvalues, marker="o")
        plt.xlabel("N components")
        plt.ylabel("Eigenvalues")
        plt.title("Eigenvalues x N components")
        plt.axvline(x=self.pca.n_components_, color="r", linestyle="--", label=f"Inflection Point: {self.pca.n_components_}")
        plt.legend()
        plt.show()
    
    def select_features_anova(self, log_columns, cubic_columns):
        """
        Select features using ANOVA and update transformation mappings
        
        Parameters:
        - log_columns (list): Columns to apply logarithmic transformation
        - cubic_columns (list): Columns to apply cubic transformation
        """
        # Initialize ANOVA test and scale the data with Robust Scaler
        anova_test = SelectKBest(score_func=f_classif, k=self.pca.n_components_)
        X_transformed_scaled = pd.DataFrame(RobustScaler().fit_transform(self.data.drop(columns=self.target)),
                                            columns=self.data.drop(columns=self.target).columns)
        
        # Fit ANOVA on the scaled data with the target variable
        anova_fit = anova_test.fit(X_transformed_scaled, self.data[self.target])
        
        # Extract feature scores and select top features based on scores
        scores_df = pd.DataFrame({"feature": X_transformed_scaled.columns, "score": anova_fit.scores_})
        features_selected = scores_df.nlargest(self.pca.n_components_, "score")["feature"].tolist()
        
        # Update lists with intersection of selected_features
        log_transformed_columns = list(set(log_columns).intersection(set(features_selected)))
        cubic_transformed_columns = list(set(cubic_columns).intersection(set(features_selected)))
        
        self.transformation_cols_anova = {
            "log_columns": log_transformed_columns,
            "cubic_columns": cubic_transformed_columns
        }
        
        self.transformation_cols_all = {
            "log_columns": log_columns,
            "cubic_columns": cubic_columns
        }
        return features_selected
    
    def save_mappings(self):
        """
        Save mappings of selected features post-ANOVA and all transformation columns to files
        
        Parameters:
        - anova_filename (str): Filename for saving ANOVA transformation mappings
        - all_filename (str): Filename for saving all transformation columns mappings
        """
        if not os.path.exists("../artifacts"):
            os.makedirs("../artifacts")

        with open("../artifacts/transformation_anova_columns.pkl", "wb") as f_anova:
            pickle.dump(self.transformation_cols_anova, f_anova)
        
        with open("../artifacts/transformation_all_columns.pkl", "wb") as f_all:
            pickle.dump(self.transformation_cols_all, f_all)