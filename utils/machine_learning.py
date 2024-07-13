import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, roc_auc_score, recall_score, roc_curve, confusion_matrix, precision_score, accuracy_score, precision_recall_curve, precision_recall_fscore_support
from sklearn.inspection import permutation_importance
import pickle

mlflow_tracking_username = os.environ.get("MLFLOW_TRACKING_USERNAME")
mlflow_tracking_password = os.environ.get("MLFLOW_TRACKING_PASSWORD")
uri = os.environ.get("uri")

class DataPreprocess:
    """
    A class for preprocessing data including feature engineering, transformation, and splitting into train-test sets.

    Methods:
    - __init__: Initializes the DataPreprocess object
    - save_preprocessor: Saves the preprocessor object to a file
    - load_preprocessor: Loads the preprocessor object from a file
    - preprocessor: Creates and returns a preprocessor pipeline for data preprocessing
    - preprocess_new_data: Preprocesses new input data, including feature engineering and transformation
    - preprocess_data: Preprocesses the input data, including feature engineering, transformation, and splitting into train-test sets
    """
    def __init__(self):
        """
        Initializes the DataPreprocess object by loading pre-saved transformation mappings
        """
        with open("../artifacts/transformation_all_columns.pkl", "rb") as f:
            self.mapping_all = pickle.load(f)

        with open("../artifacts/transformation_anova_columns.pkl", "rb") as f:
            self.mapping_anova = pickle.load(f)

    def save_preprocessor(self, preprocessor, anova=False):
        """
        Saves the preprocessor object to a file

        Parameters:
        - preprocessor: The preprocessor object to be saved
        - anova: Boolean flag indicating if the preprocessor is for ANOVA-selected columns
        """
        if not os.path.exists("../artifacts"):
            os.makedirs("../artifacts")

        if anova:
            with open("../artifacts/preprocessor_anova_columns.pkl", "wb") as f_all:
                pickle.dump(preprocessor, f_all)
        else:
            with open("../artifacts/preprocessor_all_columns.pkl", "wb") as f_all:
                pickle.dump(preprocessor, f_all)
    
    def load_preprocessor(self, anova):
        """
        Loads the preprocessor object from a file

        Parameters:
        - anova: Boolean flag indicating if the preprocessor is for ANOVA-selected columns

        Returns:
        - preprocessor: The loaded preprocessor object
        """
        if anova:
            with open("../artifacts/preprocessor_anova_columns.pkl", "rb") as f:
                preprocessor = pickle.load(f)
        else:
            with open("../artifacts/preprocessor_all_columns.pkl", "rb") as f:
                preprocessor = pickle.load(f)
        return preprocessor

    def preprocessor(self, log_cols, cbrt_cols):
        """
        Creates and returns a preprocessor pipeline for data preprocessing

        Parameters:
        - log_cols: List of column names for which log transformation is applied
        - cbrt_cols: List of column names for which cubic root transformation is applied

        Returns:
        - preprocessor: Preprocessor pipeline for data preprocessing
        """
        # Define transformers for numeric columns
        log_transformer = Pipeline(steps=[
            ("log_transformation", FunctionTransformer(np.log1p, validate=True)),
            ("scaler", RobustScaler())
        ])
        cubic_transformer = Pipeline(steps=[
            ("cbrt_transformation", FunctionTransformer(np.cbrt, validate=True)),
            ("scaler", RobustScaler())
        ])

        # Combine transformers for numeric and categorical columns
        transformers=[("num_log", log_transformer, log_cols),
                    ("num_cbrt", cubic_transformer, cbrt_cols)]

        preprocessor = ColumnTransformer(transformers=transformers, verbose_feature_names_out=False, remainder="drop")
        return preprocessor

    def preprocess_new_data(self, data, target_name, anova, log_cols, cbrt_cols):
        """
        Preprocesses new input data, including feature engineering and transformation

        Parameters:
        - data: Input DataFrame containing the raw data
        - target_name: Name of the target variable
        - anova: Boolean flag indicating if the preprocessor is for ANOVA-selected columns
        - log_cols: List of column names for which log transformation is applied
        - cbrt_cols: List of column names for which cubic root transformation is applied

        Returns:
        - X: Preprocessed feature data
        - y: Target labels
        """
        # Check if the target column contains "neg" and "pos" strings and map them to 0 and 1 respectively
        if data[target_name].dtype == object and set(data[target_name].unique()).issubset({"neg", "pos"}):
            data[target_name] = data[target_name].map({"neg": 0, "pos": 1})

        # Replace "na" string values with np.nan to mark missing values and iterate through each column in the DataFrame to handle missing values
        if "na" in data.values:
            data.replace("na", np.nan, inplace=True)
        for column in data.columns:
            if data[column].isna().any():
                # Convert column to numeric, coercing errors to NaN
                data[column] = pd.to_numeric(data[column], errors="coerce")
                # Calculate and fill missing values with the median
                median = data[column].median()
                data[column].fillna(median, inplace=True)
                try:
                    data[column] = data[column].astype("int64")
                except ValueError:
                    print("Couldn't convert to int64")

        # Load the appropriate preprocessor based on whether ANOVA-selected columns are used and apply it to the data
        preprocessor = self.load_preprocessor(anova=anova)
        X = preprocessor.transform(data.drop(columns=[target_name]))
        X = pd.DataFrame(X, columns=log_cols + cbrt_cols)
        y = data[target_name]
    
        return X, y

    def preprocess_data(self, data, target_name, test_size=None, anova=False, new_data=False):
        """
        Preprocesses the input data, including feature engineering, transformation, and splitting into train-test sets

        Parameters:
        - data: Input DataFrame containing the raw data
        - target_name: Name of the target variable
        - test_size: The proportion of the dataset to include in the test split
        - anova: Boolean flag indicating if the preprocessor is for ANOVA-selected columns
        - new_data: Boolean flag indicating if new data is being processed

        Returns:
        - X_train: Features of the training set
        - X_test: Features of the testing set
        - y_train: Target labels of the training set
        - y_test: Target labels of the testing set
        """
        # Specify columns needing log transformation, square root transformation
        log_cols = self.mapping_all["log_columns"]
        cbrt_cols = self.mapping_all["cubic_columns"]
        if anova:
            log_cols = self.mapping_anova["log_columns"]
            cbrt_cols = self.mapping_anova["cubic_columns"]

        # Process the test dataset
        if new_data:
            X, y = self.preprocess_new_data(data=data, target_name=target_name, anova=anova, log_cols=log_cols, cbrt_cols=cbrt_cols)
            return X, y

        data_process = data.drop(columns=[target_name])
        
        # Build preprocessor
        preprocessor = self.preprocessor(log_cols, cbrt_cols)

        # Fit and transform data
        data_preprocessed = preprocessor.fit_transform(data_process)

        data_preprocessed = pd.DataFrame(data_preprocessed, columns=log_cols + cbrt_cols)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(data_preprocessed, data[target_name], test_size=test_size, stratify=data[target_name], shuffle=True, random_state=42)

        # Save preprocessor if not already saved
        self.save_preprocessor(preprocessor, anova=anova)

        return X_train, X_test, y_train, y_test

class ModelTraining:
    """
    A class for training machine learning models, evaluating their performance and save the best one

    Methods:
    - __init__: Initializes the ModelTraining object
    - save_model: Saves the model to a pkl file
    - initiate_model_trainer: Initiates the model training process
    - evaluate_models: Evaluates multiple models using random search cross-validation and logs the results with MLflow
    """
    def __init__(self):
        pass

    def save_model(self, model_name, version, save_folder, save_filename):
        """
        Save the model to a pkl file

        Parameters:
        - model_name (dict): The model to save
        - version (int): The model version to save
        - save_folder (str): Folder path where the model will be saved
        - save_filename (str): Filename for the pkl file
        """
        mlflow.set_tracking_uri(uri)
        client = mlflow.tracking.MlflowClient(tracking_uri=uri)

        # Get the correct version of the registered model
        model_versions = client.search_model_versions(f"name='{model_name}'")
        latest_version = model_versions[version]
        # Construct the logged model path
        run_id = latest_version.run_id
        artifact_path = latest_version.source.split('/')[-1]
        logged_model = f'runs:/{run_id}/{artifact_path}'

        # Load the model from MLflow and saves it to a pkl file
        loaded_model = mlflow.sklearn.load_model(logged_model)
        file_path = os.path.join(save_folder, f"{save_filename}.pkl")
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        with open(file_path, 'wb') as f:
            pickle.dump(loaded_model, f)

    def initiate_model_trainer(self, train_test, experiment_name, use_smote=False):
        """
        Initiates the model training process

        Parameters:
        - train_test: A tuple containing the train-test split data in the format (X_train, y_train, X_test, y_test)
        - experiment_name: Name of the MLflow experiment where the results will be logged
        - use_smote: A boolean indicating whether to apply SMOTE for balancing the classes. Default is False
        
        Returns:
        - dict: A dictionary containing the evaluation report for each model
        """
        mlflow.set_tracking_uri(uri)
        X_train, y_train, X_test, y_test = train_test
        
        models = {
            "Logistic Regression": LogisticRegression(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42)
        }
        
        params = {
            "Logistic Regression": {
                "solver": ["liblinear", "lbfgs"],
                "penalty":["l2", "l1", "elasticnet", None], 
                "C":[1.5, 1, 0.5, 0.1]
            },
            "Random Forest":{
                "criterion":["gini", "entropy", "log_loss"],
                "max_features":["sqrt", "log2"],
                "n_estimators": [25, 50, 100],
                "max_depth": [10, 20, 30, 50]
            },
            "Gradient Boosting":{
                "loss":["log_loss", "exponential"],
                "max_features":["sqrt", "log2"],
                "n_estimators": [25, 50, 100],
                "max_depth": [10, 20, 30, 50],
                "learning_rate": [0.001, 0.01, 0.1],
            },
        }
        
        model_report = self.evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
                                           models=models, params=params, experiment_name=experiment_name, use_smote=use_smote)
        
        return model_report

    def evaluate_models(self, X_train, y_train, X_test, y_test, models, params, experiment_name, use_smote):
        """
        Evaluates multiple models using random search cross-validation and logs the results with MLflow

        Parameters:
        - X_train: Features of the training data
        - y_train: Target labels of the training data
        - X_test: Features of the testing data
        - y_test: Target labels of the testing data
        - models: A dictionary containing the models to be evaluated
        - params: A dictionary containing the hyperparameter grids for each model
        - experiment_name: Name of the MLflow experiment where the results will be logged
        - use_smote: A boolean indicating whether to apply SMOTE for balancing the classes
        
        Returns:
        - dict: A dictionary containing the evaluation report for each model
        """
        mlflow.set_experiment(experiment_name)
        report = {}
        if use_smote:
            # Apply SMOTE only to the training data
            smote = SMOTE()
            X_train, y_train = smote.fit_resample(X_train, y_train)
        for model_name, model in models.items():
            with mlflow.start_run(run_name=model_name):
                param = params[model_name]
                if model_name != "Gradient Boosting":
                    param["class_weight"] = [None] if use_smote else ["balanced"]

                rs = RandomizedSearchCV(model, param, cv=5, scoring=["recall", "f1"], refit="recall", random_state=42)
                search_result = rs.fit(X_train, y_train)
                model = search_result.best_estimator_
                y_pred = model.predict(X_test)
                mlflow.set_tags({"model_type": f"{model_name}-{experiment_name}", "smote_applied": use_smote})

                # Calculate metrics
                f1 = f1_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred)
                roc = roc_curve(y_test, model.predict_proba(X_test)[:,1])
                recall = recall_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Log metrics to MLflow
                mlflow.log_params(search_result.best_params_)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("roc_auc_score", roc_auc)
                mlflow.log_metric("recall_score", recall)
                mlflow.log_metric("precision_score", precision)
                mlflow.log_metric("accuracy_score", accuracy)
                mlflow.sklearn.log_model(model, model_name, registered_model_name=f"{model_name} - {experiment_name}")
                
                # Store the model for visualization
                report[model_name] = {"model": model, "y_pred": y_pred, "roc_auc_score": roc_auc, "roc_curve": roc}      
        return report


class MetricsVisualizations:
    """
    A class for visualizing model evaluation metrics and results

    Attributes:
    - models: A dictionary containing the trained models

    Methods:
    - __init__: Initializes the MetricsVisualizations object with a dictionary of models
    - create_subplots: Creates a figure and subplots with common settings
    - visualize_roc_curves: Visualizes ROC curves for each model
    - visualize_confusion_matrix: Visualizes confusion matrices for each model
    - plot_precision_recall_threshold: Plots precision and recall vs thresholds for each model
    - plot_feature_importance: Plots feature importance for each model
    """
    def __init__(self, models):
        """
        Initializes the MetricsVisualizations object with a dictionary of models

        Parameters:
        - models: A dictionary containing the trained models
        """
        self.models = models

    def create_subplots(self, rows, columns, figsize=(18,12)):
        """
        Creates a figure and subplots with common settings

        Parameters:
        - rows: Number of rows for subplots grid
        - columns: Number of columns for subplots grid
        - figsize: Figure size. Default is (18, 12)
        
        Returns:
        - fig: The figure object
        - ax: Array of axes objects
        """
        fig, ax = plt.subplots(rows, columns, figsize=figsize)
        ax = ax.ravel()
        return fig, ax

    def visualize_roc_curves(self):
        """
        Visualizes ROC curves for each model
        """
        plt.figure(figsize=(12, 6))
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")

        for model_name, model_data in self.models.items():
            model_roc_auc = model_data["roc_auc_score"]
            fpr, tpr, thresholds = model_data["roc_curve"]
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {model_roc_auc:.3f})")
        plt.legend()
        plt.show()

    def visualize_confusion_matrix(self, y_test, rows, columns):
        """
        Visualizes confusion matrices for each model

        Parameters:
        - y_test: True labels of the test data
        - rows: Number of rows for subplots
        - columns: Number of columns for subplots
        """
        fig, ax = self.create_subplots(rows, columns, figsize=(14, 10))
        for i, (model_name, model_data) in enumerate(self.models.items()):
            y_pred = model_data["y_pred"]
            matrix = confusion_matrix(y_test, y_pred)

            # Plot the first heatmap
            sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax[i * 2])
            ax[i * 2].set_title(f"Confusion Matrix: {model_name} - Absolute Values")
            ax[i * 2].set_xlabel("Predicted Values")
            ax[i * 2].set_ylabel("Observed values")

            # Plot the second heatmap
            sns.heatmap(matrix / np.sum(matrix), annot=True, fmt=".2%", cmap="Blues", ax=ax[i * 2 + 1])
            ax[i * 2 + 1].set_title(f"Relative Values")
            ax[i * 2 + 1].set_xlabel("Predicted Values")
            ax[i * 2 + 1].set_ylabel("Observed values")

        fig.tight_layout()
        plt.show()

    def plot_precision_recall_threshold(self, y_test, X_test, rows, columns):
        """
        Plots precision and recall vs thresholds for each model

        Parameters:
        - y_test: True labels of the test data
        - X_test: Features of the test data
        - rows: Number of rows for subplots
        - columns: Number of columns for subplots
        """
        fig, ax = self.create_subplots(rows, columns, figsize=(16, 6))
        for i, (model_name, model_data) in enumerate(self.models.items()):
            y_pred_prob = model_data["model"].predict_proba(X_test)[:,1]
            precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)

            # Plot Precision-Recall vs Thresholds for each model
            ax[i].set_title(f"Precision X Recall vs Thresholds - {model_name}")
            ax[i].plot(thresholds, precisions[:-1], "b--", label="Precision")
            ax[i].plot(thresholds, recalls[:-1], "g-", label="Recall")
            ax[i].plot([0.5, 0.5], [0, 1], 'k--')
            ax[i].set_ylabel("Score")
            ax[i].set_xlabel("Threshold")
            ax[i].legend(loc='center left')

            # Annotate precision and recall at 0.5 threshold
            y_pred = model_data["y_pred"]
            metrics = precision_recall_fscore_support(y_test, y_pred)
            precision = metrics[0][1]
            recall = metrics[1][1]
            ax[i].plot(0.5, precision, 'or')
            ax[i].annotate(f'{precision:.2f} precision', (0.51, precision))
            ax[i].plot(0.5, recall, 'or')
            ax[i].annotate(f'{recall:.2f} recall', (0.51, recall))
            ax[i].annotate('0.5 threshold', (0.39, -0.04))

        fig.tight_layout()
        plt.show()

    def plot_feature_importance(self, y_test, X_test, metric, rows, columns):
        """
        Plots feature importance for each model

        Parameters:
        - y_test: True labels of the test data
        - X_test: Features of the test data
        - metric: Metric used for evaluating feature importance
        - rows: Number of rows for subplots
        - columns: Number of columns for subplots
        """
        fig, ax = self.create_subplots(rows, columns, figsize=(16, 6))
        for i, (model_name, model_data) in enumerate(self.models.items()):
            # Calculate and sort permutation importances
            result = permutation_importance(model_data["model"], X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1, scoring=metric)
            sorted_importances_idx = result["importances_mean"].argsort()[::-1]

            # Select top 5 features
            top_features_idx = sorted_importances_idx[:5][::-1]
            top_features = X_test.columns[top_features_idx]
            importances = pd.DataFrame(result.importances[top_features_idx].T, columns=top_features)

            # Plot boxplot of feature importances
            box = importances.plot.box(vert=False, whis=10, ax=ax[i])
            box.set_title(f"Top 5 Feature Importance - {model_name}")
            box.axvline(x=0, color="k", linestyle="--")
            box.set_xlabel(f"Decay in {metric}")
            box.figure.tight_layout()

        fig.tight_layout()
        plt.show()