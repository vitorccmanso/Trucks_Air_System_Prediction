import pandas as pd

def categorize_skew(skew_value):
        """
        Categorizes the skewness of a numerical value

        This function takes a skewness value as input and categorizes it into one of three types:
        - "Negatively Skewed" if the skewness is less than -0.5
        - "Positively Skewed" if the skewness is greater than 0.5
        - "Normal" if the skewness is between -0.5 and 0.5 inclusive

        Parameters:
        - skew_value (float): The skewness value to be categorized

        Returns:
        - str: A string indicating the category of the skewness
        """
        if skew_value < -0.5:
            return "Negatively Skewed"
        elif skew_value > 0.5:
            return "Positively Skewed"
        else:
            return "Normal"

def extract_skewness_summary(data, target):
    """
    Extracts skewness information from the DataFrame data and creates a summary DataFrame

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing numerical data

    Returns:
    - skew_df (pd.DataFrame): DataFrame with skewness values and categories
    - summary_df (pd.DataFrame): Summary DataFrame with counts and mean skewness for each category
    """
    # Calculate skewness for each column
    skewness_values = data.drop(columns=[target]).skew()

    # Create DataFrame with skewness values
    skew_df = pd.DataFrame({"skew value": skewness_values})

    # Categorize skewness
    skew_df["skew_type"] = skew_df["skew value"].apply(categorize_skew)

    # Calculate summary DataFrame
    summary_df = skew_df.groupby("skew_type").agg(count=("skew value", "size"), skew_mean=("skew value", "mean"))

    return skew_df, summary_df