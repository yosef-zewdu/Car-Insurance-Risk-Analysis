import pandas as pd
import numpy as np 
from scipy.stats import zscore






# Missing values 
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Get the count of non-null values for each column
    non_null_counts = df.notnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # dtype of missing values
    mis_val_dtype = df.dtypes
    
    # missing values unique number
    unique_counts = df.nunique()
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent, mis_val_dtype, non_null_counts, unique_counts], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns={0: 'Missing Values', 1: '% of Total Values', 2: 'Dtype',3: 'Values', 4: 'Unique Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")
    

    # Return the dataframe with missing information
    return mis_val_table_ren_columns

# Non missing values 
def non_missing_values_table(df):
    # Identify non-missing value columns
    non_missing_columns = df.columns[df.notnull().all()]

    # Prepare data for the summary table
    summary_data = {
        'Column Name': [],
        'Data Type': [],
        'Total Values': [],
        'Unique Values': []
    }

    # Populate the summary data
    for column in non_missing_columns:
        summary_data['Column Name'].append(column)
        summary_data['Data Type'].append(df[column].dtype)
        summary_data['Total Values'].append(len(df[column]))
        summary_data['Unique Values'].append(df[column].nunique())

    # Create a DataFrame from the summary data
    summary_df = pd.DataFrame(summary_data)

    # Print some summary information
    print("Your selected DataFrame has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(summary_df.shape[0]) +
          " columns that have no missing values.")

    # Return the summary DataFrame
    return summary_df




# Fix outliers 
def fix_outlier(df, column):
    df[column] = np.where(df[column] > df[column].quantile(0.95), df[column].median(), df[column])
    return df[column]


# Remove outliers 
def remove_outliers(df, column_to_process, z_threshold=3):
    # Apply outlier removal to the specified column
    z_scores = zscore(df[column_to_process])
    outlier_column = column_to_process + '_Outlier'
    df[outlier_column] = (np.abs(z_scores) > z_threshold).astype(int)
    df = df[df[outlier_column] == 0]  # Keep rows without outliers

    # Drop the outlier column as it's no longer needed
    df = df.drop(columns=[outlier_column], errors='ignore')

    return df
