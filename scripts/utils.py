import pandas as pd
import numpy as np 
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    # Load Data
    file_path = '../data/MachineLearningRating_v3.txt'
    # Read the file with low_memory=False
    df = pd.read_csv(file_path, delimiter='|', low_memory=False)
    pd.set_option('display.max_columns', None)
    return df


def column_catagorize(df):
    # Categorize columns based on data types
    column_categories = {
        'numeric': [],
        'categorical': [],
        'datetime': [],
        'object': []
    }

    # Iterate over columns and categorize them
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            column_categories['numeric'].append(column)
        elif isinstance(df[column].dtype, pd.CategoricalDtype):
            column_categories['categorical'].append(column)
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            column_categories['datetime'].append(column)
        elif pd.api.types.is_object_dtype(df[column]):
            column_categories['object'].append(column)

    # Print categorized columns
    for category, columns in column_categories.items():
        print(f"\n {category.capitalize()} columns: {columns} ")
    
    return column_categories




def to_categorical(df):
    # List of columns to convert to categorical
    columns_to_convert = ['Citizenship', 'LegalType', 'Title', 'Language', 'Bank', 'AccountType',
        'MaritalStatus', 'Gender', 'Country', 'Province', 'MainCrestaZone', 'PostalCode', 'PolicyID',
        'SubCrestaZone', 'ItemType', 'VehicleType', 'make', 'Model', 'bodytype', 'UnderwrittenCoverID',
        'AlarmImmobiliser', 'TrackingDevice', 'Cylinders', 'NumberOfDoors', 'mmcode',
        'NewVehicle', 'WrittenOff', 'Rebuilt', 'Converted', 'RegistrationYear',
        'TermFrequency', 'ExcessSelected', 'CoverCategory', 'CoverType',
        'CoverGroup', 'Section', 'Product', 'StatutoryClass',
        'StatutoryRiskType']

    # Convert to categorical data type
    for column in columns_to_convert:
        df[column] = df[column].astype('category')



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

def handling_missing_values(df):
    # Drop missing values and columns 

    # Drop rows with null values less than 1%
    dff =  df.dropna(subset=['VehicleIntroDate','CapitalOutstanding','mmcode', 'VehicleType', 'make', 'kilowatts', 'NumberOfDoors', 'bodytype', 'cubiccapacity', 'Cylinders', 'Model'  ])

    # Drop columns with 100% and 99.9% null values 
    dff =  dff.drop(columns=['NumberOfVehiclesInFleet', 'CrossBorder'])

    # Filling the catagorical data types with mode

    # List of columns to fill with mode
    columns_to_fill = ['Rebuilt', 'Converted', 'WrittenOff', 'NewVehicle', 'Bank', 'AccountType', 'Gender', 'MaritalStatus']

    # Fill each column with its mode
    for column in columns_to_fill:
        mode_value = dff[column].mode()[0] 
        dff.fillna({column: mode_value}, inplace=True) 

    # Filling numerical data 
    dff['CustomValueEstimate'] = dff['CustomValueEstimate'].fillna(dff['CustomValueEstimate'].mean())
    dff['CapitalOutstanding'] = dff['CapitalOutstanding'].fillna(dff['CapitalOutstanding'].mode())


    return dff 


# Function to detect outliers using IQR and count them
def count_outliers_iqr(df):
    outlier_counts = {}
    lower_bounds = {}
    upper_bounds = {}

    for column in df.select_dtypes(include=['number']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        outlier_count = df[(df[column] < lower_bound) | (df[column] > upper_bound)].shape[0]
        
        outlier_counts[column] = outlier_count
        lower_bounds[column] = lower_bound
        upper_bounds[column] = upper_bound
    
    return lower_bounds, upper_bounds, outlier_counts



# Function to detect and cap outliers using IQR
def cap_outliers_iqr(dff):
    df = dff.copy()
    lower_bounds = {}
    upper_bounds = {}

    for column in df.select_dtypes(include=['number']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds for capping
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap the outliers
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        
    
    return df

# Example usage:
# df = pd.DataFrame(...)  # Your DataFrame
# lower_bounds, upper_bounds = cap_outliers_iqr(df)







def plot_box_outliers(dff, lower_bounds, upper_bounds, outlier_counts):
    # Create box plots for columns with outliers
    # Prepare to create box plots for columns with outliers
    num_columns = len([column for column, count in outlier_counts.items() if count > 0])
    num_rows = (num_columns + 1) // 2  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, 2, figsize=(12, num_rows * 5))  # Create subplots

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Create box plots for columns with outliers
    plot_index = 0
    for column, count in outlier_counts.items():
        if count > 0:  # Only plot columns with outliers
            sns.boxplot(x=dff[column], ax=axes[plot_index], color='skyblue')
            
            # Adding lines for upper and lower bounds
            axes[plot_index].axvline(x=upper_bounds[column], color='r', linestyle='--', label='Upper Bound')
            axes[plot_index].axvline(x=lower_bounds[column], color='g', linestyle='--', label='Lower Bound')
            
            axes[plot_index].set_title(f'Box Plot of {column} with Outlier Bounds')
            axes[plot_index].set_xlabel(column)
            axes[plot_index].legend()
            
            plot_index += 1

    # Hide any unused subplots
    for i in range(plot_index, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()





def plot_scatter_outliers(dff, lower_bounds, upper_bounds, outlier_counts):
    # Prepare to create scatter plots for columns with outliers
    num_columns = len([column for column, count in outlier_counts.items() if count > 0])
    num_rows = (num_columns + 1) // 2  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, 2, figsize=(12, num_rows * 5))  # Create subplots

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Create scatter plots for columns with outliers
    plot_index = 0
    for column, count in outlier_counts.items():
        if count > 0:  # Only plot columns with outliers
            axes[plot_index].scatter(dff.index, dff[column], color='skyblue', label='Data Points')
            
            # Adding lines for upper and lower bounds
            axes[plot_index].axhline(y=upper_bounds[column], color='r', linestyle='--', label='Upper Bound')
            axes[plot_index].axhline(y=lower_bounds[column], color='g', linestyle='--', label='Lower Bound')
            
            axes[plot_index].set_title(f'Scatter Plot of {column} with Outlier Bounds')
            axes[plot_index].set_xlabel('Index')
            axes[plot_index].set_ylabel(column)
            axes[plot_index].legend()
            
            plot_index += 1

    # Hide any unused subplots
    for i in range(plot_index, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()






