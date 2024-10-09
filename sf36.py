#!/bin/bash

import os
import pandas as pd
import argparse
import time
import subprocess
import sys
import numpy as np

def print_warning(message):
    """Prints a warning message in yellow"""
    print(f"\033[93mWarning: {message}\033[0m")

def print_error(message):
    """Prints an error message in red"""
    print(f"\033[91mError: {message}\033[0m")

def install_requirements():
    """
    This functions checks if the necessary packages are installed, and install them. 
    They are present in "requirements.txt"
    """
    print("Checking for the presence of correct packages...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt', '-q'])
    except subprocess.CalledProcessError as e:
        error_message = f"Error during package installation: {e}"
        print_error(error_message)

def import_csv_files(path):
    """
    This function imports all .csv files from a given directory, 
    checks that files and directory exist, and files are correctly structured,
    and returns a dictionary containing the DataFrames.

    Args:
        path (str): The path of the directory containing .csv files.

    Returns:
        dict: A dictionary with file names as keys and corresponding DataFrames as values.
    """
    print(f"Importing data from {path}...")

    # Check if the path exists
    if not os.path.exists(path):
        error_message = f"The path {path} does not exist."
        print_error(error_message)
        return {}

    # List of CSV files in the directory
    csv_files = [file_name for file_name in os.listdir(path) if file_name.endswith('.csv')]
    csv_file_count = len(csv_files)

    # Check if the directory contains any CSV files
    if csv_file_count == 0:
        error_message = f"The directory {path} is empty or contains no .csv files."
        print_error(error_message)
        return {}

    data_dict = {}
    expected_columns = [
        "ID", "Q1", "Q2", "Q3a", "Q3b", "Q3c", "Q3d", "Q4a", "Q4b", "Q4c", "Q5", "Q6", "Q7", "Q8",
        "Q9a", "Q9b", "Q9c", "Q9d", "Q9e", "Q9f", "Q9g", "Q9h", "Q9i", "Q9j",
        "Q10a", "Q10b", "Q10c", "Q10d", "Q10e", "Q10f", "Q10g", "Q10h", "Q10i", "Q11a", "Q11b", "Q11c", "Q11d"
    ]
    
    # Loop through the CSV files to import them
    for file_name in csv_files:
        file_path = os.path.join(path, file_name)

        df = pd.read_csv(file_path)

        # Check for missing or extra columns
        missing_columns = set(expected_columns) - set(df.columns)
        extra_columns = set(df.columns) - set(expected_columns)
        
        if missing_columns:
            print_error(f"Error: The file {file_name} is missing columns: {missing_columns}")
            return None  # Stop and return None if columns are missing
        if extra_columns:
            print_error(f"Error: The file {file_name} has extra columns: {extra_columns}")
            return None  # Stop and return None if there are extra columns

        # Check that values in columns Q1 through Q11d are either empty or numeric
        for col in expected_columns[1:]:  # Skip 'ID'
            if not df[col].apply(lambda x: pd.isnull(x) or isinstance(x, (int, float))).all():
                print_error(f"Error: Column '{col}' in {file_name} should contain only empty or numeric values.")
                return None  # Stop if column values are not valid
        
        data_dict[file_name] = df

    # Check if all CSV files were successfully imported
    if csv_file_count == len(data_dict):
        print(f"{csv_file_count} CSV files imported successfully.")
    else:
        warning_message = f"{csv_file_count} .csv files found but only {len(data_dict)} imported."
        print_warning(warning_message)
    
    return data_dict
    
def check_data_integrity(data_dict):
    """
    This function checks for duplicates, validates values in specific columns, 
    and prints the findings for data integrity in the imported DataFrames.

    Args:
        data_dict (dict): A dictionary containing DataFrames keyed by their file names.
    
    Returns:
        set: A set of DataFrame names that have integrity issues.
    """
    print("Checking data integrity...")

    problematic_dfs = set()  # To keep track of DataFrames with issues
    has_issues = False  # Track if any issues are found

    # Ignoring missing data for now, we'll handle them later
    for key, df in data_dict.items():
        # Only check rows where there are no missing values
        complete_data = df.dropna()
        if complete_data.empty:
            continue  # Skip if the entire DataFrame is empty after dropping missing values
        
    # Validating ranges in specific columns
    columns_to_check = [
        "Q1", "Q2", "Q3a", "Q3b", "Q3c", "Q3d", "Q4a", "Q4b", "Q4c", "Q5", "Q6", "Q7", "Q8",
        "Q9a", "Q9b", "Q9c", "Q9d", "Q9e", "Q9f", "Q9g", "Q9h", "Q9i", "Q9j",
        "Q10a", "Q10b", "Q10c", "Q10d", "Q10e", "Q10f", "Q10g", "Q10h", "Q10i", 
        "Q11a", "Q11b", "Q11c", "Q11d"
    ]

    acceptable_ranges = {
        'Q1': (1, 5), 'Q2': (1, 5), 'Q3a': (1, 2), 'Q3b': (1, 2), 'Q3c': (1, 2),
        'Q3d': (1, 2), 'Q4a': (1, 2), 'Q4b': (1, 2), 'Q4c': (1, 2), 'Q5': (1, 5),
        'Q6': (1, 6), 'Q7': (1, 5), 'Q8': (1, 5), 'Q9a': (1, 3), 'Q9b': (1, 3),
        'Q9c': (1, 3), 'Q9d': (1, 3), 'Q9e': (1, 3), 'Q9f': (1, 3), 'Q9g': (1, 3),
        'Q9h': (1, 3), 'Q9i': (1, 3), 'Q9j': (1, 3), 'Q10a': (1, 6), 'Q10b': (1, 6),
        'Q10c': (1, 6), 'Q10d': (1, 6), 'Q10e': (1, 6), 'Q10f': (1, 6), 'Q10g': (1, 6),
        'Q10h': (1, 6), 'Q10i': (1, 6), 'Q11a': (1, 5), 'Q11b': (1, 5), 'Q11c': (1, 5), 
        'Q11d': (1, 5)
    }

    for index, row in complete_data.iterrows():
        for col in columns_to_check:
            value = row[col]
            if pd.isnull(value):
                continue  # Skip NaN values, as they are checked separately
            if not isinstance(value, (int, float)):
                print(f"DataFrame {key}, Row {index}: Non-numeric value in {col}")
                problematic_dfs.add(key)
                has_issues = True
            elif value < acceptable_ranges[col][0] or value > acceptable_ranges[col][1]:
                print(f"DataFrame {key}, Row {index}: Value {value} in {col} is out of range.")
                problematic_dfs.add(key)
                has_issues = True
    
    if has_issues:
        error_message = "Integrity issues found. Manual check required."
        print_error(error_message)
        return
    else:
        pass

    return problematic_dfs if problematic_dfs else set()  # Return the DataFrames with integrity issues

def reorganize_columns(data_dict):
    """
    This function reorganizes and renames the columns in the DataFrames contained in the data_dict.
    
    Args:
        data_dict (dict): A dictionary with DataFrame names as keys and corresponding DataFrames as values.
    
    Returns:
        dict: A dictionary with the reorganized DataFrames.
    """
    print("Reorganizing columns order...")
    # List of columns with a prefix to be renamed
    columns_to_prefix = ['Q1', 'Q2', 'Q3a', 'Q3b', 'Q3c', 'Q3d', 'Q4a', 'Q4b', 'Q4c', 'Q5', 'Q6', 'Q7', 'Q8', 
                         'Q9a', 'Q9b', 'Q9c', 'Q9d', 'Q9e', 'Q9f', 'Q9g', 'Q9h', 'Q9i', 'Q9j', 
                         'Q10a', 'Q10b', 'Q10c', 'Q10d', 'Q10e', 'Q10f', 'Q10g', 'Q10h', 'Q10i', 
                         'Q11a', 'Q11b', 'Q11c', 'Q11d']

    # Mapping of old columns to new columns
    mappings = {
        'Old_Q9': 'New_Q3',
        'Old_Q3': 'New_Q4',
        'Old_Q4': 'New_Q5',
        'Old_Q5': 'New_Q6',
        'Old_Q6': 'New_Q7',
        'Old_Q7': 'New_Q8',
        'Old_Q10': 'New_Q9',
        'Old_Q8': 'New_Q10'
    }

    # New order of the columns
    new_order = ['ID', '1', '2', '3a', '3b', '3c', '3d', '3e', '3f', '3g', '3h', '3i', '3j', 
                 '4a', '4b', '4c', '4d', '5a', '5b', '5c', '6', '7', '8', 
                 '9a', '9b', '9c', '9d', '9e', '9f', '9g', '9h', '9i', '10', '11a', '11b', '11c', '11d']

    # Loop through each DataFrame in data_dict
    for key, df in data_dict.items():
        # Add the prefix "Old_" to the relevant columns
        for col in columns_to_prefix:
            if col in df.columns:
                df.rename(columns={col: 'Old_' + col}, inplace=True)

        # Apply mappings to rename columns with new prefixes
        for col in df.columns:
            for old, new in mappings.items():
                if col.startswith(old):
                    new_col = col.replace(old, new)
                    df.rename(columns={col: new_col}, inplace=True)

        # Remove the 'Old_' and 'New_' prefixes from the columns
        df.columns = df.columns.str.replace(r'(Old|New)_Q', '', regex=True)

        # Reorganize the columns in the new order
        try:
            df = df[new_order]
        except KeyError as e:
            print(f"Error: Missing columns {e} in DataFrame {key}. Please check the column mappings or data.")
        
        # Update the DataFrame in the dictionary
        data_dict[key] = df

    return data_dict

def recalibrate_scores(data_dict, names_with_issues):
    """
    """
    print("Recalibrating scores for some items...")

    # Reverse / recalibrate score for some items
    replacement_dicts = {
        '1': {1: 5.0, 2: 4.4, 3: 3.4, 4: 2.0, 5: 1.0},
        '6': {1: 5, 2: 4, 3: 3, 4: 2, 5: 1},
        '7': {1: 6.0, 2: 5.4, 3: 4.2, 4: 3.1, 5: 2.2, 6: 1.0},
        '9a': {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1},
        '9e': {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1},
        '9d': {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1},
        '9h': {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1},
        '11b': {1: 5, 2: 4, 3: 3, 4: 2, 5: 1},
        '11d': {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}
    }

    if names_with_issues is None:
        names_with_issues = set()  # Initialize as an empty set if None
    
    for key, df in data_dict.items():
        if key in names_with_issues:  # Skip this DataFrame if it has issues
            print(f"Skipping {key} due to integrity issues.")
            continue
        
        for col, replacement_dict in replacement_dicts.items():
            if col in df.columns:
                df[col] = df[col].replace(replacement_dict)
                data_dict[key] = df
                
        if '7' in df.columns and '8' in df.columns:
            condition = df['7'].isna()
            replacement_dict_conditional = {
                1: np.where(condition, 6.0, 5),
                2: np.where(condition, 4.75, 4),
                3: np.where(condition, 3.5, 3),
                4: np.where(condition, 2.25, 2),
                5: np.where(condition, 1.0, 1)
            }
            df['8'] = df['8'].replace(replacement_dict_conditional)
            data_dict[key] = df    

def replace_missing_by_mean(data_dict):
    column_sets = [
        ['3a', '3b', '3c', '3d', '3e', '3f', '3g', '3h', '3i', '3j'],
        ['4a', '4b', '4c', '4d'],
        ['7', '8'],
        ['1', '11a', '11b', '11c', '11d'],
        ['9a', '9e', '9g', '9i'],
        ['6', '10'],
        ['5a', '5b', '5c'],
        ['9b', '9c', '9d', '9f', '9h']
    ]
 
    print("Replacing missing data by mean...")
    
    for key, df in data_dict.items():
        for columns in column_sets:
            # Check if all columns in the set exist in the DataFrame
            valid_columns = [col for col in columns if col in df.columns]
            
            if valid_columns:
                # Compute row-wise mean ignoring NaN
                row_mean = df[valid_columns].mean(axis=1)
                
                # Fill missing values with the row mean
                df[valid_columns] = df[valid_columns].apply(lambda x: x.fillna(row_mean))

        # After attempting to fill, check for remaining missing values
        missing_values = df[df.isna().any(axis=1)]
        
        if not missing_values.empty:
            for index, row in missing_values.iterrows():
                missing_columns = row.index[row.isna()].tolist()
                print(f"ID {key}: Missing in columns: {', '.join(missing_columns)}")

def compute_raw_scales(data_dict):
    """
    Computes raw scale values based on specific column groupings.

    Parameters:
    - data_dict: Dictionary where keys are identifiers and values are pandas DataFrames containing the data.

    Returns:
    - scale_dict: Dictionary where keys are the same as input and values are DataFrames with computed raw scales.
    """
    print("Computing raw scales...")
    # Define scale columns mapping
    scale_columns = {
        'Physical Functioning': ['3a', '3b', '3c', '3d', '3e', '3f', '3g', '3h', '3i', '3j'],
        'Role-Physical': ['4a', '4b', '4c', '4d'],
        'Bodily-Pain': ['7', '8'],
        'General Health': ['1', '11a', '11b', '11c', '11d'],
        'Vitality': ['9a', '9e', '9g', '9i'],
        'Social Functioning': ['6', '10'],
        'Role-Emotional': ['5a', '5b', '5c'],
        'Mental Health': ['9b', '9c', '9d', '9f', '9h'],
        'Reported Health Transition': ['2'],
        'Mean Current Health': ['1']
    }

    scale_dict = {}

    # Iterate over each DataFrame in data_dict
    for key, df in data_dict.items():
        # Create a new DataFrame to hold scale values
        scale_df = pd.DataFrame(columns=scale_columns.keys())
        
        # Compute each scale as the sum of its corresponding columns
        for scale, columns in scale_columns.items():
            # Ensure only valid columns are used
            valid_columns = [col for col in columns if col in df.columns]
            
            if valid_columns:
                scale_df[scale] = df[valid_columns].astype(float).sum(axis=1)
            else:
                scale_df[scale] = pd.Series([None] * len(df))  # Handle missing columns with NaNs

        # Store the computed scale DataFrame in scale_dict
        scale_dict[key] = scale_df

    return scale_dict

def transform_raw_scales_to_0_100(scale_dict, replacement_dicts):
    """
    Transform raw scales to a 0-100 scale.
    """
    print("Transforming raw scales into 0-100 scales...")
    transformed_scale_dict = {}

    for key, df in scale_dict.items():
        df['Physical Functioning'] = (df['Physical Functioning'] - 10) / 20 * 100
        df['Role-Physical'] = (df['Role-Physical'] - 4) / 4 * 100
        df['Bodily-Pain'] = (df['Bodily-Pain'] - 2) / 10 * 100
        df['General Health'] = (df['General Health'] - 5) / 20 * 100
        df['Vitality'] = (df['Vitality'] - 4) / 20 * 100
        df['Social Functioning'] = (df['Social Functioning'] - 2) / 8 * 100
        df['Role-Emotional'] = (df['Role-Emotional'] - 3) / 3 * 100
        df['Mental Health'] = (df['Mental Health'] - 5) / 25 * 100
        df['Reported Health Transition'] = (df['Reported Health Transition'] - 1) / 5 * 100

        # Apply replacements
        for col, replacement_dict in replacement_dicts.items():
            if col in df.columns:
                df[col] = df[col].replace(replacement_dict)

        transformed_scale_dict[key] = df
    
    return transformed_scale_dict

def compute_composite_scores(transformed_scale_dict):
    """
    Computes composite scores based on the transformed scale DataFrames.
    """
    print("Computing composite scores...")

    composite_scores_dict = {}

    for key, df in transformed_scale_dict.items():
        # Compute PHYSICAL and MENTAL composite scores
        df['PHYSICAL'] = df[['Physical Functioning', 
                             'Role-Physical', 
                             'Bodily-Pain', 
                             'General Health']].mean(axis=1)
        
        df['MENTAL'] = df[['Vitality', 
                           'Social Functioning', 
                           'Role-Emotional', 
                           'Mental Health']].mean(axis=1)

        # Compute GLOBAL composite score
        df['GLOBAL'] = df[['Physical Functioning', 'Role-Physical', 'Bodily-Pain',
                           'General Health', 'Vitality', 'Social Functioning', 
                           'Role-Emotional', 'Mental Health']].mean(axis=1)

        # Insert the ID column at the beginning
        df.insert(0, 'ID', key)

        composite_scores_dict[key] = df

    return composite_scores_dict

def merge_data(data_dict, composite_scores_dict):
    """
    This function merges DataFrames from `data_dict` and `composite_scores_dict` based on matching keys 
    and then concatenates all the merged DataFrames vertically (axis=0).
    
    Args:
        data_dict (dict): A dictionary of DataFrames with file names as keys.
        composite_scores_dict (dict): Another dictionary of DataFrames with file names as keys.
    
    Returns:
        pd.DataFrame: A single concatenated DataFrame containing all rows and columns.
    """
    print("Merging reorganised and filled data with composite scores...")
    
    merged_dict = {}  # List to store merged DataFrames

    # Loop through the keys in data_dict
    for key in data_dict:
        if key in composite_scores_dict:  # Ensure the key exists in both dictionaries
            # Merge DataFrames on axis=1 (columns)
            merged_df = pd.merge(data_dict[key], composite_scores_dict[key], left_index=True, right_index=True, how='outer')
            merged_df = merged_df.drop(columns=['ID_y'], errors='ignore')
            merged_df.rename(columns={'ID_x': 'ID'}, inplace=True)
            merged_dict[key] = merged_df
    
    return merged_dict

def save_results(merged_dict, saving_path_ind, args):
    """
    Save results to the specified output directory.

    Parameters:
    - transformed_scale_dict (dict): Dictionary of DataFrames to save independently.
    - final_df (DataFrame): Concatenated DataFrame to save if not saving independently.
    - saving_path_ind (str): Path to save the results.
    - args (argparse.Namespace): Command line arguments including --ind.
    """
    print("Saving results...")

    # Ensure the output directory exists
    os.makedirs(saving_path_ind, exist_ok=True)

    if args.ind:
        # Save each DataFrame separately
        for key, df in merged_dict.items():
            output_file = os.path.join(saving_path_ind, f"{key}")
            df.to_csv(output_file, index=False)
            print(f"Independent file saved to: {output_file}")
    else:
        # Save the concatenated DataFrame
        final_output_file = os.path.join(saving_path_ind, "concatenated_results.csv")
        final_df = pd.concat(merged_dict, axis=0, ignore_index=True)
        final_df.to_csv(final_output_file, index=False)
        print(f"Concatenated file saved to: {final_output_file}")

def main():
    start_time = time.time()

    default_data_path = "./data"
    default_results_path = "./results"

    parser = argparse.ArgumentParser(description="Import CSV files from a directory.")

    # Input directory (default = ./data)
    parser.add_argument(
        "-d", "--directory", 
        type=str, 
        default=default_data_path, 
        help="Directory containing CSV files (default: './data')"
    )
    # Output directory (default = ./results)
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default=default_results_path, 
        help="Directory to save results (default: './results')"
    )
    # Save individual results or concatenated file
    parser.add_argument(
        "-ind", 
        action="store_true", 
        help="Save independent files"
    )

    args = parser.parse_args()

    # Use the provided directory or the default one
    path = args.directory
    saving_path_ind = args.output

    # Create output directory if it doesn't exist
    if not os.path.exists(saving_path_ind):
        os.makedirs(saving_path_ind)

    data_dict = import_csv_files(path)
    if data_dict:
        problematic_dfs = check_data_integrity(data_dict)
        reorganize_columns(data_dict)
        recalibrate_scores(data_dict, problematic_dfs)
        replace_missing_by_mean(data_dict) 
        scale_dict = compute_raw_scales(data_dict) 
        replacement_dicts = {'Mean Current Health': {5:100, 4.4:84, 3.4:61, 2:25, 1:0}}
        transformed_scale_dict = transform_raw_scales_to_0_100(scale_dict, replacement_dicts)
        composite_scores_dict = compute_composite_scores(transformed_scale_dict)
        merged_dict = merge_data(data_dict, composite_scores_dict)
        save_results(merged_dict, saving_path_ind, args)

        end_time = time.time()
        elapsed_time = end_time - start_time

    print(f"Done in {elapsed_time:.2f} seconds.")
    print('')

if __name__ == "__main__":
    main()