#!/bin/bash

import os
import pandas as pd
import argparse
import time
import subprocess
import sys
import numpy as np
import re
import glob
import shutil
from datetime import date
from datetime import datetime

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
        "ID",	"Q1",	"Q2",	"Q3a",	"Q3b",	"Q3c",	"Q3d",	"Q4a",	"Q4b",	"Q4c", "Q5",	"Q6",	"Q7",	"Q8",	
        "Q9a",	"Q9b",	"Q9c",	"Q9d",	"Q9e",	"Q9f",	"Q9g",	"Q9h",	"Q9i",	"Q9j",
        "Q10a",	"Q10b",	"Q10c",	"Q10d",	"Q10e",	"Q10f",	"Q10g",	"Q10h",	"Q10i",	"Q11a",	"Q11b",	"Q11c",	"Q11d"
    ]
    
    # Loop through the CSV files to import them
    for file_name in csv_files:
        file_path = os.path.join(path, file_name)

        df = pd.read_csv(file_path) 
        df.columns = expected_columns  

        # Reset index after dropping the first row
        df = df.reset_index(drop=True) 

        # Check for expected columns
        if not all(col in df.columns for col in expected_columns):
            print_error(f"Error: The following expected columns are missing in {file_name}: {set(expected_columns) - set(df.columns)}")
            return None 

        # Check that values in columns P1 through P16b are either empty or numeric
        for col in expected_columns[1:]:  # Skip 'ID'
            if col in df.columns:
                if not df[col].apply(lambda x: pd.isnull(x) or isinstance(x, (int, float))).all():
                    print_error(f"Error: Column '{col}' in {file_name} should contain only empty or numeric values.")
                    return None 
        
        data_dict[file_name] = df

    # Check if all CSV files were successfully imported
    if csv_file_count == len(data_dict):
        print(f"{csv_file_count} csv files imported.")
        return data_dict  
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
    print("Check data integrity...")

    names_list = []
    has_issues = False  # Track if any issues are found
    problematic_dfs = set()  # To keep track of DataFrames with issues

    # Duplicates in df
    if len(names_list) != len(set(names_list)):
        print_warning("Duplicates in provided dataframes")
        has_issues = True
        problematic_dfs.update(names_list)

    # Missing data
    for key, df in data_dict.items():
        missing_values = df[df.isna().any(axis=1)]
        if not missing_values.empty:
            for index, row in missing_values.iterrows():
                missing_columns = row.index[row.isna()]
                print(f"ID {key}: missing in {', '.join(missing_columns)}")
                problematic_dfs.add(key)
                has_issues = True

    # Correct time format: 7 days, 24 hours, 60 minutes
    columns_to_check = ["Q1",	"Q2",	"Q3a",	"Q3b",	"Q3c",	"Q3d",	"Q4a",	"Q4b",	"Q4c", "Q5",	"Q6",	"Q7",	"Q8",	
        "Q9a",	"Q9b",	"Q9c",	"Q9d",	"Q9e",	"Q9f",	"Q9g",	"Q9h",	"Q9i",	"Q9j",
        "Q10a",	"Q10b",	"Q10c",	"Q10d",	"Q10e",	"Q10f",	"Q10g",	"Q10h",	"Q10i",	"Q11a",	"Q11b",	"Q11c",	"Q11d"]

    acceptable_ranges = {
    'Q1': (1, 5),
    'Q2': (1, 5),
    'Q3a': (1, 2),
    'Q3b': (1, 2),
    'Q3c': (1, 2),
    'Q3d': (1, 2),
    'Q4a': (1, 2),
    'Q4b': (1, 2),
    'Q4c': (1, 2),
    'Q5': (1, 5),
    'Q6': (1, 6),
    'Q7': (1, 5),
    'Q8': (1, 5),
    'Q9a': (1,3), 
    'Q9b': (1,3), 
    'Q9c': (1,3), 
    'Q9d': (1,3), 
    'Q9e': (1,3), 
    'Q9f': (1,3), 
    'Q9g': (1,3),
    'Q9h': (1,3), 
    'Q9i': (1,3), 
    'Q9j': (1,3), 
    'Q10a': (1,6), 
    'Q10b': (1,6), 
    'Q10c': (1,6), 
    'Q10d': (1,6), 
    'Q10e': (1,6), 
    'Q10f': (1,6),
    'Q10g': (1,6), 
    'Q10h': (1,6), 
    'Q10i': (1,6), 
    'Q11a': (1,5), 
    'Q11b': (1,5), 
    'Q11c': (1,5), 
    'Q11d': (1,5)
    }

    aberrant_data = {}

    for key, df in data_dict.items():
        for index, row in df.iterrows():
            for col in columns_to_check:
                value = row[col]
                if value < acceptable_ranges[col][0] or value > acceptable_ranges[col][1]:
                    if key not in aberrant_data:
                        aberrant_data[key] = []
                    aberrant_data[key].append((index, col))
                    
    for key, values in aberrant_data.items():
        print_warning(f"Dataframe {key}:")
        for index, col in values:
            print_warning(f"Wrong value in {col} at {index}. Please check the correct time format: 7 days, 24 hours, 60 minutes")

    if has_issues:
        print("This has to be checked manually in raw data.")
    if not has_issues:
        print("No issues found in data integrity checks.")

    return problematic_dfs  # Return the DataFrames with integrity issues

def recalibrate_scores(data_dict, names_with_issues):
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
    print("Recalibrating scores for some items...")
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
    """Transform raw scales to a 0-100 scale."""
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
    """Computes composite scores based on the transformed scale DataFrames."""
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

# def save_results(transformed_scale_dict, final_df, saving_path_ind, args):
#     """
#     Save results to the specified output directory.

#     Parameters:
#     - transformed_scale_dict (dict): Dictionary of DataFrames to save independently.
#     - args (argparse.Namespace): Command line arguments including --ind.
#     - final_df (DataFrame): Concatenated DataFrame to save if not saving independently.
#     - saving_path_ind (str): Path to save the results.
#     """
#     # Ensure the output directory exists
#     os.makedirs(saving_path_ind, exist_ok=True)

#     if args.ind:
#         # Save each DataFrame separately
#         for key, df in transformed_scale_dict.items():
#             output_file = os.path.join(saving_path_ind, f"{key}_results.csv")
#             df.to_csv(output_file, index=False)
#             print(f"Saved independent file: {output_file}")
#     else:
#         # Save the concatenated DataFrame
#         final_output_file = os.path.join(saving_path_ind, "final_results.csv")
#         final_df.to_csv(final_output_file, index=False)
#         print(f"Saved concatenated file: {final_output_file}")


def main():
    start_time = time.time()

    default_data_path = "./data"
    default_results_path = "./results"

    parser = argparse.ArgumentParser(description="Import CSV files from a directory.")

    # Precise input directory (default = ./data)
    parser.add_argument(
        "-d", "--directory", 
        type=str, 
        default=default_data_path, 
        help="Directory containing CSV files (default: './data')"
    )
    # Precise output directory (default = ./results)
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default=default_results_path, 
        help="Directory to save results (default: './results')"
    )
    # Precise if you want individual results (default = 1 concatenated file)
    parser.add_argument(
        "--ind", 
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
        recalibrate_scores(data_dict, problematic_dfs)
        replace_missing_by_mean(data_dict) 
        scale_dict = compute_raw_scales(data_dict) 
        replacement_dicts = {'Mean Current Health': {5:100, 4.4:84, 3.4:61, 2:25, 1:0}}
        transformed_scale_dict = transform_raw_scales_to_0_100(scale_dict, replacement_dicts)
        composite_scores_dict = compute_composite_scores(transformed_scale_dict)
        print(composite_scores_dict)
        # final_df = pd.concat(transformed_scale_dict.values(), ignore_index=True)
        # save_results(transformed_scale_dict, final_df, saving_path_ind, args.ind)

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Done in {elapsed_time:.2f} seconds.")
    print('')

if __name__ == "__main__":
    main()