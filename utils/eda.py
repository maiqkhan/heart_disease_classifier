"""
EDA Utility Functions for Heart Disease Analysis
Author: Senior Data Scientist
Date: 2025-11-11

This module contains reusable functions for exploratory data analysis,
including data loading, validation, statistical tests, and quality checks.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DATA LOADING AND VALIDATION
# ============================================================================

def load_and_validate_data(filepath: str, expected_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load dataset and perform initial validation checks.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    expected_cols : list, optional
        List of expected column names for validation
    
    Returns:
    --------
    pd.DataFrame
        Loaded and validated dataframe
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Successfully loaded data from {filepath}")
        print(f"✓ Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        # Validate expected columns if provided
        if expected_cols:
            missing_cols = set(expected_cols) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing expected columns: {missing_cols}")
            print(f"✓ All {len(expected_cols)} expected columns present")
        
        return df
    
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def get_dataframe_info(df: pd.DataFrame, return_dict: bool = False) -> Optional[Dict]:
    """
    Display comprehensive dataframe information.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    return_dict : bool, default=False
        If True, return info as dictionary instead of printing
    
    Returns:
    --------
    dict or None
        If return_dict=True, returns dictionary with dataframe info
    """
    info_dict = {
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'duplicate_rows': df.duplicated().sum(),
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict()
    }
    
    if return_dict:
        return info_dict
    
    print("=" * 70)
    print("DATAFRAME INFORMATION")
    print("=" * 70)
    print(f"Shape: {info_dict['shape'][0]:,} rows × {info_dict['shape'][1]} columns")
    print(f"Memory Usage: {info_dict['memory_usage_mb']:.2f} MB")
    print(f"Duplicate Rows: {info_dict['duplicate_rows']:,}")
    print("\nColumn Data Types:")
    for col, dtype in info_dict['dtypes'].items():
        print(f"  {col:<20} {dtype}")
    print("=" * 70)


# ============================================================================
# DATA QUALITY ASSESSMENT
# ============================================================================

def assess_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate comprehensive data quality report.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    pd.DataFrame
        Quality report with metrics per column
    """
    quality_report = pd.DataFrame({
        'Column': df.columns,
        'Data_Type': df.dtypes.values,
        'Missing_Count': df.isnull().sum().values,
        'Missing_Percent': (df.isnull().sum() / len(df) * 100).values,
        'Unique_Count': [df[col].nunique() for col in df.columns],
        'Unique_Percent': [(df[col].nunique() / len(df) * 100) for col in df.columns],
    })
    
    # Add cardinality classification
    quality_report['Cardinality'] = quality_report['Unique_Percent'].apply(
        lambda x: 'High (>50%)' if x > 50 else 'Medium (5-50%)' if x > 5 else 'Low (<5%)'
    )
    
    # Add zeros count for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    quality_report['Zero_Count'] = [
        (df[col] == 0).sum() if col in numeric_cols else 0 
        for col in df.columns
    ]
    quality_report['Zero_Percent'] = [
        (df[col] == 0).sum() / len(df) * 100 if col in numeric_cols else 0 
        for col in df.columns
    ]
    
    return quality_report.round(2)


def detect_hidden_missing_values(df: pd.DataFrame, 
                                 suspicious_cols: Optional[List[str]] = None) -> Dict:
    """
    Detect potential hidden missing values (zeros, placeholders).
    Medical datasets often use 0 for missing blood pressure or cholesterol.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    suspicious_cols : list, optional
        Columns to check for suspicious zeros
    
    Returns:
    --------
    dict
        Dictionary with suspicious value findings
    """
    findings = {}
    
    if suspicious_cols is None:
        # Default medical columns that shouldn't be zero
        suspicious_cols = ['RestingBP', 'Cholesterol', 'MaxHR']
    
    for col in suspicious_cols:
        if col in df.columns:
            zero_count = (df[col] == 0).sum()
            zero_pct = (zero_count / len(df)) * 100
            
            if zero_count > 0:
                findings[col] = {
                    'zero_count': zero_count,
                    'zero_percent': round(zero_pct, 2),
                    'likely_missing': zero_pct > 1  # Flag if >1% are zeros
                }
    
    return findings


def check_domain_constraints(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Validate domain-specific constraints for medical data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with heart disease features
    
    Returns:
    --------
    dict
        Dictionary of constraint violations
    """
    violations = {}
    
    # Age constraints
    if 'Age' in df.columns:
        age_issues = []
        if (df['Age'] < 18).any():
            age_issues.append(f"{(df['Age'] < 18).sum()} records with Age < 18")
        if (df['Age'] > 100).any():
            age_issues.append(f"{(df['Age'] > 100).sum()} records with Age > 100")
        if age_issues:
            violations['Age'] = age_issues
    
    # Blood Pressure constraints
    if 'RestingBP' in df.columns:
        bp_issues = []
        if (df['RestingBP'] < 80).any() and (df['RestingBP'] > 0).any():
            bp_issues.append(f"{((df['RestingBP'] < 80) & (df['RestingBP'] > 0)).sum()} records with suspiciously low BP (<80)")
        if (df['RestingBP'] > 200).any():
            bp_issues.append(f"{(df['RestingBP'] > 200).sum()} records with very high BP (>200)")
        if bp_issues:
            violations['RestingBP'] = bp_issues
    
    # Cholesterol constraints
    if 'Cholesterol' in df.columns:
        chol_issues = []
        if (df['Cholesterol'] < 100).any() and (df['Cholesterol'] > 0).any():
            chol_issues.append(f"{((df['Cholesterol'] < 100) & (df['Cholesterol'] > 0)).sum()} records with very low cholesterol (<100)")
        if (df['Cholesterol'] > 400).any():
            chol_issues.append(f"{(df['Cholesterol'] > 400).sum()} records with very high cholesterol (>400)")
        if chol_issues:
            violations['Cholesterol'] = chol_issues
    
    # MaxHR constraints
    if 'MaxHR' in df.columns:
        hr_issues = []
        if (df['MaxHR'] < 60).any() and (df['MaxHR'] > 0).any():
            hr_issues.append(f"{((df['MaxHR'] < 60) & (df['MaxHR'] > 0)).sum()} records with very low MaxHR (<60)")
        if (df['MaxHR'] > 220).any():
            hr_issues.append(f"{(df['MaxHR'] > 220).sum()} records with MaxHR >220 (exceeds theoretical max)")
        if hr_issues:
            violations['MaxHR'] = hr_issues
    
    return violations


# ============================================================================
# STATISTICAL SUMMARY
# ============================================================================

def generate_summary_statistics(df: pd.DataFrame, 
                               include_percentiles: bool = True) -> pd.DataFrame:
    """
    Generate comprehensive statistical summary for numeric features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    include_percentiles : bool, default=True
        Whether to include additional percentiles
    
    Returns:
    --------
    pd.DataFrame
        Statistical summary
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if include_percentiles:
        percentiles = [.01, .05, .25, .50, .75, .95, .99]
    else:
        percentiles = [.25, .50, .75]
    
    summary = numeric_df.describe(percentiles=percentiles).T
    
    # Add additional statistics
    summary['skewness'] = numeric_df.skew()
    summary['kurtosis'] = numeric_df.kurtosis()
    summary['cv'] = (numeric_df.std() / numeric_df.mean()) * 100  # Coefficient of variation
    summary['iqr'] = summary['75%'] - summary['25%']
    
    return summary.round(3)


def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for categorical features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    pd.DataFrame
        Categorical features summary
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    summaries = []
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        summaries.append({
            'Column': col,
            'Unique_Values': df[col].nunique(),
            'Most_Frequent': value_counts.index[0] if len(value_counts) > 0 else None,
            'Most_Frequent_Count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
            'Most_Frequent_Percent': round((value_counts.iloc[0] / len(df)) * 100, 2) if len(value_counts) > 0 else 0,
            'Least_Frequent': value_counts.index[-1] if len(value_counts) > 0 else None,
            'Least_Frequent_Count': value_counts.iloc[-1] if len(value_counts) > 0 else 0,
            'Categories': ', '.join(map(str, value_counts.index[:5].tolist()))  # Top 5
        })
    
    return pd.DataFrame(summaries)


# ============================================================================
# TARGET VARIABLE ANALYSIS
# ============================================================================

def analyze_target_variable(df: pd.DataFrame, 
                           target_col: str = 'HeartDisease') -> Dict:
    """
    Comprehensive analysis of the target variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str, default='HeartDisease'
        Name of target column
    
    Returns:
    --------
    dict
        Dictionary with target variable metrics
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    target_counts = df[target_col].value_counts().sort_index()
    total = len(df)
    
    analysis = {
        'total_samples': total,
        'class_distribution': target_counts.to_dict(),
        'class_percentages': (target_counts / total * 100).round(2).to_dict(),
        'class_names': target_counts.index.tolist(),
        'is_binary': len(target_counts) == 2,
        'majority_class': target_counts.idxmax(),
        'minority_class': target_counts.idxmin(),
        'imbalance_ratio': round(target_counts.max() / target_counts.min(), 2),
        'is_balanced': abs(target_counts.iloc[0] - target_counts.iloc[1]) / total < 0.1 if len(target_counts) == 2 else None
    }
    
    return analysis


def print_target_summary(target_analysis: Dict):
    """
    Print formatted target variable summary.
    
    Parameters:
    -----------
    target_analysis : dict
        Output from analyze_target_variable function
    """
    print("=" * 70)
    print("TARGET VARIABLE ANALYSIS")
    print("=" * 70)
    print(f"Total Samples: {target_analysis['total_samples']:,}")
    print(f"Binary Classification: {target_analysis['is_binary']}")
    print("\nClass Distribution:")
    for class_name, count in target_analysis['class_distribution'].items():
        pct = target_analysis['class_percentages'][class_name]
        print(f"  Class {class_name}: {count:,} ({pct}%)")
    
    print(f"\nMajority Class: {target_analysis['majority_class']}")
    print(f"Minority Class: {target_analysis['minority_class']}")
    print(f"Imbalance Ratio: {target_analysis['imbalance_ratio']}:1")
    
    if target_analysis['is_balanced']:
        print("\n✓ Classes are relatively balanced (within 10% difference)")
    else:
        print("\n⚠ Classes are imbalanced - consider resampling strategies")
    print("=" * 70)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_feature_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Categorize features by type (numeric, categorical, binary).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    dict
        Dictionary with feature categories
    """
    feature_types = {
        'numeric': [],
        'categorical': [],
        'binary': []
    }
    
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            if df[col].nunique() == 2:
                feature_types['binary'].append(col)
            else:
                feature_types['numeric'].append(col)
        else:
            if df[col].nunique() == 2:
                feature_types['binary'].append(col)
            else:
                feature_types['categorical'].append(col)
    
    return feature_types


def calculate_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate detailed memory usage by column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    pd.DataFrame
        Memory usage report
    """
    memory_usage = df.memory_usage(deep=True)
    memory_df = pd.DataFrame({
        'Column': memory_usage.index,
        'Memory_Bytes': memory_usage.values,
        'Memory_MB': (memory_usage.values / 1024**2).round(4),
        'Percent_of_Total': ((memory_usage.values / memory_usage.sum()) * 100).round(2)
    })
    
    return memory_df.sort_values('Memory_Bytes', ascending=False)


def identify_constant_features(df: pd.DataFrame) -> List[str]:
    """
    Identify features with zero variance (constant values).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    list
        List of constant feature names
    """
    constant_features = []
    
    for col in df.columns:
        if df[col].nunique() == 1:
            constant_features.append(col)
    
    return constant_features


# ============================================================================
# DISPLAY HELPERS
# ============================================================================

def display_sample_data(df: pd.DataFrame, 
                       n_rows: int = 5, 
                       random: bool = False,
                       seed: int = 42) -> None:
    """
    Display sample rows from dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    n_rows : int, default=5
        Number of rows to display
    random : bool, default=False
        Whether to sample randomly
    seed : int, default=42
        Random seed for reproducibility
    """
    print(f"\n{'Random' if random else 'First'} {n_rows} rows of data:")
    print("=" * 70)
    
    if random:
        sample = df.sample(n=min(n_rows, len(df)), random_state=seed)
    else:
        sample = df.head(n_rows)
    
    print(sample.to_string())
    print("=" * 70)