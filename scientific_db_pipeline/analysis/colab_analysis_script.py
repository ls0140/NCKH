# Scientific Paper Database - EDA Analysis Script for Google Colab
# Copy and paste this entire script into a Colab cell and run it
# 
# IMPORTANT: Before running this script, install packages in a separate cell:
# !pip install pandas numpy matplotlib seaborn scipy scikit-learn plotly

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, kruskal
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

print("Libraries imported successfully!")

# Load data
print("Loading data...")
try:
    # Try to load the main dataset
    df = pd.read_csv('all_papers_with_rot_groups.csv')
    print(f"✓ Main dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    print("⚠ Main dataset not found. Please upload 'all_papers_with_rot_groups.csv'")
    df = None

try:
    # Try to load summary data
    summary_df = pd.read_csv('analysis_summary.csv')
    print(f"✓ Summary data loaded: {summary_df.shape[0]} rows, {summary_df.shape[1]} columns")
except FileNotFoundError:
    print("⚠ Summary data not found. Please upload 'analysis_summary.csv'")
    summary_df = None

if df is not None:
    print("\nDataset preview:")
    print(df.head())
    
    print("\nDataset info:")
    print(df.info())
    
    print("\nMissing values:")
    print(df.isnull().sum())

# Data Overview and Quality Assessment
if df is not None:
    print("=== DATASET OVERVIEW ===")
    print(f"Total papers: {len(df)}")
    print(f"Features: {len(df.columns)}")
    
    # ROT score distribution
    if 'rot_score' in df.columns:
        print(f"\nROT Score Statistics:")
        print(df['rot_score'].describe())
        
        print(f"\nROT Groups Distribution:")
        if 'rot_group' in df.columns:
            print(df['rot_group'].value_counts())
    
    # Feature types
    print(f"\n=== FEATURE TYPES ===")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
    
    # Data quality
    print(f"\n=== DATA QUALITY ===")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    quality_df = pd.DataFrame({
        'Missing_Count': missing_data,
        'Missing_Percent': missing_percent,
        'Data_Type': df.dtypes,
        'Unique_Values': df.nunique()
    })
    
    print(quality_df.sort_values('Missing_Percent', ascending=False))

# ROT Score Analysis
if df is not None and 'rot_score' in df.columns:
    # ROT Score Distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Histogram
    axes[0, 0].hist(df['rot_score'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('ROT Score Distribution')
    axes[0, 0].set_xlabel('ROT Score')
    axes[0, 0].set_ylabel('Frequency')
    
    # Box plot
    axes[0, 1].boxplot(df['rot_score'])
    axes[0, 1].set_title('ROT Score Box Plot')
    axes[0, 1].set_ylabel('ROT Score')
    
    # Q-Q plot
    stats.probplot(df['rot_score'].dropna(), dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
    
    # Cumulative distribution
    sorted_rot = np.sort(df['rot_score'].dropna())
    y = np.arange(1, len(sorted_rot) + 1) / len(sorted_rot)
    axes[1, 1].plot(sorted_rot, y, marker='o', markersize=2)
    axes[1, 1].set_title('Cumulative Distribution')
    axes[1, 1].set_xlabel('ROT Score')
    axes[1, 1].set_ylabel('Cumulative Probability')
    
    plt.tight_layout()
    plt.show()
    
    # ROT Groups analysis
    if 'rot_group' in df.columns:
        print("\n=== ROT GROUPS ANALYSIS ===")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Group distribution
        group_counts = df['rot_group'].value_counts()
        axes[0].pie(group_counts.values, labels=group_counts.index, autopct='%1.1f%%')
        axes[0].set_title('ROT Groups Distribution')
        
        # Group statistics
        group_stats = df.groupby('rot_group')['rot_score'].agg(['count', 'mean', 'std', 'min', 'max'])
        print(group_stats)
        
        # Box plot by group
        df.boxplot(column='rot_score', by='rot_group', ax=axes[1])
        axes[1].set_title('ROT Scores by Group')
        axes[1].set_xlabel('ROT Group')
        axes[1].set_ylabel('ROT Score')
        
        plt.tight_layout()
        plt.show()

# Categorical Feature Analysis
if df is not None and 'rot_group' in df.columns:
    # Analyze categorical features
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_features = [col for col in categorical_features if col != 'rot_group']
    
    if categorical_features:
        print(f"Analyzing {len(categorical_features)} categorical features...")
        
        # Create subplots
        n_features = len(categorical_features)
        n_cols = 2
        n_rows = (n_features + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, feature in enumerate(categorical_features):
            row = i // n_cols
            col = i % n_cols
            
            # Skip if too many unique values
            if df[feature].nunique() > 20:
                axes[row, col].text(0.5, 0.5, f'{feature}\n(Too many unique values: {df[feature].nunique()})', 
                                   ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].set_title(f'{feature} - Too Many Categories')
                continue
            
            # Create contingency table
            contingency = pd.crosstab(df[feature], df['rot_group'], normalize='index')
            
            # Plot stacked bar chart
            contingency.plot(kind='bar', stacked=True, ax=axes[row, col])
            axes[row, col].set_title(f'{feature} vs ROT Groups')
            axes[row, col].set_xlabel(feature)
            axes[row, col].set_ylabel('Proportion')
            axes[row, col].tick_params(axis='x', rotation=45)
            axes[row, col].legend(title='ROT Group')
        
        # Hide empty subplots
        for i in range(n_features, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # Statistical tests for categorical features
        print("\n=== STATISTICAL TESTS FOR CATEGORICAL FEATURES ===")
        
        for feature in categorical_features:
            if df[feature].nunique() <= 20:
                try:
                    # Chi-square test
                    contingency = pd.crosstab(df[feature], df['rot_group'])
                    chi2, p_value, dof, expected = chi2_contingency(contingency)
                    
                    print(f"\n{feature}:")
                    print(f"  Chi-square statistic: {chi2:.4f}")
                    print(f"  p-value: {p_value:.4f}")
                    print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
                    
                except Exception as e:
                    print(f"\n{feature}: Error in statistical test - {e}")
    else:
        print("No categorical features found for analysis.")

# Numeric Feature Analysis
if df is not None and 'rot_group' in df.columns:
    # Analyze numeric features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [col for col in numeric_features if col != 'rot_score']
    
    if numeric_features:
        print(f"Analyzing {len(numeric_features)} numeric features...")
        
        # Create subplots for each feature
        n_features = len(numeric_features)
        n_cols = 2
        n_rows = (n_features + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, feature in enumerate(numeric_features):
            row = i // n_cols
            col = i % n_cols
            
            # Box plot by ROT group
            df.boxplot(column=feature, by='rot_group', ax=axes[row, col])
            axes[row, col].set_title(f'{feature} by ROT Groups')
            axes[row, col].set_xlabel('ROT Group')
            axes[row, col].set_ylabel(feature)
        
        # Hide empty subplots
        for i in range(n_features, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # Statistical tests for numeric features
        print("\n=== STATISTICAL TESTS FOR NUMERIC FEATURES ===")
        
        for feature in numeric_features:
            try:
                # Group the data by ROT group
                groups = [group[feature].dropna() for name, group in df.groupby('rot_group')]
                
                # One-way ANOVA
                f_stat, p_value = f_oneway(*groups)
                
                # Kruskal-Wallis test (non-parametric alternative)
                h_stat, kw_p_value = kruskal(*groups)
                
                print(f"\n{feature}:")
                print(f"  ANOVA F-statistic: {f_stat:.4f}")
                print(f"  ANOVA p-value: {p_value:.4f}")
                print(f"  ANOVA significant: {'Yes' if p_value < 0.05 else 'No'}")
                print(f"  Kruskal-Wallis H-statistic: {h_stat:.4f}")
                print(f"  Kruskal-Wallis p-value: {kw_p_value:.4f}")
                print(f"  Kruskal-Wallis significant: {'Yes' if kw_p_value < 0.05 else 'No'}")
                
            except Exception as e:
                print(f"\n{feature}: Error in statistical test - {e}")
    else:
        print("No numeric features found for analysis.")

# Correlation Analysis
if df is not None:
    # Correlation analysis
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) > 1:
        # Calculate correlations
        correlation_matrix = numeric_df.corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        # Focus on ROT score correlations
        if 'rot_score' in correlation_matrix.columns:
            rot_correlations = correlation_matrix['rot_score'].sort_values(ascending=False)
            
            print("\n=== ROT SCORE CORRELATIONS ===")
            print(rot_correlations)
            
            # Plot top correlations
            top_correlations = rot_correlations[rot_correlations != 1.0].head(10)
            
            plt.figure(figsize=(10, 6))
            top_correlations.plot(kind='barh')
            plt.title('Top 10 Features Correlated with ROT Score')
            plt.xlabel('Correlation Coefficient')
            plt.tight_layout()
            plt.show()
    else:
        print("Not enough numeric features for correlation analysis.")

# Feature Quality Assessment
if df is not None:
    print("=== FEATURE QUALITY ASSESSMENT ===")
    
    # Create feature quality report
    quality_report = []
    
    for column in df.columns:
        if column == 'rot_group':
            continue
            
        feature_info = {
            'Feature': column,
            'Data_Type': str(df[column].dtype),
            'Missing_Count': df[column].isnull().sum(),
            'Missing_Percent': (df[column].isnull().sum() / len(df)) * 100,
            'Unique_Values': df[column].nunique(),
            'Zero_Count': (df[column] == 0).sum() if df[column].dtype in ['int64', 'float64'] else 0,
            'Zero_Percent': ((df[column] == 0).sum() / len(df)) * 100 if df[column].dtype in ['int64', 'float64'] else 0
        }
        
        # Add correlation with ROT score if numeric
        if df[column].dtype in ['int64', 'float64'] and 'rot_score' in df.columns:
            correlation = df[column].corr(df['rot_score'])
            feature_info['ROT_Correlation'] = correlation
        else:
            feature_info['ROT_Correlation'] = None
        
        quality_report.append(feature_info)
    
    quality_df = pd.DataFrame(quality_report)
    
    # Display quality report
    print("\nFeature Quality Summary:")
    print(quality_df.sort_values('Missing_Percent', ascending=False))
    
    # Identify problematic features
    print("\n=== PROBLEMATIC FEATURES ===")
    
    # High missing values
    high_missing = quality_df[quality_df['Missing_Percent'] > 50]
    if not high_missing.empty:
        print(f"\nFeatures with >50% missing values ({len(high_missing)}):")
        print(high_missing[['Feature', 'Missing_Percent']].to_string(index=False))
    
    # Low variance (mostly zeros)
    low_variance = quality_df[(quality_df['Zero_Percent'] > 80) & (quality_df['Data_Type'].isin(['int64', 'float64']))]
    if not low_variance.empty:
        print(f"\nFeatures with >80% zeros ({len(low_variance)}):")
        print(low_variance[['Feature', 'Zero_Percent']].to_string(index=False))
    
    # High cardinality categorical features
    high_cardinality = quality_df[(quality_df['Unique_Values'] > 50) & 
                                  (quality_df['Data_Type'].isin(['object', 'category']))]
    if not high_cardinality.empty:
        print(f"\nCategorical features with >50 unique values ({len(high_cardinality)}):")
        print(high_cardinality[['Feature', 'Unique_Values']].to_string(index=False))
    
    # Strong correlations with ROT
    strong_correlations = quality_df[(quality_df['ROT_Correlation'].notna()) & 
                                     (abs(quality_df['ROT_Correlation']) > 0.3)]
    if not strong_correlations.empty:
        print(f"\nFeatures with strong correlation to ROT score (|r| > 0.3) ({len(strong_correlations)}):")
        print(strong_correlations[['Feature', 'ROT_Correlation']].sort_values('ROT_Correlation', key=abs, ascending=False).to_string(index=False))

# Summary and Recommendations
if df is not None:
    print("=== ANALYSIS SUMMARY ===")
    
    print(f"\n📊 Dataset Overview:")
    print(f"   • Total papers: {len(df)}")
    print(f"   • Features: {len(df.columns)}")
    print(f"   • Numeric features: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"   • Categorical features: {len(df.select_dtypes(include=['object', 'category']).columns)}")
    
    if 'rot_score' in df.columns:
        print(f"\n🎯 ROT Score Analysis:")
        print(f"   • Mean ROT score: {df['rot_score'].mean():.3f}")
        print(f"   • Standard deviation: {df['rot_score'].std():.3f}")
        print(f"   • Range: {df['rot_score'].min():.3f} to {df['rot_score'].max():.3f}")
        
        if 'rot_group' in df.columns:
            group_dist = df['rot_group'].value_counts()
            print(f"   • Group distribution: {dict(group_dist)}")
    
    print(f"\n🔍 Data Quality Issues:")
    missing_features = quality_df[quality_df['Missing_Percent'] > 0]
    if not missing_features.empty:
        print(f"   • Features with missing data: {len(missing_features)}")
    
    zero_features = quality_df[(quality_df['Zero_Percent'] > 50) & (quality_df['Data_Type'].isin(['int64', 'float64']))]
    if not zero_features.empty:
        print(f"   • Features with >50% zeros: {len(zero_features)}")
    
    print(f"\n💡 Recommendations:")
    print(f"   1. Review features with high missing values")
    print(f"   2. Investigate features with mostly zero values")
    print(f"   3. Consider feature engineering for low-variance features")
    print(f"   4. Focus on features with strong ROT correlations for modeling")
    print(f"   5. Validate categorical feature encoding")
    
    print(f"\n✅ Analysis complete! Review the visualizations and statistics above.")

print("\n" + "="*50)
print("SCRIPT COMPLETED SUCCESSFULLY!")
print("="*50) 