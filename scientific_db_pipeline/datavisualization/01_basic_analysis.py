# ============================================================================
# BASIC DATA ANALYSIS WITH CURRENT DATA
# ============================================================================
# File: datavisualization/01_basic_analysis.py
# Purpose: Analyze ROT score distribution, publication year, and citation count

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import ttest_ind
import warnings
import io
from google.colab import files

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """Load data from uploaded CSV file"""
    print("📁 Please upload your CSV file...")
    uploaded = files.upload()
    
    filename = list(uploaded.keys())[0]
    print(f"📄 Uploaded file: {filename}")
    
    df = pd.read_csv(io.BytesIO(uploaded[filename]))
    print(f"✅ Data loaded successfully!")
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    
    return df

def analyze_rot_distribution(df):
    """Analyze ROT score distribution"""
    print("ROT SCORE DISTRIBUTION:")
    print("="*50)
    
    print(f"   • Mean: {df['rot_score'].mean():.2f}")
    print(f"   • Median: {df['rot_score'].median():.2f}")
    print(f"   • Min: {df['rot_score'].min():.2f}")
    print(f"   • Max: {df['rot_score'].max():.2f}")
    
    # Create ROT distribution plots
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Histogram
    plt.subplot(1, 2, 1)
    plt.hist(df['rot_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('ROT Score Distribution (Histogram)', fontweight='bold')
    plt.xlabel('ROT Score')
    plt.ylabel('Number of Papers')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Box plot by group
    plt.subplot(1, 2, 2)
    df.boxplot(column='rot_score', by='rot_group')
    plt.title('ROT Score by Group (Box Plot)', fontweight='bold')
    plt.xlabel('ROT Group')
    plt.ylabel('ROT Score')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_publication_year(df):
    """Analyze publication year vs ROT group"""
    print("\nPUBLICATION YEAR ANALYSIS:")
    print("="*50)
    
    year_stats = df.groupby('rot_group')['publication_year'].describe()
    print(year_stats)
    
    # Publication year plot
    plt.figure(figsize=(10, 6))
    df.boxplot(column='publication_year', by='rot_group')
    plt.title('Publication Year by ROT Group (Box Plot)', fontweight='bold')
    plt.xlabel('ROT Group')
    plt.ylabel('Publication Year')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Statistical test
    high_rot = df[df['rot_group'] == 'High ROT']['publication_year']
    low_rot = df[df['rot_group'] == 'Low ROT']['publication_year']
    
    t_stat, p_value = ttest_ind(high_rot, low_rot)
    print(f"\nT-test results:")
    print(f"   • T-statistic: {t_stat:.4f}")
    print(f"   • P-value: {p_value:.4f}")
    print(f"   • Significant: {'Yes' if p_value < 0.05 else 'No'}")

def analyze_citation_count(df):
    """Analyze citation count vs ROT group"""
    print("\nCITATION COUNT ANALYSIS:")
    print("="*50)
    
    citation_stats = df.groupby('rot_group')['citation_count'].describe()
    print(citation_stats)
    
    # Citation count plot
    plt.figure(figsize=(10, 6))
    df.boxplot(column='citation_count', by='rot_group')
    plt.title('Citation Count by ROT Group (Box Plot)', fontweight='bold')
    plt.xlabel('ROT Group')
    plt.ylabel('Citation Count')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Statistical test
    high_rot = df[df['rot_group'] == 'High ROT']['citation_count']
    low_rot = df[df['rot_group'] == 'Low ROT']['citation_count']
    
    t_stat, p_value = ttest_ind(high_rot, low_rot)
    print(f"\nT-test results:")
    print(f"   • T-statistic: {t_stat:.4f}")
    print(f"   • P-value: {p_value:.4f}")
    print(f"   • Significant: {'Yes' if p_value < 0.05 else 'No'}")

def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    print("\nCORRELATION ANALYSIS:")
    print("="*50)
    
    numeric_cols = ['rot_score', 'publication_year', 'citation_count']
    correlation = df[numeric_cols].corr()
    print(correlation)
    
    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, square=True)
    plt.title('Correlation Matrix (Heatmap)', fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print high correlations
    print("\nHigh Correlations (|r| > 0.5):")
    high_corr = []
    for i in range(len(correlation.columns)):
        for j in range(i+1, len(correlation.columns)):
            corr_val = correlation.iloc[i, j]
            if abs(corr_val) > 0.5:
                high_corr.append({
                    'Feature1': correlation.columns[i],
                    'Feature2': correlation.columns[j],
                    'Correlation': round(corr_val, 3)
                })
    
    if high_corr:
        for corr in high_corr:
            print(f"   • {corr['Feature1']} vs {corr['Feature2']}: {corr['Correlation']}")
    else:
        print("   No high correlations found.")

def main():
    """Main function to run all analyses"""
    print("BASIC DATA ANALYSIS WITH CURRENT DATA")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Check if rot_group exists
    if 'rot_group' not in df.columns:
        print("❌ Error: 'rot_group' column not found!")
        print("Available columns:", list(df.columns))
        return
    
    # Run analyses
    analyze_rot_distribution(df)
    analyze_publication_year(df)
    analyze_citation_count(df)
    create_correlation_heatmap(df)
    
    print("\n✅ Basic analysis completed!")

if __name__ == "__main__":
    main() 