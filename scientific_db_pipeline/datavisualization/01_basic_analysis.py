# ============================================================================
# BASIC DATA ANALYSIS WITH 5-CATEGORY ROT SYSTEM
# ============================================================================
# File: datavisualization/01_basic_analysis.py
# Purpose: Analyze ROT score distribution, publication year, and citation count

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import f_oneway
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
    
    # Check for final_verdict column
    if 'final_verdict' not in df.columns:
        print("❌ Error: 'final_verdict' column not found!")
        print("Please upload a file with the 5-category final_verdict column")
        return None
    
    print(f"🎯 Found 5-category system: {df['final_verdict'].unique()}")
    
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
    plt.figure(figsize=(15, 5))
    
    # Subplot 1: Histogram
    plt.subplot(1, 3, 1)
    plt.hist(df['rot_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('ROT Score Distribution (Histogram)', fontweight='bold')
    plt.xlabel('ROT Score')
    plt.ylabel('Number of Papers')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Box plot by final_verdict
    plt.subplot(1, 3, 2)
    categories = ['Very Low ROT', 'Low ROT', 'Medium ROT', 'High ROT', 'Very High ROT']
    df_filtered = df[df['final_verdict'].isin(categories)]
    df_filtered.boxplot(column='rot_score', by='final_verdict')
    plt.title('ROT Score by Category (Box Plot)', fontweight='bold')
    plt.xlabel('Final Verdict')
    plt.ylabel('ROT Score')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Violin plot
    plt.subplot(1, 3, 3)
    sns.violinplot(data=df_filtered, x='final_verdict', y='rot_score')
    plt.title('ROT Score Distribution by Category', fontweight='bold')
    plt.xlabel('Final Verdict')
    plt.ylabel('ROT Score')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def analyze_publication_year(df):
    """Analyze publication year vs final_verdict"""
    print("\nPUBLICATION YEAR ANALYSIS:")
    print("="*50)
    
    categories = ['Very Low ROT', 'Low ROT', 'Medium ROT', 'High ROT', 'Very High ROT']
    df_filtered = df[df['final_verdict'].isin(categories)]
    
    year_stats = df_filtered.groupby('final_verdict')['publication_year'].describe()
    print(year_stats)
    
    # Publication year plot
    plt.figure(figsize=(12, 6))
    df_filtered.boxplot(column='publication_year', by='final_verdict')
    plt.title('Publication Year by Category (Box Plot)', fontweight='bold')
    plt.xlabel('Final Verdict')
    plt.ylabel('Publication Year')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Statistical test (ANOVA for multiple groups)
    groups = [df_filtered[df_filtered['final_verdict'] == cat]['publication_year'] 
              for cat in categories]
    f_stat, p_value = f_oneway(*groups)
    print(f"\nANOVA results:")
    print(f"   • F-statistic: {f_stat:.4f}")
    print(f"   • P-value: {p_value:.4f}")
    print(f"   • Significant: {'Yes' if p_value < 0.05 else 'No'}")

def analyze_citation_count(df):
    """Analyze citation count vs final_verdict"""
    print("\nCITATION COUNT ANALYSIS:")
    print("="*50)
    
    categories = ['Very Low ROT', 'Low ROT', 'Medium ROT', 'High ROT', 'Very High ROT']
    df_filtered = df[df['final_verdict'].isin(categories)]
    
    citation_stats = df_filtered.groupby('final_verdict')['citation_count'].describe()
    print(citation_stats)
    
    # Citation count plot
    plt.figure(figsize=(12, 6))
    df_filtered.boxplot(column='citation_count', by='final_verdict')
    plt.title('Citation Count by Category (Box Plot)', fontweight='bold')
    plt.xlabel('Final Verdict')
    plt.ylabel('Citation Count')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Statistical test (ANOVA for multiple groups)
    groups = [df_filtered[df_filtered['final_verdict'] == cat]['citation_count'] 
              for cat in categories]
    f_stat, p_value = f_oneway(*groups)
    print(f"\nANOVA results:")
    print(f"   • F-statistic: {f_stat:.4f}")
    print(f"   • P-value: {p_value:.4f}")
    print(f"   • Significant: {'Yes' if p_value < 0.05 else 'No'}")

def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    print("\nCORRELATION ANALYSIS:")
    print("="*50)
    
    # Select numeric columns
    numeric_cols = ['rot_score', 'publication_year', 'citation_count']
    correlation_matrix = df[numeric_cols].corr()
    
    print("Correlation Matrix:")
    print(correlation_matrix)
    
    # Create heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5)
    plt.title('Correlation Heatmap', fontweight='bold')
    plt.tight_layout()
    plt.show()

def main():
    """Main analysis function"""
    print("🔬 BASIC DATA ANALYSIS WITH 5-CATEGORY ROT SYSTEM")
    print("="*60)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Run analyses
    analyze_rot_distribution(df)
    analyze_publication_year(df)
    analyze_citation_count(df)
    create_correlation_heatmap(df)
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main() 