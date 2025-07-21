# ============================================================================
# CATEGORICAL FEATURES ANALYSIS
# ============================================================================
# File: datavisualization/02_categorical_analysis.py
# Purpose: Extract and analyze categorical features from abstracts

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency
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

def extract_features_from_abstract(abstract):
    """Extract categorical features from abstract text"""
    if pd.isna(abstract) or len(str(abstract)) < 10:
        return False, False, False
    
    abstract_lower = str(abstract).lower()
    
    # Check for dataset mentions
    dataset_keywords = ['dataset', 'data set', 'corpus', 'benchmark', 'evaluation set', 
                       'training data', 'test data', 'validation data']
    mentions_dataset = any(keyword in abstract_lower for keyword in dataset_keywords)
    
    # Check for metrics mentions
    metric_keywords = ['accuracy', 'precision', 'recall', 'f1', 'f1-score', 'auc', 'roc', 
                      'bleu', 'rouge', 'perplexity', 'mse', 'mae', 'rmse', 'r-squared']
    mentions_metrics = any(keyword in abstract_lower for keyword in metric_keywords)
    
    # Check for GitHub links
    has_github_link = 'github.com' in abstract_lower or 'github.io' in abstract_lower
    
    return mentions_dataset, mentions_metrics, has_github_link

def create_categorical_features(df):
    """Create categorical features from abstract data"""
    print("CREATING CATEGORICAL FEATURES FROM ABSTRACT:")
    print("="*50)
    
    # Extract features
    print("Extracting features from abstracts...")
    features = df['abstract'].apply(extract_features_from_abstract)
    df['mentions_dataset'] = [f[0] for f in features]
    df['mentions_metrics'] = [f[1] for f in features]
    df['has_github_link'] = [f[2] for f in features]
    
    # Show results
    print(f"✅ Created categorical features:")
    print(f"   • mentions_dataset: {df['mentions_dataset'].sum()} papers ({df['mentions_dataset'].mean()*100:.1f}%)")
    print(f"   • mentions_metrics: {df['mentions_metrics'].sum()} papers ({df['mentions_metrics'].mean()*100:.1f}%)")
    print(f"   • has_github_link: {df['has_github_link'].sum()} papers ({df['has_github_link'].mean()*100:.1f}%)")
    
    return df

def analyze_categorical_features(df):
    """Analyze categorical features vs ROT group"""
    print("\nCATEGORICAL FEATURES ANALYSIS:")
    print("="*50)
    
    categorical_features = ['mentions_dataset', 'mentions_metrics', 'has_github_link']
    
    # Create stacked bar charts
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    results = []
    
    for idx, feature in enumerate(categorical_features):
        ax = axes[idx]
        
        # Create stacked bar chart
        contingency_table = pd.crosstab(df[feature], df['rot_group'], normalize='index') * 100
        contingency_table.plot(kind='bar', stacked=True, ax=ax, 
                              color=['#ff7f0e', '#1f77b4'], alpha=0.8)
        
        # Customize plot
        ax.set_title(f'{feature}\n(Stacked Bar Chart)', fontweight='bold')
        ax.set_xlabel(feature)
        ax.set_ylabel('Percentage (%)')
        ax.legend(title='ROT Group')
        ax.tick_params(axis='x', rotation=0)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', fontsize=8)
        
        # Perform Chi-square test
        observed = pd.crosstab(df[feature], df['rot_group'])
        chi2, p_value, dof, expected = chi2_contingency(observed)
        
        # Calculate Cramer's V
        n = len(df)
        min_dim = min(observed.shape) - 1
        cramer_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        
        # Determine effect size
        if cramer_v > 0.5:
            effect_size = 'Large'
        elif cramer_v > 0.3:
            effect_size = 'Medium'
        else:
            effect_size = 'Small'
        
        results.append({
            'Feature': feature,
            'Chi2_Statistic': round(chi2, 4),
            'P_Value': round(p_value, 4),
            'Cramer_V': round(cramer_v, 4),
            'Effect_Size': effect_size,
            'Significant': p_value < 0.05
        })
    
    plt.tight_layout()
    plt.show()
    
    return results

def create_grouped_bar_charts(df):
    """Create grouped bar charts for categorical features"""
    print("\nCREATING GROUPED BAR CHARTS:")
    print("="*50)
    
    categorical_features = ['mentions_dataset', 'mentions_metrics', 'has_github_link']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, feature in enumerate(categorical_features):
        ax = axes[idx]
        
        # Create grouped bar chart
        contingency_table = pd.crosstab(df[feature], df['rot_group'])
        contingency_table.plot(kind='bar', ax=ax, 
                              color=['#ff7f0e', '#1f77b4'], alpha=0.8)
        
        # Customize plot
        ax.set_title(f'{feature}\n(Grouped Bar Chart)', fontweight='bold')
        ax.set_xlabel(feature)
        ax.set_ylabel('Count')
        ax.legend(title='ROT Group')
        ax.tick_params(axis='x', rotation=0)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fontsize=8)
    
    plt.tight_layout()
    plt.show()

def print_statistical_results(results):
    """Print statistical analysis results"""
    print("\n📊 STATISTICAL ANALYSIS RESULTS:")
    print("="*50)
    
    for result in results:
        status = "✅ SIGNIFICANT" if result['Significant'] else "❌ NOT SIGNIFICANT"
        print(f"\n{result['Feature']}: {status}")
        print(f"   • Chi-square: {result['Chi2_Statistic']}")
        print(f"   • P-value: {result['P_Value']} {'(p < 0.05)' if result['Significant'] else '(p >= 0.05)'}")
        print(f"   • Cramer's V: {result['Cramer_V']} ({result['Effect_Size']} effect)")
    
    # Summary
    significant_count = sum(1 for r in results if r['Significant'])
    print(f"\n📋 SUMMARY:")
    print(f"   • Total features analyzed: {len(results)}")
    print(f"   • Significant features: {significant_count}")
    print(f"   • Success rate: {significant_count/len(results)*100:.1f}%")
    
    if significant_count >= len(results) * 0.6:
        print(f"   • 🎉 CONCLUSION: Categories have GOOD quality for verdict prediction!")
    else:
        print(f"   • ⚠️ CONCLUSION: Categories need improvement for verdict prediction!")

def detailed_feature_distribution(df):
    """Show detailed distribution of features"""
    print("\nDETAILED FEATURE DISTRIBUTION:")
    print("="*50)
    
    for feature in ['mentions_dataset', 'mentions_metrics', 'has_github_link']:
        print(f"\n{feature}:")
        print(df.groupby(['rot_group', feature]).size().unstack(fill_value=0))
        print(f"Overall: {df[feature].sum()}/{len(df)} ({df[feature].mean()*100:.1f}%)")

def main():
    """Main function to run categorical analysis"""
    print("CATEGORICAL FEATURES ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Check if required columns exist
    if 'abstract' not in df.columns:
        print("❌ Error: 'abstract' column not found!")
        print("Available columns:", list(df.columns))
        return
    
    if 'rot_group' not in df.columns:
        print("❌ Error: 'rot_group' column not found!")
        print("Available columns:", list(df.columns))
        return
    
    # Create categorical features
    df = create_categorical_features(df)
    
    # Analyze categorical features
    results = analyze_categorical_features(df)
    
    # Create grouped bar charts
    create_grouped_bar_charts(df)
    
    # Print statistical results
    print_statistical_results(results)
    
    # Show detailed distribution
    detailed_feature_distribution(df)
    
    print("\n✅ Categorical analysis completed!")

if __name__ == "__main__":
    main() 