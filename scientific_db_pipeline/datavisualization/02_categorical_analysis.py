# ============================================================================
# CATEGORICAL FEATURES ANALYSIS WITH 5-CATEGORY ROT SYSTEM
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
    
    # Check for final_verdict column
    if 'final_verdict' not in df.columns:
        print("❌ Error: 'final_verdict' column not found!")
        print("Please upload a file with the 5-category final_verdict column")
        return None
    
    print(f"🎯 Found 5-category system: {df['final_verdict'].unique()}")
    
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
    """Analyze categorical features vs final_verdict"""
    print("\nCATEGORICAL FEATURES ANALYSIS:")
    print("="*50)
    
    categories = ['Very Low ROT', 'Low ROT', 'Medium ROT', 'High ROT', 'Very High ROT']
    df_filtered = df[df['final_verdict'].isin(categories)]
    
    categorical_features = ['mentions_dataset', 'mentions_metrics', 'has_github_link']
    
    # Create stacked bar charts
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    results = []
    
    for idx, feature in enumerate(categorical_features):
        ax = axes[idx]
        
        # Create stacked bar chart
        contingency_table = pd.crosstab(df_filtered[feature], df_filtered['final_verdict'], normalize='index') * 100
        contingency_table.plot(kind='bar', stacked=True, ax=ax, alpha=0.8)
        
        # Customize plot
        ax.set_title(f'{feature.replace("_", " ").title()} vs Final Verdict', fontweight='bold')
        ax.set_xlabel(feature.replace("_", " ").title())
        ax.set_ylabel('Percentage (%)')
        ax.legend(title='Final Verdict', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Calculate chi-square test
        contingency_matrix = pd.crosstab(df_filtered[feature], df_filtered['final_verdict'])
        chi2, p_value, dof, expected = chi2_contingency(contingency_matrix)
        
        # Calculate Cramer's V
        n = len(df_filtered)
        min_dim = min(contingency_matrix.shape) - 1
        cramer_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        
        results.append({
            'Feature': feature,
            'Chi2': chi2,
            'P_value': p_value,
            'Cramer_V': cramer_v,
            'Significant': p_value < 0.05
        })
    
    plt.tight_layout()
    plt.show()
    
    return results

def create_grouped_bar_charts(df):
    """Create grouped bar charts for categorical features"""
    print("\nGROUPED BAR CHARTS:")
    print("="*50)
    
    categories = ['Very Low ROT', 'Low ROT', 'Medium ROT', 'High ROT', 'Very High ROT']
    df_filtered = df[df['final_verdict'].isin(categories)]
    
    categorical_features = ['mentions_dataset', 'mentions_metrics', 'has_github_link']
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for idx, feature in enumerate(categorical_features):
        ax = axes[idx]
        
        # Calculate percentages for each category
        feature_stats = df_filtered.groupby('final_verdict')[feature].agg(['sum', 'count'])
        feature_stats['percentage'] = (feature_stats['sum'] / feature_stats['count']) * 100
        
        # Create grouped bar chart
        bars = ax.bar(range(len(categories)), feature_stats['percentage'], 
                     color=['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728', '#9467bd'], alpha=0.8)
        
        # Customize plot
        ax.set_title(f'{feature.replace("_", " ").title()} by Category', fontweight='bold')
        ax.set_xlabel('Final Verdict')
        ax.set_ylabel('Percentage (%)')
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars with smart positioning
        for bar, percentage in zip(bars, feature_stats['percentage']):
            # Smart positioning: put label inside bar if percentage is small, outside if large
            if percentage < 5:  # Small percentages
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                       f'{percentage:.1f}%', ha='center', va='center', fontweight='bold', 
                       color='white', fontsize=10)
            else:  # Larger percentages
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def print_statistical_results(results):
    """Print statistical test results"""
    print("\nSTATISTICAL TEST RESULTS:")
    print("="*50)
    
    print(f"{'Feature':<20} {'Chi2':<10} {'P-value':<10} {'Cramer V':<10} {'Significant':<10}")
    print("-" * 60)
    
    for result in results:
        significance = "✅ YES" if result['Significant'] else "❌ NO"
        print(f"{result['Feature']:<20} {result['Chi2']:<10.2f} {result['P_value']:<10.4f} "
              f"{result['Cramer_V']:<10.3f} {significance:<10}")
    
    # Summary
    significant_features = sum(1 for r in results if r['Significant'])
    print(f"\n📊 Summary:")
    print(f"   • Total features tested: {len(results)}")
    print(f"   • Significant features: {significant_features}")
    print(f"   • Success rate: {significant_features/len(results)*100:.1f}%")



def main():
    """Main analysis function"""
    print("🔬 CATEGORICAL FEATURES ANALYSIS WITH 5-CATEGORY ROT SYSTEM")
    print("="*70)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Create categorical features
    df = create_categorical_features(df)
    
    # Run analyses
    results = analyze_categorical_features(df)
    create_grouped_bar_charts(df)
    print_statistical_results(results)
    
    print("\n✅ Categorical analysis complete!")

if __name__ == "__main__":
    main() 