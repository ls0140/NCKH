# ============================================================================
# FEATURE DISTRIBUTION ANALYSIS
# ============================================================================
# File: datavisualization/00_overall_analysis.py
# Purpose: Show detailed feature distribution tables

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
    """Extract features from abstract text with appropriate scoring"""
    if pd.isna(abstract) or len(str(abstract)) < 10:
        return 1, 1, 0
    
    abstract_lower = str(abstract).lower()
    
    # Score dataset mentions (1-10) - can have multiple mentions
    dataset_keywords = ['dataset', 'data set', 'corpus', 'benchmark', 'evaluation set', 
                       'training data', 'test data', 'validation data']
    dataset_count = sum(abstract_lower.count(keyword) for keyword in dataset_keywords)
    dataset_score = min(10, max(1, dataset_count + 1))  # At least 1, max 10
    
    # Score metrics mentions (1-10) - can have multiple mentions
    metric_keywords = ['accuracy', 'precision', 'recall', 'f1', 'f1-score', 'auc', 'roc', 
                      'bleu', 'rouge', 'perplexity', 'mse', 'mae', 'rmse', 'r-squared']
    metric_count = sum(abstract_lower.count(keyword) for keyword in metric_keywords)
    metric_score = min(10, max(1, metric_count + 1))  # At least 1, max 10
    
    # GitHub links (0 or 1) - binary feature
    has_github_link = 1 if ('github.com' in abstract_lower or 'github.io' in abstract_lower) else 0
    
    return dataset_score, metric_score, has_github_link

def create_categorical_features(df):
    """Create features from abstract data with 1-10 scoring"""
    print("CREATING FEATURES FROM ABSTRACT (1-10 SCORING):")
    print("="*50)
    
    # Extract features
    print("Extracting features from abstracts...")
    features = df['abstract'].apply(extract_features_from_abstract)
    df['dataset_score'] = [f[0] for f in features]
    df['metrics_score'] = [f[1] for f in features]
    df['has_github_link'] = [f[2] for f in features]
    
    # Show results
    print(f"✅ Created features with appropriate scoring:")
    print(f"   • dataset_score (1-10): avg={df['dataset_score'].mean():.2f}, std={df['dataset_score'].std():.2f}")
    print(f"   • metrics_score (1-10): avg={df['metrics_score'].mean():.2f}, std={df['metrics_score'].std():.2f}")
    print(f"   • has_github_link (0/1): {df['has_github_link'].sum()} papers ({df['has_github_link'].mean()*100:.1f}%)")
    
    return df

def detailed_feature_distribution(df):
    """Show detailed distribution of features with accuracy metrics"""
    print("\nDETAILED FEATURE DISTRIBUTION:")
    print("="*50)
    
    # Get categories dynamically from the data
    categories = sorted(df['final_verdict'].unique())
    df_filtered = df[df['final_verdict'].isin(categories)]
    
    # Handle mixed feature types
    numeric_features = ['dataset_score', 'metrics_score']
    binary_features = ['has_github_link']
    
    # Analyze numeric features (1-10 scoring)
    for feature in numeric_features:
        print(f"\n{feature.replace('_', ' ').title()} (1-10 Scoring):")
        print("-" * 30)
        
        # Show descriptive statistics by category
        print("Descriptive Statistics by Category:")
        stats_by_category = df_filtered.groupby('final_verdict')[feature].agg(['mean', 'std', 'min', 'max', 'count'])
        print(stats_by_category.round(2))
        
        # Show score distribution
        print(f"\nScore Distribution:")
        score_counts = df_filtered[feature].value_counts().sort_index()
        print(score_counts)
        
        # Calculate ANOVA for numeric features
        print(f"\nAccuracy Analysis:")
        print("-" * 20)
        
        from scipy.stats import f_oneway
        groups = [df_filtered[df_filtered['final_verdict'] == cat][feature].values for cat in categories]
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) >= 2:
            f_stat, p_value = f_oneway(*groups)
            
            # Calculate effect size (Eta-squared)
            ss_between = sum(len(g) * ((g.mean() - df_filtered[feature].mean()) ** 2) for g in groups)
            ss_total = sum((x - df_filtered[feature].mean()) ** 2 for x in df_filtered[feature])
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            # Calculate correlation with ROT score (if available)
            if 'rot_score' in df_filtered.columns:
                correlation = df_filtered[feature].corr(df_filtered['rot_score'])
            else:
                correlation = 0
            
            print(f"ANOVA F-statistic: {f_stat:.2f}")
            print(f"P-value: {p_value:.4f}")
            print(f"Effect size (Eta-squared): {eta_squared:.3f}")
            print(f"Correlation with ROT score: {correlation:.3f}")
            print(f"Statistical significance: {'✅ YES' if p_value < 0.05 else '❌ NO'}")
            
            # Overall feature usefulness
            if p_value < 0.05 and eta_squared > 0.06:
                print(f"Feature usefulness: ✅ GOOD (significant correlation)")
            elif p_value < 0.05:
                print(f"Feature usefulness: ⚠️ WEAK (significant but small effect)")
            else:
                print(f"Feature usefulness: ❌ POOR (no significant correlation)")
        else:
            print("Cannot perform ANOVA - insufficient groups")
            print(f"Feature usefulness: ❌ CANNOT TEST")
    
    # Analyze binary features (0/1)
    for feature in binary_features:
        print(f"\n{feature.replace('_', ' ').title()} (Binary 0/1):")
        print("-" * 30)
        
        # Create contingency table
        contingency_table = pd.crosstab(df_filtered['final_verdict'], df_filtered[feature], 
                                       margins=True, margins_name='Total')
        print("Contingency Table:")
        print(contingency_table)
        
        # Calculate percentages
        percentage_table = pd.crosstab(df_filtered['final_verdict'], df_filtered[feature], 
                                      normalize='index') * 100
        print(f"\nPercentages by category:")
        print(percentage_table.round(1))
        
        # Calculate chi-square test for binary features
        print(f"\nAccuracy Analysis:")
        print("-" * 20)
        
        from scipy.stats import chi2_contingency
        contingency_matrix = pd.crosstab(df_filtered[feature], df_filtered['final_verdict'])
        chi2, p_value, dof, expected = chi2_contingency(contingency_matrix)
        
        # Calculate Cramer's V
        n = len(df_filtered)
        min_dim = min(contingency_matrix.shape) - 1
        cramer_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        
        print(f"Chi-square: {chi2:.2f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Cramer's V: {cramer_v:.3f}")
        print(f"Statistical significance: {'✅ YES' if p_value < 0.05 else '❌ NO'}")
        
        # Overall feature usefulness
        if p_value < 0.05 and cramer_v > 0.1:
            print(f"Feature usefulness: ✅ GOOD (significant correlation)")
        elif p_value < 0.05:
            print(f"Feature usefulness: ⚠️ WEAK (significant but small effect)")
        else:
            print(f"Feature usefulness: ❌ POOR (no significant correlation)")

def main():
    """Main analysis function"""
    print("🔬 FEATURE DISTRIBUTION ANALYSIS")
    print("="*50)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Create categorical features
    df = create_categorical_features(df)
    
    # Show detailed distribution
    detailed_feature_distribution(df)
    
    print("\n✅ Feature distribution analysis complete!")

if __name__ == "__main__":
    main() 