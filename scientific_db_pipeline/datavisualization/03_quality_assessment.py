# ============================================================================
# QUALITY ASSESSMENT OF FEATURES
# ============================================================================
# File: datavisualization/03_quality_assessment.py
# Purpose: Comprehensive quality assessment of all features

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency
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

def extract_categorical_features(df):
    """Extract categorical features from abstract"""
    if 'abstract' in df.columns:
        print("Extracting categorical features from abstracts...")
        
        def extract_features(abstract):
            if pd.isna(abstract) or len(str(abstract)) < 10:
                return False, False, False
            
            abstract_lower = str(abstract).lower()
            
            # Check for dataset mentions
            dataset_keywords = ['dataset', 'data set', 'corpus', 'benchmark', 'evaluation set']
            mentions_dataset = any(keyword in abstract_lower for keyword in dataset_keywords)
            
            # Check for metrics mentions
            metric_keywords = ['accuracy', 'precision', 'recall', 'f1', 'f1-score', 'auc', 'roc', 'bleu', 'rouge', 'perplexity']
            mentions_metrics = any(keyword in abstract_lower for keyword in metric_keywords)
            
            # Check for GitHub links
            has_github_link = 'github.com' in abstract_lower or 'github.io' in abstract_lower
            
            return mentions_dataset, mentions_metrics, has_github_link
        
        features = df['abstract'].apply(extract_features)
        df['mentions_dataset'] = [f[0] for f in features]
        df['mentions_metrics'] = [f[1] for f in features]
        df['has_github_link'] = [f[2] for f in features]
        
        print("✅ Categorical features created")
    
    return df

def assess_numeric_features(df):
    """Assess quality of numeric features"""
    print("\nNUMERIC FEATURES ASSESSMENT:")
    print("="*50)
    
    numeric_features = ['rot_score', 'publication_year', 'citation_count']
    results = []
    
    for feature in numeric_features:
        if feature in df.columns:
            # Split data by ROT group
            high_rot = df[df['rot_group'] == 'High ROT'][feature].dropna()
            low_rot = df[df['rot_group'] == 'Low ROT'][feature].dropna()
            
            # T-test
            t_stat, p_value = ttest_ind(high_rot, low_rot)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(high_rot) - 1) * high_rot.var() + (len(low_rot) - 1) * low_rot.var()) / 
                               (len(high_rot) + len(low_rot) - 2))
            cohens_d = (high_rot.mean() - low_rot.mean()) / pooled_std if pooled_std != 0 else 0
            
            # Determine effect size category
            if abs(cohens_d) > 0.8:
                effect_size = 'Large'
            elif abs(cohens_d) > 0.5:
                effect_size = 'Medium'
            else:
                effect_size = 'Small'
            
            results.append({
                'Feature': feature,
                'High_ROT_Mean': round(high_rot.mean(), 3),
                'Low_ROT_Mean': round(low_rot.mean(), 3),
                'Mean_Difference': round(high_rot.mean() - low_rot.mean(), 3),
                'T_Statistic': round(t_stat, 4),
                'P_Value': round(p_value, 4),
                'Cohens_D': round(cohens_d, 3),
                'Effect_Size': effect_size,
                'Significant': p_value < 0.05
            })
    
    return results

def assess_categorical_features(df):
    """Assess quality of categorical features"""
    print("\nCATEGORICAL FEATURES ASSESSMENT:")
    print("="*50)
    
    categorical_features = ['mentions_dataset', 'mentions_metrics', 'has_github_link']
    results = []
    
    for feature in categorical_features:
        if feature in df.columns:
            # Chi-square test
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
    
    return results

def create_quality_summary(numeric_results, categorical_results):
    """Create comprehensive quality summary"""
    print("\n" + "="*80)
    print("COMPREHENSIVE QUALITY ASSESSMENT")
    print("="*80)
    
    # Dataset overview
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   • Total papers: {len(df)}")
    print(f"   • ROT Groups: {dict(df['rot_group'].value_counts())}")
    
    # Numeric features summary
    if numeric_results:
        print(f"\n🔢 NUMERIC FEATURES ANALYSIS:")
        print(f"   • Features analyzed: {len(numeric_results)}")
        significant_num = [r for r in numeric_results if r['Significant']]
        print(f"   • Significant features: {len(significant_num)}")
        
        for result in numeric_results:
            status = "✅ GOOD" if result['Significant'] else "❌ POOR"
            print(f"   • {result['Feature']}: {status}")
            print(f"     - P-value: {result['P_Value']} {'(p < 0.05)' if result['Significant'] else '(p >= 0.05)'}")
            print(f"     - Cohen's d: {result['Cohens_D']} ({result['Effect_Size']} effect)")
    
    # Categorical features summary
    if categorical_results:
        print(f"\n🔍 CATEGORICAL FEATURES ANALYSIS:")
        print(f"   • Features analyzed: {len(categorical_results)}")
        significant_cat = [r for r in categorical_results if r['Significant']]
        print(f"   • Significant features: {len(significant_cat)}")
        
        for result in categorical_results:
            status = "✅ GOOD" if result['Significant'] else "❌ POOR"
            print(f"   • {result['Feature']}: {status}")
            print(f"     - P-value: {result['P_Value']} {'(p < 0.05)' if result['Significant'] else '(p >= 0.05)'}")
            print(f"     - Cramer's V: {result['Cramer_V']} ({result['Effect_Size']} effect)")
    
    # Overall assessment
    total_features = len(numeric_results) + len(categorical_results)
    total_significant = len([r for r in numeric_results if r['Significant']]) + len([r for r in categorical_results if r['Significant']])
    
    print(f"\n💡 OVERALL ASSESSMENT:")
    print(f"   • Total features: {total_features}")
    print(f"   • Significant features: {total_significant}")
    print(f"   • Success rate: {total_significant/total_features*100:.1f}%")
    
    if total_significant >= total_features * 0.6:
        print(f"   • 🎉 EXCELLENT: Features have GOOD quality for verdict prediction!")
    elif total_significant >= total_features * 0.4:
        print(f"   • ⚠️ MODERATE: Features have MODERATE quality for verdict prediction!")
    else:
        print(f"   • ❌ POOR: Features need SIGNIFICANT improvement for verdict prediction!")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if total_significant < total_features * 0.6:
        print(f"   • Improve feature extraction methods")
        print(f"   • Add more sophisticated features")
        print(f"   • Consider feature engineering")
        print(f"   • Investigate ROT score calculation")
    else:
        print(f"   • Features are ready for model training")
        print(f"   • Consider ensemble methods")
        print(f"   • Monitor model performance")

def create_quality_visualization(numeric_results, categorical_results):
    """Create visualization of quality assessment"""
    print("\n📈 CREATING QUALITY VISUALIZATION:")
    print("="*50)
    
    # Prepare data for visualization
    all_results = []
    
    for result in numeric_results:
        all_results.append({
            'Feature': result['Feature'],
            'Type': 'Numeric',
            'P_Value': result['P_Value'],
            'Effect_Size': result['Effect_Size'],
            'Significant': result['Significant']
        })
    
    for result in categorical_results:
        all_results.append({
            'Feature': result['Feature'],
            'Type': 'Categorical',
            'P_Value': result['P_Value'],
            'Effect_Size': result['Effect_Size'],
            'Significant': result['Significant']
        })
    
    results_df = pd.DataFrame(all_results)
    
    # Create quality assessment plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: P-values
    colors = ['red' if not sig else 'green' for sig in results_df['Significant']]
    ax1.bar(range(len(results_df)), results_df['P_Value'], color=colors, alpha=0.7)
    ax1.axhline(y=0.05, color='red', linestyle='--', label='Significance threshold (0.05)')
    ax1.set_title('P-values by Feature', fontweight='bold')
    ax1.set_xlabel('Features')
    ax1.set_ylabel('P-value')
    ax1.set_xticks(range(len(results_df)))
    ax1.set_xticklabels(results_df['Feature'], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Feature types and significance
    significant_counts = results_df.groupby(['Type', 'Significant']).size().unstack(fill_value=0)
    significant_counts.plot(kind='bar', ax=ax2, color=['red', 'green'], alpha=0.7)
    ax2.set_title('Feature Quality by Type', fontweight='bold')
    ax2.set_xlabel('Feature Type')
    ax2.set_ylabel('Count')
    ax2.legend(['Not Significant', 'Significant'])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run quality assessment"""
    print("QUALITY ASSESSMENT OF FEATURES")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Check if rot_group exists
    if 'rot_group' not in df.columns:
        print("❌ Error: 'rot_group' column not found!")
        return
    
    # Extract categorical features if needed
    df = extract_categorical_features(df)
    
    # Assess features
    numeric_results = assess_numeric_features(df)
    categorical_results = assess_categorical_features(df)
    
    # Create quality summary
    create_quality_summary(numeric_results, categorical_results)
    
    # Create quality visualization
    create_quality_visualization(numeric_results, categorical_results)
    
    print("\n✅ Quality assessment completed!")

if __name__ == "__main__":
    main() 