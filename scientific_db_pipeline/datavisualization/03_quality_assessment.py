# ============================================================================
# QUALITY ASSESSMENT OF FEATURES WITH 5-CATEGORY ROT SYSTEM
# ============================================================================
# File: datavisualization/03_quality_assessment.py
# Purpose: Comprehensive quality assessment of all features

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import f_oneway, chi2_contingency
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
    
    categories = ['Very Low ROT', 'Low ROT', 'Medium ROT', 'High ROT', 'Very High ROT']
    df_filtered = df[df['final_verdict'].isin(categories)]
    
    numeric_features = ['rot_score', 'publication_year', 'citation_count']
    results = []
    
    for feature in numeric_features:
        if feature in df.columns:
            # Split data by categories
            groups = [df_filtered[df_filtered['final_verdict'] == cat][feature].dropna() 
                     for cat in categories]
            
            # ANOVA test
            f_stat, p_value = f_oneway(*groups)
            
            # Calculate effect size (Eta-squared)
            ss_between = 0
            ss_total = 0
            grand_mean = df_filtered[feature].mean()
            
            for group in groups:
                if len(group) > 0:
                    group_mean = group.mean()
                    ss_between += len(group) * (group_mean - grand_mean) ** 2
                    ss_total += sum((group - grand_mean) ** 2)
            
            eta_squared = ss_between / ss_total if ss_total != 0 else 0
            
            # Determine effect size category
            if eta_squared > 0.14:
                effect_size = 'Large'
            elif eta_squared > 0.06:
                effect_size = 'Medium'
            else:
                effect_size = 'Small'
            
            results.append({
                'Feature': feature,
                'F_statistic': f_stat,
                'P_value': p_value,
                'Eta_squared': eta_squared,
                'Effect_size': effect_size,
                'Significant': p_value < 0.05
            })
    
    return results

def assess_categorical_features(df):
    """Assess quality of categorical features"""
    print("\nCATEGORICAL FEATURES ASSESSMENT:")
    print("="*50)
    
    categories = ['Very Low ROT', 'Low ROT', 'Medium ROT', 'High ROT', 'Very High ROT']
    df_filtered = df[df['final_verdict'].isin(categories)]
    
    categorical_features = ['mentions_dataset', 'mentions_metrics', 'has_github_link']
    results = []
    
    for feature in categorical_features:
        if feature in df.columns:
            # Chi-square test
            contingency_matrix = pd.crosstab(df_filtered[feature], df_filtered['final_verdict'])
            chi2, p_value, dof, expected = chi2_contingency(contingency_matrix)
            
            # Calculate Cramer's V
            n = len(df_filtered)
            min_dim = min(contingency_matrix.shape) - 1
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
                'Chi2': chi2,
                'P_value': p_value,
                'Cramer_V': cramer_v,
                'Effect_size': effect_size,
                'Significant': p_value < 0.05
            })
    
    return results

def create_quality_summary(numeric_results, categorical_results):
    """Create comprehensive quality summary"""
    print("\nQUALITY ASSESSMENT SUMMARY:")
    print("="*50)
    
    all_results = numeric_results + categorical_results
    
    # Print results table
    print(f"{'Feature':<20} {'Test':<8} {'Statistic':<10} {'P-value':<10} {'Effect Size':<12} {'Significant':<10}")
    print("-" * 70)
    
    for result in all_results:
        if 'F_statistic' in result:
            test_type = 'ANOVA'
            statistic = result['F_statistic']
            effect_size = result['Eta_squared']
        else:
            test_type = 'Chi2'
            statistic = result['Chi2']
            effect_size = result['Cramer_V']
        
        significance = "✅ YES" if result['Significant'] else "❌ NO"
        print(f"{result['Feature']:<20} {test_type:<8} {statistic:<10.2f} {result['P_value']:<10.4f} "
              f"{effect_size:<12.3f} {significance:<10}")
    
    # Calculate success rates
    significant_numeric = sum(1 for r in numeric_results if r['Significant'])
    significant_categorical = sum(1 for r in categorical_results if r['Significant'])
    total_significant = significant_numeric + significant_categorical
    
    print(f"\n📊 SUCCESS RATES:")
    print(f"   • Numeric features: {significant_numeric}/{len(numeric_results)} ({significant_numeric/len(numeric_results)*100:.1f}%)")
    print(f"   • Categorical features: {significant_categorical}/{len(categorical_results)} ({significant_categorical/len(categorical_results)*100:.1f}%)")
    print(f"   • Overall: {total_significant}/{len(all_results)} ({total_significant/len(all_results)*100:.1f}%)")
    
    # Quality assessment
    overall_rate = total_significant / len(all_results)
    if overall_rate >= 0.6:
        quality = "EXCELLENT"
        recommendation = "Features are highly suitable for AI classification!"
    elif overall_rate >= 0.4:
        quality = "GOOD"
        recommendation = "Features are suitable for AI classification with some improvements."
    else:
        quality = "POOR"
        recommendation = "Features need significant improvement for AI classification."
    
    print(f"\n🎯 QUALITY ASSESSMENT:")
    print(f"   • Overall Quality: {quality}")
    print(f"   • Recommendation: {recommendation}")
    
    return all_results

def create_quality_visualization(numeric_results, categorical_results):
    """Create quality visualization"""
    print("\nQUALITY VISUALIZATION:")
    print("="*50)
    
    all_results = numeric_results + categorical_results
    
    # Create quality score plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Success rate by feature type
    feature_types = ['Numeric', 'Categorical']
    numeric_success = sum(1 for r in numeric_results if r['Significant']) / len(numeric_results) * 100
    categorical_success = sum(1 for r in categorical_results if r['Significant']) / len(categorical_results) * 100
    
    bars1 = ax1.bar(feature_types, [numeric_success, categorical_success], 
                    color=['#ff7f0e', '#1f77b4'], alpha=0.8)
    ax1.set_title('Success Rate by Feature Type', fontweight='bold')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars1, [numeric_success, categorical_success]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Individual feature quality
    features = [r['Feature'] for r in all_results]
    p_values = [r['P_value'] for r in all_results]
    significant = [r['Significant'] for r in all_results]
    
    colors = ['green' if sig else 'red' for sig in significant]
    bars2 = ax2.bar(range(len(features)), [-np.log10(p) if p > 0 else 10 for p in p_values], 
                    color=colors, alpha=0.8)
    ax2.set_title('Feature Quality (-log10 P-value)', fontweight='bold')
    ax2.set_xlabel('Features')
    ax2.set_ylabel('-log10(P-value)')
    ax2.set_xticks(range(len(features)))
    ax2.set_xticklabels(features, rotation=45, ha='right')
    ax2.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05 threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main quality assessment function"""
    print("🔬 QUALITY ASSESSMENT WITH 5-CATEGORY ROT SYSTEM")
    print("="*60)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Extract categorical features
    df = extract_categorical_features(df)
    
    # Assess features
    numeric_results = assess_numeric_features(df)
    categorical_results = assess_categorical_features(df)
    
    # Create summary and visualization
    all_results = create_quality_summary(numeric_results, categorical_results)
    create_quality_visualization(numeric_results, categorical_results)
    
    print("\n✅ Quality assessment complete!")

if __name__ == "__main__":
    main() 