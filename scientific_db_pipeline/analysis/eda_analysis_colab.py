# EDA Analysis for Feature Quality Evaluation - Google Colab Version
# Copy and paste this code into Google Colab cells

# ============================================================================
# CELL 1: Install Dependencies
# ============================================================================
"""
!pip install pandas matplotlib seaborn numpy scipy scikit-learn
"""

# ============================================================================
# CELL 2: Import Libraries
# ============================================================================
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("✅ Libraries imported successfully!")
"""

# ============================================================================
# CELL 3: Upload and Load Data
# ============================================================================
"""
from google.colab import files
import io

print("📁 Please upload your 'all_papers_with_rot_groups.csv' file:")
uploaded = files.upload()

# Load the data
for filename in uploaded.keys():
    if filename.endswith('.csv'):
        data = pd.read_csv(io.BytesIO(uploaded[filename]))
        print(f"✅ Loaded {filename} with {len(data)} rows")
        break

print(f"\\n📊 Data shape: {data.shape}")
print(f"📋 Columns: {list(data.columns)}")
"""

# ============================================================================
# CELL 4: Data Preview
# ============================================================================
"""
# Preview the data
print("📋 First 5 rows:")
display(data.head())

print("\\n📊 Data info:")
display(data.info())

print("\\n📈 ROT groups distribution:")
display(data['rot_group'].value_counts())
"""

# ============================================================================
# CELL 5: Data Cleaning
# ============================================================================
"""
# Check for missing values
print("🔍 Missing values:")
missing_data = data.isnull().sum()
display(missing_data[missing_data > 0])

# Remove rows with missing ROT scores
data_clean = data.dropna(subset=['rot_score'])
print(f"\\n✅ Cleaned data: {len(data_clean)} rows (removed {len(data) - len(data_clean)} rows with missing ROT scores)")

# Check ROT score distribution
print(f"\\n📊 ROT Score Statistics:")
print(f"   Mean: {data_clean['rot_score'].mean():.2f}")
print(f"   Median: {data_clean['rot_score'].median():.2f}")
print(f"   Min: {data_clean['rot_score'].min():.2f}")
print(f"   Max: {data_clean['rot_score'].max():.2f}")
"""

# ============================================================================
# CELL 6: Identify Categorical Features
# ============================================================================
"""
# Identify categorical features
categorical_features = []
for col in data_clean.columns:
    if data_clean[col].dtype == 'object' or data_clean[col].nunique() <= 10:
        if col not in ['paper_id', 'title', 'abstract', 'doi', 'source_url', 'rot_group']:
            categorical_features.append(col)

print(f"🔍 Found {len(categorical_features)} categorical features:")
for feature in categorical_features:
    unique_vals = data_clean[feature].nunique()
    print(f"   - {feature}: {unique_vals} unique values")
    if unique_vals <= 5:
        print(f"     Values: {data_clean[feature].value_counts().to_dict()}")
"""

# ============================================================================
# CELL 7: Categorical Features Analysis
# ============================================================================
"""
# Create categorical analysis plots
if categorical_features:
    n_features = len(categorical_features)
    fig, axes = plt.subplots(n_features, 2, figsize=(16, 5 * n_features))
    
    # If only one feature, make axes 2D
    if n_features == 1:
        axes = axes.reshape(1, -1)
    
    for i, feature in enumerate(categorical_features):
        print(f"📊 Analyzing {feature}...")
        
        # Stacked bar chart
        cross_tab = pd.crosstab(data_clean[feature], data_clean['rot_group'], normalize='index') * 100
        cross_tab.plot(kind='bar', stacked=True, ax=axes[i, 0], color=['#ff7f0e', '#1f77b4'])
        axes[i, 0].set_title(f'{feature} - Stacked Bar Chart', fontweight='bold')
        axes[i, 0].set_xlabel(feature)
        axes[i, 0].set_ylabel('Percentage (%)')
        axes[i, 0].legend(title='ROT Group')
        axes[i, 0].tick_params(axis='x', rotation=45)
        
        # Add percentage labels
        for c in axes[i, 0].containers:
            axes[i, 0].bar_label(c, fmt='%.1f%%', label_type='center')
        
        # Grouped bar chart
        cross_tab_counts = pd.crosstab(data_clean[feature], data_clean['rot_group'])
        cross_tab_counts.plot(kind='bar', ax=axes[i, 1], color=['#ff7f0e', '#1f77b4'])
        axes[i, 1].set_title(f'{feature} - Grouped Bar Chart', fontweight='bold')
        axes[i, 1].set_xlabel(feature)
        axes[i, 1].set_ylabel('Count')
        axes[i, 1].legend(title='ROT Group')
        axes[i, 1].tick_params(axis='x', rotation=45)
        
        # Add count labels
        for c in axes[i, 1].containers:
            axes[i, 1].bar_label(c, fmt='%d', label_type='edge')
    
    plt.tight_layout()
    plt.show()
else:
    print("❌ No categorical features found in the data")
"""

# ============================================================================
# CELL 8: Categorical Statistical Analysis
# ============================================================================
"""
# Statistical analysis for categorical features
print("📊 CATEGORICAL FEATURES STATISTICAL ANALYSIS")
print("=" * 50)

for feature in categorical_features:
    print(f"\\n🔍 {feature}:")
    print("-" * 30)
    
    # Cross-tabulation
    cross_tab = pd.crosstab(data_clean[feature], data_clean['rot_group'], margins=True)
    print("Cross-tabulation (Counts):")
    display(cross_tab)
    
    # Percentage distribution
    cross_tab_pct = pd.crosstab(data_clean[feature], data_clean['rot_group'], normalize='index') * 100
    print(f"\\nPercentage distribution by {feature}:")
    display(cross_tab_pct.round(1))
    
    # Chi-square test
    contingency_table = pd.crosstab(data_clean[feature], data_clean['rot_group'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    print(f"\\nChi-square test for independence:")
    print(f"  Chi-square statistic: {chi2:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Degrees of freedom: {dof}")
    
    if p_value < 0.05:
        print(f"  ✅ Significant relationship (p < 0.05)")
    else:
        print(f"  ❌ No significant relationship (p >= 0.05)")
"""

# ============================================================================
# CELL 9: Identify Numeric Features
# ============================================================================
"""
# Identify numeric features
numeric_features = []
for col in data_clean.columns:
    if data_clean[col].dtype in ['int64', 'float64']:
        if col not in ['paper_id', 'publication_year', 'citation_count', 'rot_score']:
            numeric_features.append(col)

print(f"🔍 Found {len(numeric_features)} numeric features:")
for feature in numeric_features:
    print(f"   - {feature}")
    print(f"     Range: [{data_clean[feature].min():.2f}, {data_clean[feature].max():.2f}]")
    print(f"     Mean: {data_clean[feature].mean():.2f}")
    print(f"     Missing: {data_clean[feature].isnull().sum()}")
"""

# ============================================================================
# CELL 10: Numeric Features Analysis
# ============================================================================
"""
# Create numeric analysis plots
if numeric_features:
    n_features = len(numeric_features)
    fig, axes = plt.subplots(n_features, 2, figsize=(16, 5 * n_features))
    
    # If only one feature, make axes 2D
    if n_features == 1:
        axes = axes.reshape(1, -1)
    
    for i, feature in enumerate(numeric_features):
        print(f"📊 Analyzing {feature}...")
        
        # Box plot
        sns.boxplot(data=data_clean, x='rot_group', y=feature, ax=axes[i, 0], palette=['#ff7f0e', '#1f77b4'])
        axes[i, 0].set_title(f'{feature} - Box Plot', fontweight='bold')
        axes[i, 0].set_xlabel('ROT Group')
        axes[i, 0].set_ylabel(feature)
        
        # Add statistics
        high_rot_data = data_clean[data_clean['rot_group'] == 'High ROT'][feature]
        low_rot_data = data_clean[data_clean['rot_group'] == 'Low ROT'][feature]
        
        high_mean = high_rot_data.mean()
        low_mean = low_rot_data.mean()
        
        axes[i, 0].text(0, axes[i, 0].get_ylim()[1] * 0.95, f'Mean: {high_mean:.2f}', 
                        ha='center', va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        axes[i, 0].text(1, axes[i, 0].get_ylim()[1] * 0.95, f'Mean: {low_mean:.2f}', 
                        ha='center', va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Violin plot
        sns.violinplot(data=data_clean, x='rot_group', y=feature, ax=axes[i, 1], palette=['#ff7f0e', '#1f77b4'])
        axes[i, 1].set_title(f'{feature} - Violin Plot', fontweight='bold')
        axes[i, 1].set_xlabel('ROT Group')
        axes[i, 1].set_ylabel(feature)
        
        # Add sample sizes
        high_rot_count = len(high_rot_data)
        low_rot_count = len(low_rot_data)
        
        axes[i, 1].text(0, axes[i, 1].get_ylim()[1] * 0.98, f'n={high_rot_count}', ha='center', va='top', fontweight='bold')
        axes[i, 1].text(1, axes[i, 1].get_ylim()[1] * 0.98, f'n={low_rot_count}', ha='center', va='top', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
else:
    print("❌ No numeric features found in the data")
"""

# ============================================================================
# CELL 11: Numeric Statistical Analysis
# ============================================================================
"""
# Statistical analysis for numeric features
print("📊 NUMERIC FEATURES STATISTICAL ANALYSIS")
print("=" * 50)

for feature in numeric_features:
    print(f"\\n🔍 {feature}:")
    print("-" * 30)
    
    # Group by ROT and calculate statistics
    high_rot_data = data_clean[data_clean['rot_group'] == 'High ROT'][feature]
    low_rot_data = data_clean[data_clean['rot_group'] == 'Low ROT'][feature]
    
    print(f"High ROT (n={len(high_rot_data)}):")
    print(f"  Mean: {high_rot_data.mean():.2f}")
    print(f"  Median: {high_rot_data.median():.2f}")
    print(f"  Std: {high_rot_data.std():.2f}")
    print(f"  Range: [{high_rot_data.min():.2f}, {high_rot_data.max():.2f}]")
    
    print(f"\\nLow ROT (n={len(low_rot_data)}):")
    print(f"  Mean: {low_rot_data.mean():.2f}")
    print(f"  Median: {low_rot_data.median():.2f}")
    print(f"  Std: {low_rot_data.std():.2f}")
    print(f"  Range: [{low_rot_data.min():.2f}, {low_rot_data.max():.2f}]")
    
    # T-test for difference in means
    t_stat, p_value = stats.ttest_ind(high_rot_data, low_rot_data)
    print(f"\\nT-test for difference in means:")
    print(f"  T-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print(f"  ✅ Significant difference (p < 0.05)")
    else:
        print(f"  ❌ No significant difference (p >= 0.05)")
    
    # Mann-Whitney U test
    u_stat, u_p_value = stats.mannwhitneyu(high_rot_data, low_rot_data, alternative='two-sided')
    print(f"\\nMann-Whitney U test:")
    print(f"  U-statistic: {u_stat:.4f}")
    print(f"  p-value: {u_p_value:.4f}")
    
    if u_p_value < 0.05:
        print(f"  ✅ Significant distribution difference (p < 0.05)")
    else:
        print(f"  ❌ No significant distribution difference (p >= 0.05)")
"""

# ============================================================================
# CELL 12: Correlation Analysis
# ============================================================================
"""
# Create correlation heatmap
if len(numeric_features) > 1:
    # Add ROT score to numeric features for correlation
    correlation_features = numeric_features + ['rot_score']
    
    # Calculate correlation matrix
    corr_matrix = data_clean[correlation_features].corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
    
    plt.title('Correlation Heatmap - Numeric Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Show ROT score correlations
    print("🔗 Correlation with ROT Score:")
    rot_correlations = corr_matrix['rot_score'].sort_values(ascending=False)
    display(rot_correlations)
else:
    print("❌ Not enough numeric features for correlation analysis")
"""

# ============================================================================
# CELL 13: Summary and Recommendations
# ============================================================================
"""
# Generate comprehensive summary
print("📊 COMPREHENSIVE EDA ANALYSIS SUMMARY")
print("=" * 50)
print(f"\\n📈 Data Overview:")
print(f"   Total papers analyzed: {len(data_clean)}")
print(f"   High ROT papers: {len(data_clean[data_clean['rot_group'] == 'High ROT'])}")
print(f"   Low ROT papers: {len(data_clean[data_clean['rot_group'] == 'Low ROT'])}")
print(f"   ROT score range: [{data_clean['rot_score'].min():.2f}, {data_clean['rot_score'].max():.2f}]")
print(f"   Median ROT score: {data_clean['rot_score'].median():.2f}")

print(f"\\n🔍 Features Analyzed:")
print(f"   Categorical features: {len(categorical_features)}")
for feature in categorical_features:
    print(f"     - {feature}")

print(f"   Numeric features: {len(numeric_features)}")
for feature in numeric_features:
    print(f"     - {feature}")

print(f"\\n💡 Key Insights:")
print(f"   1. Review the plots above for visual patterns")
print(f"   2. Check statistical significance from the tests above")
print(f"   3. Look for features that strongly correlate with ROT scores")
print(f"   4. Consider which features best discriminate between High/Low ROT")

print(f"\\n🎯 Recommendations:")
print(f"   1. Focus on features with significant statistical relationships")
print(f"   2. Consider feature engineering based on insights")
print(f"   3. Validate findings with domain experts")
print(f"   4. Use insights to improve your feature extraction process")

print(f"\\n✅ Analysis Complete!")
print(f"📊 You now have comprehensive insights into your feature quality.")
"""

# ============================================================================
# CELL 14: Download Results (Optional)
# ============================================================================
"""
# Save summary statistics to CSV
summary_data = []

# Categorical features summary
for feature in categorical_features:
    cross_tab = pd.crosstab(data_clean[feature], data_clean['rot_group'])
    chi2, p_value, dof, expected = chi2_contingency(cross_tab)
    
    summary_data.append({
        'feature': feature,
        'type': 'categorical',
        'chi2_statistic': chi2,
        'p_value': p_value,
        'significant': p_value < 0.05
    })

# Numeric features summary
for feature in numeric_features:
    high_rot_data = data_clean[data_clean['rot_group'] == 'High ROT'][feature]
    low_rot_data = data_clean[data_clean['rot_group'] == 'Low ROT'][feature]
    
    t_stat, t_p_value = stats.ttest_ind(high_rot_data, low_rot_data)
    u_stat, u_p_value = stats.mannwhitneyu(high_rot_data, low_rot_data, alternative='two-sided')
    
    summary_data.append({
        'feature': feature,
        'type': 'numeric',
        't_statistic': t_stat,
        't_p_value': t_p_value,
        'u_statistic': u_stat,
        'u_p_value': u_p_value,
        'significant_t': t_p_value < 0.05,
        'significant_u': u_p_value < 0.05
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('feature_quality_analysis.csv', index=False)

print("📊 Summary statistics saved to 'feature_quality_analysis.csv'")
display(summary_df)

# Download the file
from google.colab import files
files.download('feature_quality_analysis.csv')
""" 