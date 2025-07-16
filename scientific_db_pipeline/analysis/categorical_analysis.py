# categorical_analysis.py
# Analysis script for categorical features vs final verdict (ROT groups)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from database.schema import get_engine, Paper, PaperFeatures
import logging

# Configure logging and plotting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data_from_database():
    """Load paper features and ROT groups from database"""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Query to get papers with features and ROT groups
        query = """
        SELECT 
            p.paper_id,
            p.rot_score,
            p.rot_group,
            f.mentions_dataset,
            f.mentions_metrics,
            f.has_github_link,
            f.final_verdict_category
        FROM papers p
        JOIN paper_features f ON p.paper_id = f.paper_id
        WHERE p.rot_score IS NOT NULL
        ORDER BY p.rot_score DESC
        """
        
        df = pd.read_sql(query, engine)
        
        if df.empty:
            logging.error("No data found with both features and ROT scores!")
            return None
        
        logging.info(f"Loaded {len(df)} papers with features and ROT scores")
        
        # If rot_group is not in database, create it based on median ROT
        if 'rot_group' not in df.columns or df['rot_group'].isna().all():
            median_rot = df['rot_score'].median()
            df['rot_group'] = df['rot_score'].apply(lambda x: 'High ROT' if x >= median_rot else 'Low ROT')
            logging.info(f"Created ROT groups using median ROT score: {median_rot:.2f}")
        
        return df
        
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None
    finally:
        session.close()

def create_stacked_bar_chart(data, feature, ax, title):
    """Create a stacked bar chart for a categorical feature vs ROT groups"""
    # Count occurrences
    cross_tab = pd.crosstab(data[feature], data['rot_group'], normalize='index') * 100
    
    # Create stacked bar chart
    cross_tab.plot(kind='bar', stacked=True, ax=ax, color=['#ff7f0e', '#1f77b4'])
    
    # Customize the plot
    ax.set_title(f'{title} - Stacked Bar Chart', fontsize=14, fontweight='bold')
    ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.legend(title='ROT Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=45)
    
    # Add percentage labels on bars
    for c in ax.containers:
        ax.bar_label(c, fmt='%.1f%%', label_type='center')
    
    # Add total count annotations
    total_counts = data.groupby(feature).size()
    for i, (feature_val, count) in enumerate(total_counts.items()):
        ax.text(i, 102, f'n={count}', ha='center', va='bottom', fontweight='bold')

def create_grouped_bar_chart(data, feature, ax, title):
    """Create a grouped bar chart for a categorical feature vs ROT groups"""
    # Count occurrences
    cross_tab = pd.crosstab(data[feature], data['rot_group'])
    
    # Create grouped bar chart
    cross_tab.plot(kind='bar', ax=ax, color=['#ff7f0e', '#1f77b4'])
    
    # Customize the plot
    ax.set_title(f'{title} - Grouped Bar Chart', fontsize=14, fontweight='bold')
    ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.legend(title='ROT Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=45)
    
    # Add count labels on bars
    for c in ax.containers:
        ax.bar_label(c, fmt='%d', label_type='edge')
    
    # Add total count annotations
    total_counts = data.groupby(feature).size()
    for i, (feature_val, count) in enumerate(total_counts.items()):
        ax.text(i, ax.get_ylim()[1] * 1.02, f'Total: {count}', 
                ha='center', va='bottom', fontweight='bold')

def analyze_categorical_features():
    """Main function to analyze all categorical features"""
    # Load data
    data = load_data_from_database()
    if data is None:
        return
    
    # Define categorical features and their display names
    categorical_features = {
        'mentions_dataset': 'Mentions Dataset',
        'mentions_metrics': 'Mentions Metrics', 
        'has_github_link': 'Has GitHub Link',
        'final_verdict_category': 'Final Verdict Category'
    }
    
    # Filter out features that don't exist in the data
    available_features = {k: v for k, v in categorical_features.items() if k in data.columns}
    
    logging.info(f"Analyzing {len(available_features)} categorical features")
    
    # Create subplots for each feature
    n_features = len(available_features)
    fig, axes = plt.subplots(n_features, 2, figsize=(16, 5 * n_features))
    
    # If only one feature, make axes 2D
    if n_features == 1:
        axes = axes.reshape(1, -1)
    
    # Generate plots for each feature
    for i, (feature, title) in enumerate(available_features.items()):
        logging.info(f"Processing feature: {feature}")
        
        # Skip if feature has too many unique values (likely not categorical)
        unique_count = data[feature].nunique()
        if unique_count > 10:
            logging.warning(f"Skipping {feature} - too many unique values ({unique_count})")
            continue
        
        # Create stacked bar chart
        create_stacked_bar_chart(data, feature, axes[i, 0], title)
        
        # Create grouped bar chart
        create_grouped_bar_chart(data, feature, axes[i, 1], title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('analysis/categorical_features_analysis.png', dpi=300, bbox_inches='tight')
    logging.info("✅ Saved categorical analysis plot to 'analysis/categorical_features_analysis.png'")
    
    # Show the plot
    plt.show()
    
    # Print summary statistics
    print_summary_statistics(data, available_features)

def print_summary_statistics(data, features):
    """Print summary statistics for categorical features"""
    print("\n" + "="*60)
    print("📊 CATEGORICAL FEATURES SUMMARY STATISTICS")
    print("="*60)
    
    for feature, title in features.items():
        if feature not in data.columns:
            continue
            
        print(f"\n🔍 {title}:")
        print("-" * 40)
        
        # Cross-tabulation
        cross_tab = pd.crosstab(data[feature], data['rot_group'], margins=True)
        print("Cross-tabulation (Counts):")
        print(cross_tab)
        
        # Percentage distribution
        cross_tab_pct = pd.crosstab(data[feature], data['rot_group'], normalize='index') * 100
        print(f"\nPercentage distribution by {title}:")
        print(cross_tab_pct.round(1))
        
        # Chi-square test for independence
        from scipy.stats import chi2_contingency
        contingency_table = pd.crosstab(data[feature], data['rot_group'])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        print(f"\nChi-square test for independence:")
        print(f"  Chi-square statistic: {chi2:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Degrees of freedom: {dof}")
        
        if p_value < 0.05:
            print(f"  ✅ Significant relationship (p < 0.05)")
        else:
            print(f"  ❌ No significant relationship (p >= 0.05)")

def create_individual_feature_plots(data):
    """Create individual plots for each feature for better detail"""
    features = {
        'mentions_dataset': 'Mentions Dataset',
        'mentions_metrics': 'Mentions Metrics', 
        'has_github_link': 'Has GitHub Link',
        'final_verdict_category': 'Final Verdict Category'
    }
    
    for feature, title in features.items():
        if feature not in data.columns:
            continue
            
        # Create individual figure for each feature
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Stacked bar chart
        create_stacked_bar_chart(data, feature, ax1, title)
        
        # Grouped bar chart
        create_grouped_bar_chart(data, feature, ax2, title)
        
        plt.tight_layout()
        
        # Save individual plot
        filename = f'analysis/{feature}_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logging.info(f"✅ Saved {title} analysis to '{filename}'")
        
        plt.show()

if __name__ == "__main__":
    print("🔍 Starting Categorical Features Analysis...")
    print("="*50)
    
    # Create analysis directory if it doesn't exist
    import os
    os.makedirs('analysis', exist_ok=True)
    
    # Run the analysis
    data = load_data_from_database()
    if data is not None:
        analyze_categorical_features()
        create_individual_feature_plots(data)
        
        print("\n🎉 Analysis complete! Check the 'analysis/' folder for saved plots.")
    else:
        print("❌ Failed to load data. Please check your database connection.") 