# numeric_analysis.py
# Analysis script for numeric features vs final verdict (ROT groups)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from database.schema import get_engine, Paper, PaperFeatures
import logging
from scipy import stats

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
            f.abstract_word_count,
            f.avg_sentence_length,
            f.readability_flesch_score,
            f.jargon_score
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

def create_box_plot(data, feature, ax, title):
    """Create a box plot for a numeric feature vs ROT groups"""
    # Create box plot
    sns.boxplot(data=data, x='rot_group', y=feature, ax=ax, palette=['#ff7f0e', '#1f77b4'])
    
    # Customize the plot
    ax.set_title(f'{title} - Box Plot', fontsize=14, fontweight='bold')
    ax.set_xlabel('ROT Group', fontsize=12)
    ax.set_ylabel(title, fontsize=12)
    
    # Add statistics annotations
    high_rot_data = data[data['rot_group'] == 'High ROT'][feature]
    low_rot_data = data[data['rot_group'] == 'Low ROT'][feature]
    
    # Calculate statistics
    high_mean = high_rot_data.mean()
    low_mean = low_rot_data.mean()
    high_median = high_rot_data.median()
    low_median = low_rot_data.median()
    
    # Add mean and median annotations
    ax.text(0, ax.get_ylim()[1] * 0.95, f'Mean: {high_mean:.2f}\nMedian: {high_median:.2f}', 
            ha='center', va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.text(1, ax.get_ylim()[1] * 0.95, f'Mean: {low_mean:.2f}\nMedian: {low_median:.2f}', 
            ha='center', va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

def create_violin_plot(data, feature, ax, title):
    """Create a violin plot for a numeric feature vs ROT groups"""
    # Create violin plot
    sns.violinplot(data=data, x='rot_group', y=feature, ax=ax, palette=['#ff7f0e', '#1f77b4'])
    
    # Customize the plot
    ax.set_title(f'{title} - Violin Plot', fontsize=14, fontweight='bold')
    ax.set_xlabel('ROT Group', fontsize=12)
    ax.set_ylabel(title, fontsize=12)
    
    # Add sample size annotations
    high_rot_count = len(data[data['rot_group'] == 'High ROT'])
    low_rot_count = len(data[data['rot_group'] == 'Low ROT'])
    
    ax.text(0, ax.get_ylim()[1] * 0.98, f'n={high_rot_count}', ha='center', va='top', fontweight='bold')
    ax.text(1, ax.get_ylim()[1] * 0.98, f'n={low_rot_count}', ha='center', va='top', fontweight='bold')

def analyze_numeric_features():
    """Main function to analyze all numeric features"""
    # Load data
    data = load_data_from_database()
    if data is None:
        return
    
    # Define numeric features and their display names
    numeric_features = {
        'abstract_word_count': 'Abstract Word Count',
        'avg_sentence_length': 'Average Sentence Length',
        'readability_flesch_score': 'Readability Score (Flesch)',
        'jargon_score': 'Jargon Score'
    }
    
    # Filter out features that don't exist in the data
    available_features = {k: v for k, v in numeric_features.items() if k in data.columns}
    
    logging.info(f"Analyzing {len(available_features)} numeric features")
    
    # Create subplots for each feature
    n_features = len(available_features)
    fig, axes = plt.subplots(n_features, 2, figsize=(16, 5 * n_features))
    
    # If only one feature, make axes 2D
    if n_features == 1:
        axes = axes.reshape(1, -1)
    
    # Generate plots for each feature
    for i, (feature, title) in enumerate(available_features.items()):
        logging.info(f"Processing feature: {feature}")
        
        # Create box plot
        create_box_plot(data, feature, axes[i, 0], title)
        
        # Create violin plot
        create_violin_plot(data, feature, axes[i, 1], title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('analysis/numeric_features_analysis.png', dpi=300, bbox_inches='tight')
    logging.info("✅ Saved numeric analysis plot to 'analysis/numeric_features_analysis.png'")
    
    # Show the plot
    plt.show()
    
    # Print summary statistics
    print_summary_statistics(data, available_features)

def print_summary_statistics(data, features):
    """Print summary statistics for numeric features"""
    print("\n" + "="*60)
    print("📊 NUMERIC FEATURES SUMMARY STATISTICS")
    print("="*60)
    
    for feature, title in features.items():
        if feature not in data.columns:
            continue
            
        print(f"\n🔍 {title}:")
        print("-" * 40)
        
        # Group by ROT and calculate statistics
        high_rot_data = data[data['rot_group'] == 'High ROT'][feature]
        low_rot_data = data[data['rot_group'] == 'Low ROT'][feature]
        
        print(f"High ROT (n={len(high_rot_data)}):")
        print(f"  Mean: {high_rot_data.mean():.2f}")
        print(f"  Median: {high_rot_data.median():.2f}")
        print(f"  Std: {high_rot_data.std():.2f}")
        print(f"  Range: [{high_rot_data.min():.2f}, {high_rot_data.max():.2f}]")
        
        print(f"\nLow ROT (n={len(low_rot_data)}):")
        print(f"  Mean: {low_rot_data.mean():.2f}")
        print(f"  Median: {low_rot_data.median():.2f}")
        print(f"  Std: {low_rot_data.std():.2f}")
        print(f"  Range: [{low_rot_data.min():.2f}, {low_rot_data.max():.2f}]")
        
        # T-test for difference in means
        t_stat, p_value = stats.ttest_ind(high_rot_data, low_rot_data)
        print(f"\nT-test for difference in means:")
        print(f"  T-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"  ✅ Significant difference (p < 0.05)")
        else:
            print(f"  ❌ No significant difference (p >= 0.05)")
        
        # Mann-Whitney U test for difference in distributions
        u_stat, u_p_value = stats.mannwhitneyu(high_rot_data, low_rot_data, alternative='two-sided')
        print(f"\nMann-Whitney U test:")
        print(f"  U-statistic: {u_stat:.4f}")
        print(f"  p-value: {u_p_value:.4f}")
        
        if u_p_value < 0.05:
            print(f"  ✅ Significant distribution difference (p < 0.05)")
        else:
            print(f"  ❌ No significant distribution difference (p >= 0.05)")

def create_individual_feature_plots(data):
    """Create individual plots for each feature for better detail"""
    features = {
        'abstract_word_count': 'Abstract Word Count',
        'avg_sentence_length': 'Average Sentence Length',
        'readability_flesch_score': 'Readability Score (Flesch)',
        'jargon_score': 'Jargon Score'
    }
    
    for feature, title in features.items():
        if feature not in data.columns:
            continue
            
        # Create individual figure for each feature
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Box plot
        create_box_plot(data, feature, ax1, title)
        
        # Violin plot
        create_violin_plot(data, feature, ax2, title)
        
        plt.tight_layout()
        
        # Save individual plot
        filename = f'analysis/{feature}_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logging.info(f"✅ Saved {title} analysis to '{filename}'")
        
        plt.show()

def create_correlation_heatmap(data):
    """Create correlation heatmap for all numeric features"""
    numeric_features = [
        'rot_score', 'abstract_word_count', 'avg_sentence_length', 
        'readability_flesch_score', 'jargon_score'
    ]
    
    # Filter available features
    available_features = [f for f in numeric_features if f in data.columns]
    
    if len(available_features) < 2:
        logging.warning("Not enough numeric features for correlation analysis")
        return
    
    # Calculate correlation matrix
    corr_matrix = data[available_features].corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
    
    plt.title('Correlation Heatmap - Numeric Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('analysis/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    logging.info("✅ Saved correlation heatmap to 'analysis/correlation_heatmap.png'")
    
    plt.show()

if __name__ == "__main__":
    print("🔍 Starting Numeric Features Analysis...")
    print("="*50)
    
    # Create analysis directory if it doesn't exist
    import os
    os.makedirs('analysis', exist_ok=True)
    
    # Run the analysis
    data = load_data_from_database()
    if data is not None:
        analyze_numeric_features()
        create_individual_feature_plots(data)
        create_correlation_heatmap(data)
        
        print("\n🎉 Analysis complete! Check the 'analysis/' folder for saved plots.")
    else:
        print("❌ Failed to load data. Please check your database connection.") 