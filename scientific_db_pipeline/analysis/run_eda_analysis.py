# run_eda_analysis.py
# Main script to run comprehensive EDA analysis for feature quality evaluation

import os
import sys
import logging
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from categorical_analysis import analyze_categorical_features, create_individual_feature_plots
from numeric_analysis import analyze_numeric_features, create_individual_feature_plots as create_numeric_plots, create_correlation_heatmap
from categorical_analysis import load_data_from_database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis/eda_analysis.log'),
        logging.StreamHandler()
    ]
)

def run_comprehensive_eda():
    """Run comprehensive EDA analysis for all features"""
    print("🔍 COMPREHENSIVE EDA ANALYSIS FOR FEATURE QUALITY EVALUATION")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create analysis directory
    os.makedirs('analysis', exist_ok=True)
    
    # Load data once
    print("📊 Loading data from database...")
    data = load_data_from_database()
    
    if data is None:
        print("❌ Failed to load data. Please check your database connection.")
        return
    
    print(f"✅ Loaded {len(data)} papers with features and ROT scores")
    print(f"   High ROT papers: {len(data[data['rot_group'] == 'High ROT'])}")
    print(f"   Low ROT papers: {len(data[data['rot_group'] == 'Low ROT'])}")
    print()
    
    # Run categorical analysis
    print("📈 Running Categorical Features Analysis...")
    print("-" * 50)
    try:
        analyze_categorical_features()
        create_individual_feature_plots(data)
        print("✅ Categorical analysis completed successfully!")
    except Exception as e:
        print(f"❌ Error in categorical analysis: {e}")
        logging.error(f"Categorical analysis error: {e}")
    
    print()
    
    # Run numeric analysis
    print("📊 Running Numeric Features Analysis...")
    print("-" * 50)
    try:
        analyze_numeric_features()
        create_numeric_plots(data)
        create_correlation_heatmap(data)
        print("✅ Numeric analysis completed successfully!")
    except Exception as e:
        print(f"❌ Error in numeric analysis: {e}")
        logging.error(f"Numeric analysis error: {e}")
    
    print()
    
    # Generate summary report
    print("📋 Generating Summary Report...")
    print("-" * 50)
    generate_summary_report(data)
    
    print()
    print("🎉 EDA Analysis Complete!")
    print("📁 Check the 'analysis/' folder for all generated plots and reports.")
    print(f"📝 Log file: analysis/eda_analysis.log")

def generate_summary_report(data):
    """Generate a comprehensive summary report"""
    report_file = 'analysis/eda_summary_report.txt'
    
    with open(report_file, 'w') as f:
        f.write("COMPREHENSIVE EDA ANALYSIS SUMMARY REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Data overview
        f.write("DATA OVERVIEW\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total papers analyzed: {len(data)}\n")
        f.write(f"High ROT papers: {len(data[data['rot_group'] == 'High ROT'])}\n")
        f.write(f"Low ROT papers: {len(data[data['rot_group'] == 'Low ROT'])}\n")
        f.write(f"ROT score range: [{data['rot_score'].min():.2f}, {data['rot_score'].max():.2f}]\n")
        f.write(f"Median ROT score: {data['rot_score'].median():.2f}\n\n")
        
        # Feature availability
        f.write("FEATURE AVAILABILITY\n")
        f.write("-" * 25 + "\n")
        categorical_features = ['mentions_dataset', 'mentions_metrics', 'has_github_link', 'final_verdict_category']
        numeric_features = ['abstract_word_count', 'avg_sentence_length', 'readability_flesch_score', 'jargon_score']
        
        f.write("Categorical features:\n")
        for feature in categorical_features:
            if feature in data.columns:
                missing_pct = (data[feature].isna().sum() / len(data)) * 100
                f.write(f"  ✓ {feature}: {missing_pct:.1f}% missing\n")
            else:
                f.write(f"  ✗ {feature}: Not available\n")
        
        f.write("\nNumeric features:\n")
        for feature in numeric_features:
            if feature in data.columns:
                missing_pct = (data[feature].isna().sum() / len(data)) * 100
                f.write(f"  ✓ {feature}: {missing_pct:.1f}% missing\n")
            else:
                f.write(f"  ✗ {feature}: Not available\n")
        
        f.write("\n")
        
        # Key insights
        f.write("KEY INSIGHTS\n")
        f.write("-" * 15 + "\n")
        
        # Categorical insights
        f.write("Categorical Features:\n")
        for feature in categorical_features:
            if feature in data.columns:
                cross_tab = data.groupby([feature, 'rot_group']).size().unstack(fill_value=0)
                f.write(f"  {feature}:\n")
                f.write(f"    {cross_tab.to_string()}\n")
                f.write(f"    Chi-square test needed for significance\n\n")
        
        # Numeric insights
        f.write("Numeric Features:\n")
        for feature in numeric_features:
            if feature in data.columns:
                high_rot_mean = data[data['rot_group'] == 'High ROT'][feature].mean()
                low_rot_mean = data[data['rot_group'] == 'Low ROT'][feature].mean()
                f.write(f"  {feature}:\n")
                f.write(f"    High ROT mean: {high_rot_mean:.2f}\n")
                f.write(f"    Low ROT mean: {low_rot_mean:.2f}\n")
                f.write(f"    Difference: {abs(high_rot_mean - low_rot_mean):.2f}\n")
                f.write(f"    T-test needed for significance\n\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 18 + "\n")
        f.write("1. Review the generated plots for visual patterns\n")
        f.write("2. Check statistical significance using the provided tests\n")
        f.write("3. Consider feature engineering based on insights\n")
        f.write("4. Validate findings with domain experts\n")
        f.write("5. Use insights to improve feature extraction\n\n")
        
        # Files generated
        f.write("FILES GENERATED\n")
        f.write("-" * 18 + "\n")
        f.write("Plots:\n")
        f.write("  - categorical_features_analysis.png\n")
        f.write("  - numeric_features_analysis.png\n")
        f.write("  - correlation_heatmap.png\n")
        f.write("  - Individual feature plots (*_analysis.png)\n")
        f.write("Reports:\n")
        f.write("  - eda_analysis.log\n")
        f.write("  - eda_summary_report.txt (this file)\n")
    
    print(f"✅ Summary report saved to: {report_file}")

def main():
    """Main function with error handling"""
    try:
        run_comprehensive_eda()
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logging.error(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main() 