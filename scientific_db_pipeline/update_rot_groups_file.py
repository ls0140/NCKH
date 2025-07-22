# update_rot_groups_file.py

import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def update_rot_groups_file():
    """
    Updates the original all_papers_with_rot_groups.csv file to add 5-category final_verdict.
    """
    try:
        # Read the original CSV file
        input_file = 'sorted_rot/all_papers_with_rot_groups.csv'
        df = pd.read_csv(input_file)
        
        logging.info(f"Loaded {len(df)} papers from {input_file}")
        
        # Get ROT scores and calculate percentiles for 5 categories
        rot_scores = df['rot_score'].dropna()
        
        # Calculate percentiles for 5 equal groups (20% each)
        percentiles = [20, 40, 60, 80]
        thresholds = np.percentile(rot_scores, percentiles)
        
        logging.info(f"ROT Score thresholds for 5 categories:")
        logging.info(f"  Very Low ROT: < {thresholds[0]:.2f}")
        logging.info(f"  Low ROT: {thresholds[0]:.2f} - {thresholds[1]:.2f}")
        logging.info(f"  Medium ROT: {thresholds[1]:.2f} - {thresholds[2]:.2f}")
        logging.info(f"  High ROT: {thresholds[2]:.2f} - {thresholds[3]:.2f}")
        logging.info(f"  Very High ROT: > {thresholds[3]:.2f}")
        
        # Create the 5-category classification
        def classify_rot(rot_score):
            if pd.isna(rot_score):
                return 'Unknown'
            elif rot_score < thresholds[0]:
                return 'Very Low ROT'
            elif rot_score < thresholds[1]:
                return 'Low ROT'
            elif rot_score < thresholds[2]:
                return 'Medium ROT'
            elif rot_score < thresholds[3]:
                return 'High ROT'
            else:
                return 'Very High ROT'
        
        # Add the final_verdict column to the existing dataframe
        df['final_verdict'] = df['rot_score'].apply(classify_rot)
        
        # Count papers in each category
        category_counts = df['final_verdict'].value_counts()
        logging.info(f"\nPapers in each category:")
        for category, count in category_counts.items():
            percentage = (count / len(df)) * 100
            logging.info(f"  {category}: {count} papers ({percentage:.1f}%)")
        
        # Save back to the original file (overwrite it)
        df.to_csv(input_file, index=False)
        
        logging.info(f"\nUpdated original file: {input_file}")
        
        # Create summary statistics
        summary_data = []
        for category in ['Very Low ROT', 'Low ROT', 'Medium ROT', 'High ROT', 'Very High ROT']:
            category_papers = df[df['final_verdict'] == category]
            if len(category_papers) > 0:
                avg_rot = category_papers['rot_score'].mean()
                avg_citations = category_papers['citation_count'].mean()
                avg_year = category_papers['publication_year'].mean()
                summary_data.append({
                    'Category': category,
                    'Count': len(category_papers),
                    'Percentage': (len(category_papers) / len(df)) * 100,
                    'Avg_ROT_Score': avg_rot,
                    'Avg_Citations': avg_citations,
                    'Avg_Publication_Year': avg_year
                })
        
        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_file = 'sorted_rot/five_category_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        
        logging.info(f"Saved summary to: {summary_file}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error updating ROT groups file: {e}")
        return False

if __name__ == '__main__':
    update_rot_groups_file() 