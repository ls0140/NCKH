# calculate_verdict.py

import logging
import pandas as pd # type: ignore
from sqlalchemy import text
from database.schema import get_engine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_and_store_verdict():
    engine = get_engine()

    try:
        logging.info("Reading paper features from the database...")
        df = pd.read_sql("SELECT * FROM paper_features", engine, index_col='paper_id')

        if df.empty:
            logging.warning("The 'paper_features' table is empty. Please run feature_extraction.py first.")
            return

        # --- Verdict Calculation Logic (0-10 scale) ---
        # Normalize numerical features to a 0-1 scale
        normalized_readability = df['readability_flesch_score'].clip(0, 100) / 100.0
        normalized_jargon = df['jargon_score'].clip(0, 1) # Jargon score is already 0-1

        # Define weights for each component
        weights = {
            'readability': 6,   # Clarity is important
            'jargon': 4         # Technical depth is also important
        }

        # Calculate the final score, converting booleans to 1s and 0s
        df['final_verdict_score'] = (
            df['has_github_link'].astype(int) * weights['github'] +
            df['mentions_metrics'].astype(int) * weights['metrics'] +
            df['mentions_dataset'].astype(int) * weights['dataset'] +
            normalized_readability * weights['readability'] +
            normalized_jargon * weights['jargon']
        )
        
        # --- Categorize the Score ---
        def get_category(score):
            if score >= 5.5: return 'Excellent'
            if score >= 4.0: return 'Good'
            if score >= 3.0: return 'Average'
            return 'Poor'

        df['final_verdict_category'] = df['final_verdict_score'].apply(get_category)
        
        # --- Update the Database ---
        logging.info(f"Updating database with new verdict scores for {len(df)} papers...")
        update_data = df[['final_verdict_score', 'final_verdict_category']].reset_index()
        
        with engine.begin() as connection:
            for _, row in update_data.iterrows():
                connection.execute(
                    text("""
                    UPDATE paper_features
                    SET final_verdict_score = :score, final_verdict_category = :category
                    WHERE paper_id = :pid
                    """),
                    {'score': row['final_verdict_score'], 'category': row['final_verdict_category'], 'pid': row['paper_id']}
                )
        
        logging.info("✅ Final verdict update complete!")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)

if __name__ == '__main__':
    calculate_and_store_verdict()