# calculate_verdict.py

import logging
import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text # Import the text function
from database.schema import get_engine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_and_store_verdict():
    """
    Calculates a final verdict score based on existing features and
    updates the database.
    """
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        logging.info("Reading all paper features from the database...")
        query = "SELECT * FROM paper_features"
        df = pd.read_sql(query, engine, index_col='paper_id')

        # --- Verdict Calculation Logic ---
        df['normalized_readability'] = df['readability_flesch_score'] / 100.0
        df['normalized_jargon'] = 1 - (df['jargon_score'].clip(upper=15) / 15.0)

        weights = {
            'github': 3.0,
            'metrics': 2.5,
            'dataset': 2.0,
            'readability': 1.5,
            'jargon': 1.0,
        }

        df['final_verdict_score'] = (
            df['has_github_link'] * weights['github'] +
            df['mentions_metrics'] * weights['metrics'] +
            df['mentions_dataset'] * weights['dataset'] +
            df['normalized_readability'] * weights['readability'] +
            df['normalized_jargon'] * weights['jargon']
        )
        
        # --- Categorize the Score ---
        def get_category(score):
            if score >= 8.0:
                return 'Excellent'
            elif score >= 6.0:
                return 'Good'
            elif score >= 4.0:
                return 'Average'
            else:
                return 'Poor'

        df['final_verdict_category'] = df['final_verdict_score'].apply(get_category)
        
        # --- Update the Database ---
        logging.info("Updating the database with final verdict scores...")
        update_data = df[['final_verdict_score', 'final_verdict_category']].reset_index()
        
        with engine.begin() as connection:
            for _, row in update_data.iterrows():
                # Corrected: Wrap the SQL string in text()
                connection.execute(
                    text("""
                    UPDATE paper_features
                    SET final_verdict_score = :score, final_verdict_category = :category
                    WHERE paper_id = :pid
                    """),
                    {'score': row['final_verdict_score'], 'category': row['final_verdict_category'], 'pid': row['paper_id']}
                )
        
        logging.info("✅ Final verdict calculation and update complete!")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        session.rollback()
    finally:
        session.close()

if __name__ == '__main__':
    calculate_and_store_verdict()