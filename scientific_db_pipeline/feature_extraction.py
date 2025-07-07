# feature_extraction.py

import logging
import re
import pandas as pd
import textstat
from sqlalchemy import insert, Column, Integer, String, Boolean, Float, ForeignKey
from sqlalchemy.orm import sessionmaker
from database.schema import get_engine, Base

# --- Configuration: Feature Keywords ---
JARGON_LIST = [
    'neural network', 'deep learning', 'transformer', 'gan', 'convolutional',
    'algorithm', 'python', 'tensorflow', 'pytorch', 'backpropagation',
    'gradient descent', 'hyperparameter', 'regularization', 'svm', 'big data'
]
DATASET_KEYWORDS = [
    'dataset', 'corpus', 'benchmark', 'data set', 'training data', 'validation set'
]
METRIC_KEYWORDS = [
    'accuracy', 'precision', 'recall', 'f1-score', 'f1 score', 'auc', 'roc', 'mean squared error', 'mse'
]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the table schema within the script
class PaperFeatures(Base):
    __tablename__ = 'paper_features'
    paper_id = Column(Integer, ForeignKey('papers.paper_id'), primary_key=True)
    abstract_word_count = Column(Integer)
    avg_sentence_length = Column(Float)
    readability_flesch_score = Column(Float)
    jargon_score = Column(Float)
    mentions_dataset = Column(Boolean)
    mentions_metrics = Column(Boolean)
    has_github_link = Column(Boolean)


def extract_and_store_features():
    """
    Connects to the DB, extracts features from paper abstracts,
    and stores them in the `paper_features` table.
    """
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        query = """
        SELECT p.paper_id, p.abstract
        FROM papers p
        LEFT JOIN paper_features f ON p.paper_id = f.paper_id
        WHERE f.paper_id IS NULL;
        """
        
        logging.info("Reading papers without features from the database...")
        df_papers = pd.read_sql(query, engine)

        # Add these three lines for debugging
        print("--- DEBUG: SQL QUERY BEING EXECUTED ---")
        print(query)
        print("---------------------------------------")
        
        # This line should already be in your code
        df_papers = pd.read_sql(query, engine)

        if df_papers.empty:
            logging.info("✅ All papers already have features.")
            return

        logging.info(f"Found {len(df_papers)} new papers. Starting feature extraction... ⚙️")
        
        new_features = []
        for _, row in df_papers.iterrows():
            paper_id = int(row['paper_id'])
            abstract = str(row['abstract'])

            if not abstract or pd.isna(abstract) or len(abstract.split()) < 20:
                continue

            word_count = len(abstract.split())
            abstract_lower = abstract.lower()

            # --- Feature Calculation ---
            feature_obj = {
                'paper_id': paper_id,
                'abstract_word_count': word_count,
                'avg_sentence_length': textstat.avg_sentence_length(abstract),  # type: ignore
                'readability_flesch_score': textstat.flesch_reading_ease(abstract),  # type: ignore
                'jargon_score': (sum(1 for term in JARGON_LIST if term in abstract_lower) / word_count) * 100,
                'mentions_dataset': any(keyword in abstract_lower for keyword in DATASET_KEYWORDS),
                'mentions_metrics': any(keyword in abstract_lower for keyword in METRIC_KEYWORDS),
                'has_github_link': bool(re.search(r'github\.com', abstract_lower))
            }
            new_features.append(feature_obj)

        if not new_features:
            logging.warning("No valid features were generated.")
            return

        logging.info(f"Saving {len(new_features)} new feature sets...")
        session.execute(insert(PaperFeatures), new_features)
        session.commit()
        
        logging.info("🚀 Successfully saved new feature sets. Stage 2 is complete!")

    except Exception as e:
        logging.error(f"An error occurred during feature extraction: {e}", exc_info=True)
        session.rollback()
    finally:
        session.close()

if __name__ == '__main__':
    extract_and_store_features()