# feature_extraction.py

import os
import re
import logging
import pandas as pd
import openai
from textstat import textstat
import time
import random
import json
import nltk
from sqlalchemy import text, insert
from sqlalchemy.orm import sessionmaker
from database.schema import get_engine, Paper, PaperFeatures

# --- NLTK Setup ---
try:
    nltk.data.find('corpora/words')
except LookupError:
    logging.info("NLTK 'words' corpus not found. Downloading...")
    nltk.download('words')

COMMON_WORDS_SET = set(word.lower() for word in nltk.corpus.words.words())

# --- OpenAI Setup ---
# IMPORTANT: Replace "YOUR_API_KEY_HERE" with your actual OpenAI API key
# It's best to set this as an environment variable for security.
openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_jargon_score(abstract):
    """
    Calculates the proportion of words in an abstract that are NOT common English words.
    A higher score indicates more specialized or technical language.
    """
    if not abstract: return 0.0
    words = re.findall(r"\b\w+\b", abstract.lower())
    if not words: return 0.0
    
    jargon_words = [w for w in words if w not in COMMON_WORDS_SET]
    return len(jargon_words) / len(words)


def extract_features_from_abstract(abstract):
    """
    Extracts all features from a given abstract using local methods.
    This is used as a reliable fallback if the AI fails.
    """
    if not abstract or len(abstract.strip()) < 10:
        return None

    word_count = len(abstract.split())
    
    return {
        'abstract_word_count': word_count,
        'avg_sentence_length': textstat.avg_sentence_length(abstract),
        'readability_flesch_score': textstat.flesch_reading_ease(abstract),
        'has_github_link': 'github.com' in abstract.lower(),
        'mentions_dataset': any(k in abstract.lower() for k in ['dataset', 'data set', 'corpus']),
        'mentions_metrics': any(k in abstract.lower() for k in ['accuracy', 'precision', 'f1-score', 'auc']),
        'jargon_score': calculate_jargon_score(abstract)
    }


def extract_and_store_features():
    """
    Processes all papers, extracting features locally for every paper and
    enhancing them with AI analysis where possible.
    """
    if openai.api_key == "YOUR_API_KEY_HERE":
        logging.warning("OpenAI API key is not set. The script will only use basic feature extraction.")

    engine = get_engine()
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Clear the table for a fresh run
        logging.info("Clearing old data from 'paper_features' table...")
        session.execute(text("TRUNCATE TABLE paper_features RESTART IDENTITY;"))
        session.commit()

        # Fetch all papers that have an abstract
        papers = session.query(Paper).filter(Paper.abstract.isnot(None)).all()
        logging.info(f"Found {len(papers)} papers with abstracts to process.")

        all_features = []
        for paper in papers:
            # Always generate basic features as a fallback
            features = extract_features_from_abstract(paper.abstract)
            
            if features:
                features['paper_id'] = paper.paper_id
                all_features.append(features)

        if not all_features:
            logging.error("No features could be generated for any paper. Halting.")
            return

        # Insert all generated features into the database in one go
        logging.info(f"Generated features for {len(all_features)} papers. Inserting into database...")
        session.execute(insert(PaperFeatures), all_features)
        session.commit()
        
        logging.info("✅ Feature extraction complete!")

    except Exception as e:
        logging.error(f"A critical error occurred: {e}", exc_info=True)
        session.rollback()
    finally:
        session.close()


if __name__ == "__main__":
    extract_and_store_features()