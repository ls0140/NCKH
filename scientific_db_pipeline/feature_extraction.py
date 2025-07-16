# feature_extraction.py

import logging
import re
import pandas as pd
import textstat
from sqlalchemy import insert, Column, Integer, String, Boolean, Float, ForeignKey
from sqlalchemy.orm import sessionmaker
from database.schema import get_engine, Base

<<<<<<< HEAD
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
=======
# Try to ensure the word list is available
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

# Load the set of common English words (lowercase)
COMMON_WORDS_SET = set(word.lower() for word in nltk_words.words())

# Optionally, add more common words or domain-specific stopwords


def calculate_jargon_score(abstract, common_words_set=COMMON_WORDS_SET):
    """
    Calculate the proportion of words in the abstract that are NOT in the common English word list.
    Returns a float between 0 and 1.
    """
    if not abstract or len(abstract.strip()) < 10:
        return 0.0
    # Tokenize words (simple split, remove punctuation)
    words = re.findall(r"\b\w+\b", abstract.lower())
    if not words:
        return 0.0
    jargon_words = [w for w in words if w not in common_words_set]
    return len(jargon_words) / len(words)

# Configure OpenAI
openai.api_key = "API KEY TOP SECRET"
>>>>>>> b766010 (Top secret area 51)

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
        
<<<<<<< HEAD
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
=======
        for i, paper in enumerate(papers):
            try:
                # Always process and overwrite features
                # Extract basic features first
                basic_features = extract_basic_features(paper.abstract)
                if not basic_features:
                    skipped_count += 1
                    continue
                
                # Use AI to get additional features (with rate limiting)
                ai_features = None
                if i % 3 == 0:  # Only process every 3rd paper with AI to reduce API calls
                    ai_features = analyze_abstract_with_ai(paper.abstract)
                    if ai_features:
                        ai_processed_count += 1
                        # Add longer delay after AI processing
                        time.sleep(2 + random.uniform(0, 1))
                
                if ai_features:
                    # Combine basic and AI features
                    combined_features = {**basic_features, **ai_features}
                else:
                    # Use only basic features with defaults
                    combined_features = {
                        **basic_features,
                        'jargon_score': 0
                    }
                
                # Update or create features record
                existing_features = session.query(PaperFeatures).filter_by(paper_id=paper.paper_id).first()
                if existing_features:
                    # Overwrite all fields
                    for key, value in combined_features.items():
                        if hasattr(existing_features, key):
                            setattr(existing_features, key, value)
                    else:
                    # Create new record
                    features = PaperFeatures(
                        paper_id=paper.paper_id,
                        **combined_features
                    )
                    session.add(features)
                
                processed_count += 1
                
                # Commit every 5 papers to avoid memory issues
                if processed_count % 5 == 0:
                    session.commit()
                    logging.info(f"Processed {processed_count}/{len(papers)} papers (AI: {ai_processed_count}, Skipped: {skipped_count})")
                    # Add delay between batches
                    time.sleep(1 + random.uniform(0, 0.5))
                
            except Exception as e:
                logging.error(f"Error processing paper {paper.paper_id}: {e}")
                skipped_count += 1
>>>>>>> 3f86cd5 (Update ALOT of thingS)
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
<<<<<<< HEAD
        
        logging.info("🚀 Successfully saved new feature sets. Stage 2 is complete!")
=======
        logging.info(f"Feature extraction complete. Processed {processed_count} papers, AI enhanced: {ai_processed_count}, Skipped: {skipped_count}")
>>>>>>> 3f86cd5 (Update ALOT of thingS)

    except Exception as e:
        logging.error(f"An error occurred during feature extraction: {e}", exc_info=True)
        session.rollback()
    finally:
        session.close()

<<<<<<< HEAD
if __name__ == '__main__':
    extract_and_store_features()
=======
def analyze_feature_reliability():
    """
    Analyze the reliability of extracted features
    """
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Get all features
        features = session.query(PaperFeatures).all()
        
        if not features:
            logging.info("No features found in database")
            return
        
        # Convert to DataFrame for analysis
        data = []
        for feature in features:
            data.append({
                'paper_id': feature.paper_id,
                'abstract_word_count': feature.abstract_word_count,
                'readability_flesch_score': feature.readability_flesch_score,
                'avg_sentence_length': feature.avg_sentence_length,
                'jargon_score': feature.jargon_score,
                'mentions_dataset': feature.mentions_dataset,
                'mentions_metrics': feature.mentions_metrics,
                'has_github_link': feature.has_github_link
            })
        
        df = pd.DataFrame(data)
        
        # Basic statistics
        logging.info("=== Feature Reliability Analysis ===")
        logging.info(f"Total papers with features: {len(df)}")
        
        for column in df.columns:
            if column != 'paper_id':
                if df[column].dtype in ['int64', 'float64']:
                    logging.info(f"{column}: mean={df[column].mean():.2f}, std={df[column].std():.2f}, range=[{df[column].min():.2f}, {df[column].max():.2f}]")
        else:
                    logging.info(f"{column}: {df[column].value_counts().to_dict()}")
        
        # Check for missing values
        missing_data = df.isnull().sum()
        logging.info(f"Missing values: {missing_data.to_dict()}")
        
    except Exception as e:
        logging.error(f"Error in feature analysis: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze_feature_reliability()
    else:
        extract_and_store_features()
>>>>>>> 3f86cd5 (Update ALOT of thingS)
