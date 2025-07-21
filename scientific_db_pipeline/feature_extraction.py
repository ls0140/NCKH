# feature_extraction.py

import os
import re
import logging
import pandas as pd
import numpy as np
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from database.schema import get_engine, Paper, PaperFeatures
import openai
from textstat import textstat
import time
import random
import json
import nltk
from nltk.corpus import words as nltk_words

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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_abstract_with_ai(abstract, max_retries=3):
    """
    Use OpenAI to analyze abstract and extract features with retry logic
    """
    for attempt in range(max_retries):
        try:
            prompt = f"""
            Analyze this scientific paper abstract and provide the following features as JSON:
            
            Abstract: {abstract}
            
            Please return ONLY a JSON object with these exact keys:
            {{
                "jargon_score": <percentage of technical terms, 0-100>,
                "readability_flesch_score": <Flesch reading ease score, 0-100>,
                "avg_sentence_length": <average words per sentence>,
                "abstract_word_count": <total word count>,
                "mentions_dataset": <true/false if mentions datasets>,
                "mentions_metrics": <true/false if mentions evaluation metrics>,
                "has_github_link": <true/false if contains GitHub link>
            }}
            
            Be precise and return only the JSON object.
            """
            
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a scientific text analyzer. Return only JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            
            result = json.loads(content)
            
            # Convert boolean strings to actual booleans
            result['mentions_dataset'] = bool(result['mentions_dataset'])
            result['mentions_metrics'] = bool(result['mentions_metrics'])
            result['has_github_link'] = bool(result['has_github_link'])
            
            return result
            
        except openai.RateLimitError:
            wait_time = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff
            logging.warning(f"Rate limit hit, waiting {wait_time:.1f} seconds (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_time)
            continue
        except Exception as e:
            logging.error(f"AI analysis failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(1)
    
    return None

def extract_basic_features(abstract):
    """
    Extract basic features without AI, including real jargon score
    """
    if not abstract or len(abstract.strip()) < 10:
        return None
    
    # Basic text statistics
    abstract_word_count = len(abstract.split())
    sentences = re.split(r'[.!?]+', abstract)
    sentences = [s.strip() for s in sentences if s.strip()]
    avg_sentence_length = abstract_word_count / len(sentences) if sentences else 0
    
    # Readability score
    try:
        readability_flesch_score = textstat.flesch_reading_ease(abstract)
    except:
        readability_flesch_score = 0
    
    # Check for GitHub links
    has_github_link = 'github.com' in abstract.lower() or 'github.io' in abstract.lower()
    
    # Check for dataset mentions
    dataset_keywords = ['dataset', 'data set', 'corpus', 'benchmark', 'evaluation set']
    mentions_dataset = any(keyword in abstract.lower() for keyword in dataset_keywords)
    
    # Check for evaluation metrics
    eval_keywords = ['accuracy', 'precision', 'recall', 'f1', 'f1-score', 'auc', 'roc', 'bleu', 'rouge', 'perplexity']
    mentions_metrics = any(keyword in abstract.lower() for keyword in eval_keywords)
    
    # Calculate jargon score
    jargon_score = calculate_jargon_score(abstract)
    
    return {
        'abstract_word_count': abstract_word_count,
        'avg_sentence_length': avg_sentence_length,
        'readability_flesch_score': readability_flesch_score,
        'has_github_link': has_github_link,
        'mentions_dataset': mentions_dataset,
        'mentions_metrics': mentions_metrics,
        'jargon_score': jargon_score
    }

def extract_and_store_features():
    """
    Extract features from paper abstracts and store in database with better rate limiting
    """
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Get all papers with abstracts
        papers = session.query(Paper).filter(Paper.abstract.isnot(None)).all()
        logging.info(f"Found {len(papers)} papers with abstracts")
        
        processed_count = 0
        ai_processed_count = 0
        skipped_count = 0
        
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
                continue
        
        session.commit()
        logging.info(f"Feature extraction complete. Processed {processed_count} papers, AI enhanced: {ai_processed_count}, Skipped: {skipped_count}")

    except Exception as e:
        logging.error(f"Error in feature extraction: {e}")
        session.rollback()
    finally:
        session.close()

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