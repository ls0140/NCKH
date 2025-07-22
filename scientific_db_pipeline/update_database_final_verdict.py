# update_database_final_verdict.py

import logging
import pandas as pd
import numpy as np
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from database.schema import Paper, get_engine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def update_database_final_verdict():
    """
    Updates the papers table in the database to add final_verdict column with 5-category classification.
    """
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # First, add the final_verdict column to the papers table if it doesn't exist
        logging.info("Adding final_verdict column to papers table...")
        
        # Check if column already exists
        result = session.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'papers' AND column_name = 'final_verdict'
        """))
        
        if not result.fetchone():
            # Add the column
            session.execute(text("ALTER TABLE papers ADD COLUMN final_verdict VARCHAR(50)"))
            session.commit()
            logging.info("Added final_verdict column to papers table")
        else:
            logging.info("final_verdict column already exists")
        
        # Get all papers with ROT scores
        papers = session.query(Paper).filter(Paper.rot_score.isnot(None)).all()
        logging.info(f"Found {len(papers)} papers with ROT scores")
        
        if not papers:
            logging.warning("No papers found with ROT scores")
            return False
        
        # Calculate percentiles for 5 categories
        rot_scores = [paper.rot_score for paper in papers]
        percentiles = [20, 40, 60, 80]
        thresholds = np.percentile(rot_scores, percentiles)
        
        logging.info(f"ROT Score thresholds for 5 categories:")
        logging.info(f"  Very Low ROT: < {thresholds[0]:.2f}")
        logging.info(f"  Low ROT: {thresholds[0]:.2f} - {thresholds[1]:.2f}")
        logging.info(f"  Medium ROT: {thresholds[1]:.2f} - {thresholds[2]:.2f}")
        logging.info(f"  High ROT: {thresholds[2]:.2f} - {thresholds[3]:.2f}")
        logging.info(f"  Very High ROT: > {thresholds[3]:.2f}")
        
        # Classify each paper
        def classify_rot(rot_score):
            if rot_score < thresholds[0]:
                return 'Very Low ROT'
            elif rot_score < thresholds[1]:
                return 'Low ROT'
            elif rot_score < thresholds[2]:
                return 'Medium ROT'
            elif rot_score < thresholds[3]:
                return 'High ROT'
            else:
                return 'Very High ROT'
        
        # Update each paper with its final_verdict
        updated_count = 0
        for paper in papers:
            final_verdict = classify_rot(paper.rot_score)
            paper.final_verdict = final_verdict
            updated_count += 1
        
        session.commit()
        logging.info(f"Updated {updated_count} papers with final_verdict")
        
        # Show distribution
        logging.info("\nFinal verdict distribution in database:")
        result = session.execute(text("""
            SELECT final_verdict, COUNT(*) as count 
            FROM papers 
            WHERE final_verdict IS NOT NULL 
            GROUP BY final_verdict 
            ORDER BY count DESC
        """))
        
        for row in result:
            logging.info(f"  {row.final_verdict}: {row.count} papers")
        
        return True
        
    except Exception as e:
        logging.error(f"Error updating database: {e}")
        session.rollback()
        return False
    finally:
        session.close()

if __name__ == '__main__':
    update_database_final_verdict() 