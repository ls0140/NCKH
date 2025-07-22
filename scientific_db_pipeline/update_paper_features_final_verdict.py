# update_paper_features_final_verdict.py

import logging
import pandas as pd
import numpy as np
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from database.schema import get_engine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def update_paper_features_final_verdict():
    """
    Updates the paper_features table in the database to add final_verdict column with 5-category classification.
    """
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # First, add the final_verdict column to the paper_features table if it doesn't exist
        logging.info("Adding final_verdict column to paper_features table...")
        
        # Check if column already exists
        result = session.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'paper_features' AND column_name = 'final_verdict'
        """))
        
        if not result.fetchone():
            # Add the column
            session.execute(text("ALTER TABLE paper_features ADD COLUMN final_verdict VARCHAR(50)"))
            session.commit()
            logging.info("Added final_verdict column to paper_features table")
        else:
            logging.info("final_verdict column already exists in paper_features table")
        
        # Get all paper_features with their corresponding ROT scores from papers table
        logging.info("Getting paper_features with ROT scores...")
        result = session.execute(text("""
            SELECT pf.paper_id, p.rot_score
            FROM paper_features pf
            JOIN papers p ON pf.paper_id = p.paper_id
            WHERE p.rot_score IS NOT NULL
        """))
        
        paper_rot_scores = {row.paper_id: row.rot_score for row in result}
        logging.info(f"Found {len(paper_rot_scores)} paper_features with ROT scores")
        
        if not paper_rot_scores:
            logging.warning("No paper_features found with ROT scores")
            return False
        
        # Calculate percentiles for 5 categories
        rot_scores = list(paper_rot_scores.values())
        percentiles = [20, 40, 60, 80]
        thresholds = np.percentile(rot_scores, percentiles)
        
        logging.info(f"ROT Score thresholds for 5 categories:")
        logging.info(f"  Very Low ROT: < {thresholds[0]:.2f}")
        logging.info(f"  Low ROT: {thresholds[0]:.2f} - {thresholds[1]:.2f}")
        logging.info(f"  Medium ROT: {thresholds[1]:.2f} - {thresholds[2]:.2f}")
        logging.info(f"  High ROT: {thresholds[2]:.2f} - {thresholds[3]:.2f}")
        logging.info(f"  Very High ROT: > {thresholds[3]:.2f}")
        
        # Classify each paper feature
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
        
        # Update each paper feature with its final_verdict
        updated_count = 0
        for paper_id, rot_score in paper_rot_scores.items():
            final_verdict = classify_rot(rot_score)
            session.execute(text("""
                UPDATE paper_features 
                SET final_verdict = :final_verdict 
                WHERE paper_id = :paper_id
            """), {'final_verdict': final_verdict, 'paper_id': paper_id})
            updated_count += 1
        
        session.commit()
        logging.info(f"Updated {updated_count} paper_features with final_verdict")
        
        # Show distribution
        logging.info("\nFinal verdict distribution in paper_features table:")
        result = session.execute(text("""
            SELECT final_verdict, COUNT(*) as count 
            FROM paper_features 
            WHERE final_verdict IS NOT NULL 
            GROUP BY final_verdict 
            ORDER BY count DESC
        """))
        
        for row in result:
            logging.info(f"  {row.final_verdict}: {row.count} papers")
        
        return True
        
    except Exception as e:
        logging.error(f"Error updating paper_features table: {e}")
        session.rollback()
        return False
    finally:
        session.close()

if __name__ == '__main__':
    update_paper_features_final_verdict() 