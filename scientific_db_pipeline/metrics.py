# metrics.py

import datetime
import logging
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from database.schema import Paper, get_engine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a session factory
engine = get_engine()
Session = sessionmaker(bind=engine)

def calculate_and_store_rot():
    """
    Calculates the ROT (Rate of Citation) score for all papers in the database
    and updates their records. This function should be run after data collection is complete.
    """
    session = Session()
    logging.info("Starting ROT score calculation for all papers.")

    try:
        current_year = datetime.datetime.now().year

        # Fetch all papers that have the necessary data to calculate ROT
        papers_to_update = session.query(Paper).filter(
            Paper.citation_count.isnot(None),
            Paper.publication_year.isnot(None)
        ).all()

        if not papers_to_update:
            logging.warning("No papers found with both citation_count and publication_year. Cannot calculate ROT.")
            return

        updated_count = 0
        for paper in papers_to_update:
            # Explicitly check for required attributes, as hinted by the original comment.
            if paper.publication_year is not None and paper.citation_count is not None:
                # Calculate the age of the paper. Add 1 to avoid division by zero for papers
                # published in the current year.
                age = current_year - paper.publication_year + 1
                
                if age > 0:
                    # Calculate the ROT score
                    rot_score = paper.citation_count / age
                    paper.rot_score = rot_score  # Assuming your 'Paper' model has a 'rot_score' field
                    updated_count += 1
                else:
                    # Handle cases of invalid publication years (e.g., in the future)
                    logging.warning(f"Skipping paper {paper.id} with invalid publication year: {paper.publication_year}")

        if updated_count > 0:
            session.commit()
            logging.info(f"Successfully calculated ROT for and updated {updated_count} papers.")
        else:
            logging.info("No papers were updated.")

    except SQLAlchemyError as e:
        logging.error(f"A database error occurred: {e}")
        session.rollback() # Roll back changes on error
    except Exception as e:
        logging.error(f"An unexpected error occurred during ROT calculation: {e}")
        session.rollback()
    finally:
        session.close() # Ensure the session is always closed

# Example of how to run the function
if __name__ == '__main__':
    calculate_and_store_rot()