# clear_database.py

import logging
from sqlalchemy.orm import sessionmaker
from database.schema import Paper, Author, PaperAuthors, CitationHistory, get_engine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a session factory
engine = get_engine()
Session = sessionmaker(bind=engine)

def clear_database():
    """
    Clears all data from the database tables.
    """
    session = Session()
    
    try:
        print("Clearing database...")
        
        # Delete in order to respect foreign key constraints
        session.query(CitationHistory).delete()
        session.query(PaperAuthors).delete()
        session.query(Paper).delete()
        session.query(Author).delete()
        
        session.commit()
        print("Database cleared successfully!")
        
    except Exception as e:
        logging.error(f"Error clearing database: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == '__main__':
    clear_database() 