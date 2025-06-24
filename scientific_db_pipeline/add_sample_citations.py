# add_sample_citations.py

import random
import logging
from sqlalchemy.orm import sessionmaker
from database.schema import Paper, get_engine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a session factory
engine = get_engine()
Session = sessionmaker(bind=engine)

def add_sample_citations():
    """
    Adds sample citation data to papers for demonstration purposes.
    In a real scenario, you would fetch this data from citation databases.
    """
    session = Session()
    
    try:
        # Get all papers
        papers = session.query(Paper).all()
        
        if not papers:
            logging.warning("No papers found in database.")
            return False
            
        logging.info(f"Adding sample citations to {len(papers)} papers...")
        
        updated_count = 0
        for paper in papers:
            # Generate realistic citation counts based on publication year
            # Newer papers tend to have fewer citations, older papers more
            current_year = 2025
            paper_age = current_year - paper.publication_year + 1
            
            # Base citation count based on age and some randomness
            if paper_age <= 1:
                # Very recent papers: 0-50 citations
                base_citations = random.randint(0, 50)
            elif paper_age <= 3:
                # Recent papers: 10-200 citations
                base_citations = random.randint(10, 200)
            elif paper_age <= 5:
                # Medium age papers: 50-500 citations
                base_citations = random.randint(50, 500)
            else:
                # Older papers: 100-2000 citations
                base_citations = random.randint(100, 2000)
            
            # Add some randomness and make some papers more "trending"
            if random.random() < 0.1:  # 10% chance of being a "trending" paper
                base_citations *= random.uniform(2, 5)
            
            # Add some noise
            final_citations = int(base_citations * random.uniform(0.8, 1.2))
            
            paper.citation_count = final_citations
            updated_count += 1
            
        session.commit()
        logging.info(f"Successfully added sample citations to {updated_count} papers.")
        return True
        
    except Exception as e:
        logging.error(f"Error adding sample citations: {e}")
        session.rollback()
        return False
    finally:
        session.close()

def main():
    """
    Main function to add sample citation data.
    """
    print("=== ADDING SAMPLE CITATION DATA ===")
    print("This script adds realistic sample citation data to your papers")
    print("for demonstration of the ROT analysis functionality.")
    print()
    
    if add_sample_citations():
        print("Sample citations added successfully!")
        print("You can now run the ROT analysis with: python rot_analyzer.py")
    else:
        print("Failed to add sample citations.")

if __name__ == '__main__':
    main() 