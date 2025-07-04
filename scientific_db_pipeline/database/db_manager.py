# database/db_manager.py

import datetime
import logging
from sqlalchemy.orm import sessionmaker
from .schema import Paper, Author, PaperAuthors, CitationHistory, get_engine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a session factory
engine = get_engine()
Session = sessionmaker(bind=engine)

def add_or_update_paper(paper_data: dict):
    """
    Adds a new paper or updates an existing one based on its source URL.
    This function implements the "upsert" logic. 
    """
    session = Session()
    try:
        # Check if a paper with this source URL already exists 
        # Use source_url instead of DOI since arXiv papers don't have DOIs
        existing_paper = session.query(Paper).filter(Paper.source_url == paper_data.get('source_url')).first()

        if existing_paper:
            # Paper exists, update its information 
            logging.info(f"Updating existing paper with URL: {paper_data['source_url']}")
            existing_paper.citation_count = paper_data.get('citation_count')
            existing_paper.last_updated = datetime.datetime.utcnow()
            paper_to_update = existing_paper
        else:
            # Paper does not exist, insert a new record 
            logging.info(f"Inserting new paper: {paper_data['title']}")
            new_paper = Paper(
                title=paper_data['title'],
                abstract=paper_data['abstract'],
                publication_year=paper_data['publication_year'],
                doi=paper_data.get('doi'),
                source_url=paper_data['source_url'],
                pdf_url=paper_data.get('pdf_url'),
                source_db=paper_data['source_db'],
                citation_count=paper_data.get('citation_count'),
                last_updated=datetime.datetime.utcnow()
            )
            session.add(new_paper)
            session.flush()  # Flush to get the new_paper.paper_id
            paper_to_update = new_paper

            # Handle authors
            for i, author_name in enumerate(paper_data['authors']):
                # Check if author exists
                author = session.query(Author).filter(Author.full_name == author_name).first()
                if not author:
                    author = Author(full_name=author_name)
                    session.add(author)
                    session.flush() # Flush to get the new author_id
                
                # Create the link between paper and author 
                paper_author_link = PaperAuthors(
                    paper_id=paper_to_update.paper_id,
                    author_id=author.author_id,
                    author_order=i + 1
                )
                session.add(paper_author_link)

        # Add a record to citation history for trend analysis 
        if paper_to_update.citation_count is not None:
             history_entry = CitationHistory(
                 paper_id=paper_to_update.paper_id,
                 check_date=datetime.date.today(),
                 citation_count=paper_to_update.citation_count
             )
             session.add(history_entry)

        session.commit()
    except Exception as e:
        logging.error(f"Database error for paper '{paper_data.get('title')}': {e}", exc_info=True)
        session.rollback()
    finally:
        session.close()