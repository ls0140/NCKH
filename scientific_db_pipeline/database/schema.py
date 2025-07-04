# database/schema.py

import datetime
from typing import List, Optional

from sqlalchemy import (
    create_engine,
    ForeignKey,
    String,
    Text,
    Integer,
    Float,
    TIMESTAMP,
    Date,
    UniqueConstraint,
    Column
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# Define the base class for our declarative models
Base = declarative_base()

# Linking table for the many-to-many relationship between papers and authors
class PaperAuthors(Base):
    __tablename__ = 'paper_authors'
    paper_id = Column(ForeignKey('papers.paper_id'), primary_key=True)
    author_id = Column(ForeignKey('authors.author_id'), primary_key=True)
    author_order = Column(Integer, nullable=False)

    # Establish the bidirectional relationship
    paper = relationship("Paper", back_populates="author_associations")
    author = relationship("Author", back_populates="paper_associations")

class Paper(Base):
    __tablename__ = 'papers'
    
    paper_id = Column(Integer, primary_key=True)
    title = Column(Text)
    abstract = Column(Text)
    publication_year = Column(Integer, index=True)
    doi = Column(String(255), unique=True, index=True)
    source_url = Column(Text)
    pdf_url = Column(Text)
    source_db = Column(String(50))
    citation_count = Column(Integer)
    last_updated = Column(TIMESTAMP)
    rot_score = Column(Float, index=True)

    # Relationship to the PaperAuthors association table
    author_associations = relationship("PaperAuthors", back_populates="paper")
    
    def __repr__(self):
        return f"<Paper(title='{self.title[:30]}...', doi='{self.doi}')>"

class Author(Base):
    __tablename__ = 'authors'
    
    author_id = Column(Integer, primary_key=True)
    full_name = Column(String(255), index=True)
    affiliation = Column(Text)

    # Relationship to the PaperAuthors association table
    paper_associations = relationship("PaperAuthors", back_populates="author")
    
    # Ensure that an author's full name is unique
    __table_args__ = (UniqueConstraint('full_name'),)

    def __repr__(self):
        return f"<Author(full_name='{self.full_name}')>"

class CitationHistory(Base):
    """
    This table stores snapshots of citation counts over time for historical analysis.
    """
    __tablename__ = 'citation_history'
    
    history_id = Column(Integer, primary_key=True)
    paper_id = Column(ForeignKey('papers.paper_id'))
    check_date = Column(Date)
    citation_count = Column(Integer)
    
    def __repr__(self):
        return f"<CitationHistory(paper_id={self.paper_id}, date='{self.check_date}', count={self.citation_count})>"

def get_engine():
    """Creates a database engine instance."""
    # Replace with your actual database credentials
    DATABASE_URL = "postgresql://postgres:123@localhost:5432/scientific_papers"
    engine = create_engine(DATABASE_URL)
    return engine

def create_tables(engine):
    """Creates all the tables in the database based on the schema."""
    # The following line drops all tables before creating them.
    # Use with caution, as it will delete all data.
    # Base.metadata.drop_all(engine) 
    Base.metadata.create_all(engine)
    print("Tables created successfully.")

if __name__ == '__main__':
    # This allows you to re-create the tables by running `python database/schema.py`
    # Warning: If your tables already exist, this will not update them.
    # You may need to drop them first for changes to apply.
    engine = get_engine()
    create_tables(engine)