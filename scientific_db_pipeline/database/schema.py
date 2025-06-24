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
<<<<<<< HEAD
    UniqueConstraint
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Mapped, mapped_column
=======
    UniqueConstraint,
    Column
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
>>>>>>> 133d86d (Done 1.2)

# Define the base class for our declarative models
Base = declarative_base()

# Linking table for the many-to-many relationship between papers and authors
class PaperAuthors(Base):
    __tablename__ = 'paper_authors'
<<<<<<< HEAD
    paper_id: Mapped[int] = mapped_column(ForeignKey('papers.paper_id'), primary_key=True)
    author_id: Mapped[int] = mapped_column(ForeignKey('authors.author_id'), primary_key=True)
    author_order: Mapped[int] = mapped_column(Integer, nullable=False)

    # Establish the bidirectional relationship
    paper: Mapped["Paper"] = relationship(back_populates="author_associations")
    author: Mapped["Author"] = relationship(back_populates="paper_associations")
=======
    paper_id = Column(ForeignKey('papers.paper_id'), primary_key=True)
    author_id = Column(ForeignKey('authors.author_id'), primary_key=True)
    author_order = Column(Integer, nullable=False)

    # Establish the bidirectional relationship
    paper = relationship("Paper", back_populates="author_associations")
    author = relationship("Author", back_populates="paper_associations")
>>>>>>> 133d86d (Done 1.2)

class Paper(Base):
    __tablename__ = 'papers'
    
<<<<<<< HEAD
    # Use Mapped and mapped_column for explicit typing
    paper_id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    abstract: Mapped[Optional[str]] = mapped_column(Text)
    publication_year: Mapped[Optional[int]] = mapped_column(Integer, index=True)
    doi: Mapped[Optional[str]] = mapped_column(String(255), unique=True, index=True)
    source_url: Mapped[Optional[str]] = mapped_column(Text)
    pdf_url: Mapped[Optional[str]] = mapped_column(Text)
    source_db: Mapped[Optional[str]] = mapped_column(String(50))
    citation_count: Mapped[Optional[int]] = mapped_column(Integer)
    last_updated: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP)
    rot_score: Mapped[Optional[float]] = mapped_column(Float, index=True)

    # Relationship to the PaperAuthors association table
    author_associations: Mapped[List[PaperAuthors]] = relationship(back_populates="paper")
=======
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
>>>>>>> 133d86d (Done 1.2)
    
    def __repr__(self):
        return f"<Paper(title='{self.title[:30]}...', doi='{self.doi}')>"

class Author(Base):
    __tablename__ = 'authors'
    
<<<<<<< HEAD
    author_id: Mapped[int] = mapped_column(primary_key=True)
    full_name: Mapped[str] = mapped_column(String(255), index=True)
    affiliation: Mapped[Optional[str]] = mapped_column(Text)

    # Relationship to the PaperAuthors association table
    paper_associations: Mapped[List[PaperAuthors]] = relationship(back_populates="author")
=======
    author_id = Column(Integer, primary_key=True)
    full_name = Column(String(255), index=True)
    affiliation = Column(Text)

    # Relationship to the PaperAuthors association table
    paper_associations = relationship("PaperAuthors", back_populates="author")
>>>>>>> 133d86d (Done 1.2)
    
    # Ensure that an author's full name is unique
    __table_args__ = (UniqueConstraint('full_name'),)

    def __repr__(self):
        return f"<Author(full_name='{self.full_name}')>"

class CitationHistory(Base):
    """
    This table stores snapshots of citation counts over time for historical analysis.
    """
    __tablename__ = 'citation_history'
    
<<<<<<< HEAD
    history_id: Mapped[int] = mapped_column(primary_key=True)
    paper_id: Mapped[int] = mapped_column(ForeignKey('papers.paper_id'))
    check_date: Mapped[datetime.date] = mapped_column(Date)
    citation_count: Mapped[int] = mapped_column(Integer)
=======
    history_id = Column(Integer, primary_key=True)
    paper_id = Column(ForeignKey('papers.paper_id'))
    check_date = Column(Date)
    citation_count = Column(Integer)
>>>>>>> 133d86d (Done 1.2)
    
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