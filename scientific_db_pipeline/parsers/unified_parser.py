# parsers/unified_parser.py

import logging
import feedparser
from typing import Optional
from datetime import datetime

def parse_arxiv_entry(entry: feedparser.FeedParserDict) -> Optional[dict]:
    """
    Transforms a feedparser entry object from the arXiv API into a standardized dictionary.
    
    Args:
        entry: An entry from a feedparser-parsed Atom feed.

    Returns:
        A dictionary containing cleaned and structured paper data, or None if parsing fails.
    """
    try:
        # Extract author names into a simple list
        authors = [author.name for author in entry.authors]

        # Find the PDF link
        pdf_url = None
        source_url = None
        for link in entry.links:
            # The 'source_url' is the link to the abstract page
            if link.get('rel') == 'alternate' and link.get('type') == 'text/html':
                source_url = link.href
            # The 'pdf_url' is the direct link to the PDF
            if link.get('title') == 'pdf':
                pdf_url = link.href
        
        # If the abstract page link wasn't found, fall back to the main entry id
        if not source_url:
            source_url = entry.id

        # The publication date is a time.struct_time object.
        # We add '# type: ignore' to suppress the incorrect Pylance error.
        publication_year = entry.published_parsed.tm_year  # type: ignore

        # The unified, canonical data model
        paper_data = {
            'title': entry.title,
            'abstract': entry.summary,
            'authors': authors,
            'publication_year': publication_year,
            'doi': entry.get('doi'),
            'source_url': source_url,
            'pdf_url': pdf_url,
            'source_db': 'arXiv',
            'citation_count': None,
        }
        return paper_data
    except Exception as e:
        logging.error(f"Failed to parse feedparser entry '{getattr(entry, 'id', 'Unknown')}': {e}", exc_info=True)
        return None