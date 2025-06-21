# collectors/arxiv_collector.py

import logging
import requests
import feedparser
import time
from urllib.parse import urlencode
from parsers.unified_parser import parse_arxiv_entry
from database.db_manager import add_or_update_paper

def fetch_arxiv_papers(max_results: int = 500):
    """
    Fetches papers by making a direct HTTP request to the arXiv API and parsing the Atom XML response.
    """
    base_url = 'http://export.arxiv.org/api/query?'

    page_size = 100
    start = 0

    # **THE FINAL QUERY**
    # We are now targeting the specific, high-traffic sub-categories for AI and ML.
    # This will yield the rich set of results you're looking for.
    query = '(cat:cs.LG OR cat:cs.AI OR cat:cs.CV OR cat:cs.CL OR cat:cs.NE) AND all:"machine learning"'
    
    logging.info(f"Starting direct API fetch with targeted query: '{query}'")

    papers_fetched = 0
    # We will fetch up to max_results papers
    while papers_fetched < max_results:
        results_this_page = min(page_size, max_results - papers_fetched)

        params = {
            'search_query': query,
            'start': start,
            'max_results': results_this_page,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        url = base_url + urlencode(params)
        logging.info(f"Requesting URL: {url}")
        
        try:
            response = requests.get(url)
            if response.status_code != 200:
                logging.error(f"HTTP Error {response.status_code}. Response: {response.text}")
                break

            feed = feedparser.parse(response.content)

            if feed.entries and 'errors' in feed.entries[0].id:
                error_summary = feed.entries[0].summary
                logging.error(f"API returned an error: {error_summary}")
                break

            if not feed.entries:
                logging.info("No more entries found. This may be the end of the results.")
                break

            if papers_fetched == 0:
                 logging.info("SUCCESS! API returned valid results. Processing...")

            for entry in feed.entries:
                parsed_data = parse_arxiv_entry(entry)
                if parsed_data:
                    add_or_update_paper(parsed_data)
                # We count the paper fetched even if parsing fails to advance the page
                papers_fetched += 1
            
            logging.info(f"Processed {papers_fetched} of {max_results} papers.")

            # Set the start for the next page
            start += page_size
            
            time.sleep(2.0)

        except requests.exceptions.RequestException as e:
            logging.error(f"A network error occurred: {e}", exc_info=True)
            break
            
    logging.info(f"Finished. Total papers fetched: {papers_fetched}")