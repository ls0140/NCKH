# main.py

import logging
from collectors.arxiv_collector import fetch_arxiv_papers
from metrics import calculate_and_store_rot

# Configure logging to file and console for robustness 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)

def run_pipeline():
    """
    Executes the full data collection and processing pipeline.
    """
    logging.info("=============================================")
    logging.info("Pipeline run started.")
    logging.info("=============================================")

    # --- 1. Data Collection ---
    # The complex query is now built inside the collector function.
    # We only need to specify the max number of results here.
<<<<<<< HEAD
    MAX_RESULTS = 500
=======
    MAX_RESULTS = 50  # Reduced from 500 to get a smaller dataset
>>>>>>> 133d86d (Done 1.2)
    
    # Call the updated function
    fetch_arxiv_papers(max_results=MAX_RESULTS)
    
    # --- 2. Metric Calculation ---
    logging.info("Skipping ROT calculation as arXiv does not provide citation data.")
    # calculate_and_store_rot()

    logging.info("=============================================")
    logging.info("Pipeline run finished.")
    logging.info("=============================================")


if __name__ == '__main__':
    run_pipeline()