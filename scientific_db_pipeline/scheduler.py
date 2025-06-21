# scheduler.py

import logging
from apscheduler.schedulers.blocking import BlockingScheduler
from main import run_pipeline

# Configure logging for the scheduler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scheduler.log"),
        logging.StreamHandler()
    ]
)

def scheduled_job():
    """The job to be executed by the scheduler."""
    logging.info("Scheduler triggered the pipeline job.")
    try:
        run_pipeline()
    except Exception as e:
        logging.critical(f"The scheduled pipeline job failed catastrophically: {e}", exc_info=True)

if __name__ == '__main__':
    # Create a scheduler that will block the process
    scheduler = BlockingScheduler()
    
    # Schedule the job to run once every day at 2 AM
    scheduler.add_job(scheduled_job, 'interval', days=1, start_date='2025-06-18 02:00:00')
    
    logging.info("Scheduler started. The pipeline will run daily at 2 AM.")
    logging.info("Press Ctrl+C to exit.")
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logging.info("Scheduler stopped.")