# simple_rot_analyzer.py

import logging
import datetime
import csv
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func, desc
from database.schema import Paper, get_engine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a session factory
engine = get_engine()
Session = sessionmaker(bind=engine)

def calculate_rot_scores():
    """
    Calculates ROT (Rate of Citation) scores for all papers and updates the database.
    ROT = citation_count / paper_age
    """
    session = Session()
    logging.info("Starting ROT score calculation...")
    
    try:
        current_year = datetime.datetime.now().year
        
        # Get all papers with citation data
        papers = session.query(Paper).filter(
            Paper.citation_count.isnot(None),
            Paper.publication_year.isnot(None)
        ).all()
        
        if not papers:
            logging.warning("No papers found with citation data. Cannot calculate ROT.")
            return False
            
        updated_count = 0
        for paper in papers:
            age = current_year - paper.publication_year + 1
            if age > 0:
                rot_score = paper.citation_count / age
                paper.rot_score = rot_score
                updated_count += 1
                
        session.commit()
        logging.info(f"Successfully calculated ROT for {updated_count} papers.")
        return True
        
    except Exception as e:
        logging.error(f"Error calculating ROT scores: {e}")
        session.rollback()
        return False
    finally:
        session.close()

def analyze_rot_distribution():
    """
    Analyzes the distribution of ROT scores and suggests thresholds for High/Low ROT.
    """
    session = Session()
    
    try:
        # Get ROT statistics
        stats = session.query(
            func.count(Paper.paper_id).label('total_papers'),
            func.avg(Paper.rot_score).label('avg_rot'),
            func.min(Paper.rot_score).label('min_rot'),
            func.max(Paper.rot_score).label('max_rot')
        ).filter(Paper.rot_score.isnot(None)).first()
        
        if not stats or stats.total_papers == 0:
            logging.warning("No papers with ROT scores found.")
            return None
            
        # Calculate median manually since some PostgreSQL versions don't support PERCENTILE_CONT
        all_rot_scores = session.query(Paper.rot_score).filter(
            Paper.rot_score.isnot(None)
        ).order_by(Paper.rot_score).all()
        
        rot_scores = [score[0] for score in all_rot_scores]
        median_rot = rot_scores[len(rot_scores) // 2] if rot_scores else 0
        
        logging.info(f"ROT Statistics:")
        logging.info(f"  Total papers: {stats.total_papers}")
        logging.info(f"  Average ROT: {stats.avg_rot:.2f}")
        logging.info(f"  Median ROT: {median_rot:.2f}")
        logging.info(f"  Min ROT: {stats.min_rot:.2f}")
        logging.info(f"  Max ROT: {stats.max_rot:.2f}")
        
        return {
            'total_papers': stats.total_papers,
            'avg_rot': stats.avg_rot,
            'median_rot': median_rot,
            'min_rot': stats.min_rot,
            'max_rot': stats.max_rot
        }
        
    except Exception as e:
        logging.error(f"Error analyzing ROT distribution: {e}")
        return None
    finally:
        session.close()

def group_papers_by_rot(high_rot_threshold=None, low_rot_threshold=None):
    """
    Groups papers into High ROT and Low ROT categories.
    
    Args:
        high_rot_threshold: Papers with ROT >= this value are High ROT
        low_rot_threshold: Papers with ROT <= this value are Low ROT
    """
    session = Session()
    
    try:
        # If no thresholds provided, use median as separator
        if high_rot_threshold is None or low_rot_threshold is None:
            all_rot_scores = session.query(Paper.rot_score).filter(
                Paper.rot_score.isnot(None)
            ).order_by(Paper.rot_score).all()
            
            rot_scores = [score[0] for score in all_rot_scores]
            median_rot = rot_scores[len(rot_scores) // 2] if rot_scores else 0
            
            if high_rot_threshold is None:
                high_rot_threshold = median_rot
            if low_rot_threshold is None:
                low_rot_threshold = median_rot
        
        # Get High ROT papers
        high_rot_papers = session.query(Paper).filter(
            Paper.rot_score >= high_rot_threshold
        ).order_by(desc(Paper.rot_score)).all()
        
        # Get Low ROT papers
        low_rot_papers = session.query(Paper).filter(
            Paper.rot_score <= low_rot_threshold
        ).order_by(Paper.rot_score).all()
        
        # Get papers in the middle (if any)
        middle_papers = session.query(Paper).filter(
            Paper.rot_score > low_rot_threshold,
            Paper.rot_score < high_rot_threshold
        ).all()
        
        logging.info(f"\n=== ROT GROUPING RESULTS ===")
        logging.info(f"High ROT threshold: {high_rot_threshold:.2f}")
        logging.info(f"Low ROT threshold: {low_rot_threshold:.2f}")
        logging.info(f"High ROT papers: {len(high_rot_papers)}")
        logging.info(f"Low ROT papers: {len(low_rot_papers)}")
        logging.info(f"Middle papers: {len(middle_papers)}")
        
        return {
            'high_rot': high_rot_papers,
            'low_rot': low_rot_papers,
            'middle': middle_papers,
            'thresholds': {
                'high': high_rot_threshold,
                'low': low_rot_threshold
            }
        }
        
    except Exception as e:
        logging.error(f"Error grouping papers by ROT: {e}")
        return None
    finally:
        session.close()

def print_paper_details(papers, group_name):
    """
    Prints detailed information about papers in a group.
    """
    print(f"\n=== {group_name.upper()} ROT PAPERS ===")
    print(f"Total: {len(papers)} papers")
    print("-" * 80)
    
    for i, paper in enumerate(papers[:10], 1):  # Show first 10 papers
        print(f"{i}. Title: {paper.title[:60]}...")
        print(f"   ROT Score: {paper.rot_score:.2f}")
        print(f"   Citations: {paper.citation_count}")
        print(f"   Year: {paper.publication_year}")
        print(f"   DOI: {paper.doi}")
        print(f"   Abstract: {paper.abstract[:100]}..." if paper.abstract else "   Abstract: None")
        print()
    
    if len(papers) > 10:
        print(f"... and {len(papers) - 10} more papers")

def export_rot_groups_to_csv():
    """
    Exports the ROT groups to CSV files for easy analysis.
    """
    session = Session()
    
    try:
        # Get all papers with ROT scores
        papers = session.query(Paper).filter(
            Paper.rot_score.isnot(None)
        ).all()
        
        if not papers:
            logging.warning("No papers with ROT scores found.")
            return False
        
        # Calculate median for grouping
        rot_scores = [paper.rot_score for paper in papers]
        rot_scores.sort()
        median_rot = rot_scores[len(rot_scores) // 2]
        
        # Prepare data for CSV export
        high_rot_data = []
        low_rot_data = []
        
        for paper in papers:
            paper_data = {
                'paper_id': paper.paper_id,
                'title': paper.title,
                'abstract': paper.abstract,
                'publication_year': paper.publication_year,
                'citation_count': paper.citation_count,
                'rot_score': paper.rot_score,
                'doi': paper.doi,
                'source_url': paper.source_url
            }
            
            if paper.rot_score >= median_rot:
                paper_data['rot_group'] = 'High ROT'
                high_rot_data.append(paper_data)
            else:
                paper_data['rot_group'] = 'Low ROT'
                low_rot_data.append(paper_data)
        
        # Export High ROT papers
        if high_rot_data:
            with open('high_rot_papers.csv', 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['paper_id', 'title', 'abstract', 'publication_year', 
                             'citation_count', 'rot_score', 'doi', 'source_url', 'rot_group']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(high_rot_data)
        
        # Export Low ROT papers
        if low_rot_data:
            with open('low_rot_papers.csv', 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['paper_id', 'title', 'abstract', 'publication_year', 
                             'citation_count', 'rot_score', 'doi', 'source_url', 'rot_group']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(low_rot_data)
        
        # Export all papers combined
        all_papers_data = high_rot_data + low_rot_data
        if all_papers_data:
            with open('all_papers_with_rot_groups.csv', 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['paper_id', 'title', 'abstract', 'publication_year', 
                             'citation_count', 'rot_score', 'doi', 'source_url', 'rot_group']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_papers_data)
        
        logging.info(f"Exported {len(high_rot_data)} High ROT papers to 'high_rot_papers.csv'")
        logging.info(f"Exported {len(low_rot_data)} Low ROT papers to 'low_rot_papers.csv'")
        logging.info(f"Exported all {len(all_papers_data)} papers to 'all_papers_with_rot_groups.csv'")
        
        return True
        
    except Exception as e:
        logging.error(f"Error exporting to CSV: {e}")
        return False
    finally:
        session.close()

def main():
    """
    Main function to run the complete ROT analysis and grouping.
    """
    print("=== SCIENTIFIC PAPERS ROT ANALYSIS (SIMPLE VERSION) ===")
    print("This script will analyze your papers and group them by ROT (Reference of Time)")
    print()
    
    # Step 1: Calculate ROT scores
    print("Step 1: Calculating ROT scores...")
    if not calculate_rot_scores():
        print("Failed to calculate ROT scores. Exiting.")
        return
    
    # Step 2: Analyze distribution
    print("\nStep 2: Analyzing ROT distribution...")
    stats = analyze_rot_distribution()
    if not stats:
        print("Failed to analyze ROT distribution. Exiting.")
        return
    
    # Step 3: Group papers
    print("\nStep 3: Grouping papers by ROT...")
    groups = group_papers_by_rot()
    if not groups:
        print("Failed to group papers. Exiting.")
        return
    
    # Step 4: Print details
    print_paper_details(groups['high_rot'], 'High ROT')
    print_paper_details(groups['low_rot'], 'Low ROT')
    
    # Step 5: Export to CSV
    print("\nStep 4: Exporting results to CSV files...")
    export_rot_groups_to_csv()
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("You can now:")
    print("1. View the CSV files in your project directory")
    print("2. Open pgAdmin 4 and run queries on your database")
    print("3. Use the ROT groups for your team analysis")

if __name__ == '__main__':
    main() 