# create_clean_csv.py

import csv
import re
from sqlalchemy.orm import sessionmaker
from database.schema import Paper, get_engine

# Create a session factory
engine = get_engine()
Session = sessionmaker(bind=engine)

def clean_text(text):
    """Clean text by removing extra whitespace and line breaks"""
    if not text:
        return ""
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove line breaks
    text = text.replace('\n', ' ').replace('\r', ' ')
    # Strip leading/trailing whitespace
    text = text.strip()
    return text

def create_clean_csv():
    """
    Creates a clean CSV file that displays properly in Excel.
    """
    session = Session()
    
    try:
        # Get all papers with ROT scores
        papers = session.query(Paper).filter(
            Paper.rot_score.isnot(None)
        ).order_by(Paper.rot_score.desc()).all()
        
        if not papers:
            print("No papers with ROT scores found.")
            return False
        
        # Calculate median for grouping
        rot_scores = [paper.rot_score for paper in papers]
        rot_scores.sort()
        median_rot = rot_scores[len(rot_scores) // 2]
        
        # Prepare data for CSV export
        high_rot_data = []
        low_rot_data = []
        
        for paper in papers:
            # Clean the abstract text
            clean_abstract = clean_text(paper.abstract)
            clean_title = clean_text(paper.title)
            
            paper_data = {
                'paper_id': paper.paper_id,
                'title': clean_title[:100] + "..." if len(clean_title) > 100 else clean_title,
                'abstract': clean_abstract[:200] + "..." if len(clean_abstract) > 200 else clean_abstract,
                'publication_year': paper.publication_year,
                'citation_count': paper.citation_count,
                'rot_score': round(paper.rot_score, 2),
                'doi': paper.doi or "N/A",
                'source_url': paper.source_url
            }
            
            if paper.rot_score >= median_rot:
                paper_data['quality_group'] = 'High ROT'
                high_rot_data.append(paper_data)
            else:
                paper_data['quality_group'] = 'Low ROT'
                low_rot_data.append(paper_data)
        
        # Export High ROT papers (clean version)
        if high_rot_data:
            with open('high_quality_papers_clean.csv', 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['paper_id', 'title', 'abstract', 'publication_year', 
                             'citation_count', 'rot_score', 'doi', 'source_url', 'quality_group']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(high_rot_data)
        
        # Export Low ROT papers (clean version)
        if low_rot_data:
            with open('low_quality_papers_clean.csv', 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['paper_id', 'title', 'abstract', 'publication_year', 
                             'citation_count', 'rot_score', 'doi', 'source_url', 'quality_group']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(low_rot_data)
        
        # Export all papers combined (clean version)
        all_papers_data = high_rot_data + low_rot_data
        if all_papers_data:
            with open('all_papers_clean.csv', 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['paper_id', 'title', 'abstract', 'publication_year', 
                             'citation_count', 'rot_score', 'doi', 'source_url', 'quality_group']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_papers_data)
        
        print(f"✅ Created clean CSV files:")
        print(f"   📄 high_quality_papers_clean.csv - {len(high_rot_data)} high quality papers")
        print(f"   📄 low_quality_papers_clean.csv - {len(low_rot_data)} lower quality papers")
        print(f"   📄 all_papers_clean.csv - {len(all_papers_data)} total papers")
        print()
        print("🎯 Open 'high_quality_papers_clean.csv' in Excel to see the filtered results!")
        
        return True
        
    except Exception as e:
        print(f"Error creating clean CSV: {e}")
        return False
    finally:
        session.close()

if __name__ == '__main__':
    create_clean_csv() 