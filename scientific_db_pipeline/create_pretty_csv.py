# create_pretty_csv.py

import pandas as pd
import numpy as np

def create_pretty_csv_files():
    """
    Creates prettier versions of the three main CSV files with better formatting.
    """
    print("Creating prettier CSV files with better formatting...")
    
    # Read the existing files
    all_papers = pd.read_csv('sorted_rot/all_papers_with_rot_groups.csv')
    high_rot = pd.read_csv('sorted_rot/high_rot_papers.csv')
    low_rot = pd.read_csv('sorted_rot/low_rot_papers.csv')
    
    # Function to format abstracts for better readability
    def format_abstract(abstract):
        if pd.isna(abstract) or abstract is None:
            return "No abstract available"
        
        # Truncate long abstracts and add ellipsis
        if len(str(abstract)) > 200:
            return str(abstract)[:200] + "..."
        return str(abstract)
    
    # Function to format titles for better readability
    def format_title(title):
        if pd.isna(title) or title is None:
            return "No title available"
        
        # Truncate long titles and add ellipsis
        if len(str(title)) > 80:
            return str(title)[:80] + "..."
        return str(title)
    
    # Function to format ROT score
    def format_rot_score(rot_score):
        if pd.isna(rot_score):
            return "N/A"
        return f"{rot_score:.2f}"
    
    # Function to format citation count
    def format_citations(citations):
        if pd.isna(citations):
            return "N/A"
        return f"{int(citations):,}"
    
    # Function to format publication year
    def format_year(year):
        if pd.isna(year):
            return "N/A"
        return str(int(year))
    
    # Create prettier versions
    def create_pretty_dataframe(df, group_name):
        # Create a copy for formatting
        pretty_df = df.copy()
        
        # Apply formatting functions
        pretty_df['Title'] = pretty_df['title'].apply(format_title)
        pretty_df['Abstract'] = pretty_df['abstract'].apply(format_abstract)
        pretty_df['Publication Year'] = pretty_df['publication_year'].apply(format_year)
        pretty_df['Citations'] = pretty_df['citation_count'].apply(format_citations)
        pretty_df['ROT Score'] = pretty_df['rot_score'].apply(format_rot_score)
        pretty_df['DOI'] = pretty_df['doi'].fillna('N/A')
        pretty_df['Source URL'] = pretty_df['source_url'].fillna('N/A')
        
        # Select and reorder columns for better presentation
        columns_order = [
            'paper_id',
            'Title', 
            'Abstract',
            'Publication Year',
            'Citations',
            'ROT Score',
            'DOI',
            'Source URL',
            'rot_group'
        ]
        
        # Filter columns that exist
        existing_columns = [col for col in columns_order if col in pretty_df.columns]
        pretty_df = pretty_df[existing_columns]
        
        # Rename columns for better presentation
        column_mapping = {
            'paper_id': 'Paper ID',
            'rot_group': 'ROT Group'
        }
        pretty_df = pretty_df.rename(columns=column_mapping)
        
        # Sort by ROT score (descending for high ROT, ascending for low ROT)
        if group_name == 'high':
            pretty_df = pretty_df.sort_values('ROT Score', ascending=False)
        elif group_name == 'low':
            pretty_df = pretty_df.sort_values('ROT Score', ascending=True)
        
        return pretty_df
    
    # Create prettier versions
    pretty_all_papers = create_pretty_dataframe(all_papers, 'all')
    pretty_high_rot = create_pretty_dataframe(high_rot, 'high')
    pretty_low_rot = create_pretty_dataframe(low_rot, 'low')
    
    # Save prettier versions
    pretty_all_papers.to_csv('sorted_rot/all_papers_pretty.csv', index=False, encoding='utf-8')
    pretty_high_rot.to_csv('sorted_rot/high_rot_papers_pretty.csv', index=False, encoding='utf-8')
    pretty_low_rot.to_csv('sorted_rot/low_rot_papers_pretty.csv', index=False, encoding='utf-8')
    
    # Also create a summary statistics file
    summary_stats = {
        'Metric': [
            'Total Papers',
            'High ROT Papers',
            'Low ROT Papers',
            'Average ROT Score',
            'Median ROT Score',
            'Highest ROT Score',
            'Lowest ROT Score',
            'Average Citations',
            'Median Citations'
        ],
        'Value': [
            len(all_papers),
            len(high_rot),
            len(low_rot),
            f"{all_papers['rot_score'].mean():.2f}",
            f"{all_papers['rot_score'].median():.2f}",
            f"{all_papers['rot_score'].max():.2f}",
            f"{all_papers['rot_score'].min():.2f}",
            f"{all_papers['citation_count'].mean():.1f}",
            f"{all_papers['citation_count'].median():.1f}"
        ]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv('sorted_rot/analysis_summary.csv', index=False, encoding='utf-8')
    
    print("✅ Created prettier CSV files:")
    print("   - all_papers_pretty.csv")
    print("   - high_rot_papers_pretty.csv") 
    print("   - low_rot_papers_pretty.csv")
    print("   - analysis_summary.csv")
    
    # Print some statistics
    print(f"\n📊 Summary Statistics:")
    print(f"   Total Papers: {len(all_papers)}")
    print(f"   High ROT Papers: {len(high_rot)}")
    print(f"   Low ROT Papers: {len(low_rot)}")
    print(f"   Average ROT Score: {all_papers['rot_score'].mean():.2f}")
    print(f"   Median ROT Score: {all_papers['rot_score'].median():.2f}")

if __name__ == '__main__':
    create_pretty_csv_files() 