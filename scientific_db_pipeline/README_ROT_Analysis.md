# Scientific Papers ROT Analysis Guide

## What is ROT (Reference of Time)?

ROT (Rate of Citation) measures how quickly a paper is being cited relative to its age:
- **ROT = citation_count / paper_age**
- **High ROT**: Papers that are being cited frequently relative to their age (trending, influential)
- **Low ROT**: Papers that are being cited less frequently relative to their age (less trending)

## Your Current System

Your pipeline collects scientific papers from arXiv and stores them in a PostgreSQL database with these tables:
- `papers` - contains titles, abstracts, publication years, citations, etc.
- `authors` - author information
- `paper_authors` - links papers to authors
- `citation_history` - citation tracking over time

## How to View Your Data in pgAdmin 4

### Step 1: Connect to Database
1. Open pgAdmin 4
2. Create a new server connection with these details:
   - **Host**: `localhost`
   - **Port**: `5432`
   - **Database**: `scientific_papers`
   - **Username**: `postgres`
   - **Password**: `123`

### Step 2: View Your Data
1. Navigate to: `Servers` → `Your Server` → `Databases` → `scientific_papers` → `Schemas` → `public` → `Tables`
2. Right-click on `papers` table → "View/Edit Data" → "All Rows"
3. This shows all your papers with their abstracts (the "paragraphs" you mentioned)

### Step 3: Run Analysis Queries
1. Open the Query Tool (SQL icon)
2. Copy and paste queries from `pgadmin_queries.sql`
3. Run them to analyze your data

## How to Group Papers by ROT

### Option 1: Use the Python Script (Recommended)
```bash
# Install pandas if you haven't already
pip install pandas

# Run the ROT analysis
python rot_analyzer.py
```

This will:
- Calculate ROT scores for all papers
- Group them into High ROT and Low ROT
- Export results to CSV files:
  - `high_rot_papers.csv`
  - `low_rot_papers.csv`
  - `all_papers_with_rot_groups.csv`

### Option 2: Use pgAdmin 4 Queries
1. Open `pgadmin_queries.sql` in pgAdmin 4
2. Run queries 4-8 to calculate ROT and group papers
3. Use queries 7-8 to view High ROT and Low ROT papers separately

## Understanding Your Results

### High ROT Papers
- Papers with above-median citation rates
- These are likely trending or influential in their field
- Good candidates for current research focus

### Low ROT Papers
- Papers with below-median citation rates
- May be older, niche, or less impactful
- Could be interesting for historical analysis

## Key Queries for Your Team Work

### View All Papers with ROT Groups
```sql
-- Run this in pgAdmin 4 to see all papers grouped by ROT
WITH rot_stats AS (
    SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY rot_score) as median_rot
    FROM papers 
    WHERE rot_score IS NOT NULL
)
SELECT 
    p.title,
    p.abstract,
    p.publication_year,
    p.citation_count,
    p.rot_score,
    CASE 
        WHEN p.rot_score >= rs.median_rot THEN 'High ROT'
        ELSE 'Low ROT'
    END as rot_group
FROM papers p, rot_stats rs
WHERE p.rot_score IS NOT NULL
ORDER BY p.rot_score DESC;
```

### Count Papers in Each Group
```sql
-- See how many papers are in each ROT group
WITH rot_stats AS (
    SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY rot_score) as median_rot
    FROM papers 
    WHERE rot_score IS NOT NULL
)
SELECT 
    CASE 
        WHEN p.rot_score >= rs.median_rot THEN 'High ROT'
        ELSE 'Low ROT'
    END as rot_group,
    COUNT(*) as paper_count
FROM papers p, rot_stats rs
WHERE p.rot_score IS NOT NULL
GROUP BY 
    CASE 
        WHEN p.rot_score >= rs.median_rot THEN 'High ROT'
        ELSE 'Low ROT'
    END;
```

## Next Steps for Your Team

1. **Run the ROT analysis**: `python rot_analyzer.py`
2. **Review the CSV files** in your project directory
3. **Use pgAdmin 4** to explore specific papers in each group
4. **Share the results** with your team using the exported CSV files
5. **Analyze patterns** in High ROT vs Low ROT papers

## Troubleshooting

### If you get database connection errors:
- Make sure PostgreSQL is running
- Check that the database `scientific_papers` exists
- Verify your connection credentials

### If ROT scores are NULL:
- Run the ROT calculation query in pgAdmin 4 (query #4)
- Or run `python rot_analyzer.py` to calculate all scores

### If you need to re-run data collection:
```bash
python main.py
```

## Files Created

- `rot_analyzer.py` - Main analysis script
- `pgadmin_queries.sql` - SQL queries for pgAdmin 4
- `high_rot_papers.csv` - High ROT papers (after running analysis)
- `low_rot_papers.csv` - Low ROT papers (after running analysis)
- `all_papers_with_rot_groups.csv` - All papers with ROT groups 