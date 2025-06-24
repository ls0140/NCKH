-- pgAdmin_queries.sql
-- Useful SQL queries to run in pgAdmin 4 for analyzing your scientific papers

-- 1. View all papers with their basic information
SELECT 
    paper_id,
    title,
    publication_year,
    citation_count,
    rot_score,
    doi,
    source_url
FROM papers 
ORDER BY publication_year DESC;

-- 2. Count total papers in database
SELECT COUNT(*) as total_papers FROM papers;

-- 3. View papers with abstracts (your "paragraphs")
SELECT 
    paper_id,
    title,
    abstract,
    publication_year,
    citation_count,
    rot_score
FROM papers 
WHERE abstract IS NOT NULL
ORDER BY publication_year DESC;

-- 4. Calculate ROT scores for papers that don't have them yet
-- (This updates the database with ROT calculations)
UPDATE papers 
SET rot_score = citation_count / (EXTRACT(YEAR FROM CURRENT_DATE) - publication_year + 1)
WHERE citation_count IS NOT NULL 
  AND publication_year IS NOT NULL 
  AND rot_score IS NULL;

-- 5. View ROT statistics
SELECT 
    COUNT(*) as total_papers,
    AVG(rot_score) as avg_rot,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY rot_score) as median_rot,
    MIN(rot_score) as min_rot,
    MAX(rot_score) as max_rot
FROM papers 
WHERE rot_score IS NOT NULL;

-- 6. Group papers by ROT (High vs Low)
-- First, get the median ROT to use as threshold
WITH rot_stats AS (
    SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY rot_score) as median_rot
    FROM papers 
    WHERE rot_score IS NOT NULL
)
SELECT 
    p.paper_id,
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

-- 7. View High ROT papers only
WITH rot_stats AS (
    SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY rot_score) as median_rot
    FROM papers 
    WHERE rot_score IS NOT NULL
)
SELECT 
    p.paper_id,
    p.title,
    p.abstract,
    p.publication_year,
    p.citation_count,
    p.rot_score
FROM papers p, rot_stats rs
WHERE p.rot_score >= rs.median_rot
ORDER BY p.rot_score DESC;

-- 8. View Low ROT papers only
WITH rot_stats AS (
    SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY rot_score) as median_rot
    FROM papers 
    WHERE rot_score IS NOT NULL
)
SELECT 
    p.paper_id,
    p.title,
    p.abstract,
    p.publication_year,
    p.citation_count,
    p.rot_score
FROM papers p, rot_stats rs
WHERE p.rot_score < rs.median_rot
ORDER BY p.rot_score ASC;

-- 9. Count papers in each ROT group
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
    COUNT(*) as paper_count,
    AVG(p.rot_score) as avg_rot_in_group
FROM papers p, rot_stats rs
WHERE p.rot_score IS NOT NULL
GROUP BY 
    CASE 
        WHEN p.rot_score >= rs.median_rot THEN 'High ROT'
        ELSE 'Low ROT'
    END;

-- 10. View papers by publication year with ROT groups
WITH rot_stats AS (
    SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY rot_score) as median_rot
    FROM papers 
    WHERE rot_score IS NOT NULL
)
SELECT 
    p.publication_year,
    CASE 
        WHEN p.rot_score >= rs.median_rot THEN 'High ROT'
        ELSE 'Low ROT'
    END as rot_group,
    COUNT(*) as paper_count
FROM papers p, rot_stats rs
WHERE p.rot_score IS NOT NULL
GROUP BY p.publication_year, 
    CASE 
        WHEN p.rot_score >= rs.median_rot THEN 'High ROT'
        ELSE 'Low ROT'
    END
ORDER BY p.publication_year DESC, rot_group;

-- 11. Find papers with highest ROT scores (top 10)
SELECT 
    paper_id,
    title,
    abstract,
    publication_year,
    citation_count,
    rot_score
FROM papers 
WHERE rot_score IS NOT NULL
ORDER BY rot_score DESC
LIMIT 10;

-- 12. Find papers with lowest ROT scores (bottom 10)
SELECT 
    paper_id,
    title,
    abstract,
    publication_year,
    citation_count,
    rot_score
FROM papers 
WHERE rot_score IS NOT NULL
ORDER BY rot_score ASC
LIMIT 10; 