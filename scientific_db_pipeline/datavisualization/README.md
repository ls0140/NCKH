# Data Visualization Package - 5-Category ROT System

## Overview
This package contains comprehensive data visualization and quality assessment tools for scientific paper analysis with a **5-category ROT (Rate of Citation) scoring system**.

## Files Structure

```
datavisualization/
├── 01_basic_analysis.py          # Basic data analysis with 5-category system
├── 02_categorical_analysis.py    # Categorical features analysis with 5 categories
├── 03_quality_assessment.py      # Comprehensive quality assessment
└── README.md                     # This file
```

## 5-Category ROT System

Your data now uses a **5-category classification system**:
- **Very Low ROT**: Papers with lowest citation rates relative to age
- **Low ROT**: Papers with low citation rates
- **Medium ROT**: Papers with moderate citation rates
- **High ROT**: Papers with high citation rates
- **Very High ROT**: Papers with highest citation rates (trending papers)

## Usage Instructions

### 1. Basic Analysis (`01_basic_analysis.py`)
**Purpose:** Analyze ROT score distribution, publication year, and citation count across 5 categories

**Features:**
- ROT Score Distribution (Histogram + Box Plot + Violin Plot)
- Publication Year vs Category Analysis
- Citation Count vs Category Analysis
- Correlation Matrix (Heatmap)
- Statistical Tests (ANOVA for multiple groups)

**How to use:**
1. Copy code to Google Colab
2. Upload your CSV file with `final_verdict` column
3. Run the script
4. View generated plots and statistics

### 2. Categorical Analysis (`02_categorical_analysis.py`)
**Purpose:** Extract and analyze categorical features from abstracts

**Features:**
- Extract features from abstracts (datasets, metrics, GitHub links)
- Stacked Bar Charts for categorical features across 5 categories
- Grouped Bar Charts with percentage values
- Chi-square tests and Cramer's V analysis
- Statistical significance assessment

**How to use:**
1. Copy code to Google Colab
2. Upload CSV file with 'abstract' and 'final_verdict' columns
3. Run the script
4. View categorical analysis results

### 3. Quality Assessment (`03_quality_assessment.py`)
**Purpose:** Comprehensive quality assessment of all features

**Features:**
- Combined analysis of numeric and categorical features
- Quality scoring and recommendations
- Visualization of feature quality
- Success rate calculation
- Detailed recommendations for improvement

**How to use:**
1. Copy code to Google Colab
2. Upload CSV file with 'final_verdict' column
3. Run the script
4. Review quality assessment report

## Required Data Format

Your CSV file should contain these columns:
- `paper_id`: Unique identifier for each paper
- `title`: Paper title
- `abstract`: Paper abstract (for categorical analysis)
- `publication_year`: Year of publication
- `citation_count`: Number of citations
- `rot_score`: ROT score (0-100 or higher)
- **`final_verdict`**: 5-category classification ('Very Low ROT', 'Low ROT', 'Medium ROT', 'High ROT', 'Very High ROT')

## Expected Outputs

### Visualizations
1. **Distribution Plots:** Histograms, box plots, and violin plots
2. **Comparison Plots:** Box plots comparing 5 categories
3. **Correlation Plots:** Heatmaps showing relationships
4. **Categorical Plots:** Stacked and grouped bar charts
5. **Quality Plots:** Feature quality assessment

### Statistical Results
1. **Descriptive Statistics:** Mean, median, min, max for each category
2. **Statistical Tests:** ANOVA (for numeric), Chi-square tests (for categorical)
3. **Effect Sizes:** Eta-squared (numeric), Cramer's V (categorical)
4. **Significance Levels:** P-values and significance indicators

### Quality Assessment
1. **Feature Quality Scores:** Individual feature assessments
2. **Overall Success Rate:** Percentage of significant features
3. **Recommendations:** Suggestions for improvement
4. **Conclusions:** Final quality verdict

## Interpretation Guide

### P-values
- **p < 0.05:** Feature is significant (✅ GOOD)
- **p >= 0.05:** Feature is not significant (❌ POOR)

### Effect Sizes
- **Large:** Strong relationship with target categories
- **Medium:** Moderate relationship with target categories
- **Small:** Weak relationship with target categories

### Success Rate
- **≥ 60%:** Excellent quality for model training
- **40-60%:** Moderate quality, needs improvement
- **< 40%:** Poor quality, significant improvement needed

## Troubleshooting

### Common Issues
1. **Missing final_verdict column:** Ensure your CSV has the 5-category final_verdict column
2. **Data format:** Check that numeric columns contain numbers
3. **Empty abstracts:** Categorical analysis requires abstract text
4. **Encoding issues:** Use UTF-8 encoding for text data

### Error Messages
- **"final_verdict column not found":** Add the 5-category classification to your data
- **"abstract column not found":** Add abstract text for categorical analysis
- **"No categorical features found":** Check if abstract extraction worked

## Recommendations

### For Better Results
1. **Improve feature extraction:** Use more sophisticated NLP techniques
2. **Add more features:** Consider additional paper characteristics
3. **Feature engineering:** Create composite features
4. **Data quality:** Ensure clean, consistent data

### For Model Training
1. **Use significant features only:** Focus on features with p < 0.05
2. **Consider feature selection:** Remove redundant features
3. **Monitor performance:** Track model accuracy with different feature sets
4. **Iterate:** Continuously improve feature quality

## Contact
For questions or issues, refer to the main project documentation or contact the development team. 