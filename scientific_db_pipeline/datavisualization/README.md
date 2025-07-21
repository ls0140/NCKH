# Data Visualization Package

## Overview
This package contains comprehensive data visualization and quality assessment tools for scientific paper analysis with ROT (Rate of Topic) scoring.

## Files Structure

```
datavisualization/
├── 01_basic_analysis.py          # Basic data analysis with current data
├── 02_categorical_analysis.py    # Categorical features analysis
├── 03_quality_assessment.py      # Comprehensive quality assessment
└── README.md                     # This file
```

## Usage Instructions

### 1. Basic Analysis (`01_basic_analysis.py`)
**Purpose:** Analyze ROT score distribution, publication year, and citation count

**Features:**
- ROT Score Distribution (Histogram + Box Plot)
- Publication Year vs ROT Group Analysis
- Citation Count vs ROT Group Analysis
- Correlation Matrix (Heatmap)
- Statistical Tests (T-tests)

**How to use:**
1. Copy code to Google Colab
2. Upload your CSV file
3. Run the script
4. View generated plots and statistics

### 2. Categorical Analysis (`02_categorical_analysis.py`)
**Purpose:** Extract and analyze categorical features from abstracts

**Features:**
- Extract features from abstracts (datasets, metrics, GitHub links)
- Stacked Bar Charts for categorical features
- Grouped Bar Charts
- Chi-square tests and Cramer's V analysis
- Statistical significance assessment

**How to use:**
1. Copy code to Google Colab
2. Upload CSV file with 'abstract' column
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
2. Upload CSV file
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
- `rot_group`: ROT group classification ('High ROT' or 'Low ROT')

## Expected Outputs

### Visualizations
1. **Distribution Plots:** Histograms and box plots
2. **Comparison Plots:** Box plots comparing groups
3. **Correlation Plots:** Heatmaps showing relationships
4. **Categorical Plots:** Stacked and grouped bar charts
5. **Quality Plots:** Feature quality assessment

### Statistical Results
1. **Descriptive Statistics:** Mean, median, min, max
2. **Statistical Tests:** T-tests, Chi-square tests
3. **Effect Sizes:** Cohen's d, Cramer's V
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
- **Large:** Strong relationship with target
- **Medium:** Moderate relationship with target
- **Small:** Weak relationship with target

### Success Rate
- **≥ 60%:** Excellent quality for model training
- **40-60%:** Moderate quality, needs improvement
- **< 40%:** Poor quality, significant improvement needed

## Troubleshooting

### Common Issues
1. **Missing columns:** Ensure your CSV has required columns
2. **Data format:** Check that numeric columns contain numbers
3. **Empty abstracts:** Categorical analysis requires abstract text
4. **Encoding issues:** Use UTF-8 encoding for text data

### Error Messages
- **"rot_group column not found":** Add ROT group classification to your data
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