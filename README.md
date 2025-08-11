# Entrepreneurship Education NLP Analysis

This project analyzes entrepreneurship education programs using NLP techniques to assess competency development and correlate program characteristics with Enactus competition performance.

## Overview

The analysis uses BART-MNLI (facebook/bart-large-mnli) to classify text from Canadian university entrepreneurship programs across 10 key competencies. The pipeline processes program descriptions, teaching methods, co-curricular activities, and performance data to generate quantitative competency scores and visualizations.

## Data Sources

The analysis processes four text fields from each institution:
- Undergraduate Entrepreneurship Courses & Programs
- Teaching and Assessment Methods
- Co-Curricular Activities  
- Annual Report + Enactus Impact Data

## Competency Framework

The system evaluates 10 entrepreneurial competencies:

1. **Innovation**: creativity, ideation, design thinking, problem solving
2. **Entrepreneurship**: venture creation, business model, startup, business plan
3. **Social Impact**: social enterprise, sustainability, community development
4. **Leadership**: team building, collaboration, project management, communication
5. **Financial Literacy**: budgeting, financial planning, investment, funding
6. **Marketing**: marketing strategy, customer acquisition, branding
7. **Risk Management**: risk assessment, uncertainty, decision making
8. **Technology**: digital transformation, technology adoption, e-commerce
9. **Networking**: networking, partnerships, stakeholder engagement
10. **Ethics**: business ethics, corporate responsibility, ethical decision making

## Methodology

### Text Preprocessing
- Stopword removal using NLTK
- Lemmatization and tokenization
- Punctuation and short word filtering
- Case normalization

### BART-MNLI Classification
The scoring system uses three approaches with weighted combination:
- **Direct Classification (50%)**: Competency name as label
- **Keyword Classification (30%)**: Top 6 keywords per competency, averaged
- **Hypothesis Testing (20%)**: Explicit entailment checking

Final scores range from 0.0 to 1.0, representing competency presence strength.

### Performance Scoring
Competition performance is quantified as:
- National Champion: +10 points
- National placement: +5-8 points  
- Regional Champion: +5 points
- Regional placement: +2 points
- Additional achievements: +1 point each

## Pipeline

1. **Data Loading**: Read CSV data and clean institution records
2. **Text Preprocessing**: Apply NLP preprocessing to all text fields
3. **Competency Classification**: Run BART-MNLI scoring for each competency
4. **Aggregation**: Calculate institution-level competency profiles
5. **Metrics Calculation**: Compute diversity indices and performance correlations
6. **Visualization Generation**: Create heatmaps and dashboard
7. **Report Generation**: Output summary statistics and rankings

## Visualizations

### `competency_distribution_heatmap.png`
Institution × competency heatmap showing competency scores across all universities. Darker colors indicate stronger competency presence.

### `program_area_emphasis_heatmap.png` 
Competency × program area heatmap showing where each competency is most emphasized (courses, teaching methods, co-curricular activities, or impact reports).

### `regional_comparison_heatmap.png`
Region × competency heatmap comparing average competency scores across Western, Central, and Eastern Canadian institutions.

### `performance_correlation_heatmap.png`
Single-column heatmap showing correlation coefficients between each competency and Enactus competition performance scores.

### `comprehensive_dashboard.png`
Multi-panel dashboard containing:
- Top 5 most comprehensive programs (bar chart)
- Top 5 innovation-focused programs (bar chart) 
- Most balanced competency portfolios (bar chart)
- Top 5 social impact programs (bar chart)
- Regional competency profiles (grouped bar chart)
- Program depth vs text complexity (scatter plot)
- Competency correlation matrix (triangular heatmap)

## Installation and Usage

```bash
conda activate eee
pip install transformers pandas matplotlib seaborn scikit-learn textstat nltk networkx scipy

python entrepreneurship_nlp_analysis.py
```

## Output Files

All outputs are saved to the `viz/` directory:
- `competency_distribution_heatmap.png`
- `program_area_emphasis_heatmap.png`
- `regional_comparison_heatmap.png`
- `performance_correlation_heatmap.png`
- `comprehensive_dashboard.png`
- `entrepreneurship_analysis_report.txt`

## Key Metrics

- **Competency Diversity Index**: Shannon entropy of competency distribution
- **Comprehensiveness Score**: Mean across all competency scores
- **Innovation Quotient**: Average of Innovation, Technology, and Entrepreneurship scores
- **Text Complexity**: Flesch reading ease and grade level scores

## Requirements

- Python 3.7+
- transformers
- pandas
- matplotlib
- seaborn
- scikit-learn
- textstat
- nltk
- networkx
- scipy


---
*Built with love, caffeine, and way too much time spent tweaking heatmap color schemes.*