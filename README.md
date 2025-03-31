# Text Analysis Project

## Overview
This project performs text analysis on articles to extract various metrics including sentiment scores, readability indices, and other text statistics. It downloads articles from URLs, processes them, and generates an Excel output with the analysis results.

## Approach

1. **Data Extraction**
   - Reads article URLs from `Input.xlsx`
   - Downloads article content using BeautifulSoup and requests
   - Cleans HTML and extracts main article text
   - Stores articles in the `articles` directory

2. **Text Processing**
   - Removes stop words using custom lists from `StopWords`
   - Uses positive/negative word dictionaries from `MasterDictionary`
   - Tokenizes text into sentences and words using NLTK

3. **Analysis & Metrics**
   - Calculates sentiment scores (positive/negative)
   - Computes readability metrics (Fog Index, complex words)
   - Analyzes text statistics (word count, syllables, etc.)
   - Generates personal pronoun counts and average word lengths

## Setup & Installation

1. Clone the repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```python
import nltk
nltk.download('punkt')
```

## Dependencies
- pandas - Data manipulation and analysis
- requests - HTTP library for downloading articles
- beautifulsoup4 - Web scraping and HTML parsing
- nltk - Natural Language Processing tools
- openpyxl - Excel file handling

## Project Structure
```
├── analyze_articles.py  # Main analysis script
├── text_analysis.py    # Article downloading and analysis
├── Input.xlsx         # Input file with URLs
├── articles/          # Downloaded article texts
├── MasterDictionary/  # Sentiment word lists
└── StopWords/        # Stop word lists
```

## Usage

1. Ensure your `Input.xlsx` contains the URLs to analyze
2. To download and analyze articles:
```bash
python text_analysis.py
```

3. To analyze existing articles:
```bash
python analyze_articles.py
```

The script will:
- Process each article
- Calculate metrics
- Generate `Output Data Structure.xlsx` with results

## Output
The analysis generates an Excel file with the following metrics for each article:
- Sentiment scores (positive/negative)
- Polarity and subjectivity scores
- Average sentence length
- Complex word percentages
- Fog index
- Word counts and syllable statistics
- Personal pronoun counts
- Average word length