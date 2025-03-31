# Text Analysis Project

Article analysis tool that extracts content and calculates readability metrics using NLP techniques.

## Technical Stack

- **Python 3.8+** - Core implementation
- **newspaper3k** - Article extraction with HTML cleaning
- **NLTK** - Natural language processing and tokenization
- **pandas** - Data handling and Excel I/O
- **concurrent.futures** - Parallel processing implementation
- **tqdm** - Progress tracking

## Features

- Parallel processing of multiple articles simultaneously
- Extracts clean article content using newspaper3k
- Calculates sentiment scores, readability metrics and text statistics
- Progress bar shows extraction and analysis status
- Results saved to Excel file

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Article-Insight-Analyzer
```

2. Create and activate virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required NLTK data:
```python
import nltk
nltk.download('punkt')
```

## Usage

1. Place your input Excel file as `Input.xlsx` with columns:
   - URL_ID: Unique identifier for each article
   - URL: Web URL of the article

2. Run the analysis:
```python
python text_analysis.py
```

3. Results will be saved to `Output.xlsx` with the following metrics:

- Sentiment Scores
  - POSITIVE SCORE
  - NEGATIVE SCORE 
  - POLARITY SCORE
  - SUBJECTIVITY SCORE

- Readability Metrics
  - AVG SENTENCE LENGTH
  - PERCENTAGE OF COMPLEX WORDS
  - FOG INDEX
  - AVG NUMBER OF WORDS PER SENTENCE
  - COMPLEX WORD COUNT
  - WORD COUNT
  - SYLLABLE PER WORD

- Text Statistics
  - PERSONAL PRONOUNS
  - AVG WORD LENGTH

## Implementation Details

### Text Extraction (`extract_article()`)
- Uses `newspaper3k` for content extraction
- Custom cleaning pipeline:
  - HTML tag removal
  - Whitespace normalization
  - Related content removal
  - Article/title separation

### Text Analysis (`TextAnalyzer` class)
- Sentiment Analysis:
  ```
  Positive Score = Count of positive words
  Negative Score = Count of negative words
  Polarity Score = (Pos - Neg) / (Pos + Neg + 0.000001)
  Subjectivity Score = (Pos + Neg) / (Word Count + 0.000001)
  ```

## Project Structure

```
├── text_analysis.py      # Main script with article extraction and analysis
├── StopWords/           # Stop words files
├── MasterDictionary/    # Positive and negative word lists
│   ├── positive-words.txt
│   └── negative-words.txt
├── articles/            # Extracted article text files
├── Input.xlsx           # Input file with URLs
└── requirements.txt     # Python dependencies
```