import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import time

# Set NLTK data path
nltk.data.path = [os.path.join(os.path.expanduser('~'), 'AppData', 'Roaming', 'nltk_data')]

def setup_nltk():
    try:
        nltk.download('punkt', download_dir=nltk.data.path[0])
        nltk.download('averaged_perceptron_tagger', download_dir=nltk.data.path[0])
        nltk.data.load('tokenizers/punkt/english.pickle')
    except LookupError as e:
        print(f"Error downloading NLTK data: {e}")
        print("Falling back to basic tokenization...")

# Call setup at start
setup_nltk()

class TextAnalyzer:
    def __init__(self):
        self.positive_words = self._load_words('MasterDictionary/positive-words.txt')
        self.negative_words = self._load_words('MasterDictionary/negative-words.txt')
        self.stop_words = self._load_stop_words()

    def _load_words(self, filepath):
        words = set()
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    words.update(line.strip() for line in f if line.strip() and not line.startswith(';'))
                break
            except UnicodeDecodeError:
                continue
        return words

    def _load_stop_words(self):
        stop_words = set()
        for file in os.listdir('StopWords'):
            if file.endswith('.txt'):
                stop_words.update(self._load_words(f'StopWords/{file}'))
        return stop_words

    def _count_syllables(self, word):
        """
        Modified syllable counting to handle exceptions
        """
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        
        # Handle special endings
        if word.endswith('es') or word.endswith('ed'):
            word = word[:-2]
            
        # Count vowel groups
        prev_char_is_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_char_is_vowel:
                count += 1
            prev_char_is_vowel = is_vowel
            
        # Handle special cases
        if word.endswith('e'):
            count -= 1
        if count == 0:
            count = 1
            
        return count

    def analyze_text(self, text):
        if not text:
            return self._get_empty_metrics()

        try:
            # Split into sentences for analysis
            sentences = sent_tokenize(text)
            raw_words = word_tokenize(text)
            
            # Clean and tokenize text for sentiment analysis
            cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
            words = word_tokenize(cleaned_text)
            words = [word for word in words if word not in self.stop_words and word.isalnum()]
            
            if not words:
                return self._get_empty_metrics()

            # Sentiment Analysis
            positive_score = sum(1 for word in words if word in self.positive_words)
            negative_score = sum(1 for word in words if word in self.negative_words) * -1

            # Calculate scores
            total_sentiment = positive_score + abs(negative_score)
            polarity_score = (positive_score + negative_score) / (total_sentiment + 0.000001)
            subjectivity_score = total_sentiment / (len(words) + 0.000001)

            # Calculate metrics
            avg_sentence_length = len(raw_words) / len(sentences) if sentences else 0
            complex_words = [word for word in words if self._count_syllables(word) > 2]
            percent_complex_words = len(complex_words) / len(words) if words else 0
            fog_index = 0.4 * (avg_sentence_length + percent_complex_words)

            # Additional metrics
            syllable_count = sum(self._count_syllables(word) for word in words)
            syllable_per_word = syllable_count / len(words) if words else 0
            personal_pronouns = len(re.findall(r'\b(I|we|my|ours|us)\b(?!\s+(?:states|state))', text, re.I))
            avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

            return {
                'POSITIVE SCORE': positive_score,
                'NEGATIVE SCORE': abs(negative_score),
                'POLARITY SCORE': polarity_score,
                'SUBJECTIVITY SCORE': subjectivity_score,
                'AVG SENTENCE LENGTH': avg_sentence_length,
                'PERCENTAGE OF COMPLEX WORDS': percent_complex_words,
                'FOG INDEX': fog_index,
                'AVG NUMBER OF WORDS PER SENTENCE': avg_sentence_length,
                'COMPLEX WORD COUNT': len(complex_words),
                'WORD COUNT': len(words),
                'SYLLABLE PER WORD': syllable_per_word,
                'PERSONAL PRONOUNS': personal_pronouns,
                'AVG WORD LENGTH': avg_word_length
            }
        except Exception as e:
            print(f"Error analyzing text: {e}")
            return self._get_empty_metrics()

    def _get_empty_metrics(self):
        return {
            'POSITIVE SCORE': 0,
            'NEGATIVE SCORE': 0,
            'POLARITY SCORE': 0,
            'SUBJECTIVITY SCORE': 0,
            'AVG SENTENCE LENGTH': 0,
            'PERCENTAGE OF COMPLEX WORDS': 0,
            'FOG INDEX': 0,
            'AVG NUMBER OF WORDS PER SENTENCE': 0,
            'COMPLEX WORD COUNT': 0,
            'WORD COUNT': 0,
            'SYLLABLE PER WORD': 0,
            'PERSONAL PRONOUNS': 0,
            'AVG WORD LENGTH': 0
        }

def extract_article(url):
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
        'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36',
    ]
    
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0',
    }

    def clean_text(text):
        text = re.sub(r'[\n\r\t]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^\w\s.,!?"\'-]', '', text)
        return text

    def clean_title(title):
        if not title:
            return ""
        title = re.sub(r'<[^>]+>', '', str(title))
        title = re.sub(r'\s+', ' ', title).strip()
        title = re.sub(r'[^\w\s.,!?"|:-]', '', title)
        return title

    def extract_main_content(soup):
        content_selectors = [
            ('article', {}),
            ('div', {'class_': ['article-body', 'article-content', 'story-content']}),
            ('div', {'itemprop': 'articleBody'}),
            ('div', {'class_': lambda x: x and 'article' in str(x).lower()}),
            ('div', {'role': 'main'})
        ]
        
        for tag, attrs in content_selectors:
            content = soup.find(tag, **attrs)
            if content:
                paragraphs = []
                for p in content.find_all(['p', 'h2', 'h3', 'h4']):
                    text = clean_text(p.get_text())
                    if text and len(text.split()) > 4:
                        paragraphs.append(text)
                
                if len(' '.join(paragraphs).split()) > 100:
                    return ' '.join(paragraphs)
        
        all_paragraphs = []
        for p in soup.find_all('p'):
            text = clean_text(p.get_text())
            if text and len(text.split()) > 4:
                all_paragraphs.append(text)
        
        if len(' '.join(all_paragraphs).split()) > 100:
            return ' '.join(all_paragraphs)
        
        return ""

    return ""

def analyze_existing_articles():
    input_df = pd.read_excel('Input.xlsx')
    analyzer = TextAnalyzer()
    results = []
    
    for _, row in input_df.iterrows():
        url_id = str(row['URL_ID'])
        article_path = f'articles/{url_id}.txt'
        
        print(f"Analyzing article {url_id}")
        
        try:
            with open(article_path, 'r', encoding='utf-8') as f:
                text = f.read()
            metrics = analyzer.analyze_text(text)
            
            results.append({
                'URL_ID': url_id,
                'URL': row['URL'],
                **metrics
            })
        except Exception as e:
            print(f"Error analyzing {article_path}: {e}")
            continue
    
    columns = [
        'URL_ID', 'URL',
        'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 'SUBJECTIVITY SCORE',
        'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX',
        'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT', 'WORD COUNT',
        'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH'
    ]
    
    output_df = pd.DataFrame(results)[columns]
    output_df.to_excel('output.xlsx', index=False)
    print("Analysis complete! Results saved to 'output.xlsx'")

def main():
    if os.path.exists('articles') and len(os.listdir('articles')) > 0:
        print("Articles folder exists. Moving directly to analysis...")
        analyze_existing_articles()
        return

    if not os.path.exists('articles'):
        os.makedirs('articles')

    input_df = pd.read_excel('Input.xlsx')
    analyzer = TextAnalyzer()
    results = []

    for _, row in input_df.iterrows():
        url_id = str(row['URL_ID'])
        url = row['URL']
        
        print(f"Processing URL_ID: {url_id}")
        article_text = extract_article(url)
        
        with open(f'articles/{url_id}.txt', 'w', encoding='utf-8') as f:
            f.write(article_text)
        
        analysis_results = analyzer.analyze_text(article_text)
        results.append({
            'URL_ID': url_id,
            'URL': url,
            **analysis_results
        })

    output_df = pd.DataFrame(results)
    output_df.to_excel('output.xlsx', index=False)
    print("Analysis complete! Results saved to 'output.xlsx'")

if __name__ == '__main__':
    main()
