import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from newspaper import Article, Config
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

nltk.data.path = [os.path.join(os.path.expanduser('~'), 'AppData', 'Roaming', 'nltk_data')]

class TextAnalyzer:
    def __init__(self):
        self.positive_words = self._load_words('MasterDictionary/positive-words.txt')
        self.negative_words = self._load_words('MasterDictionary/negative-words.txt')
        self.stop_words = self._load_stop_words()

    def _load_words(self, filepath):
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    return set(line.strip() for line in f if line.strip() and not line.startswith(';'))
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error reading {filepath}: {str(e)}")
                return set()
        print(f"Failed to read {filepath} with any encoding")
        return set()

    def _load_stop_words(self):
        stop_words = set()
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        for file in os.listdir('StopWords'):
            if file.endswith('.txt'):
                file_loaded = False
                for encoding in encodings:
                    try:
                        with open(f'StopWords/{file}', 'r', encoding=encoding) as f:
                            stop_words.update(line.strip() for line in f if line.strip())
                            file_loaded = True
                            break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        print(f"Error reading {file}: {str(e)}")
                        break
                if not file_loaded:
                    print(f"Failed to read {file} with any encoding")
        return stop_words

    def _count_syllables(self, word):
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index-1] not in vowels:
                count += 1
        if word.endswith('e'):
            count -= 1
        if count == 0:
            count += 1
        return count

    def analyze_text(self, text):
        if not text:
            return self._get_empty_metrics()
        try:
            sentences = sent_tokenize(text)
            words = word_tokenize(text.lower())
        except Exception as e:
            print(f"NLTK tokenization failed, using basic splitting: {e}")
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            words = [w.strip().lower() for w in text.split() if w.strip()]
        words = [word for word in words if word.isalnum() and word not in self.stop_words]
        positive_score = sum(1 for word in words if word in self.positive_words)
        negative_score = sum(1 for word in words if word in self.negative_words)
        if not words:
            return self._get_empty_metrics()
        polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
        subjectivity_score = (positive_score + negative_score) / (len(words) + 0.000001)
        avg_sentence_length = len(words) / max(len(sentences), 1)
        complex_words = [word for word in words if self._count_syllables(word) > 2]
        percent_complex_words = len(complex_words) / len(words)
        fog_index = 0.4 * (avg_sentence_length + percent_complex_words)
        pronouns = len(re.findall(r'\b(I|we|my|ours|us)\b', text, re.I))
        avg_word_length = sum(len(word) for word in words) / len(words)
        return {
            'POSITIVE SCORE': positive_score,
            'NEGATIVE SCORE': negative_score,
            'POLARITY SCORE': polarity_score,
            'SUBJECTIVITY SCORE': subjectivity_score,
            'AVG SENTENCE LENGTH': avg_sentence_length,
            'PERCENTAGE OF COMPLEX WORDS': percent_complex_words,
            'FOG INDEX': fog_index,
            'AVG NUMBER OF WORDS PER SENTENCE': avg_sentence_length,
            'COMPLEX WORD COUNT': len(complex_words),
            'WORD COUNT': len(words),
            'SYLLABLE PER WORD': sum(self._count_syllables(word) for word in words) / len(words),
            'PERSONAL PRONOUNS': pronouns,
            'AVG WORD LENGTH': avg_word_length
        }

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
    config = Config()
    config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    config.request_timeout = 30
    config.fetch_images = False
    config.memoize_articles = False
    
    try:
        article = Article(url, config=config)
        article.download()
        time.sleep(2)
        article.parse()
        
        # Clean and extract only title and main text
        title = article.title.strip() if article.title else ""
        text = article.text.strip() if article.text else ""
        
        # Additional cleaning steps
        # Remove any remaining HTML
        text = re.sub(r'<[^>]+>', '', text)
        # Remove multiple newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # Remove any "Related Articles" or similar sections
        text = re.sub(r'Related Articles.*', '', text, flags=re.IGNORECASE|re.DOTALL)
        text = re.sub(r'Recommended Stories.*', '', text, flags=re.IGNORECASE|re.DOTALL)
        
        # Only return if we have actual content
        if text and not text.isspace():
            return f"{title}\n\n{text}"
        else:
            print(f"No valid content found for {url}")
            return ""
            
    except Exception as e:
        print(f"Error extracting from {url}: {str(e)}")
        return ""

def process_article(row, analyzer):
    url_id = str(row['URL_ID'])
    url = row['URL']
    try:
        article_text = extract_article(url)
        
        # Save article text
        with open(f'articles/{url_id}.txt', 'w', encoding='utf-8') as f:
            f.write(article_text)
            
        # Analyze text
        analysis_results = analyzer.analyze_text(article_text)
        
        return {
            'URL_ID': url_id,
            'URL': url,
            **analysis_results
        }
    except Exception as e:
        print(f"Error processing {url_id}: {e}")
        return None

def main():
    if not os.path.exists('articles'):
        os.makedirs('articles')
        
    analyzer = TextAnalyzer()
    input_df = pd.read_excel('Input.xlsx')
    results = []
    
    # Use ThreadPoolExecutor for parallel processing
    # Adjust max_workers based on your system capabilities
    with ThreadPoolExecutor(max_workers=20) as executor:
        # Create future tasks
        future_to_url = {
            executor.submit(process_article, row, analyzer): row 
            for _, row in input_df.iterrows()
        }
        
        # Process results as they complete with progress bar
        with tqdm(total=len(future_to_url), desc="Processing articles") as pbar:
            for future in as_completed(future_to_url):
                result = future.result()
                if result:
                    results.append(result)
                pbar.update(1)

    # Create output DataFrame and save
    output_df = pd.DataFrame(results)
    output_df.to_excel('Output.xlsx', index=False)
    print("Analysis complete! Results saved to 'Output.xlsx'")

if __name__ == '__main__':
    main()
