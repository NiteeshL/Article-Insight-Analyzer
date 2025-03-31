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

# Download required NLTK data with error handling
def setup_nltk():
    try:
        nltk.download('punkt', download_dir=nltk.data.path[0])
        nltk.download('averaged_perceptron_tagger', download_dir=nltk.data.path[0])
        # Create tokenizer explicitly
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
            # Try NLTK tokenization first
            sentences = sent_tokenize(text)
            words = word_tokenize(text.lower())
        except Exception as e:
            print(f"NLTK tokenization failed, using basic splitting: {e}")
            # Fallback to basic tokenization
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            words = [w.strip().lower() for w in text.split() if w.strip()]

        # Filter out stop words and punctuation
        words = [word for word in words if word.isalnum() and word not in self.stop_words]
        
        # Calculate scores
        positive_score = sum(1 for word in words if word in self.positive_words)
        negative_score = sum(1 for word in words if word in self.negative_words)
        
        # Handle empty text case
        if not words:
            return self._get_empty_metrics()
        
        # Calculate metrics
        polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
        subjectivity_score = (positive_score + negative_score) / (len(words) + 0.000001)
        
        # More metrics
        avg_sentence_length = len(words) / max(len(sentences), 1)
        complex_words = [word for word in words if self._count_syllables(word) > 2]
        percent_complex_words = len(complex_words) / len(words)
        fog_index = 0.4 * (avg_sentence_length + percent_complex_words)
        
        # Count personal pronouns
        pronouns = len(re.findall(r'\b(I|we|my|ours|us)\b', text, re.I))
        
        # Average word length
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
    # List of User-Agents to try
    user_agents = [
        # Chrome
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        # Edge
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
        # Chrome Mobile
        'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36',
    ]
    
    # Headers that mimic a browser
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0',
    }

    def clean_text(text):
        # Remove extra whitespace, newlines and tabs
        text = re.sub(r'[\n\r\t]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove any HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?"\'-]', '', text)
        return text

    def clean_title(title):
        # Clean the title but preserve the full text
        if not title:
            return ""
        # Remove HTML tags
        title = re.sub(r'<[^>]+>', '', str(title))
        # Remove extra whitespace
        title = re.sub(r'\s+', ' ', title).strip()
        # Remove special characters but keep punctuation and separators
        title = re.sub(r'[^\w\s.,!?"|:-]', '', title)
        return title

    def extract_main_content(soup):
        # First try specific article containers
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
                # Get all paragraphs from the content
                paragraphs = []
                for p in content.find_all(['p', 'h2', 'h3', 'h4']):
                    text = clean_text(p.get_text())
                    # Only include substantial paragraphs
                    if text and len(text.split()) > 4:
                        paragraphs.append(text)
                
                if len(' '.join(paragraphs).split()) > 100:  # Minimum 100 words
                    return ' '.join(paragraphs)
        
        # Fallback: try to find the largest cluster of paragraphs
        all_paragraphs = []
        for p in soup.find_all('p'):
            text = clean_text(p.get_text())
            if text and len(text.split()) > 4:
                all_paragraphs.append(text)
        
        if len(' '.join(all_paragraphs).split()) > 100:
            return ' '.join(all_paragraphs)
        
        return ""

    for user_agent in user_agents:
        try:
            headers['User-Agent'] = user_agent
            time.sleep(2)
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            if 'Not Acceptable' in response.text:
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove all unwanted elements first
            for element in soup.find_all([
                'script', 'style', 'nav', 'header', 'footer', 'iframe',
                'aside', 'form', 'noscript', 'figure', 'figcaption'
            ]):
                element.decompose()
            
            # Get complete title from title tag
            title = ''
            if soup.title:
                title = clean_title(soup.title.string)
            
            # Only use alternative title sources if no title tag found
            if not title:
                for title_elem in [
                    soup.find('meta', property='og:title'),
                    soup.find('meta', property='twitter:title'),
                    soup.find('h1'),
                    soup.find(class_=['article-title', 'entry-title', 'post-title'])
                ]:
                    if title_elem:
                        if title_elem.get('content'):
                            title = clean_title(title_elem['content'])
                        else:
                            title = clean_title(title_elem.get_text())
                        break
            
            # Extract main content
            article_text = extract_main_content(soup)
            
            if article_text:
                return f"{title}\n\n{article_text}"
            
        except Exception as e:
            print(f"Error extracting from {url}: {str(e)}")
            continue
    
    print(f"Failed to extract article from {url}")
    return ""

def main():
    # Check if articles directory exists and has content
    if os.path.exists('articles') and len(os.listdir('articles')) > 0:
        print("Articles folder already exists. Skipping download phase...")
        print("Moving directly to analysis phase...")
        # Call the analysis script
        import analyze_articles
        analyze_articles.main()
        return

    # If no articles exist, continue with extraction process
    # Create articles directory if it doesn't exist
    if not os.path.exists('articles'):
        os.makedirs('articles')

    # Initialize analyzer
    analyzer = TextAnalyzer()

    # Read input Excel file
    input_df = pd.read_excel('Input.xlsx')
    
    # Initialize results list
    results = []

    # Process each URL
    for _, row in input_df.iterrows():
        url_id = str(row['URL_ID'])
        url = row['URL']
        
        print(f"Processing URL_ID: {url_id}")
        
        # Extract article text
        article_text = extract_article(url)
        
        # Save article text
        with open(f'articles/{url_id}.txt', 'w', encoding='utf-8') as f:
            f.write(article_text)
        
        # Analyze text
        analysis_results = analyzer.analyze_text(article_text)
        
        # Combine input data with analysis results
        result = {
            'URL_ID': url_id,
            'URL': url,
            **analysis_results
        }
        results.append(result)
    
    # Create output DataFrame
    output_df = pd.DataFrame(results)
    
    # Save to Excel with new filename
    output_df.to_excel('output.xlsx', index=False)
    print("Analysis complete! Results saved to 'output.xlsx'")

if __name__ == '__main__':
    main()
