import pandas as pd
import os
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Set NLTK data path
nltk.data.path = [os.path.join(os.path.expanduser('~'), 'AppData', 'Roaming', 'nltk_data')]

class ArticleAnalyzer:
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

    def analyze_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return self.analyze_text(text)
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None

    def analyze_text(self, text):
        if not text:
            return None

        # Split into sentences for analysis
        sentences = sent_tokenize(text)
        
        # Get raw words (before cleaning) for sentence length calculations
        raw_words = word_tokenize(text)
        
        # Clean and tokenize text for sentiment analysis
        # Remove punctuation and convert to lower case
        cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = word_tokenize(cleaned_text)
        
        # Remove stop words
        words = [word for word in words if word not in self.stop_words and word.isalnum()]
        
        # 1. Sentiment Analysis
        positive_score = sum(1 for word in words if word in self.positive_words)
        negative_score = sum(1 for word in words if word in self.negative_words) * -1  # Multiply by -1 as per requirements
        
        # Calculate polarity and subjectivity scores
        total_sentiment = positive_score + abs(negative_score)
        polarity_score = (positive_score + negative_score) / (total_sentiment + 0.000001)
        subjectivity_score = total_sentiment / (len(words) + 0.000001)
        
        # 2. Readability Analysis
        avg_sentence_length = len(raw_words) / len(sentences) if sentences else 0
        
        # Count complex words (more than 2 syllables)
        complex_words = [word for word in words if self._count_syllables(word) > 2]
        percent_complex_words = len(complex_words) / len(words) if words else 0
        
        # Fog Index
        fog_index = 0.4 * (avg_sentence_length + percent_complex_words)
        
        # 3. Average Words Per Sentence
        avg_words_per_sentence = avg_sentence_length  # Same as avg_sentence_length
        
        # 4. Complex Word Count
        complex_word_count = len(complex_words)
        
        # 5. Word Count (cleaned words)
        word_count = len(words)
        
        # 6. Syllable Count Per Word
        syllable_count = sum(self._count_syllables(word) for word in words)
        syllable_per_word = syllable_count / len(words) if words else 0
        
        # 7. Personal Pronouns (excluding "US")
        personal_pronouns = len(re.findall(r'\b(I|we|my|ours|us)\b(?!\s+(?:states|state))', text, re.I))
        
        # 8. Average Word Length
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        return {
            'POSITIVE SCORE': positive_score,
            'NEGATIVE SCORE': abs(negative_score),  # Return absolute value
            'POLARITY SCORE': polarity_score,
            'SUBJECTIVITY SCORE': subjectivity_score,
            'AVG SENTENCE LENGTH': avg_sentence_length,
            'PERCENTAGE OF COMPLEX WORDS': percent_complex_words,
            'FOG INDEX': fog_index,
            'AVG NUMBER OF WORDS PER SENTENCE': avg_words_per_sentence,
            'COMPLEX WORD COUNT': complex_word_count,
            'WORD COUNT': word_count,
            'SYLLABLE PER WORD': syllable_per_word,
            'PERSONAL PRONOUNS': personal_pronouns,
            'AVG WORD LENGTH': avg_word_length
        }

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

def main():
    # Read input data
    input_df = pd.read_excel('Input.xlsx')
    
    # Initialize analyzer
    analyzer = ArticleAnalyzer()
    
    # Analyze each article
    results = []
    for _, row in input_df.iterrows():
        url_id = str(row['URL_ID'])
        article_path = f'articles/{url_id}.txt'
        
        print(f"Analyzing article {url_id}")
        
        # Get metrics
        metrics = analyzer.analyze_file(article_path)
        
        if metrics:
            results.append({
                'URL_ID': url_id,
                'URL': row['URL'],
                **metrics
            })
        else:
            print(f"Failed to analyze {url_id}")
    
    # Create output DataFrame with exact column order
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

if __name__ == '__main__':
    main()
