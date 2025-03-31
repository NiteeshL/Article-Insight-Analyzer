import pandas as pd
import os
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

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

        sentences = sent_tokenize(text)
        raw_words = word_tokenize(text)
        cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = word_tokenize(cleaned_text)
        words = [word for word in words if word not in self.stop_words and word.isalnum()]
        positive_score = sum(1 for word in words if word in self.positive_words)
        negative_score = sum(1 for word in words if word in self.negative_words) * -1
        total_sentiment = positive_score + abs(negative_score)
        polarity_score = (positive_score + negative_score) / (total_sentiment + 0.000001)
        subjectivity_score = total_sentiment / (len(words) + 0.000001)
        avg_sentence_length = len(raw_words) / len(sentences) if sentences else 0
        complex_words = [word for word in words if self._count_syllables(word) > 2]
        percent_complex_words = len(complex_words) / len(words) if words else 0
        fog_index = 0.4 * (avg_sentence_length + percent_complex_words)
        avg_words_per_sentence = avg_sentence_length
        complex_word_count = len(complex_words)
        word_count = len(words)
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
            'AVG NUMBER OF WORDS PER SENTENCE': avg_words_per_sentence,
            'COMPLEX WORD COUNT': complex_word_count,
            'WORD COUNT': word_count,
            'SYLLABLE PER WORD': syllable_per_word,
            'PERSONAL PRONOUNS': personal_pronouns,
            'AVG WORD LENGTH': avg_word_length
        }

    def _count_syllables(self, word):
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        if word.endswith('es') or word.endswith('ed'):
            word = word[:-2]
        prev_char_is_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_char_is_vowel:
                count += 1
            prev_char_is_vowel = is_vowel
        if word.endswith('e'):
            count -= 1
        if count == 0:
            count = 1
        return count

def main():
    input_df = pd.read_excel('Input.xlsx')
    analyzer = ArticleAnalyzer()
    results = []
    for _, row in input_df.iterrows():
        url_id = str(row['URL_ID'])
        article_path = f'articles/{url_id}.txt'
        print(f"Analyzing article {url_id}")
        metrics = analyzer.analyze_file(article_path)
        if metrics:
            results.append({
                'URL_ID': url_id,
                'URL': row['URL'],
                **metrics
            })
        else:
            print(f"Failed to analyze {url_id}")
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
