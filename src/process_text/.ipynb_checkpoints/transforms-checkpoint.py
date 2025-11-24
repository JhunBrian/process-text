import regex as re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import contractions
from textblob import TextBlob
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet

__all__ = ["Compose", "LowerCase", "RemovePunctuations", "Tokenize", "RemoveStopWords", "Stem", "Lemmatize",
           "RemoveSpecialCharacters", "RemoveEmojis", "HandleMispellings"
          ]


class LowerCase:
    def __call__(self, text):
        return text.lower()

class RemovePunctuations:
    def __init__(self):
        self.punctuations = string.punctuation
        
    def __call__(self, text):
        return text.translate(str.maketrans('', '', self.punctuations))

class Tokenize:
    def __init__(self, tokenize_fn=None):
        self.tokenize_fn = tokenize_fn

    def __call__(self, text):
        if self.tokenize_fn == None:
            def tokenize_fn(text):
                return text.split(' ')

            return tokenize_fn(text)


class RemoveStopWords:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def __call__(self, text):
        words = text.split(' ')
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)


class Stem:
    def __init__(self, stemmer=PorterStemmer()):
        self.stemmer = stemmer

    def __call__(self, text):
        words = text.split(' ')
        stemmed = [self.stemmer.stem(word) for word in words]
        return ' '.join(stemmed)


class Lemmatize:
    def __init__(self, lemmatizer=None):
        # Set default lemmatizer to WordNetLemmatizer if not provided
        self.lemmatizer = lemmatizer or WordNetLemmatizer()

    def __call__(self, text):
        words = word_tokenize(text)  # Tokenize text into words
        pos_tagged = pos_tag(words)  # POS tag the tokenized words
        lemmatized = [
            self.lemmatizer.lemmatize(word, self.get_wordnet_pos(pos)) or word
            for word, pos in pos_tagged
        ]
        return ' '.join(lemmatized)

    @staticmethod
    def get_wordnet_pos(treebank_tag):
        """Converts POS tag from treebank to WordNet format."""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN


class RemoveSpecialCharacters:
    def __init__(self, pattern=None):
        if pattern == None:
            self.pattern = r'[^a-zA-Z0-9\s]'
        else:
            self.pattern = pattern
            
    def __call__(self, text):
        return re.sub(self.pattern, '', text)


class RemoveEmojis:
    def __call__(self, text):
        emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
        "\U0001F700-\U0001F77F"  # Alchemical Symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed Characters
        "]+", flags=re.UNICODE
        )
        return emoji_pattern.sub(r'', text)


class HandleMispellings:
    def __call__(self, text):
        words = text.split(' ')
        corrected = [str(TextBlob(word).correct()) for word in words]
        return ' '.join(corrected)


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, text):
        for t in self.transforms:
            text = t(text)

        return text