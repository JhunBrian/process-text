# process-text

A simple and lightweight text preprocessing toolkit for NLP pipelines.

## Features
- Lowercasing  
- Punctuation removal  
- Tokenization  
- Stopword removal  
- Stemming  
- Lemmatization with POS tagging  
- Emoji removal  
- Special character filtering  
- Misspelling correction using TextBlob  
- Compose multiple transformations into a pipeline  

## Installation

```bash
pip install akoang-library


## Usage

```from process_text import transforms

string = "Hello World 🫠"

trans = transforms.Compose([
  transforms.LowerCase(),
  transforms.RemoveEmojis()
])

print(trans(string))
