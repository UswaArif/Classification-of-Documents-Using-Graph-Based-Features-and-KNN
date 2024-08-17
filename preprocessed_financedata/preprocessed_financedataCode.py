import os
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from docx import Document

# Function to preprocess text
def preprocess_text(text):
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Get English stopwords
    stop_words = set(stopwords.words('english'))
    
    # Remove stopwords
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # Initialize Porter Stemmer
    porter = PorterStemmer()
    
    # Stem words
    stemmed_tokens = [porter.stem(word) for word in filtered_tokens]
    
    # Return preprocessed text
    return ' '.join(stemmed_tokens)

# Directory containing document files
docs_directory = 'D:\\6 semester\\GT project\\financedata'
preprocessed_directory = 'D:\\6 semester\\GT project\\preprocessed_financedata'

# Iterate over document files in the directory
for filename in os.listdir(docs_directory):
    if filename.endswith('.docx'):
        # Read document
        doc_path = os.path.join(docs_directory, filename)
        document = Document(doc_path)
        
        # Extract text from document
        text = ''
        for paragraph in document.paragraphs:
            text += paragraph.text + ' '
        
        # Preprocess text
        preprocessed_text = preprocess_text(text)
        
        # Save preprocessed data to a new file
        preprocessed_filename = f'preprocessed_{filename[:-5]}.docx'  # Modify the extension as needed
        preprocessed_filepath = os.path.join(preprocessed_directory, preprocessed_filename)
        with open(preprocessed_filepath, 'w', encoding='utf-8') as f:
            f.write(preprocessed_text)
        
        print(f'Preprocessed data saved to {preprocessed_filepath}')
