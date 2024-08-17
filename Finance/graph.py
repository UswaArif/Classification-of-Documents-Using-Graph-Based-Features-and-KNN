from docx import Document
import networkx as nx
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

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
    return stemmed_tokens 

# Function to read preprocessed data from a document file
def read_preprocessed_data(file_path):
    try:
        document = Document(file_path)
        # Extract text from paragraphs
        text = '\n'.join([paragraph.text for paragraph in document.paragraphs])
        return preprocess_text(text)
    except Exception as e:
        print(f"Error reading preprocessed data from document: {e}")
        return None

# Function to create a directed graph from the list of tokens
def create_directed_graph(tokens):
    try:
        G = nx.DiGraph()
        for i in range(len(tokens) - 1):
            source = tokens[i]
            target = tokens[i + 1]
            G.add_edge(source, target)
        return G
    except Exception as e:
        print(f"Error creating directed graph: {e}")
        return None

# Function to generate directed graphs for multiple documents
def generate_graphs(file_paths):
    for i, file_path in enumerate(file_paths, start=1):
        print(f"Generating graph for document {i}/{len(file_paths)}")
        # Read preprocessed data from the document
        tokens = read_preprocessed_data(file_path)
        if tokens is None:
            continue
        
        # Create a directed graph
        G = create_directed_graph(tokens)
        if G is None:
            continue
        
        # Visualize the directed graph
        plt.figure(figsize=(10, 6))
        nx.draw(G, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_weight='bold')
        plt.title(f"Directed Graph of Terms from Document {i}")
        plt.show()

# Main function
def main():
    # File paths of the documents containing preprocessed data
    file_paths = [
        "d:\\6 semester\\GT project\\financedata\\scraped_data_1.docx",
        "d:\\6 semester\\GT project\\financedata\\scraped_data_2.docx",
        "d:\\6 semester\\GT project\\financedata\\scraped_data_3.docx",
        "d:\\6 semester\\GT project\\financedata\\scraped_data_4.docx",
        "d:\\6 semester\\GT project\\financedata\\scraped_data_5.docx",
        "d:\\6 semester\\GT project\\financedata\\scraped_data_6.docx",
        "d:\\6 semester\\GT project\\financedata\\scraped_data_7.docx",
        "d:\\6 semester\\GT project\\financedata\\scraped_data_8.docx",
        "d:\\6 semester\\GT project\\financedata\\scraped_data_9.docx",
        "d:\\6 semester\\GT project\\financedata\\scraped_data_10.docx",
        "d:\\6 semester\\GT project\\financedata\\scraped_data_11.docx",
        "d:\\6 semester\\GT project\\financedata\\scraped_data_12.docx",
    ]

    # Generate directed graphs for each document
    generate_graphs(file_paths)

if __name__ == "__main__":
    main()
