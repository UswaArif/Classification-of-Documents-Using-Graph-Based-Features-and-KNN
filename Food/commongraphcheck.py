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
    
# Function to calculate the maximal common subgraph size between two graphs
def calculate_maximal_common_subgraph_size(graph_1, graph_2):
    num_nodes_g1 = len(graph_1)
    num_nodes_g2 = len(graph_2)
    common_nodes = set(graph_1.nodes()) & set(graph_2.nodes())
    common_subgraph_size = len(graph_1.subgraph(common_nodes))
    max_size = max(num_nodes_g1, num_nodes_g2)
    distance = 1 - (common_subgraph_size / max_size)
    return distance

# Function to create graphs for each file
def create_graphs_for_files(file_paths):
    graphs = []
    for file_path in file_paths:
        tokens = read_preprocessed_data(file_path)
        if tokens is not None:
            graph = create_directed_graph(tokens)
            if graph is not None:
                graphs.append(graph)
    return graphs

# Function to calculate distances between all pairs of graphs
def calculate_distances_between_graphs(graphs):
    distances = {}
    for i in range(len(graphs)):
        for j in range(i+1, len(graphs)):
            graph_1 = graphs[i]
            graph_2 = graphs[j]
            distance = calculate_maximal_common_subgraph_size(graph_1, graph_2)
            distances[(i, j)] = distance
    return distances

# Main function
def main():
    # File paths of the documents containing preprocessed data
    file_paths = [
        "d:\\6 semester\\GT project\\fooddata\\scraped_data_1.docx",
        "d:\\6 semester\\GT project\\fooddata\\scraped_data_2.docx",
        "d:\\6 semester\\GT project\\fooddata\\scraped_data_3.docx",
        "d:\\6 semester\\GT project\\fooddata\\scraped_data_4.docx",
        "d:\\6 semester\\GT project\\fooddata\\scraped_data_5.docx",
        "d:\\6 semester\\GT project\\fooddata\\scraped_data_6.docx",
        "d:\\6 semester\\GT project\\fooddata\\scraped_data_7.docx",
        "d:\\6 semester\\GT project\\fooddata\\scraped_data_8.docx",
        "d:\\6 semester\\GT project\\fooddata\\scraped_data_9.docx",
        "d:\\6 semester\\GT project\\fooddata\\scraped_data_10.docx",
        "d:\\6 semester\\GT project\\fooddata\\scraped_data_11.docx",
        "d:\\6 semester\\GT project\\fooddata\\scraped_data_12.docx",
        # Add paths to other files as needed
    ]

    # Create graphs for each file
    graphs = create_graphs_for_files(file_paths)

    # Calculate distances between all pairs of graphs
    distances = calculate_distances_between_graphs(graphs)

    # Print distances
    for (i, j), distance in distances.items():
        print(f"Distance between Graph {i+1} and Graph {j+1}: {distance}")

if __name__ == "__main__":
    main()
