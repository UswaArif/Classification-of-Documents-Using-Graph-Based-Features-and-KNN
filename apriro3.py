#giving only one frequent subgraph an selecting according to itself
from docx import Document
import networkx as nx
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from itertools import combinations

# Function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    porter = PorterStemmer()
    stemmed_tokens = [porter.stem(word) for word in filtered_tokens]
    return stemmed_tokens

# Function to read preprocessed data from a document file
def read_preprocessed_data(file_path):
    try:
        document = Document(file_path)
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

# Function to convert directed graph to undirected graph
def convert_to_undirected(G):
    try:
        return G.to_undirected()
    except Exception as e:
        print(f"Error converting graph to undirected: {e}")
        return None
    
# Function to find frequent subgraphs using the Apriori algorithm
def apriori_algorithm(graphs, min_support):
    frequent_subgraphs = []
    support_counts = {}  # Dictionary to store support counts for each subgraph
    
    # Extract all unique edges from the graphs
    edges = set()
    for graph in graphs:
        edges.update(graph.edges())
    
    # Generate candidate subgraphs of larger size
    candidate_subgraphs = []
    for size in range(2, min(len(edges) + 1, 4)):  # Adjust size range
        candidate_subgraphs.extend(combinations(edges, size))
    
    # Calculate support for candidate subgraphs
    for graph in graphs:
        for subgraph in candidate_subgraphs:
            if set(subgraph).issubset(graph.edges()):
                support_counts[subgraph] = support_counts.get(subgraph, 0) + 1
    
    # Filter frequent subgraphs based on minimum support
    for subgraph, support in support_counts.items():
        if support >= min_support:
            frequent_subgraphs.append(subgraph)
    
    return frequent_subgraphs, support_counts

# Read preprocessed data from the document
file_path = "d:\\6 semester\\GT project\\fooddata\\scraped_data_1.docx"
tokens = read_preprocessed_data(file_path)

# Create a directed graph
if tokens:
    G = create_directed_graph(tokens)

    # Convert directed graph to undirected
    G_undirected = convert_to_undirected(G)

    # Define the training set of graphs
    training_graphs = [G_undirected]  # For simplicity, using only one graph as an example

    # Define the minimum support threshold
    min_support = 1  # Adjust as needed

    # Call the Apriori algorithm to find frequent subgraphs
    frequent_subgraphs, support_counts = apriori_algorithm(training_graphs, min_support)

    # Initialize variable to store the selected frequent subgraph
    selected_subgraph = None

    # Check each subgraph for frequent
    for subgraph in frequent_subgraphs:
        # Check if the subgraph meets the minimum support threshold
        if support_counts[subgraph] >= min_support:
            selected_subgraph = subgraph
            break

    # Display the selected frequent subgraph
    if selected_subgraph:
        plt.figure(figsize=(8, 6))
        G_selected = nx.Graph(selected_subgraph)
        nx.draw(G_selected, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_weight='bold')
        plt.title("Selected Frequent Subgraph")
        plt.show()
    else:
        print("No frequent subgraph selected.")
else:
    print("Error: Unable to read preprocessed data.")
