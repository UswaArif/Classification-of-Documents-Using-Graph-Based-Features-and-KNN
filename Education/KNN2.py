from docx import Document
import networkx as nx
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Function to preprocess text
def preprocess_text(text):
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Get English stopwords
    stop_words = set(stopwords.words('english'))
    #print(stop_words)
    # Remove stopwords
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # Initialize Porter Stemmer new instance
    porter = PorterStemmer()
    
    # Stem words
    stemmed_tokens = [porter.stem(word) for word in filtered_tokens]
    
    # Return preprocessed text
    return stemmed_tokens

# Function to read preprocessed data from a document file
def read_preprocessed_data(file_path):
    try:
        document = Document(file_path)
        # Extract text from paragraphs. Extract each paragraph in a line
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

    '''
    common_subgraph = graph_1.subgraph(common_nodes)
    pos = nx.spring_layout(common_subgraph)  # Positions for all nodes
    nx.draw_networkx_nodes(common_subgraph, pos, node_size=700)
    nx.draw_networkx_edges(common_subgraph, pos, width=2, alpha=0.5, edge_color='b')
    nx.draw_networkx_labels(common_subgraph, pos, font_size=20, font_family='sans-serif')
    plt.title('Common Subgraph')
    plt.axis('off')
    plt.show()'''
    
    return distance


# Function to classify test documents using KNN algorithm
'''def classify_knn(test_graph, training_graphs, labels, k):
    distances = []
    for i, train_graph in enumerate(training_graphs):
        distance = calculate_maximal_common_subgraph_size(test_graph, train_graph)
        distances.append((distance, labels[i]))  # Store distance along with label

    # Sort training graphs based on distances
    distances.sort(key=lambda x: x[0])

    # Select k-nearest neighbors
    neighbors = distances[:k]

    # Get the class labels of the k-nearest neighbors
    labels_of_neighbors = [label for _, label in neighbors]

    # Find the majority class label
    majority_label = Counter(labels_of_neighbors).most_common(1)[0][0]

    return majority_label'''

def annotate_documents(test_file_paths):
    true_labels = []
    for file_path in test_file_paths:
        print(f"Document: {file_path}")
        label = input("Enter the label for this document: ")
        true_labels.append(label)
    return true_labels

# Function to create TF-IDF vectors from text data
def create_tfidf_vectors(file_paths, vectorizer=None):
    documents = []
    for file_path in file_paths:
        tokens = read_preprocessed_data(file_path)
        if tokens is not None:
            documents.append(' '.join(tokens))
    
    if vectorizer is None:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
    else:
        tfidf_matrix = vectorizer.transform(documents)
    
    return tfidf_matrix, vectorizer

# Function to classify test documents using SVM
def classify_svm(test_file_paths, training_tfidf_matrix, labels, vectorizer):
    test_tfidf_matrix, _ = create_tfidf_vectors(test_file_paths, vectorizer)
    X_train, X_test, y_train, y_test = train_test_split(training_tfidf_matrix, labels, test_size=0.2, random_state=42)
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    predicted_labels = svm_model.predict(test_tfidf_matrix)
    return predicted_labels

def extract_graph_features(graph):
    # Extract features from the graph
    features = []
    # Example feature: Average node degree
    avg_degree = sum(dict(graph.degree()).values()) / len(graph)
    features.append(avg_degree)
    return features

# Main function
def main():

    topics = ['finance', 'food', 'education']
    features = []
    labels = []

    # Compute distances between graphs within each topic
    for topic in topics:
        for doc_num1 in range(1, 13):  # 12 training documents per topic
            file_path1 = f"D:\\6 semester\\GT project\\{topic}data\\scraped_data_{doc_num1}.docx"
            words1 = read_preprocessed_data(file_path1)
            graph1 = create_directed_graph(words1)
            if graph1 is not None:
                for doc_num2 in range(13, 16):  # 3 testing documents per topic
                    file_path2 = f"D:\\6 semester\\GT project\\{topic}data\\scraped_data_{doc_num2}.docx"  
                    words2 = read_preprocessed_data(file_path2)
                    graph2 = create_directed_graph(words2)
                    if graph2 is not None:  
                        distance = calculate_maximal_common_subgraph_size(graph1, graph2)
                        features.append([distance])
                        labels.append(topic)
    print(len(features))
    desired_test_samples = 9  
    # Calculate the test size based on the desired number of samples
    test_size = desired_test_samples / len(features)

    # Perform the train-test split from scikit-learn with the updated test size
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)
    print("X_train:", X_train)
    print("X_test:", X_test)
    print("y_train:", y_train)
    print("y_test:", y_test)

    # Train k-NN model
    #k=3 means we have three classes in labels of topics
    k = 3  
    knn_classifier = KNeighborsClassifier(n_neighbors=k)

    knn_classifier.fit(X_train, y_train)

    # Predict labels for the test set
    y_pred = knn_classifier.predict(X_test)
    print(y_pred)
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=topics)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=topics, yticklabels=topics)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Calculate precision, recall, and F1 score for k-NN
    precision_knn = precision_score(y_test, y_pred, average='weighted')
    recall_knn = recall_score(y_test, y_pred, average='weighted')
    f1_score_knn = f1_score(y_test, y_pred, average='weighted')

    print("Precision (k-NN):", precision_knn)
    print("Recall (k-NN):", recall_knn)
    print("F1 Score (k-NN):", f1_score_knn)

    # Perform the train-test split with the updated test size
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)

    # Concatenate tokens into strings 
    X_train_strings = [' '.join(map(str, tokens)) for tokens in X_train]

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train_strings)

    # Train k-NN model
    k = 3  
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train_tfidf, y_train)

    # Concatenate tokens into strings for test documents
    X_test_strings = [' '.join(map(str, tokens)) for tokens in X_test]

    # Transform test data into TF-IDF vectors using the same vectorizer
    X_test_tfidf = vectorizer.transform(X_test_strings)

    # Predict labels for the test set
    y_pred = knn_classifier.predict(X_test_tfidf)
    print(" ")
    print("  ")
    print("Vector-Based:")
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    #print("Accuracy:", accuracy)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=topics)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=topics, yticklabels=topics)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Calculate precision, recall, and F1 score for k-NN
    precision_knn = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall_knn = recall_score(y_test, y_pred, average='weighted')
    f1_score_knn = f1_score(y_test, y_pred, average='weighted')

    print("Precision (k-NN):", precision_knn)
    print("Recall (k-NN):", recall_knn)
    print("F1 Score (k-NN):", f1_score_knn)

if __name__ == "__main__":
    main()
