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

# Function to classify test documents using KNN algorithm
def classify_knn(test_graph, training_graphs, labels, k):
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

    return majority_label

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

    # Labels for each document (assuming binary classification, modify as needed)
    # Assign arbitrary labels to each document
    labels = [f"class{i}" for i in range(len(file_paths))]

    # Create graphs and extract labels for each file
    graphs = []
    for file_path in file_paths:
        tokens = read_preprocessed_data(file_path)
        if tokens is not None:
            graph = create_directed_graph(tokens)
            if graph is not None:
                graphs.append(graph)

    # File paths of the test documents
    test_file_paths = [
        "d:\\6 semester\\GT project\\financedata\\scraped_data_13.docx",
        "d:\\6 semester\\GT project\\financedata\\scraped_data_14.docx",
        "d:\\6 semester\\GT project\\financedata\\scraped_data_15.docx",
    ]

    # Lists to store predicted labels
    predicted_labels = []

    # Iterate over each test document
    for test_document_path in test_file_paths:
        # Load and preprocess test document
        test_tokens = read_preprocessed_data(test_document_path)
        if test_tokens is None:
            continue

        # Create graph for test document
        test_graph = create_directed_graph(test_tokens)
        if test_graph is None:
            continue

        # Classify test document using KNN
        k = 3  # Number of neighbors to consider
        predicted_label = classify_knn(test_graph, graphs, labels, k)
        predicted_labels.append(predicted_label)
        print(f"Predicted label for test document {test_document_path} is: {predicted_label}")

    true_labels = ["class8", "class11", "class11"]  

    # Compute confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Display confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

        # Extract TP, FP, TN, FN values
    TP = conf_matrix[1, 1]  # True Positives
    FP = conf_matrix[0, 1]  # False Positives
    TN = conf_matrix[0, 0]  # True Negatives
    FN = conf_matrix[1, 0]  # False Negatives

    print("True Positives (TP):", TP)
    print("False Positives (FP):", FP)
    print("True Negatives (TN):", TN)
    print("False Negatives (FN):", FN)

# Calculate accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
    print("Accuracy:", accuracy)

    # Calculate precision
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    print("Precision:", precision)

    # Calculate recall
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    print("Recall:", recall)

    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    print("F1 Score:", f1_score)

    # Compute TF-IDF vectors for SVM
    training_tfidf_matrix, vectorizer = create_tfidf_vectors(file_paths)

    # Classify test documents using SVM
    predicted_labels_svm = classify_svm(test_file_paths, training_tfidf_matrix, labels, vectorizer)

    # Compute confusion matrix for SVM
    conf_matrix_svm = confusion_matrix(true_labels, predicted_labels_svm)

    # Display confusion matrix for SVM
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_svm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (SVM)")
    plt.show()

    # Extract TP, FP, TN, FN values for SVM
    TP_svm = conf_matrix_svm[1, 1]  # True Positives
    FP_svm = conf_matrix_svm[0, 1]  # False Positives
    TN_svm = conf_matrix_svm[0, 0]  # True Negatives
    FN_svm = conf_matrix_svm[1, 0]  # False Negatives

    print("True Positives (TP) for SVM:", TP_svm)
    print("False Positives (FP) for SVM:", FP_svm)
    print("True Negatives (TN) for SVM:", TN_svm)
    print("False Negatives (FN) for SVM:", FN_svm)

    # Calculate accuracy for SVM
    accuracy_svm = (TP_svm + TN_svm) / (TP_svm + TN_svm + FP_svm + FN_svm) if (TP_svm + TN_svm + FP_svm + FN_svm) != 0 else 0
    print("Accuracy for SVM:", accuracy_svm)

    # Calculate precision for SVM
    precision_svm = TP_svm / (TP_svm + FP_svm) if (TP_svm + FP_svm) != 0 else 0
    print("Precision for SVM:", precision_svm)

    # Calculate recall for SVM
    recall_svm = TP_svm / (TP_svm + FN_svm) if (TP_svm + FN_svm) != 0 else 0
    print("Recall for SVM:", recall_svm)

    # Calculate F1 score for SVM
    f1_svm = 2 * (precision_svm * recall_svm) / (precision_svm + recall_svm) if (precision_svm + recall_svm) != 0 else 0
    print("F1 Score for SVM:", f1_svm)

if __name__ == "__main__":
    main()
