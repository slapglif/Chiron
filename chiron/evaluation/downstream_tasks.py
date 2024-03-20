
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def semantic_similarity_prediction(embeddings, labels):
    # Split data into training and testing sets
    train_embeddings, test_embeddings, train_labels, test_labels = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42
    )

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(train_embeddings, train_labels)

    # Make predictions on the test set
    predictions = model.predict(test_embeddings)

    # Evaluate the model
    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, average="weighted")

    return accuracy, f1


def text_classification(embeddings, labels):
    # Split data into training and testing sets
    train_embeddings, test_embeddings, train_labels, test_labels = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42
    )

    # Train an SVM classifier
    classifier = SVC()
    classifier.fit(train_embeddings, train_labels)

    # Make predictions on the test set
    predictions = classifier.predict(test_embeddings)

    # Evaluate the classifier
    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, average="weighted")

    return accuracy, f1