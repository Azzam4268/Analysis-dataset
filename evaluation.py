from sklearn.metrics import accuracy_score, classification_report  # For evaluation metrics
import matplotlib.pyplot as plt  # For creating visualizations
import pandas as pd  # For better table display
import numpy as np  # For numerical operations

def evaluate_model(trained_data, file_path): #Evaluate the trained model on the test data and print results.


    model = trained_data["model"]
    X_test = trained_data["X_test"]
    y_test = trained_data["y_test"]

    # Predict labels for the test data
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Generate a detailed classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Convert the classification report into a pandas DataFrame for better visualization
    report_df = pd.DataFrame(report).transpose()

    # Print the results
    print(f"Results for {file_path}:")
    print(f"Accuracy: {accuracy:.2%}")
    print("Classification Report:")
    print(report_df)
    print("-" * 50)

    return accuracy, report


def plot_accuracy(accuracies, labels): #    Create and save a bar chart comparing the accuracies of datasets.


    plt.figure(figsize=(8, 5))
    plt.bar(labels, accuracies, color=['blue', 'green'])  # Colors for the bars
    plt.ylim(0.9, 1.0)  # Set y-axis range for better visualization
    plt.title("Accuracy Comparison Between Datasets", fontsize=14)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xlabel("Dataset", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Annotate the bars with accuracy values
    for i, v in enumerate(accuracies):
        plt.text(i, v - 0.005, f"{v:.2%}", ha="center", fontsize=10, color="white")

    # Save the plot as an image
    plt.tight_layout()
    plt.savefig("accuracy_comparison.png")
    print("Accuracy comparison plot saved as 'accuracy_comparison.png'")


def plot_classification_report(report, dataset_name): #    Create and save a bar chart showing precision, recall, and F1-score for each class.


    labels = list(report.keys())[:-3]  # Ignore 'accuracy', 'macro avg', and 'weighted avg'
    precision = [report[label]['precision'] for label in labels]
    recall = [report[label]['recall'] for label in labels]
    f1_score = [report[label]['f1-score'] for label in labels]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(15, 6))
    plt.bar(x - width, precision, width, label='Precision', color='blue')
    plt.bar(x, recall, width, label='Recall', color='green')
    plt.bar(x + width, f1_score, width, label='F1-Score', color='orange')

    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Scores', fontsize=12)
    plt.title(f'Classification Metrics for {dataset_name}', fontsize=14)
    plt.xticks(x, labels, rotation=90, fontsize=10)
    plt.legend(fontsize=12)
    plt.tight_layout()

    filename = f"{dataset_name.replace(' ', '_').lower()}_classification_report.png"
    plt.savefig(filename)
    print(f"Classification report plot saved as '{filename}'")
