import pandas as pd  # For data manipulation
from sklearn.model_selection import train_test_split  # To split datasets into training and testing sets
from sklearn.pipeline import make_pipeline  # To create a pipeline for scaling and modeling
from sklearn.preprocessing import StandardScaler  # For scaling feature data
from sklearn.linear_model import LogisticRegression  # The machine learning model used for classification

def train_model(file_path): #    Train a logistic regression model using the dataset provided in the file path.

    dataset = pd.read_csv(file_path)  # Load the dataset from the CSV file

    # Separate features (X) and labels (y)
    X = dataset.drop('class', axis=1)  # Drop the 'class' column to get the features
    y = dataset['class']  # The 'class' column contains the labels

    # Split the data into training (70%) and testing (30%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create a pipeline with data scaling and logistic regression
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))

    # Train the model using the training data
    pipeline.fit(X_train, y_train)

    # Return the trained model and testing data for evaluation
    return {
        "model": pipeline,  # The trained logistic regression model
        "X_test": X_test,   # Test features
        "y_test": y_test    # Test labels
    }
