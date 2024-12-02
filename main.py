from model_training import train_model
from evaluation import evaluate_model, plot_accuracy, plot_classification_report

# Define the file paths for the datasets
file_ArSL = "ArSL_dataset_100 Frame.csv"  # Path to the 100-frame dataset
file_ASL = "ASL_dataset_100 Frame.csv"  # Path to the 300-frame dataset

# Train models for each dataset
model_ArSL= train_model(file_ArSL)  # Train a model using the 100-frame dataset
model_ASL = train_model(file_ASL)  # Train a model using the 300-frame dataset

# Evaluate the trained models and display the results
accuracy_ArSL, report_ArSL = evaluate_model(model_ArSL, file_ArSL)
accuracy_ASL, report_ASL = evaluate_model(model_ASL, file_ASL)

# Plot a comparison of accuracies between the two datasets
plot_accuracy([accuracy_ArSL, accuracy_ASL], ["ArSL", "ASL"])

# Plot detailed classification metrics for each dataset
plot_classification_report(report_ArSL, "100 ArSL Frames Dataset")
plot_classification_report(report_ASL, "100 ASL Frames Dataset")
