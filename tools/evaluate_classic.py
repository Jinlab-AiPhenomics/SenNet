import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load datasets (replace with your file paths)
corn_2023 = pd.read_excel('2023corn-class-predictions.xlsx')
oil_2023 = pd.read_excel('2023oil-class-predictions.xlsx')
corn_2024 = pd.read_excel('2024corn-class-predictions.xlsx')
oil_2024 = pd.read_excel('2024oil-class-predictions.xlsx')

# Combine all datasets
combined_data = pd.concat([
    corn_2023[['yield_class', 'Predicted_Yield_Class']],
    oil_2023[['yield_class', 'Predicted_Yield_Class']],
    corn_2024[['yield_class', 'Predicted_Yield_Class']],
    oil_2024[['yield_class', 'Predicted_Yield_Class']]
])


# Function to calculate metrics for each class
def calculate_class_metrics(data, true_col, pred_col):
    y_true = data[true_col]
    y_pred = data[pred_col]
    classes = sorted(y_true.unique())

    # Initialize a dictionary to store metrics for each class
    metrics = {cls: {"Precision": None, "Recall": None, "F1 Score": None, "Accuracy": None} for cls in classes}

    for cls in classes:
        # Binary mask for current class
        y_true_binary = (y_true == cls).astype(int)
        y_pred_binary = (y_pred == cls).astype(int)

        # Calculate precision, recall, F1 for current class
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_binary, y_pred_binary, average="binary", zero_division=0
        )

        # Calculate accuracy for current class
        accuracy = accuracy_score(y_true_binary, y_pred_binary)

        # Store metrics
        metrics[cls]["Precision"] = precision
        metrics[cls]["Recall"] = recall
        metrics[cls]["F1 Score"] = f1
        metrics[cls]["Accuracy"] = accuracy

    # Convert to DataFrame for display
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.index.name = "Class"
    return metrics_df


# Calculate metrics for the combined dataset
combined_class_metrics = calculate_class_metrics(combined_data, 'yield_class', 'Predicted_Yield_Class')


# Function to plot with gradient colors by metrics and class
def visualize_class_metrics_gradient(metrics_df, dataset_name):
    # Set up the bar plot for grouped data
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = metrics_df.columns  # Metrics (Precision, Recall, F1 Score, Accuracy)
    classes = metrics_df.index  # Classes (0, 1, 2)

    x = np.arange(len(metrics))  # X-axis positions for metrics
    bar_width = 0.25  # Width of each bar
    opacity = 0.8

    # Colors based on the class: darker for class 2, medium for 1, light for 0
    colors = {0: "#d4e6f1", 1: "#76b7eb", 2: "#2a81cb"}

    # 更新图例标签
    legend_labels = {0: "Low", 1: "Mid", 2: "High"}

    for i, cls in enumerate(classes):
        offset = (i - 1) * bar_width  # Offset each class group
        ax.bar(x + offset, metrics_df.loc[cls], bar_width,
               alpha=opacity, label=legend_labels[cls], color=colors[cls])

    # 更新图例位置
    ax.legend(loc="upper left", fontsize=15)

    # Add labels and title
    # ax.set_title(f"{dataset_name} Metrics with Gradient Colors", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score", fontsize=23)
    ax.set_xlabel("Metrics", fontsize=23)
    ax.tick_params(axis='x', labelsize=23)  # Adjust X-axis tick font size
    ax.tick_params(axis='y', labelsize=23)
    plt.ylim(0, 1)  # All metrics are percentages (0 to 1)
    plt.tight_layout()
    plt.show()


# Visualize the combined metrics with gradient colors
visualize_class_metrics_gradient(combined_class_metrics, "Combined Data (All Years)")
