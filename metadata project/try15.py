import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk, font

# Function to load the dataset
def load_data(file_path):
    """Loads the dataset from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        print("Metadata loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None

# Function to preprocess the metadata
def preprocess_data(data):
    """Encodes target labels and selects metadata features for modeling."""
    data['classification'] = LabelEncoder().fit_transform(data['classification'])
    features = [
        "usage_counter", "prio", "vm_pgoff", "task_size", "mm_users",
        "map_count", "min_flt", "maj_flt", "utime", "stime"
    ]  # Metadata features
    X = data[features]
    y = data['classification']
    return X, y

# Function to split the data into training and testing sets
def split_data(X, y, test_size=0.3):
    """Splits the dataset into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print("Metadata split into training and testing sets.")
    return X_train, X_test, y_train, y_test

# Function to perform Randomized Search for hyperparameter tuning
def perform_random_search(X_train, y_train):
    """Performs Randomized Search to find the best hyperparameters."""
    param_dist = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rfc = RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1)
    random_search = RandomizedSearchCV(rfc, param_distributions=param_dist, cv=3, scoring='roc_auc', n_iter=10, verbose=0, n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)
    
    print("Randomized search completed.")
    return random_search

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    """Evaluates the model and returns performance metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    print("Model evaluation completed.")
    return conf_matrix, classification_rep, auc_roc, y_pred, y_pred_proba

# Function to plot ROC Curve
def plot_roc_curve(y_test, y_pred_proba, auc_roc):
    """Plots the Receiver Operating Characteristic (ROC) curve."""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='orange', label=f'AUC-ROC = {auc_roc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Threat Detection')
    plt.legend()
    plt.show()
    print("ROC Curve plotted.")

# Function to plot Confusion Matrix
def plot_confusion_matrix(conf_matrix):
    """Plots the confusion matrix using Seaborn."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Threat'], yticklabels=['Normal', 'Threat'])
    plt.title('Confusion Matrix for Metadata Analysis')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    print("Confusion matrix plotted.")

# Function to show results in a GUI
def show_results(root, best_params, auc_roc, classification_rep, y_pred):
    """Displays evaluation results in a Tkinter GUI window."""
    threat_count = (y_pred == 1).sum()
    normal_count = (y_pred == 0).sum()
    
    results_message = f"""
    Metadata Analysis Results:
    --------------------------
    Best Parameters: {best_params}
    AUC-ROC Score: {auc_roc:.2f}
    
    Classification Report:
    {classification_rep}
    
    Results Summary:
    --------------------------
    Threats Detected: {threat_count}
    Normal Metadata: {normal_count}

    Judgment Factors:
    - Login Time
    - Data Arrived
    - Data Sent
    - Usage Time
    - Location (if available)
    """

    results_window = tk.Toplevel(root)
    results_window.title("Analysis Results")
    results_window.geometry("500x400")

    title_font = font.Font(family="Helvetica", size=16, weight="bold")
    ttk.Label(results_window, text="Metadata Analysis Results", font=title_font).pack(pady=10)

    results_text = tk.Text(results_window, wrap="word", height=15, width=50, font=("Arial", 11))
    results_text.insert("1.0", results_message)
    results_text.config(state="disabled")
    results_text.pack(padx=10, pady=10)

    ttk.Button(results_window, text="Close", command=results_window.destroy).pack(pady=10)

# Main GUI Window
def main_gui(best_params, auc_roc, classification_rep, y_pred):
    """Creates the main GUI window for the metadata analysis system."""
    root = tk.Tk()
    root.title("Potential Threat Detection System")
    root.geometry("500x250")

    title_font_main = font.Font(family="Helvetica", size=18, weight="bold")
    tk.Label(root, text="Metadata Analysis Tool", font=title_font_main).pack(pady=10)

    ttk.Button(root, text="Show Results", command=lambda: show_results(root, best_params, auc_roc, classification_rep, y_pred)).pack(pady=20)

    root.mainloop()

# Main execution flow
if __name__ == "__main__":
    file_path = r'c:\Users\abhis\Downloads\archive\Malware dataset.csv'
    data = load_data(file_path)

    if data is not None:
        X, y = preprocess_data(data)
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        random_search = perform_random_search(X_train, y_train)
        best_params = random_search.best_params_
        model = random_search.best_estimator_

        conf_matrix, classification_rep, auc_roc, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)

        plot_roc_curve(y_test, y_pred_proba, auc_roc)
        plot_confusion_matrix(conf_matrix)

        main_gui(best_params, auc_roc, classification_rep, y_pred)
