import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Step 1: Simulate Metadata Dataset
# Columns: UserID, FileAccessCount, NetworkUsage, LoginAttempts, Label (0 = Normal, 1 = Anomalous)
np.random.seed(42)
data = {
    "UserID": [f"User_{i}" for i in range(1, 101)],
    "FileAccessCount": np.random.randint(1, 500, 100),
    "NetworkUsage": np.random.randint(100, 10000, 100),
    "LoginAttempts": np.random.randint(1, 10, 100),
    "Label": np.random.choice([0, 1], size=100, p=[0.9, 0.1])  # 10% anomalies
}

df = pd.DataFrame(data)

# Step 2: Preprocessing
X = df[["FileAccessCount", "NetworkUsage", "LoginAttempts"]]
y = df["Label"]

# Define pipeline with normalization, PCA, and Random Forest Classifier
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('clf', RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=5))
])

# Perform cross-validation
scores = cross_val_score(pipe, X, y, cv=5, scoring='f1_macro')
print("Cross-validation scores: ", scores)
print("Average cross-validation score: ", np.mean(scores))

# Step 3: Evaluate model on test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
y_pred_proba = pipe.predict_proba(X_test)[:, 1]

# Print evaluation metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nF1 Score:")
print(f1_score(y_test, y_pred))

print("\nAUC-ROC Score:")
print(roc_auc_score(y_test, y_pred_proba))

# Step 4: Visualize Results
# Add prediction results to the test dataset for visualization
X_test_df = pd.DataFrame(X_test, columns=["FileAccessCount", "NetworkUsage", "LoginAttempts"])
X_test_df["Label"] = y_test
X_test_df["Predicted"] = y_pred
X_test_df["Predicted_Proba"] = y_pred_proba

# Plot Network Usage vs. File Access Count with anomaly detection
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=X_test_df["FileAccessCount"],
    y=X_test_df["NetworkUsage"],
    hue=X_test_df["Predicted"],
    style=X_test_df["Label"],
    palette={0: "green", 1: "red"}
)
plt.title("Anomaly Detection Visualization")
plt.xlabel("File Access Count")
plt.ylabel("Network Usage")
plt.legend(title="Predicted Label")
plt.show()

# Plot predicted probability distribution
plt.figure(figsize=(10, 6))
sns.histplot(X_test_df["Predicted_Proba"], bins=10, kde=True)
plt.title("Predicted Probability Distribution")
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.show()

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'clf__n_estimators': [10, 50, 100, 200],
    'clf__max_depth': [5, 10, 15],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 5, 10]
}
grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='f1_macro')
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Evaluate the best model on the test set
best_pipe = grid_search.best_estimator_
y_pred_best = best_pipe.predict(X_test)
y_pred_proba_best = best_pipe.predict_proba(X_test)[:, 1]
print("Best model's accuracy:", accuracy_score(y_test, y_pred_best))
print("Best model's F1 score:", f1_score(y_test, y_pred_best))
print("Best model's AUC-ROC score:", roc_auc_score(y_test, y_pred_proba_best))