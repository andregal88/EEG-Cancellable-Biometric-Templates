import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the combined features dataset
df = pd.read_csv("../output/features/all_features_dataset.csv")

# Define which runs are for training and which for testing
train_runs = ["R01", "R02", "R03", "R04"]
test_runs = ["R05", "R06", "R07", "R08", "R09", "R10", "R11", "R12", "R13", "R14"]

train_df = df[df["run"].isin(train_runs)]
test_df = df[df["run"].isin(test_runs)]

# Select features (drop subject/run/epoch_file columns)
X_train = train_df.drop(columns=["subject", "run", "epoch_file"])
y_train = train_df["subject"]
X_test = test_df.drop(columns=["subject", "run", "epoch_file"])
y_test = test_df["subject"]

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = clf.predict(X_test_scaled)

# Print classification results
print(classification_report(y_test, y_pred))