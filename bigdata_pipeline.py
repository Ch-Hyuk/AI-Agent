import os
import matplotlib.pyplot as plt
import dask.dataframe as dd
from dask_ml.linear_model import LogisticRegression
from dask_ml.model_selection import train_test_split
from dask_ml.metrics import accuracy_score
from sklearn.datasets import make_classification


# Generate synthetic big dataset
n_samples = int(os.environ.get('N_SAMPLES', '10000'))
features, labels = make_classification(
    n_samples=n_samples,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    random_state=42,
)

# Convert to Dask DataFrame
import pandas as pd
import numpy as np

df = pd.DataFrame(features)
df['label'] = labels
ddf = dd.from_pandas(df, npartitions=8)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    ddf.drop('label', axis=1), ddf['label'], test_size=0.2, random_state=42
)

# Convert dataframes to dask arrays for dask-ml estimators
X_train_da = X_train.to_dask_array(lengths=True)
X_test_da = X_test.to_dask_array(lengths=True)
y_train_da = y_train.to_dask_array(lengths=True)
y_test_da = y_test.to_dask_array(lengths=True)

# Train Logistic Regression model
model = LogisticRegression(max_iter=100)
model.fit(X_train_da, y_train_da)

# Predict and evaluate
preds = model.predict(X_test_da)
accuracy = accuracy_score(y_test_da, preds)
print(f"Accuracy: {accuracy:.4f}")

# Collect small sample for visualization
sample = X_test.head(1000)
sample_labels = y_test.head(1000)
sample_da = dd.from_pandas(sample, npartitions=1).to_dask_array(lengths=True)
sample_preds = model.predict(sample_da).compute()

# Visualize predicted vs actual labels
plt.figure(figsize=(8, 6))
plt.scatter(range(len(sample_preds)), sample_preds, color='blue', label='Predicted', alpha=0.6)
plt.scatter(range(len(sample_labels)), sample_labels, color='red', label='Actual', alpha=0.6)
plt.title('Predicted vs Actual Labels (Sample)')
plt.xlabel('Sample Index')
plt.ylabel('Label')
plt.legend()
plt.tight_layout()
plt.savefig('results.png')
print('Saved plot to results.png')
