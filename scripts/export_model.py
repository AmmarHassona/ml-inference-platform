from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model_artifacts"
MODEL_DIR.mkdir(exist_ok=True)

NUMERICAL_FEATURES = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]

# load data
print("Loading Adult Income dataset...")
adult = fetch_openml("adult", version=2, as_frame=True)
X = adult.data[NUMERICAL_FEATURES].astype(float).to_numpy()
y = (adult.target == ">50K").astype(int).to_numpy()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train models
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(x_train, y_train)

gb_clf = GradientBoostingClassifier(random_state=42)
gb_clf.fit(x_train, y_train)

# predict and evaluate
rf_y_pred = rf_clf.predict(x_test)
gb_y_pred = gb_clf.predict(x_test)

print("Random Forest Classifier Accuracy:", metrics.accuracy_score(y_test, rf_y_pred))
print("Gradient Boosting Classifier Accuracy:", metrics.accuracy_score(y_test, gb_y_pred))

# save training features for data drift detection
np.save(MODEL_DIR / "reference_features.npy", x_train)

# set float array of shape (batch_size, features)
initial_type = [("float_input", FloatTensorType([None, 6]))]
onnx_model_rf = convert_sklearn(rf_clf, initial_types=initial_type)
onnx_model_gb = convert_sklearn(gb_clf, initial_types=initial_type)

# save model onnx
with open(MODEL_DIR / "model_v1.onnx", "wb") as f:
    f.write(onnx_model_rf.SerializeToString())

with open(MODEL_DIR / "model_v2.onnx", "wb") as f:
    f.write(onnx_model_gb.SerializeToString())

print("Model artifacts saved to", MODEL_DIR)