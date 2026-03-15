"""
This serves as a test script that trains simple models using scikit-learn on the standard iris dataset. 
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

import numpy as np

# load data
iris = load_iris()
x, y = iris.data, iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# train models
rf_clf = RandomForestClassifier(n_estimators = 100, random_state = 42)
rf_clf.fit(x_train, y_train)

gb_clf = GradientBoostingClassifier(random_state = 42)
gb_clf.fit(x_train, y_train)

# predict and evaluate
rf_y_pred = rf_clf.predict(x_test)
gb_y_pred = gb_clf.predict(x_test)

print("Random Forest Classifier Accuracy: ", metrics.accuracy_score(y_test, rf_y_pred))
print("Gradient Boosting Classifier Accuracy: ", metrics.accuracy_score(y_test, gb_y_pred))

# save training features for data drift detection
np.save("model_artifacts/reference_features.npy", x_train)

# set float array of shape (batch_size, features)
initial_type = [("float_input", FloatTensorType([None, 4]))]
onnx_model_rf = convert_sklearn(rf_clf, initial_types = initial_type)
onnx_model_gb = convert_sklearn(gb_clf, initial_types = initial_type)

# save model onnx
with open("model_artifacts/model_v1.onnx", "wb") as f:
    f.write(onnx_model_rf.SerializeToString())

with open("model_artifacts/model_v2.onnx", "wb") as f:
    f.write(onnx_model_gb.SerializeToString())