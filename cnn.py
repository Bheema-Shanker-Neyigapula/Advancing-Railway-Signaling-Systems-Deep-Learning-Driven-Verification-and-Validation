import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import IsolationForest
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Deep learning approach using CNN
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=128, epochs=5, validation_split=0.1)

# Traditional machine learning approach using K-Nearest Neighbors (KNN)
X_train_flatten = X_train.reshape(X_train.shape[0], -1)
X_test_flatten = X_test.reshape(X_test.shape[0], -1)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_flatten, np.argmax(y_train, axis=1))

# Anomaly detection using Isolation Forest
X_train_flatten = X_train_flatten[:10000]  # Reduce training set size for anomaly detection

# Deep learning anomaly detection using Isolation Forest
dl_predictions = model.predict(X_test)
dl_predicted_labels = np.argmax(dl_predictions, axis=1)
dl_if_model = IsolationForest(contamination=0.01)
dl_if_model.fit(dl_predicted_labels.reshape(-1, 1))
dl_if_predictions = dl_if_model.predict(dl_predicted_labels.reshape(-1, 1))
dl_if_precision = precision_score(np.argmax(y_test, axis=1), dl_if_predictions, average='macro')
dl_if_recall = recall_score(np.argmax(y_test, axis=1), dl_if_predictions, average='macro')  # Change average='binary' to average='macro'

# K-Nearest Neighbors anomaly detection using Isolation Forest
knn_predictions = knn_model.predict(X_test_flatten)
knn_if_model = IsolationForest(contamination=0.01)
knn_if_model.fit(knn_predictions.reshape(-1, 1))
knn_if_predictions = knn_if_model.predict(knn_predictions.reshape(-1, 1))
knn_if_precision = precision_score(np.argmax(y_test, axis=1), knn_if_predictions, average='macro')
knn_if_recall = recall_score(np.argmax(y_test, axis=1), knn_if_predictions, average='macro')  # Change average='binary' to average='macro'


# Evaluate the deep learning model for fault diagnosis
dl_predictions = model.predict(X_test)
dl_predicted_labels = np.argmax(dl_predictions, axis=1)
dl_true_labels = np.argmax(y_test, axis=1)
dl_report = classification_report(dl_true_labels, dl_predicted_labels, output_dict=True)

# Traditional machine learning approach using K-Nearest Neighbors (KNN)
knn_predictions = knn_model.predict(X_test_flatten)
knn_true_labels = np.argmax(y_test, axis=1)
knn_report = classification_report(knn_true_labels, knn_predictions, output_dict=True)

# Bar plot comparing the accuracy between the deep learning approach (CNN) and K-Nearest Neighbors (KNN) models
labels = ['Deep Learning (CNN)', 'K-Nearest Neighbors (KNN)']
accuracy_scores = [model.evaluate(X_test, y_test, verbose=0)[1], accuracy_score(knn_true_labels, knn_predictions)]

plt.figure(figsize=(8, 6))
plt.bar(labels, accuracy_scores)
plt.ylim(0, 1)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison: Deep Learning (CNN) vs. K-Nearest Neighbors (KNN)')
#plt.show()

# Plotting the precision and recall rates for anomaly detection
labels = ['Deep Learning (CNN)', 'K-Nearest Neighbors (KNN)']
precision_scores = [dl_if_precision, knn_if_precision]
recall_scores = [dl_if_recall, knn_if_recall]

plt.figure(figsize=(10, 6))
x = np.arange(len(labels))
width = 0.35

plt.bar(x - width / 2, precision_scores, width, label='Precision')
plt.bar(x + width / 2, recall_scores, width, label='Recall')

plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Precision and Recall Scores for Anomaly Detection')
plt.xticks(x, labels)
plt.legend()
#plt.show()

# Plotting the fault diagnosis reports
dl_f1_scores = [dl_report[str(i)]['f1-score'] for i in range(10)]
knn_f1_scores = [knn_report[str(i)]['f1-score'] for i in range(10)]

plt.figure(figsize=(10, 6))
x = np.arange(10)

plt.bar(x - width / 2, dl_f1_scores, width, label='Deep Learning (CNN)')
plt.bar(x + width / 2, knn_f1_scores, width, label='K-Nearest Neighbors (KNN)')

plt.xlabel('Classes')
plt.ylabel('F1-Score')
plt.title('Fault Diagnosis F1-Scores by Class')
plt.xticks(x, [str(i) for i in range(10)])
plt.legend()
plt.show()
