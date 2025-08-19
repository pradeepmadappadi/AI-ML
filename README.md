2. Logistic Regression (Classification)
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Dataset
iris = load_iris()
X, y = iris.data, iris.target

# Model
clf = LogisticRegression(max_iter=200)
clf.fit(X, y)

print("Predicted:", clf.predict([X[0]]))
print("Actual:", y[0])

3. K-Means Clustering (Unsupervised Learning)
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Data
X, _ = make_blobs(n_samples=100, centers=3, random_state=42)

# Model
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

print("Cluster Centers:\n", kmeans.cluster_centers_)
print("Labels:", kmeans.labels_[:10])

4. Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

# Dataset
iris = load_iris()
X, y = iris.data, iris.target

# Model
clf = DecisionTreeClassifier()
clf.fit(X, y)

print("Prediction:", clf.predict([X[0]])
