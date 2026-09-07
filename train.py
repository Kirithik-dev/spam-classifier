import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# LOADING DATA
df = pd.read_csv("data/spam.csv", encoding="latin-1")
print(f"Loaded {len(df)} data points.")
df = df[["v1","v2"]]
df.columns = ["label", "text"]
X = df["text"]
Y = df["label"]

# SPLITTING DATA
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2 , random_state=46, stratify=Y)

# VECTORIZING DATA
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# TRAINING MODEL
model = LogisticRegression()
model.fit(X_train,Y_train)

# EVALUATING MODEL
predictions = model.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)
print(f"Model accuracy: {accuracy*100:.2f}%")

# SAVING MODEL AND VECTORIZER
with open("model.pkl", "wb") as model_file:
  pickle.dump(model, model_file)
print("Model saved to model.pkl")
