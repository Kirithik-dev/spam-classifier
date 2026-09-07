import sys
import pickle

# 1. Load the saved model and vectorizer
with open("model.pkl", "rb") as model_file:
    model, vectorizer = pickle.load(model_file)

# 2. Get the text from the command line
text = sys.argv[1] 

# 3. Convert the text into TF-IDF numbers (same way training data was converted)
text_vec = vectorizer.transform([text])

# 4. Make a prediction
prediction = model.predict(text_vec)[0]

# 5. Print the prediction
print(f"Text: {text}")
print(f"Prediction: {prediction}")