import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv("emails.csv")

# Separate input and output
X = data["text"]
y = data["label"]

# Convert text into numbers
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vectorized, y)

# Test with user input
while True:
    msg = input("Enter email: ")
    msg_vector = vectorizer.transform([msg])
    prediction = model.predict(msg_vector)
    print("Prediction:", prediction[0])