from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pickle

# Sample data
texts = [
    "Win money now",
    "Free offer available",
    "Hello how are you",
    "Let's meet tomorrow"
]

labels = [1, 1, 0, 0]

# Create model
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train model
model.fit(texts, labels)

# Save model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model Saved Successfully")