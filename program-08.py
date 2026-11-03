
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Dataset
reviews = [
    "I loved this movie, it was fantastic",
    "The product is excellent and very useful",
    "Amazing experience, I really enjoyed it",
    "The service was wonderful",
    "Very happy with my purchase",
    "This product is terrible",
    "I hated this movie",
    "Very disappointing experience",
    "The product broke after one day",
    "Poor quality and bad service"
]

# 2. Labels: 1 = Positive, 0 = Negative
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    reviews,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# 4. Convert text into TF-IDF features
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# 6. Predict test data
y_pred = model.predict(X_test_tfidf)

# 7. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy * 100, "%")

# 8. Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 9. Test new reviews
new_reviews = [
    "The movie was amazing and I loved it",
    "The product was awful and disappointing",
    "I am very happy with this purchase"
]

new_reviews_tfidf = vectorizer.transform(new_reviews)
predictions = model.predict(new_reviews_tfidf)

# 10. Display results
for review, prediction in zip(new_reviews, predictions):
    sentiment = "Positive" if prediction == 1 else "Negative"

    print("\nReview:", review)
    print("Sentiment:", sentiment)
