# Step 1: Import libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Step 2: Sample data
texts = ["Win money now!", "Hello friend, how are you?", "Free offer just for you", "Let's catch up tomorrow"]
labels = ["spam", "ham", "spam", "ham"]

# Step 3: Convert text to features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25)

# Step 5: Train Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 6: Test with a new message
sample = vectorizer.transform(["Free vacation offer"])
print("Prediction:", model.predict(sample)[0])
