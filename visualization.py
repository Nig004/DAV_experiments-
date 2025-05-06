import matplotlib.pyplot as plt

# Sample labels from a spam filter
labels = ["spam", "ham", "spam", "ham", "spam", "ham", "ham", "spam"]

# Count labels
from collections import Counter
label_counts = Counter(labels)

# Separate data
categories = list(label_counts.keys())
counts = list(label_counts.values())

# Bar plot
plt.bar(categories, counts, color=['red', 'green'])
plt.title("Spam vs Ham Messages")
plt.xlabel("Category")
plt.ylabel("Number of Messages")
plt.show()
