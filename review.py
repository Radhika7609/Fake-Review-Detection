import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load the datasetcsv

dataframe = pd.read_csv(r"C:\Users\radhi\OneDrive\Desktop\PROJECT 2\review1.csv", encoding='ISO-8859-1')


# Print the first few rows to check your data
print(dataframe.head().to_string(encoding='utf-8', errors='ignore'))
 

# Split features and labels
x = dataframe["text_"]  # Assuming 'text_' contains the review text
y = dataframe["label"]   # Assuming 'label' contains the labels

# Check for NaN values and drop them while keeping both x and y aligned
nan_mask = x.isnull() | y.isnull()  # Create a mask for NaNs
x = x[~nan_mask]  # Keep rows where the mask is False
y = y[~nan_mask]  # Keep rows where the mask is False

# Ensure x and y are reset to have consistent indexing
x = x.reset_index(drop=True)
y = y.reset_index(drop=True)

# Now x and y should have the same length
print(f"Cleaned dataset size: {len(x)} reviews and {len(y)} labels")

# Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50)

# Create the TF-IDF vectorizer and fit it to the training data
tfidf_vectorizer = TfidfVectorizer()  # Initialize the vectorizer
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)  # Fit and transform the training data

# Train the model
classifier = SVC()
classifier.fit(x_train_tfidf, y_train)

# Save the model
with open(r"C:\Users\radhi\OneDrive\Desktop\myapp\model.pkl", "wb") as model_file:
    pickle.dump(classifier, model_file)

# Save the TF-IDF vectorizer
with open(r"C:\Users\radhi\OneDrive\Desktop\myapp\tfidf_vectorizer.pkl", "wb") as vec_file:
    pickle.dump(tfidf_vectorizer, vec_file)

print("Model and TF-IDF vectorizer saved successfully.")
