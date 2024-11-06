from flask import Flask, render_template, request
import pickle
import sqlite3

app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
model = None
tfidf_vectorizer = None

try:
    with open(r"C:\Users\radhi\OneDrive\Desktop\myapp\model.pkl", 'rb') as model_file:
        model = pickle.load(model_file)  # Load the model correctly using load
    with open(r"C:\Users\radhi\OneDrive\Desktop\myapp\tfidf_vectorizer.pkl", 'rb') as vec_file:
        tfidf_vectorizer = pickle.load(vec_file)  # Load the vectorizer correctly
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")

# Initialize SQLite database
def init_db():
    with sqlite3.connect('database.db') as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS reviews
                     (id INTEGER PRIMARY KEY, review TEXT, prediction TEXT)''')
        conn.commit()

init_db()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']

    # Check if model and vectorizer are loaded
    if model is None or tfidf_vectorizer is None:
        return "Model or vectorizer not loaded. Please check the server logs.", 500

    # Preprocess review: vectorize the input
    try:
        review_vectorized = tfidf_vectorizer.transform([review])  # Vectorize the input review
        prediction = model.predict(review_vectorized)[0]  # Make the prediction
    except Exception as e:
        return f"Error during prediction: {e}", 500

    # Save to database
    with sqlite3.connect('database.db') as conn:
        c = conn.cursor()
        c.execute('INSERT INTO reviews (review, prediction) VALUES (?, ?)', (review, prediction))
        conn.commit()

    return render_template('results.html', review=review, prediction=prediction)

@app.route('/admin')
def admin():
    with sqlite3.connect('database.db') as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM reviews')
        reviews = c.fetchall()
    return render_template('admin.html', reviews=reviews)

if __name__ == '__main__':
    app.run(debug=True)
