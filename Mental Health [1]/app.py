import pickle
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from flask_mysqldb import MySQL

app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'admin'
app.config['MYSQL_DB'] = 'recommendation'

#Initialize MySQL
mysql = MySQL(app)

# Load the classification model and TF-IDF vectorizer
classification_model = pickle.load(open('models/log_reg_classifier.pkl', 'rb'))
'''
cleaned_df=pd.read_csv("dataset/cleaned_dataset.csv")
cleaned_df.drop(cleaned_df[cleaned_df['cleaned_statement'].isnull()==True].index,axis=0,inplace=True)

# Assume 'labels' is a column indicating the target for each row in combined_df
x = cleaned_df[['cleaned_statement','num_of_characters','num_of_sentences']]  # Text features
y = cleaned_df['status'] # Target variable

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=101)

# Initialize TF-IDF Vectorizer and fit/transform on the 'tokens' column
vectorizer = TfidfVectorizer(max_features=42009)
x_train_tfidf = vectorizer.fit_transform(x_train['cleaned_statement'])
x_test_tfidf = vectorizer.transform(x_test['cleaned_statement'])
'''
tfidf_vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('frontend.html')

@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Get the text input from the form
        input_text = request.form.get('Sentence')

        print(input_text)

        # Transform the text input to the TF-IDF representation
        input_text_transformed = tfidf_vectorizer.transform([input_text])

        # Predict the sentiment label
        prediction = classification_model.predict(input_text_transformed)

        # Fetch the latest recommendations from the database
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM mental_health WHERE objects = %s", (prediction[0],))
        recommendations = cursor.fetchall()
        cursor.close()

        # Return the result to the template
        return render_template('frontend.html', result=prediction[0],recommendations=recommendations)

    else:
        return render_template('frontend.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
