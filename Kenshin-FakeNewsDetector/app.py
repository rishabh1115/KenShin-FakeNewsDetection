from flask import Flask, request, render_template
import joblib
import re
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]',' ',text) 
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]   
    return ' '.join(tokens)

Vector=joblib.load(open("vectorizer_ngram.pkl","rb"))
model=joblib.load(open("fake_news_model_ngram.pkl","rb"))

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/how')
def how():
    return render_template('how.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method=="POST":
        news = str(request.form['news'])
        cleaned_news= preprocess(news)
        prediction = model.predict(Vector.transform([cleaned_news]))[0]
        label = "Real" if prediction == 1 else "Fake"
        print(prediction)

        return render_template('result.html', prediction_text="The Given News or News Headline is {label}".format(label=label))
    else:
        return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)
