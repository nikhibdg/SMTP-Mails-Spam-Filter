from flask import Flask, render_template,request,flash
from forms import ContactForm, SignupForm
from flask_mail import Message, Mail
#from intro_to_flask import app
#from models import db
from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np

# import HashingVectorizer from local dir
from vectorizer import vect

mail = Mail()
app = Flask(__name__)
app.secret_key = 'development key'

app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 465
app.config["MAIL_USE_SSL"] = True
app.config["MAIL_USERNAME"] = '***@***.com'
app.config["MAIL_PASSWORD"] = '****'
mail.init_app(app)

######## Preparing the Classifier
cur_dir = os.path.dirname(__file__)

clf = pickle.load(open( "pkl_objects/classifier.pkl", "rb" ))
db = os.path.join(cur_dir, 'reviews.sqlite')

def classify(document):
    label = {0: 'Ham', 1: 'Spam'}
    X = vect.transform([document])
    y = clf.predict(X)[0]
		#print y
    proba = np.max(clf.predict_proba(X))
    return label[y], proba

@app.route('/')
def home():
  return render_template('home.html')

@app.route('/about')
def about():
  return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
  form = ContactForm()

  if request.method == 'POST':
    if form.validate() == False:
      flash('All fields are required.')
      return render_template('contact.html', form=form)
    else:
        reviews = request.form['message']
        y,prob = classify(reviews)
        print(y,prob)
        if (y=='Spam'):
            flash('Message is spam')
            return render_template('spam.html',form=form)
        else:
            msg = Message(form.subject.data, sender='nikhilbdg@gmail', recipients=['nikhilbdg@gmail.com'])
            msg.body = """
            From:%s:%s;
            %s
            """%(form.name.data,form.email.data,form.message.data)
            mail.send(msg)
            return render_template('contact.html', success=True)

 
  else:
    return render_template('contact.html', form=form)

	

@app.route('/testdb')
def testdb():
  if db.session.query("1").from_statement("SELECT 1").all():
    return 'It works.'
  else:
    return 'Something is broken.'

@app.route('/signup', methods=['GET', 'POST'])
def signup():
  form = SignupForm()
   
  if request.method == 'POST':
    if form.validate() == False:
      return render_template('signup.html', form=form)
    else:  
      return "[1] Create a new user [2] sign in the user [3] redirect to the user's profile"
   
  elif request.method == 'GET':
    return render_template('signup.html', form=form)

if __name__ == '__main__':
  app.run(debug=True)
