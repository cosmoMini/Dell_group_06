from flask import Flask, flash, render_template, request, session
import os
from rest_api import callRestApi
from rest_api import makecalc1

app = Flask(__name__)


@app.route('/')
def hello_world():
    callRestApi()
    return render_template ('index.html')
    #if not session.get('logged_in'):
     #   return render_template('login.html')
    #else:
     #   return render_template('index.html')

@app.route('/api', methods=['POST'])
def callRest():
    return makecalc1()

@app.route('/dashboard', methods=['POST'])
def do_admin_login():
    if request.form['password'] == 'password' and request.form['username'] == 'admin':
        session['logged_in'] = True
    else:
        flash('wrong password!')
    return render_template('track_your_customer.html');

@app.route('/complaint', methods=['POST'])
def do_complaint():
    return render_template('complaint.html');

if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(debug=False)