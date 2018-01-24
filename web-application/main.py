#!/usr/bin/env python
from flask import Flask, render_template, flash, request, jsonify, Markup
import matplotlib.pyplot as plt
import logging, io, base64, os
from wtforms import Form, TextField, TextAreaField, validators, FloatField, IntegerField, StringField, SubmitField
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
 

# default traveler constants
DEFAULT_EMBARKED = 'Southampton'
DEFAULT_FARE = 33
DEFAULT_AGE = 30
DEFAULT_GENDER = 'Female'
DEFAULT_TITLE = 'Mrs.'
DEFAULT_CLASS = 'Second'
DEFAULT_CABIN = 'C'
DEFAULT_SIBSP = 0
DEFAULT_PARCH = 0

# initializing constant vars
average_survival_rate = 0
titanic_df = None
features = None
# logistic regression modeling
lr_model = LogisticRegression()

app = Flask(__name__)

class ReusableForm(Form):
	selected_embarked = TextField('selected_embarked', validators=[validators.required()])
	selected_fare = IntegerField('selected_fare', validators=[validators.required()])
	selected_age = IntegerField('selected_age', validators=[validators.required()])
	selected_gender =TextField('selected_gender', validators=[validators.required()])
	selected_title =TextField('selected_title', validators=[validators.required()])
	selected_class =TextField('selected_class', validators=[validators.required()])
	selected_cabin = TextField('selected_cabin', validators=[validators.required()])
	selected_sibsp =IntegerField('selected_sibsp', validators=[validators.required()])
	selected_parch =IntegerField('selected_parch', validators=[validators.required()])

def prepare_data_for_model(raw_dataframe):
    # dummy all categorical fields
    dataframe_dummy = pd.get_dummies(raw_dataframe)
    # remove the nan colums in dataframe as most are outcome variable and we can't use them
    dataframe_dummy = dataframe_dummy.dropna()
    return (dataframe_dummy)

@app.before_first_request
def startup():
    global titanic_df, average_survival_rate, lr_model, features

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(BASE_DIR, 'titanic3.csv')
    titanic_df = pd.read_csv(src)

    # get title
    titanic_df['title'] = [ln.split()[1] for ln in titanic_df['name'].values]
    titanic_df['title'].value_counts()
    titanic_df['title'] = [title if title in ['Mr.', 'Miss.', 'Mrs.', 'Master.', 'Dr.', 'Rev.'] else 'Unknown'
                           for title in titanic_df['title'].values ]

    # strip first letter from cabin number if there
    titanic_df['cabin'] = titanic_df['cabin'].replace(np.NaN, 'U')
    titanic_df['cabin'] = [ln[0] for ln in titanic_df['cabin'].values]
    titanic_df['cabin'] = titanic_df['cabin'].replace('U', 'Unknown')

    titanic_df['isfemale'] = np.where(titanic_df['sex'] == 'female', 1, 0)

    # drop features not needed for model
    titanic_df = titanic_df[[f for f in list(titanic_df) if f not in ['sex', 'name', 'boat','body', 'ticket', 'home.dest']]]

    # make pclass actual categorical column
    titanic_df['pclass'] = np.where(titanic_df['pclass'] == 1, 'First',
                                    np.where(titanic_df['pclass'] == 2, 'Second', 'Third'))

    # get average survival rate
    average_survival_rate = np.mean(titanic_df['survived']) * 100

    titanic_df['embarked'] = titanic_df['embarked'].replace(np.NaN, 'Unknown')

    # prepare training data
    titanic_ready_df = prepare_data_for_model(titanic_df)

    #from sklearn.metrics import accuracy_score
    features = [feat for feat in list(titanic_ready_df) if feat != 'survived']
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(titanic_ready_df[features], titanic_ready_df[['survived']], test_size=0.5, random_state=42)

    # fit model only once
    lr_model.fit(X_train, y_train.values.ravel())

@app.route("/", methods=['POST', 'GET'])
def submit_new_profile():
    form = ReusableForm(request.form)
    logging.warning(form.errors)
    model_results = ''
    if request.method == 'POST':
        selected_embarked = request.form['selected_embarked']
        selected_fare = request.form['selected_fare']
        selected_age = request.form['selected_age']
        selected_gender = request.form['selected_gender']
        selected_title = request.form['selected_title']
        selected_class = request.form['selected_class']
        selected_cabin = request.form['selected_cabin']
        selected_sibsp = request.form['selected_sibsp']
        selected_parch = request.form['selected_parch']

        # assign new variables to live data for prediction
        x_predict_pclass = selected_class
        x_predict_is_female = 1 if selected_gender == 'Female' else 0
        x_predict_age = int(selected_age)
        x_predict_sibsp = int(selected_sibsp)
        x_predict_parch = int(selected_parch)
        x_predict_fare = int(selected_fare)
        x_predict_cabin = selected_cabin
        x_predict_embarked = selected_embarked[0]
        x_predict_title = selected_title

        titanic_df_tmp = titanic_df.copy()
        titanic_df_tmp = titanic_df_tmp[['pclass', 'age', 'sibsp', 'parch', 'fare', 'cabin', 'embarked', 'title', 'isfemale', 'survived']]

        titanic_fictional_df = pd.DataFrame([[x_predict_pclass,
                                             x_predict_age,
                                             x_predict_sibsp,
                                             x_predict_parch,
                                             x_predict_fare,
                                             x_predict_cabin,
                                             x_predict_embarked,
                                             x_predict_title,
                                             x_predict_is_female,
                                             0]], columns = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'cabin', 'embarked', 'title', 'isfemale', 'survived'])

        titanic_df_tmp = pd.concat([titanic_fictional_df, titanic_df_tmp], ignore_index=True)
        titanic_df_tmp = prepare_data_for_model(titanic_df_tmp)

        Y_pred = lr_model.predict_proba(titanic_df_tmp[features].head(1))
        probability_of_surviving_fictional_character = Y_pred[0][1] * 100

        fig = plt.figure()
        objects = ('Average Survival Rate', 'Fictional Traveler')
        y_pos = np.arange(len(objects))
        performance = [average_survival_rate, probability_of_surviving_fictional_character]

        ax = fig.add_subplot(111)
        colors = ['gray', 'blue']
        plt.bar(y_pos, performance, align='center', color = colors, alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.axhline(average_survival_rate, color="r")
        plt.ylim([0,100])
        plt.ylabel('Survival Probability')
        plt.title('How Did Your Fictional Traveler Do? \n ' + str(round(probability_of_surviving_fictional_character,2)) + '% of Surviving!')
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return render_template('index.html',
        	model_results = model_results,
        	model_plot = Markup('<img src="data:image/png;base64,{}">'.format(plot_url)),
        	selected_embarked = selected_embarked,
        	selected_fare = selected_fare,
        	selected_age = selected_age,
        	selected_gender = selected_gender,
        	selected_title = selected_title,
        	selected_class = selected_class,
        	selected_cabin = selected_cabin,
        	selected_sibsp = selected_sibsp,
        	selected_parch = selected_parch)
    else:
        # set default passenger settings
        return render_template('index.html',
            model_results = '',
            model_plot = '',
            selected_embarked = DEFAULT_EMBARKED,
            selected_fare = DEFAULT_FARE,
            selected_age = DEFAULT_AGE,
            selected_gender = DEFAULT_GENDER,
            selected_title = DEFAULT_TITLE,
            selected_class = DEFAULT_CLASS,
            selected_cabin = DEFAULT_CABIN,
            selected_sibsp = DEFAULT_SIBSP,
            selected_parch = DEFAULT_PARCH)

if __name__=='__main__':
	app.run(debug=False)
