from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle

# load the model from disk
#loaded_model=pickle.load(open('XGBoostregressionmodel.pkl', 'rb'))  #loading the pickle file
#app = Flask(__name__)  #starting the flask

#loaded_model= pickle.load(open('random_forest_regression_model.pkl', 'rb'))
#app = Flask(__name__)

#loaded_model= pickle.load(open('ridge_regression_model.pkl', 'rb'))
#app = Flask(__name__)

loaded_model= pickle.load(open('decision_regression_model.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')  #default home page,backslash
#defining 2 functions, home and predict 
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])  #POST is like, you click the button and prediction works
def predict():
    df=pd.read_csv('real_2018.csv') #reading the file
    my_prediction=loaded_model.predict(df.iloc[:,:-1].values) #removing the last col as it is DV, .values coz the OP should come as a array(during training it was converted into array)
    my_prediction=my_prediction.tolist()
    return render_template('result.html',prediction = my_prediction)  #it goes to result.html and fectches the result


if __name__ == '__main__':
	app.run(debug=True)
    
    
    