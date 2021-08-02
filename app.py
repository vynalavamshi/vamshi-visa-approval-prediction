import numpy as np 
import pandas as pd
from flask import Flask, request,render_template
import pickle
app=Flask(__name__)
model=pickle.load(open('Visa.pkl','rb'))
@app.route('/')
def home():
    return render_template('homepage.html')
@app.route('/predict')
def predict():
    return render_template('predict.html')
@app.route('/y_predict',methods=['POST'])
def y_predict():

    input_features=[x for x in request.form.values()]
    features_value=[np.array(input_features)]
    features_name=['FULL_TIME_POSITION','PREVAILING_WAGE','YEAR','SOC_N']
    df=pd.DataFrame(features_value,columns=features_name)
    output=model.predict(df)
    output=np.argmax(output)
    print(output)
    
    return render_template('predict.html',prediction_text=output)

if __name__=='__main__':
    app.run(debug=True)
