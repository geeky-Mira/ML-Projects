from flask import Flask, render_template, request
import pandas as pd
import pickle

app= Flask(__name__)
data = pd.read_csv('Indian_Metropolitan_house_price_data.csv')
pipe = pickle.load(open('RandomForestModel.pkl','rb'))

@app.route('/')
def index():
    states = sorted(data['State'].unique())
    
    return render_template('index.html',states=states)
    

@app.route('/predict',methods=['POST'])
def predict():
    bhk =  request.form.get('BHK')
    sqft =  request.form.get('total_sqft')
    state = request.form.get('state')
     #print(city_tier,bhk,sqft)
    
    input = pd.DataFrame([[bhk,sqft,state,]],columns=['BHK','SQUARE_FT','State'])
    prediction = pipe.predict(input)[0] * 100000
    return str(prediction)
if __name__=="__main__":
    app.run(debug=True,port=5001)