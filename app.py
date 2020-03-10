import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model_amazon_optimization_pickle.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method=="POST":
        budgetinrupees=int(request.form["budgetinrupees"])
        targetingtype=int(request.form["targetingtype"])
        biddingstrategy=int(request.form["biddingstrategy"])
        impressions=int(request.form["impressions"])
        clicks=int(request.form["clicks"])
        clickthroughrate=float(request.form["clickthroughrate"])
        spendinrs=float(request.form["spendinrs"])
        cpc=float(request.form["cpc"])
        daystotalorders14=int(request.form["14daystotalorders"])
        acosspentinperc=float(request.form["acosspentinperc"])
        daystotalsales14=float(request.form["14daystotalsales"])
        campaign_product=int(request.form["campaign_product"])
        campaign_class=int(request.form["campaign_class"])
        priceofproduct=float(request.form["priceofproduct"])
        totalremainingbalance=float(daystotalsales14)-float(spendinrs)
    int_features = [budgetinrupees,targetingtype,biddingstrategy,impressions,clicks,clickthroughrate,spendinrs,
                    cpc,daystotalorders14,acosspentinperc,daystotalsales14,campaign_product,campaign_class,priceofproduct,totalremainingbalance]
    print("The features are:")
    print(int_features)
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]
    if(output==0):
        return render_template('index.html', prediction_text='The parameter  are optimal and can be made more optimal. However, We are running in profits.')
    else:
        return render_template('index.html', prediction_text='Warning!we are running in loss.Parameters need strict optimizations.')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)