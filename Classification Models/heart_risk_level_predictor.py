from flask import Flask,render_template,request
import joblib
import numpy as np

model=joblib.load('heart_risk_prediction.sav')

app=Flask(__name__) #application

@app.route('/')
def index():

	return render_template('patient_details.html')

@app.route('/getresults',methods=['POST'])
def getresults():

	result=request.form 


	name=result['name']
	age=float(result['age'])
	gender=float(result['gender'])
	cp=float(result['cp'])
	trestbps=float(result['trestbps'])
	chol=float(result['chol'])
	fbs=float(result['fbs'])
	restecg=float(result['restecg'])
	thalach=float(result['thalach'])
	exang=float(result['exang'])
	oldpeak=float(result['oldpeak'])
	slope=float(result['slope'])
	ca=float(result['ca'])
	thal=float(result['thal'])
	

	test_data=np.array([age,gender,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]).reshape(1,-1)


	prediction=model.predict(test_data)
	
	resultDict={"name":name,"risk":prediction[0]}

	return render_template('patient_results.html',results=resultDict)

app.run(debug=True)