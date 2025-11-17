
import requests 

url = 'http://localhost:9696/predict'


patient = {
  "Age": 49,
  "Sex": "F",
  "ChestPainType": "NAP",
  "RestingBP": 160,
  "Cholesterol": 180,
  "FastingBS": 0,
  "RestingECG": "Normal",
  "MaxHR": 156,
  "ExerciseAngina": "N",
  "Oldpeak": 1,
  "ST_Slope": "Flat"
}

response = requests.post(url, json=patient)
predictions = response.json() 

print(predictions)