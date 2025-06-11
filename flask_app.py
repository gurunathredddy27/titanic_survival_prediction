from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('titanic_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

  
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs 
        age = float(request.form['Age'])
        # if fare >= 1000 or 0:
        #     return 'Enter valid fare'
        # else:
        fare = float(request.form['Fare'])
        sex_male = int(request.form['Sex_male'])
        pclass = int(request.form['Pclass'])
        embarked_q = int(request.form['Embarked_Q'])
        embarked_s = int(request.form['Embarked_S'])

        final_features = np.array([[age, fare, pclass, sex_male, embarked_q, embarked_s]])
        prediction = model.predict(final_features)

        result = "Survived" if prediction[0] == 1 else "Did NOT survive"
    except Exception as e:
        result = f"Error: {e}"

    return render_template('index.html', prediction_text=f'Prediction: {result}')


if __name__ == "__main__":
    app.run(debug=True)