# 🚢 Titanic Survival Prediction (Flask Web App)

This is a **Machine Learning Web Application** built using **Flask** that predicts whether a passenger would survive the Titanic disaster based on key features. The model is trained using a **Random Forest Classifier** on the Titanic dataset and presented through a simple HTML interface.

---

## Objective

To predict the **survival status** of a passenger using the following input features:

- Age  
- Fare  
- Passenger Class (Pclass)  
- Gender (as binary: male = 1, female = 0)  
- Port of Embarkation (Q, S encoded as binary variables)

---

## Machine Learning Model

- Algorithm: **Random Forest Classifier**
- File: `titanic_model.pkl` (generated via `pickle`)
- Preprocessing:
  - One-hot encoding for `Sex` and `Embarked`
  - Selected features: `Age`, `Fare`, `Pclass`, `Sex_male`, `Embarked_Q`, `Embarked_S`
  - NaN values dropped

---

## Features

- Clean web UI using HTML/CSS
- Prediction handled via Flask backend
- Real-time output: “**Survived**” or “**Did NOT Survive**”
- Input validation with basic error handling

---

## How to Run Locally

### Clone the Repository

```bash
git clone https://github.com/yourusername/titanic-survival-flask.git
cd titanic-survival-flask
