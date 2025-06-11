import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('titanic.csv')

# One-hot encoding
df['Sex_male'] = (df['Sex'] == 'male').astype(int)
df['Embarked_Q'] = (df['Embarked'] == 'Q').astype(int)
df['Embarked_S'] = (df['Embarked'] == 'S').astype(int)

# Select only reveland features
features = ['Age', 'Fare', 'Pclass', 'Sex_male', 'Embarked_Q', 'Embarked_S']
df = df[features + ['Survived']].dropna()

X = df[features]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Saveing model in pickle
with open('titanic_model.pkl', 'wb') as f:
    pickle.dump(model, f)
