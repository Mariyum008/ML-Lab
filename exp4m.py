import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("./titanic.csv")
pd.read_csv("./titanic.csv")
df.head()

df.isnull().sum()

df['age'].fillna(value=df['age'].mean(), inplace=True)
df['fare'].fillna(value=df['fare'].mean(), inplace=True)
df['embarked'].fillna(value=df['embarked'].mode()[0], inplace=True)
df.drop(labels = ['cabin', 'name', 'ticket'], axis= 1, inplace=True)
df.head()

df = pd.get_dummies(df, columns=['sex', 'embarked'], drop_first = True)
df.head()

from sklearn.model_selection import train_test_split
X = df.drop('survived', axis=1)
y=df['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression (max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print (f"Accuracy: {accuracy_score (y_test, y_pred)}")
print (f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")
print (f"Classification Report: \n{classification_report (y_test, y_pred)}")

import tkinter as tk
from tkinter import ttk
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
root = tk.Tk()
root.title("Logistic Regression Results MARIYUM")
report = classification_report (y_test, model.predict(X_test), output_dict=True)
report_text = pd.DataFrame (report).transpose().round(2).to_string()
text = tk. Text (root, height=10, width=60)
text.insert(tk. END, report_text)
text.pack()
gender_label = ttk. Label(root, text="Select Gender: ")
gender_label.pack(pady=5)
gender = ttk.Combobox (root, values=["male", "female"])
gender.pack(pady=5)
pclass_label = ttk. Label(root, text="Select Pclass:")
pclass_label.pack(pady=5)
pclass = ttk.Combobox (root, values=[1, 2, 3])
pclass.pack(pady=5)
def show_survivors():
  filtered = data[(data['sex'] == gender.get()) & (data['pclass'] == int(pclass.get())) & (data['survived'] == 1)]
  result_text = f"Survivors: {len(filtered)}"
  result_label.config(text=result_text)
button = ttk.Button(root, text="Show Survivors", command=show_survivors)
button.pack(pady=10)
result_label = ttk. Label(root, text="")
result_label.pack(pady=5)
root.mainloop()