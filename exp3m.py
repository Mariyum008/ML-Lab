import numpy as np
import tkinter as tk
from tkinter import ttk
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
# Name: Mariyum
data = load_diabetes()
X = data.data[:, :2]
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

class MyLinearRegression:
  def __init__(self):
    self.intercept_ = None
    self.coef_ = None
  def fit(self, X_train, y_train):
    X_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    self.coef_ = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y_train
    self.intercept_ = self.coef_[0]
    self.coef_ = self.coef_[1:]
  def predict(self, X_test):
    X_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    return X_b @ np.concatenate(([self.intercept_], self.coef_))
my_model = MyLinearRegression()
my_model.fit(X_train, y_train)
y_pred_my = my_model.predict(X_test)
r2_my = r2_score(y_test, y_pred_my)
intercept_my = my_model.intercept_
coefficients_my = my_model.coef_

class RegressionApp:
  def __init__(self, root):
    self.root = root
    self.root.title("Multiple Linear Regression Mariyum")
    self.create_widgets()

  def create_widgets(self):
    ttk.Label(self.root, text="Input 1:").grid(column=0, row=0, padx=10, pady=10)
    self.feature1_var = tk.DoubleVar()
    self.feature1_entry = ttk.Entry(self.root, textvariable=self.feature1_var)
    self.feature1_entry.grid(column=1, row=0, padx=10, pady=10)
    ttk.Label(self.root, text="Input 2:").grid(column=0, row=1, padx=10, pady=10)
    self.feature2_var = tk.DoubleVar()
    self.feature2_entry = ttk.Entry(self.root, textvariable=self.feature2_var)

    self.feature2_entry.grid(column=1, row=1, padx=10, pady=10)
    self.calc_button = ttk.Button(self.root, text="Calculate", command=self.calculate)
    self.calc_button.grid(column=0, row=4, columnspan=2, pady=10)
    # Result Labels
    ttk.Label(self.root, text="Prediction:").grid(column=0, row=5, padx=10, pady=10)
    self.result_var = tk.StringVar()
    self.result_label = ttk.Label(self.root, textvariable=self.result_var)
    self.result_label.grid(column=1, row=5, padx=10, pady=10)
    # Custom Model Info
    ttk.Label(self.root, text="R² Score:").grid(column=0, row=6, padx=10, pady=10)
    self.r2_var = tk.StringVar(value=f"{r2_my:.4f}")
    self.r2_label = ttk.Label(self.root, textvariable=self.r2_var)
    self.r2_label.grid(column=1, row=6, padx=10, pady=10)
    ttk.Label(self.root, text="Intercept:").grid(column=0, row=7, padx=10, pady=10)
    self.intercept_var = tk.StringVar(value=f"{intercept_my:.4f}")
    self.intercept_label = ttk.Label(self.root, textvariable=self.intercept_var)
    self.intercept_label.grid(column=1, row=7, padx=10, pady=10)
    ttk.Label(self.root, text="Coefficients:").grid(column=0, row=8, padx=10, pady=10)
    self.coef_var = tk.StringVar(value=str(coefficients_my))
    self.coef_label = ttk.Label(self.root, textvariable=self.coef_var)
    self.coef_label.grid(column=1, row=8, padx=10, pady=10)
  def calculate(self):
    feature1 = self.feature1_var.get()
    feature2 = self.feature2_var.get()
    features = np.array([[feature1, feature2]])
    prediction = my_model.predict(features)[0]
    self.result_var.set(f"{prediction:.2f}")

root = tk.Tk()
app = RegressionApp(root)
root.mainloop()

import numpy as np
import tkinter as tk
from tkinter import ttk
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
# Name: Mariyum
data = load_diabetes()
X = data.data[:, :2]
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

class MyLinearRegression:
  def __init__(self):
    self.intercept_ = None
    self.coef_ = None
  def fit(self, X_train, y_train):
    X_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    self.coef_ = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y_train
    self.intercept_ = self.coef_[0]
    self.coef_ = self.coef_[1:]
  def predict(self, X_test):
    X_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    return X_b @ np.concatenate(([self.intercept_], self.coef_))
my_model = MyLinearRegression()
my_model.fit(X_train, y_train)
y_pred_my = my_model.predict(X_test)
r2_my = r2_score(y_test, y_pred_my)
intercept_my = my_model.intercept_
coefficients_my = my_model.coef_

class RegressionApp:
  def __init__(self, root):
    self.root = root
    self.root.title("Multiple Linear Regression Mariyum")
    self.create_widgets()

  def create_widgets(self):
    ttk.Label(self.root, text="Input 1:").grid(column=0, row=0, padx=10, pady=10)
    self.feature1_var = tk.DoubleVar()
    self.feature1_entry = ttk.Entry(self.root, textvariable=self.feature1_var)
    self.feature1_entry.grid(column=1, row=0, padx=10, pady=10)
    ttk.Label(self.root, text="Input 2:").grid(column=0, row=1, padx=10, pady=10)
    self.feature2_var = tk.DoubleVar()
    self.feature2_entry = ttk.Entry(self.root, textvariable=self.feature2_var)

    self.feature2_entry.grid(column=1, row=1, padx=10, pady=10)
    self.calc_button = ttk.Button(self.root, text="Calculate", command=self.calculate)
    self.calc_button.grid(column=0, row=4, columnspan=2, pady=10)
    # Result Labels
    ttk.Label(self.root, text="Prediction:").grid(column=0, row=5, padx=10, pady=10)
    self.result_var = tk.StringVar()
    self.result_label = ttk.Label(self.root, textvariable=self.result_var)
    self.result_label.grid(column=1, row=5, padx=10, pady=10)
    # Custom Model Info
    ttk.Label(self.root, text="R² Score:").grid(column=0, row=6, padx=10, pady=10)
    self.r2_var = tk.StringVar(value=f"{r2_my:.4f}")
    self.r2_label = ttk.Label(self.root, textvariable=self.r2_var)
    self.r2_label.grid(column=1, row=6, padx=10, pady=10)
    ttk.Label(self.root, text="Intercept:").grid(column=0, row=7, padx=10, pady=10)
    self.intercept_var = tk.StringVar(value=f"{intercept_my:.4f}")
    self.intercept_label = ttk.Label(self.root, textvariable=self.intercept_var)
    self.intercept_label.grid(column=1, row=7, padx=10, pady=10)
    ttk.Label(self.root, text="Coefficients:").grid(column=0, row=8, padx=10, pady=10)
    self.coef_var = tk.StringVar(value=str(coefficients_my))
    self.coef_label = ttk.Label(self.root, textvariable=self.coef_var)
    self.coef_label.grid(column=1, row=8, padx=10, pady=10)
  def calculate(self):
    feature1 = self.feature1_var.get()
    feature2 = self.feature2_var.get()
    features = np.array([[feature1, feature2]])
    prediction = my_model.predict(features)[0]
    self.result_var.set(f"{prediction:.2f}")

root = tk.Tk()
app = RegressionApp(root)
root.mainloop()