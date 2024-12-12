from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
def home(request):
    return render(request, 'home.html')
def predict(request):
    return render(request, 'predict.html')
def result(request):
    data = pd.read_csv(r"F:\Diabetes_Project\diabetes.csv")
    x = data.drop('Outcome', axis=1)
    y = data['Outcome']
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
    loaded_model = joblib.load('svm_model.joblib')
    # clf = svm.SVC(kernel='linear', C=1.0)
    # clf.fit(x_train, y_train)
    #model = LogisticRegression()
    #model.fit(x_train, y_train)
    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])
    y_pred = loaded_model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])
    #pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])
    #pred = clf.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])
    result3 = y_pred
    result1 = ""
    if result3 == [0]:
        result1 = "Positive"
    else:
        result1 = "Negative"
    return render(request, "predict.html", {"result2": result1})
