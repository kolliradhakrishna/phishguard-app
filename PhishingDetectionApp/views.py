from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn import svm
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import io
import urllib, base64
from .models import BlockedUrl

global results_data
results_data = {}

X = np.load("model/X.txt.npy")
Y = np.load("model/Y.txt.npy")
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]


with open('model/tfidf.txt', 'rb') as file:
    tfidf = pickle.load(file)
file.close()
X = tfidf.fit_transform(X).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

if os.path.exists('model/svm.txt'):
    with open('model/svm.txt', 'rb') as file:
        svm_cls = pickle.load(file)
    file.close()
else:
    svm_cls = svm.SVC()
    svm_cls.fit(X_train, y_train)
    with open('model/svm.txt', 'wb') as file:
        pickle.dump(svm_cls, file)
    file.close()

if os.path.exists('model/lgbm.txt'):
    with open('model/lgbm.txt', 'rb') as file:
        lgbm_cls = pickle.load(file)
    file.close()
else:
    lgbm_cls = LGBMClassifier()
    lgbm_cls.fit(X_train, y_train)
    with open('model/lgbm.txt', 'wb') as file:
        pickle.dump(lgbm_cls, file)
    file.close()

with open('model/rf.txt', 'rb') as file:
    rf_cls = pickle.load(file)
file.close()    

from matplotlib.figure import Figure

def RunSVM(request):
    if request.method == 'GET':
        global X_train, X_test, y_train, y_test
        
        predict = svm_cls.predict(X_test)
        acc = accuracy_score(y_test,predict)*100
        p = precision_score(y_test,predict,average='macro') * 100
        r = recall_score(y_test,predict,average='macro') * 100
        f = f1_score(y_test,predict,average='macro') * 100
        
        # Local variable instead of global
        current_results = {}
        current_results['SVM'] = {
            'accuracy': acc,
            'precision': p,
            'recall': r,
            'fscore': f
        }
        
        output = ""
        for model_name, metrics in current_results.items():
            output += f'<tr><td><font size="" color="black">{model_name}</td>'
            output += f'<td><font size="" color="black">{metrics["accuracy"]}</td>'
            output += f'<td><font size="" color="black">{metrics["precision"]}</td>'
            output += f'<td><font size="" color="black">{metrics["recall"]}</td>'
            output += f'<td><font size="" color="black">{metrics["fscore"]}</td>'

        LABELS = ['Normal URL','Phishing URL']
        conf_matrix = confusion_matrix(y_test, predict)
        
        # Thread-safe plotting using OO API
        fig = Figure(figsize=(6, 6))
        ax = fig.subplots()
        sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g", ax=ax);
        ax.set_ylim([0,2])
        ax.set_title("SVM Confusion matrix") 
        ax.set_ylabel('True class') 
        ax.set_xlabel('Predicted class') 
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = urllib.parse.quote(string)
        chart_data = uri
        
        context= {'data':output, 'chart_data': chart_data, 'chart_title': 'SVM Confusion Matrix'}
        return render(request, 'ViewOutput.html', context)
    return render(request, 'ViewOutput.html', {})     

def RunLGBM(request):
    if request.method == 'GET':
        global X_train, X_test, y_train, y_test
        
        predict = lgbm_cls.predict(X_test)
        acc = accuracy_score(y_test,predict)*100
        p = precision_score(y_test,predict,average='macro') * 100
        r = recall_score(y_test,predict,average='macro') * 100
        f = f1_score(y_test,predict,average='macro') * 100
        
        # Local variable
        current_results = {}
        current_results['LightGBM'] = {
            'accuracy': acc,
            'precision': p,
            'recall': r,
            'fscore': f
        }
        
        output = ""
        for model_name, metrics in current_results.items():
            output += f'<tr><td><font size="" color="black">{model_name}</td>'
            output += f'<td><font size="" color="black">{metrics["accuracy"]}</td>'
            output += f'<td><font size="" color="black">{metrics["precision"]}</td>'
            output += f'<td><font size="" color="black">{metrics["recall"]}</td>'
            output += f'<td><font size="" color="black">{metrics["fscore"]}</td>'

        LABELS = ['Normal URL','Phishing URL']
        conf_matrix = confusion_matrix(y_test, predict) 
        
        # Thread-safe plotting using OO API
        fig = Figure(figsize=(6, 6))
        ax = fig.subplots()
        sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g", ax=ax);
        ax.set_ylim([0,2])
        ax.set_title("LightGBM Confusion matrix") 
        ax.set_ylabel('True class') 
        ax.set_xlabel('Predicted class') 
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = urllib.parse.quote(string)
        chart_data = uri
        
        context= {'data':output, 'chart_data': chart_data, 'chart_title': 'LightGBM Confusion Matrix'}
        return render(request, 'ViewOutput.html', context)
    return render(request, 'ViewOutput.html', {})    



def getData(arr):
    data = ""
    for i in range(len(arr)):
        arr[i] = arr[i].strip()
        if len(arr[i]) > 0:
            data += arr[i]+" "
    return data.strip()        

def PredictAction(request):
    if request.method == 'POST':
        global rf_cls, tfidf
        url_input = request.POST.get('t1', '')
        test = []
        if not url_input:
            return render(request, 'Predict.html', {'msg': 'Please enter a valid URL.'})
        arr = url_input.split("/")
        if len(arr) > 0:
            data = getData(arr)

            test.append(data)
            test = tfidf.transform(test).toarray()


            predict = rf_cls.predict(test)

            predict = predict[0]
            output = ""
            url= ""
            if predict == 0:
                url = url_input
                output = " Given URL Predicted as Genuine"
            if predict == 1:
                url=url_input
                output = " PHISHING Detected in Given URL"
            context= {'url':url,'msg':output}
            return render(request, 'Predict.html', context)
    return render(request, 'Predict.html', {})




def index(request):
    return render(request, 'index.html', {})

def Predict(request):
    return render(request, 'Predict.html', {})
    
def AdminLogin(request):
    if request.user.is_authenticated:
        return render(request, 'AdminScreen.html', {'data': f'Welcome {request.user.username}'})
    return render(request, 'AdminLogin.html', {})

from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required

def AdminLoginAction(request):
    if request.method == 'POST':
        user = request.POST.get('t1', '')
        password = request.POST.get('t2', '')
        
        user_obj = authenticate(username=user, password=password)
        
        if user_obj is not None:
            login(request, user_obj)
            context = {'data': 'Welcome ' + user}
            return render(request, 'AdminScreen.html', context)
        else:
            context = {'data': 'Invalid Login'}
            return render(request, 'AdminLogin.html', context)
    
    if request.user.is_authenticated:
        return render(request, 'AdminScreen.html', {'data': f'Welcome {request.user.username}'})
        
    return render(request, 'AdminLogin.html', {})

@login_required(login_url='AdminLogin')
def ViewBlockedUrls(request):
    data = BlockedUrl.objects.all().order_by('-blocked_date')
    context = {"data": data}
    return render(request, 'ViewBlockedUrls.html', context)


@login_required(login_url='AdminLogin')
def blockurl(request):
    # Security fix: Ideally should be POST, but keeping GET for now per user requirements, 
    # just adding auth protection.
    if request.method == 'GET':
        url = request.GET.get('url')
        if url:
             BlockedUrl.objects.create(url=url)
             
        alt = "<script>window.alert('Url Blocked Successfully..!!!')</script>"
        data = BlockedUrl.objects.all().order_by('-blocked_date')
        
        context = {"data": data, 'alt': alt}
        return render(request, 'ViewBlockedUrls.html', context)
    
    data = BlockedUrl.objects.all().order_by('-blocked_date')
    return render(request, 'ViewBlockedUrls.html', {"data": data})

# Protect model runs as well
@login_required(login_url='AdminLogin')
def RunSVM_Protected(request):
   return RunSVM(request)

@login_required(login_url='AdminLogin') 
def RunLGBM_Protected(request):
   return RunLGBM(request)






