from django.shortcuts import render
from . import ml_predict

def home(request):
    return render(request, 'index.html')

def result(request):
    user_input = request.POST
    prediction = ml_predict.prediction_model(user_input)
    return render(request, 'result.html', {'prediction':prediction})
