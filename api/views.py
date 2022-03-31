from django.shortcuts import render

# Create your views here.
import numpy as np
import pandas as pd
from .apps import ApiConfig
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view
import joblib
import json
import sys
import os

@api_view(["POST"])
def post(request):
        data = request.data
        height = data['Height']
        gender = data['Gender']
        if gender == 'Male':
            gender = 0
        elif gender == 'Female':
            gender = 1
        else:
            return Response("Gender field is invalid", status=400)
        lin_reg_model = ApiConfig.model
        weight_predicted = lin_reg_model.predict([[gender, height]])[0][0]
        weight_predicted = np.round(weight_predicted, 1)
        response_dict = {"Predicted Weight (kg)": weight_predicted}
        return Response(response_dict, status=200)