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
from spam import model
@api_view(["POST"])
def post(request):
        data = request.data
        SMS = data['SMS']
        model_sms = model.Spam_model(SMS)
        response = model_sms.predict_from_naive_bayes_model()
        return Response(response, status=200)