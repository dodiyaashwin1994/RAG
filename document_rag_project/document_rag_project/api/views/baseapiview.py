from rest_framework.views import APIView
from rest_framework import status
from rest_framework.response import Response
from django.conf import Settings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class BaseAPIView(APIView):
    pass