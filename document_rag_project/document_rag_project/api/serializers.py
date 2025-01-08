from rest_framework import serializers
from .models import Document, DocumentSelection

class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ['id', 'title', 'content', 'embeddings', 'created_at']
        read_only_fields = ['id', 'embeddings', 'created_at']


class DocumentSelectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = DocumentSelection
        fields = ['id', 'user_email', 'document']
