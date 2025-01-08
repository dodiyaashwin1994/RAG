from django.urls import path
from .views import DocumentIngestionView, QnAView

urlpatterns = [
    path("ingest/", DocumentIngestionView.as_view(), name="document-ingest"),
    path("qna/", QnAView.as_view(), name="qna"),
]
