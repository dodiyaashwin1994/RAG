from django.db import models


class Document(models.Model):
    text = models.TextField()
    embedding = models.JSONField()  # Store embeddings as JSON
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Document: {self.text[:50]}"


class QuestionAnswer(models.Model):
    question = models.TextField()
    answer = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Q: {self.question[:50]} | A: {self.answer[:50]}"

