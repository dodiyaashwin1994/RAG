from .baseapiview import *
from api.models import Document, QuestionAnswer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
import torch

# Load Q&A model and tokenizer globally
QNA_MODEL_NAME = "facebook/bart-large-cnn"
qna_tokenizer = AutoTokenizer.from_pretrained(QNA_MODEL_NAME)
qna_model = AutoModelForSeq2SeqLM.from_pretrained(QNA_MODEL_NAME)

EMBEDDING_MODEL_NAME = "bert-base-uncased"  # Model for embeddings
embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)


class QnAView(APIView):
    def post(self, request, *args, **kwargs):
        try:
            question = request.data.get("question")
            if not question:
                return Response({"error": "Question is required."}, status=status.HTTP_400_BAD_REQUEST)

            # Retrieve all embeddings from the database
            documents = Document.objects.all()
            if not documents:
                return Response({"error": "No documents available for Q&A."}, status=status.HTTP_404_NOT_FOUND)

            # Load embeddings and texts
            embeddings = torch.tensor([doc.embedding for doc in documents])
            texts = [doc.text for doc in documents]

            if len(texts) == 0 or len(embeddings) == 0:
                return Response({"error": "No documents with valid embeddings found."}, status=status.HTTP_404_NOT_FOUND)

            # Encode the question
            inputs = qna_tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                # Convert to float for mean operation
                question_embedding = inputs["input_ids"].float().mean(dim=1)


            # Compute cosine similarity between question and document embeddings
            similarity_scores = torch.nn.functional.cosine_similarity(question_embedding, embeddings)

            if similarity_scores.numel() == 0:
                return Response({"error": "No similarity scores could be computed."}, status=status.HTTP_404_NOT_FOUND)

            # Get the most relevant document
            top_doc_index = torch.argmax(similarity_scores).item()

            if top_doc_index >= len(texts):
                return Response({"error": "No relevant document found for the given question."}, status=status.HTTP_404_NOT_FOUND)

            relevant_text = texts[top_doc_index]

            # Generate the answer using the Q&A model
            qna_inputs = qna_tokenizer.encode_plus(
                f"question: {question} context: {relevant_text}",
                return_tensors="pt",
                max_length=1024,
                truncation=True,
            )
            summary_ids = qna_model.generate(qna_inputs["input_ids"], max_length=200, num_beams=4, early_stopping=True)
            answer = qna_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            # Save the question and answer
            QuestionAnswer.objects.create(question=question, answer=answer)

            return Response({"question": question, "answer": answer}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
