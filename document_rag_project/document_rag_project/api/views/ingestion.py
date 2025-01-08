from .baseapiview import *
from api.models import Document
from transformers import AutoTokenizer, AutoModel

# Load the tokenizer and model
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)


class DocumentIngestionView(APIView):
    def post(self, request, *args, **kwargs):
        try:
            document_text = request.data.get("text")
            if not document_text:
                return Response({"error": "Document text is required."}, status=status.HTTP_400_BAD_REQUEST)

            # Generate embeddings
            inputs = tokenizer(document_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling for sentence embeddings

            embedding_vector = embeddings.squeeze().tolist()

            # Save document to the database
            Document.objects.create(text=document_text, embedding=embedding_vector)
            return Response({"message": "Document ingested successfully."}, status=status.HTTP_201_CREATED)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
