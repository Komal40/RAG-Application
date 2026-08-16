import os
from google import genai
from dotenv import load_dotenv
# import google.generativeai as genai

load_dotenv()

class LLMService:
    def __init__(self, model_name: str = "models/gemini-3.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing in .env file")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate_rag_response(self, query: str, context_chunks: list[str]) -> str:
        """
        Combines retrieved vector context with user query for grounded generation using Gemini.
        """
        combined_context = "\n\n---\n\n".join(context_chunks)

        prompt = f"""
You are an enterprise AI assistant answering questions based strictly on the provided document context.

Rules:
1. Answer using ONLY the facts mentioned in the context below.
2. If the answer cannot be determined from the context, clearly state: 'I cannot find this information in the uploaded document.'
3. Keep the response professional, concise, and structured.

Context:
{combined_context}

Question: {query}
"""

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        return response.text