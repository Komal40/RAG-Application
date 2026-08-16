from typing import List
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.models.document import Document, DocumentChunk
from app.services.embedding_service import get_embedding_provider
from app.services.llm_service import LLMService

class RAGService:
    def __init__(self):
        self.embedding_provider = get_embedding_provider()
        self.llm_service = LLMService()

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Splits raw text into overlapping chunks to maintain context continuity across boundaries.
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start += (chunk_size - overlap)
            
        return [c for c in chunks if c]

    def process_and_store_document(self, db: Session, filename: str, full_text: str):
        """
        Processes uploaded document: creates record, generates chunks, computes vector embeddings,
        and performs bulk insert into pgvector.
        """
        # 1. Document entry in DB
        doc = Document(filename=filename)
        db.add(doc)
        db.commit()
        db.refresh(doc)

        # 2. Text Chunking
        raw_chunks = self.chunk_text(full_text)

        # 3. Batch Embedding Generation
        embeddings = self.embedding_provider.get_embeddings(raw_chunks)

        # 4. Save Chunks + Vectors to pgvector
        chunk_objects = []
        for index, (chunk_text, vector) in enumerate(zip(raw_chunks, embeddings)):
            chunk_obj = DocumentChunk(
                document_id=doc.id,
                chunk_index=index,
                content=chunk_text,
                embedding=vector
            )
            chunk_objects.append(chunk_obj)

        db.bulk_save_objects(chunk_objects)
        db.commit()

        return {
            "document_id": doc.id,
            "filename": filename,
            "total_chunks": len(chunk_objects)
        }

    def search_similar_chunks(self, db: Session, query: str, top_k: int = 4, document_id: int = None):
        """
        Executes Vector Cosine Similarity Search using pgvector operator (<=>)
        """
        query_embedding = self.embedding_provider.get_embedding(query)

        # Cosine Distance Query using pgvector
        stmt = select(
            DocumentChunk,
            DocumentChunk.embedding.cosine_distance(query_embedding).label("distance")
        )

        if document_id:
            stmt = stmt.where(DocumentChunk.document_id == document_id)

        stmt = stmt.order_by("distance").limit(top_k)
        results = db.execute(stmt).all()

        search_results = []
        for chunk, distance in results:
            search_results.append({
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "similarity_score": round(1 - distance, 4)
            })

        return search_results

    def generate_answer_from_docs(self, db: Session, query: str, top_k: int = 4, document_id: int = None):
        """
        Full RAG Pipeline: Retrieves relevant vector chunks and generates grounded LLM answer.
        """
        # 1. Retrieve top-k relevant chunks
        raw_results = self.search_similar_chunks(db=db, query=query, top_k=top_k, document_id=document_id)
        
        if not raw_results:
            return {
                "answer": "No relevant context found in the uploaded documents.",
                "sources": []
            }

        contexts = [item["content"] for item in raw_results]

        # 2. Call Gemini LLM Service with retrieved contexts
        llm_answer = self.llm_service.generate_rag_response(query=query, context_chunks=contexts)

        return {
            "answer": llm_answer,
            "sources": contexts
        }