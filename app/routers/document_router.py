from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from pypdf import PdfReader
import io
from app.schemas import SearchQueryRequest,AskQuestionRequest
from app.database import get_db
from app.services.rag_service import RAGService

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])
rag_service = RAGService()

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        pdf_bytes = await file.read()
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        extracted_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"

        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="Empty PDF file")

        result = rag_service.process_and_store_document(
            db=db, 
            filename=file.filename, 
            full_text=extracted_text
        )

        return {"status": "Success", "data": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/search")
def search_documents(
    request: SearchQueryRequest, 
    top_k: int = 4, 
    document_id: int | None = None,
    db: Session = Depends(get_db)
):
    """
    User-facing search endpoint. 
    Accepts only 'query' in body. Retrieves top matching document chunks.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    raw_results = rag_service.search_similar_chunks(
        db=db,
        query=request.query,
        top_k=top_k,
        document_id=document_id
    )

    # Simplified user-facing response format
    simplified_results = [item["content"] for item in raw_results]

    return {
        "query": request.query,
        "results": simplified_results
    }




@router.post("/ask")
def ask_question(
    request: AskQuestionRequest,
    top_k: int = 4,
    db: Session = Depends(get_db)
):
    """
    RAG Endpoint: Retrieves relevant document chunks and generates a grounded LLM answer using Gemini API.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    result = rag_service.generate_answer_from_docs(
        db=db,
        query=request.query,
        top_k=top_k
    )

    return {
        "query": request.query,
        "answer": result["answer"],
        "sources": result["sources"]
    }