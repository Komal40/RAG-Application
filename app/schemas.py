from pydantic import BaseModel, Field
from datetime import datetime

class UploadResponse(BaseModel):
    status: str
    document_id: int
    filename: str
    total_chunks_created: int
    preview_chunk: str

    class Config:
        from_attributes = True


class SearchQueryRequest(BaseModel):
    query: str = Field(..., example="Summarize the key points in the document")



class AskQuestionRequest(BaseModel):
    query: str = Field(..., example="What skills and professional experience are mentioned in the document?")
