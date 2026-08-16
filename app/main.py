from fastapi import FastAPI
from sqlalchemy import text
from app.database import engine, Base
from app.routers import document_router

app = FastAPI(
    title="Enterprise AI RAG Engine",
    description="Production-grade Document Intelligence Platform",
    version="1.0.0"
)

# Enable Pgvector extension & create DB tables on boot
@app.on_event("startup")
def startup_event():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
    Base.metadata.create_all(bind=engine)

# Include Routers
app.include_router(document_router.router)

@app.get("/")
def health_check():
    return {"status": "Healthy", "service": "RAG Backend Engine"}