from pypdf import PdfReader
from io import BytesIO

class PDFService:
    @staticmethod
    def extract_text_and_chunk(file_bytes: bytes, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """
        Reads PDF bytes, extracts raw text, and splits into overlapping chunks.
        """
        pdf_file = BytesIO(file_bytes)
        reader = PdfReader(pdf_file)
        
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + " "

        # Clean whitespace
        full_text = " ".join(full_text.split())

        # Overlapping Chunking Algorithm
        chunks = []
        start = 0
        text_length = len(full_text)

        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = full_text[start:end]
            chunks.append(chunk)
            
            # Step forward by (chunk_size - overlap)
            start += (chunk_size - overlap)

        return chunks