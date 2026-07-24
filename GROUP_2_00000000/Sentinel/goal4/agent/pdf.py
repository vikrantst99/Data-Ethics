"""
Read plain text out of a PDF so a proposal can be uploaded instead of typed.

extract_text() accepts a file path, raw bytes or a file-like object (e.g. a
Streamlit upload). It tries pdfplumber first and falls back to pypdf. If neither
is installed, it raises a clear error the UI can show.
"""

import io


def _as_stream(source):
    # Normalise path / bytes / file-like into something pdf libraries can open
    if isinstance(source, (bytes, bytearray)):
        return io.BytesIO(source)
    if hasattr(source, "read"):
        data = source.read()
        return io.BytesIO(data if isinstance(data, (bytes, bytearray))
                          else data.encode("utf-8"))
    return source


def extract_text(source):
    stream = _as_stream(source)
    # Preferred: pdfplumber (good at layout-aware text)
    try:
        import pdfplumber
        with pdfplumber.open(stream) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        text = "\n".join(pages).strip()
        if text:
            return text
    except Exception:
        pass

    # Fallback: pypdf
    try:
        if hasattr(stream, "seek"):
            stream.seek(0)
        from pypdf import PdfReader
        reader = PdfReader(stream)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()
    except Exception as error:
        raise RuntimeError(
            "Could not read the PDF. Install 'pdfplumber' or 'pypdf'. "
            "Original error: " + str(error))
