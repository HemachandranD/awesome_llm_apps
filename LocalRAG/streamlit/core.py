def check_valid_file(file: BytesIO) -> str:
    """Reads an uploaded file and returns a File object"""
    if file.name.lower().endswith(".docx"):
        return "DOCX"
    elif file.name.lower().endswith(".pdf"):
        return "PDF"
    elif file.name.lower().endswith(".txt"):
        return "Text"
    elif file.name.lower().endswith(".md"):
        return "Markdown"
    else:
        raise NotImplementedError(f"File type {file.name.split('.')[-1]} not supported")