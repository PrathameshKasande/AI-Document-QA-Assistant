from src.pdf_loader import load_multiple_pdfs

def test_pdf_loader():

    text = load_multiple_pdfs(
        "data/uploaded_pdfs"
    )

    assert isinstance(text, str)

    assert len(text) > 0