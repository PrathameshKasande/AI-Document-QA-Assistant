from pypdf import PdfReader
import os


def load_multiple_pdfs(folder_path):

    text = ""

    for file_name in os.listdir(folder_path):

        if file_name.endswith(".pdf"):

            pdf_path = os.path.join(
                folder_path,
                file_name
            )

            pdf_reader = PdfReader(pdf_path)

            for page in pdf_reader.pages:

                page_text = page.extract_text()

                if page_text:

                    text += page_text

    return text