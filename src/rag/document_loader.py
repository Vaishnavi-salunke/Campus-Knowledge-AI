import os
from pypdf import PdfReader
import pytesseract
from pdf2image import convert_from_path

# Tesseract Path
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def load_documents(folder_path):

    documents = []

    for filename in os.listdir(folder_path):

        file_path = os.path.join(folder_path, filename)

        # TXT Files
        if filename.lower().endswith(".txt"):

            try:
                with open(file_path, "r", encoding="utf-8") as file:

                    content = file.read()

                    if content.strip():

                        documents.append(
                            {
                                "filename": filename,
                                "content": content
                            }
                        )

                        print(f"TXT Loaded: {filename}")
                        print(f"Characters extracted: {len(content)}")

            except Exception as e:
                print(f"Error reading TXT file {filename}: {e}")

        # PDF Files
        elif filename.lower().endswith(".pdf"):

            content = ""

            try:

                reader = PdfReader(file_path)

                for page in reader.pages:

                    text = page.extract_text()

                    if text:
                        content += text + "\n"

            except Exception as e:
                print(f"Error reading PDF {filename}: {e}")

            # OCR Fallback
            if len(content.strip()) < 100:

                print(f"OCR Processing: {filename}")

                try:

                    images = convert_from_path(file_path)

                    for image in images:

                        text = pytesseract.image_to_string(image)

                        if text:
                            content += text + "\n"

                except Exception as e:
                    print(f"OCR Failed for {filename}: {e}")

            print(f"\nPDF Loaded: {filename}")
            print(f"Characters extracted: {len(content)}")
            print("-" * 50)

            if content.strip():

                documents.append(
                    {
                        "filename": filename,
                        "content": content
                    }
                )

    print(f"\nTotal Documents Loaded: {len(documents)}")

    return documents