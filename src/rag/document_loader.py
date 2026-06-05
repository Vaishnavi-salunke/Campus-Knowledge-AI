from pathlib import Path


def load_documents(data_folder="data"):
    documents = []

    data_path = Path(data_folder)

    if not data_path.exists():
        return documents

    for file in data_path.glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            documents.append(
                {
                    "filename": file.name,
                    "content": f.read()
                }
            )

    return documents