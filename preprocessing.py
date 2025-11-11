import os
import fitz # from PyMuPDF
from tqdm.auto import tqdm # for progress bar. pip install tqdm
import requests

def text_formatter(text: str) -> str:
    """ Performs minor formatting of the text."""
    cleaned_text = text.replace("\n", " ").strip()

    return cleaned_text

def open_and_read_pdf(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages_and_text = []

    for page_num, page in enumerate(tqdm(doc)):
        page_text = page.get_text()
        page_text = text_formatter(page_text)
        pages_and_text.append({"page_number": page_num -41, 
                               "page_char_count": len(page_text),
                               "page_word_count": len(page_text.split(" ")),
                               "page_sentence_count_raw": len(page_text.split(". ")),
                               "page_token_count": len(page_text) / 4, # 1 token = ~ 4 charachters
                               "page_text": page_text})

    return pages_and_text

def check_path(file_path: str, url: str):
    if not os.path.exists(file_path):
        print("[INFO] File does not exist, downloading...")

        # The local path to save the PDF file
        filename = file_path

        # Send a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            
            # Save the PDF file to the local path
            with open(filename, "wb") as f:
                f.write(response.content)
            
            print(f"[INFO] Successfully downloaded and saved as {filename}")
        else:
            print(f"[INFO] Failed to download the PDF file from {url}. Status code: {response.status_code}")

    else:
        print(f"[INFO] File already exists at {file_path}")

def split_list(input_list: list[str], slice_size: int) -> list[list[str]]: 
    """ A function to split lists of texts recurisively into chunk size.
        e.g. [20] -> [10, 10] or [25] -> [10, 10, 5]
    """
    return [input_list[i:i+slice_size] for i in range(0, len(input_list), slice_size)]