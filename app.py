# Create a simple RAG application from scratch
import os
import requests
from preprocessing import open_and_read_pdf, check_path, split_list
import pandas as pd
from spacy.lang.en import English
from tqdm.auto import tqdm # shows progress bar. pip install tqdm
from sentence_transformers import SentenceTransformer

# Get PDF document path
pdf_path = "data/human-nutrition-text.pdf"

# Download PDF document
url = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"
check_path(pdf_path, url)

pages_and_texts = open_and_read_pdf(pdf_path)

# import random
# print(random.sample(pages_and_texts, k=3))

nlp = English()

# Add a sentencizer pipeline, see https://spacy.io/api/sentencizer
nlp.add_pipe("sentencizer")

for item in tqdm(pages_and_texts):
    item["sentences"] = list(nlp(item["page_text"]).sents)

    # Make sure all the sentences are strings (the default type is spaCy datatype)
    item["sentences"] = [str(sentence) for sentence in item["sentences"]]

    # Count the centences
    item["page_sentence_count_spacy"] = len(item["sentences"])

# print(pages_and_texts[938])
df = pd.DataFrame(pages_and_texts)
print(df.describe().round(2))

# Chunking our sentences together
# Define split size to turn groups of sentences into chunks
num_sentence_chunk_size = 10

# test_list = list(range(25))
# print(split_list(test_list, num_sentence_chunk_size))

# Loop through pages and texts and split sentences into chunks
for item in tqdm(pages_and_texts):
    item["sentence_chunks"] = split_list(item["sentences"], num_sentence_chunk_size)
    item["num_chunks"] = len(item["sentence_chunks"])
# print(pages_and_texts[938])
df = pd.DataFrame(pages_and_texts)
# print(df.describe().round(2))

# Splitting each chunk into its own item
import re
pages_and_chunks = []
for item in tqdm(pages_and_texts):
    for chunk in item["sentence_chunks"]:
        # Join the sentences together into a paragraph-like structure, aka join the list of sentences into one paragraph
        joined_sentence_chunk = "".join(chunk).replace("  ", " ").strip()
        joined_sentence_chunk = re.sub(r"\.([A-Z])", r". \1", joined_sentence_chunk) # ".A" => ". A"

        pages_and_chunks.append({
            "page_number": item["page_number"],
            "sentence_chunk": joined_sentence_chunk,
            "chunk_char_count": len(joined_sentence_chunk),
            "chunk_word_count": len([word for word in joined_sentence_chunk.split(" ")]),
            "chunk_token_count": len(joined_sentence_chunk) / 4
        })

# print(len(pages_and_chunks))
# print(pages_and_chunks[387])
df = pd.DataFrame(pages_and_chunks)
# print(df.describe().round(2))

### Filter chunks of texts for short chunks (they don't contain much useful information)
# Show random chunks with under 30 tokens in length
min_token_count = 30
# for row in df[df["chunk_token_count"] <= min_token_count].sample(5).iterrows():
#    print(f'Chunk token count: {row[1]["chunk_token_count"]} | Text: {row[1]["sentence_chunk"]}')

pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_count].to_dict(orient="records")
# print(pages_and_chunks_over_min_token_len[367])

# Embedding
embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cpu")

"""
# Create a list of sentences
sentences = ["Sentence Transformers provides an easy way to create embeddings.", 
            "Sentences can be embedded one by one or in a list.",
            "I like horses!"]

# Sentence are embedded/encoded by model.encode()
embeddings = embedding_model.encode(sentences)
embeddings_dict = dict(zip(sentences, embeddings))

# See the embeddings
for sentence, embedding in embeddings_dict.items():
    print(f"Sentence: {sentence}")
    print(f"Embedding: {embedding}")
    print("")

print(embeddings[0].shape)
"""

embedding_model.to("cpu")
# embedding_model.to("cuda") # much faster


# Embed each chucnk one by one
for item in tqdm(pages_and_chunks_over_min_token_len):
    item["embedding"] = embedding_model.encode(item["sentence_chunk"])


# Make in batches (faster)
text_chunks = [item["sentence_chunk"] for item in pages_and_chunks_over_min_token_len]

# Embed all texts in batches
text_chunks_embeddings = embedding_model.encode(text_chunks, 
                                                batch_size=32, # you can experiment to find which batch size leads to best performance
                                                convert_to_tensor=True)

# save embeddings to file. If your embedding database is really large (e.g. over 100K-1M samples) you might want to look into using a vector database
text_chunks_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len)
embeddings_df_save_path = "text_chunks_embeddings.csv"
text_chunks_embeddings_df.to_csv(embeddings_df_save_path, index=False)

# import saved file and view
text_chunks_embeddings_df_load = pd.read_csv(embeddings_df_save_path)
print(text_chunks_embeddings_df_load.head())

# RAG goal: Retrieve relevant passages based on a query