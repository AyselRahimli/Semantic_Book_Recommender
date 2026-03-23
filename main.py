import os
import gradio as gr
import pandas as pd
import numpy as np

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


books = pd.read_csv("books_sentimental.csv")

books["thumbnail"] = books["thumbnail"].fillna("")
books["large_thumbnail"] = np.where(
    books["thumbnail"].str.strip() == "",
    "not_found.jpg",
    books["thumbnail"] + "&fife=w800"
)

raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=0
)

documents = text_splitter.split_documents(raw_documents)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

persist_dir = "chroma_books_db"

if os.path.exists(persist_dir):
    db_books = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model
    )
else:
    db_books = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_dir
    )


def retrieve_semantic_recommendations(
    query: str,
    category: str = "All",
    tone: str = "All",
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)

    books_list = []
    for rec in recs:
        parts = rec.page_content.strip('"').split()
        if len(parts) > 0 and parts[0].isdigit():
            books_list.append(int(parts[0]))

    book_recs = books[books["isbn13"].isin(books_list)].copy()

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].copy()

    tone_map = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Angry": "anger",
        "Suspenseful": "fear",
        "Sad": "sadness"
    }

    if tone != "All":
        tone_col = tone_map.get(tone)
        if tone_col in book_recs.columns:
            book_recs = book_recs.sort_values(by=tone_col, ascending=False)
        else:
            print(f"Column '{tone_col}' not found. Available columns: {book_recs.columns.tolist()}")

    return book_recs.head(final_top_k)


def format_authors(authors: str) -> str:
    authors = str(authors)
    authors_split = authors.split(";")

    if len(authors_split) == 2:
        return f"{authors_split[0]} and {authors_split[1]}"
    elif len(authors_split) > 2:
        return f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
    else:
        return authors


def recommend_books(query: str, category: str, tone: str):
    try:
        recommendations = retrieve_semantic_recommendations(query, category, tone)

        if recommendations.empty:
            return "### No recommendations found 😔", []

        results = []

        for _, row in recommendations.iterrows():
            description = str(row["description"])
            truncated_description = " ".join(description.split()[:30]) + "..."

            authors_str = format_authors(row["authors"])

            image = row.get("large_thumbnail", "not_found.jpg")
            if pd.isna(image) or str(image).strip() == "":
                image = "not_found.jpg"

            caption = f"{row['title']} by {authors_str}: {truncated_description}"
            results.append((image, caption))

        return "", results

    except Exception as e:
        print("ERROR:", e)
        return f"### Error: {str(e)}", []


categories = ["All"] + sorted(books["simple_categories"].dropna().unique().tolist())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks() as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(
            label="Please enter a description of a book:",
            placeholder="e.g., A story about forgiveness"
        )
        category_dropdown = gr.Dropdown(
            choices=categories,
            label="Select a category:",
            value="All"
        )
        tone_dropdown = gr.Dropdown(
            choices=tones,
            label="Select an emotional tone:",
            value="All"
        )
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output_message = gr.Markdown()
    output_gallery = gr.Gallery(label="Recommended books", columns=8, rows=2)

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=[output_message, output_gallery]
    )

if __name__ == "__main__":
    dashboard.launch()
