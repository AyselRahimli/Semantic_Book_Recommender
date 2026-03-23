# Semantic Book Recommender System

An end-to-end **AI-powered book recommendation system** that combines **semantic search**, **zero-shot classification**, and **emotion-based filtering** to provide personalized book suggestions.

---

## Project Overview

This project builds a **content-based recommendation system** using modern NLP techniques.
Instead of relying on keywords, it understands the **meaning (semantics)** of book descriptions.

Users can:

* Describe a book they want
* Filter by category (Fiction / Nonfiction)
* Filter by emotional tone (Happy, Sad, Suspenseful, etc.)
* Receive visually rich recommendations with thumbnails


## 📂 Project Pipeline

### 1️⃣ Data Collection

* Dataset: **7k Books with Metadata**
* Fields used:

  * Title
  * Authors
  * Description
  * Rating
  * Categories
  * Thumbnail

---

### 2️⃣ Data Cleaning & Preprocessing

* Removed missing values:

  * `description`
  * `num_pages`
  * `average_rating`
  * `published_year`

* Filtered low-quality data:

  * Removed books with descriptions < 25 words

* Created new features:

  * `title_and_subtitle`
  * `tagged_description` (ISBN + description)

* Handled missing thumbnails:

```python
books["large_thumbnail"] = np.where(
    books["thumbnail"].isna(),
    "cover-not-found.jpg",
    books["thumbnail"]
)
```

---

### 3️⃣ Exploratory Data Analysis (EDA)

* Missing value visualization (heatmaps)
* Correlation analysis (Spearman)
* Description length analysis
* Category distribution

---

### 4️⃣ Zero-Shot Classification

Used Hugging Face model:

```
facebook/bart-large-mnli
```

#### Purpose:

* Convert messy categories → simple labels:

  * Fiction
  * Nonfiction

#### Process:

1. Evaluate model using **300 Fiction + 300 Nonfiction samples**
2. Compute accuracy
3. Predict missing categories
4. Fill missing values in dataset

---

### 5️⃣ Emotion Analysis

Each book is tagged with emotional scores:

* Joy
* Sadness
* Anger
* Fear
* Surprise

Used for **tone-based filtering** in recommendations.

---

### 6️⃣ Semantic Embeddings

Used local model:

```
sentence-transformers/all-MiniLM-L6-v2
```

#### Why?

* Fast
* Lightweight
* No API cost
* Optimized for semantic similarity

---

### 7️⃣ Vector Database (Chroma)

* Stored embeddings for all books
* Enabled fast similarity search
* Used persistent storage to avoid recomputation

---

### 8️⃣ Recommendation Logic

#### Step-by-step:

1. User inputs query
2. Convert query → embedding
3. Retrieve top-k similar books
4. Filter by category (optional)
5. Rank by emotional tone (optional)
6. Return top results

---

### 9️⃣ User Interface (Gradio)

Built an interactive dashboard:

* Text input for query
* Dropdown for category
* Dropdown for emotional tone
* Gallery output with:

  * Book cover
  * Title
  * Authors
  * Short description


## ▶️ Run the App

```bash
python main.py
```

---

## 📸 Example Query

```
"A story about love and forgiveness"
```

👉 Output:

* Semantically similar books
* Filtered by category
* Sorted by emotional tone

---


