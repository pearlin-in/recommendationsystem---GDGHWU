# Build Your Own Recommendation System

Type in a movie, song or book you like and get the 5 most similar items back. Along the way you will learn **unsupervised learning**, **vector embeddings** and **cosine similarity**, the same core ideas behind the recommendation features of Netflix, Spotify and Goodreads.

> No installation, no GPU, no paid services. Everything runs in a free Google Colab notebook.

---

## Table of contents

1. [What you will build](#1-what-you-will-build)
2. [Prerequisites](#2-prerequisites)
3. [Google Colab setup (free tier)](#3-google-colab-setup-free-tier)
4. [Datasets](#4-datasets)
5. [Repository structure](#5-repository-structure)
6. [Session 1: Foundations and TF-IDF baseline](#6-session-1-foundations-and-tf-idf-baseline)
7. [Session 2: Embeddings, evaluation and app](#7-session-2-embeddings-evaluation-and-app)
8. [Starter code](#8-starter-code)
9. [Troubleshooting](#10-troubleshooting)
10. [Stretch goals](#11-stretch-goals)
11. [Credits and licenses](#13-credits-and-licenses)

---

## 1. What you will build

A **content-based recommender**: it looks at what an item *is* (genres, description, keywords, audio features) rather than what other users did.

```
 "Inception"  ->  [ turn into a vector ]  ->  [ compare with every other vector ]  ->  top 5 closest
```

| Step | Idea | Tool |
|---|---|---|
| 1. Clean the data | Pick useful columns, fix missing values | pandas |
| 2. Represent items as vectors | TF-IDF first, then neural sentence embeddings | scikit-learn, sentence-transformers |
| 3. Measure similarity | Cosine similarity (angle between vectors) | scikit-learn, NumPy |
| 4. Recommend | Sort by similarity and return the top 5 | NumPy |
| 5. Ship it | Search box with results, shareable link | Gradio |

**Why this is unsupervised learning:** nobody tells the model what the "right" recommendation is. There are no labels. It discovers structure (which items are close together) from the data alone.

---

## 2. Prerequisites

- A laptop and a Google account
- Basic Python (variables, loops, functions). Pandas is taught during the workshop.
- No machine learning background needed

---

## 3. Google Colab setup (free tier)

### 3.1 First-time setup

1. Go to [colab.research.google.com](https://colab.research.google.com) and sign in with your Google account.
2. Click **File > Open notebook > GitHub tab**, paste the repository URL, and open `notebooks/session1_starter.ipynb`.
3. Click **File > Save a copy in Drive** so your changes are saved to your own account.
4. Check the runtime: **Runtime > Change runtime type**. Leave it on **CPU** (the default). Everything in this workshop is designed to work on CPU. If a GPU (T4) is available you can switch to it to speed up Session 2 but do not depend on it.
5. Click **Connect** (top right). Wait for the green check mark.

### 3.2 Run your first cell

```python
import sys
print(sys.version)
```

Press `Shift + Enter` to run a cell.

### 3.3 Install the extra libraries

Colab already includes pandas, NumPy, scikit-learn, matplotlib and `datasets`. We only need two extra packages:

```python
!pip install -q sentence-transformers gradio
```

> If you see a prompt to **restart the runtime** after installing, click **Runtime > Restart session** and then run the cells from the top again (skip the install cell).

### 3.4 Optional: save your work to Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

Save expensive outputs (such as embeddings) so you do not recompute them after a disconnect:

```python
import numpy as np
np.save('/content/drive/MyDrive/movie_embeddings.npy', emb)
# later:
emb = np.load('/content/drive/MyDrive/movie_embeddings.npy')
```

---

## 4. Datasets

All datasets below are **free, need no API key or Kaggle login, and download directly inside Colab**. They were chosen so that every step works on the free tier.

### 4.1 Comparison at a glance

| Domain | Dataset | Approx. size | Text description? | Best technique | Free-tier fit | Recommended? |
|---|---|---|---|---|---|---|
| **Movies** | TMDB 5000 Movies | ~4,800 movies | Yes (plot overview, keywords, tagline) | TF-IDF, then sentence embeddings | Excellent | **Primary pick** |
| **Songs** | Spotify Tracks Dataset | ~114k rows, 125 genres, ~20 MB | No (numeric audio features + genre) | Scaled audio features + cosine | Good (with subsampling) | **Primary pick** |
| **Books** | Goodbooks-10k (extended with descriptions) | ~10,000 books | Yes (description, genres) | TF-IDF, then sentence embeddings | Good | **Primary pick** |
| Movies | MovieLens latest-small | ~9,700 movies, ~100k ratings | No (genres + user tags only) | Collaborative filtering | Excellent | Optional extra |
| Books | Goodbooks-10k (original) | 10,000 books | No (title, author, year, user tags) | Tag-based TF-IDF | Excellent | Fallback |

### 4.2 Why these picks

**Movies: TMDB 5000.** It is the classic choice for this exact workshop because it has real text (plot summaries and keywords), which is what makes the jump from TF-IDF to sentence embeddings visible. It is small enough that a full 4,800 x 4,800 similarity matrix is only about 90 MB.

**Songs: Spotify Tracks Dataset.** Large and clean, with 125 genres and audio features per track (danceability, energy, valence, tempo, and so on). Important difference: **it has no lyrics or descriptions**, so songs are represented by *numeric audio vectors* rather than text. This is actually a useful teaching moment: vectors do not have to come from text, and cosine similarity works the same way. Do not try to compute a full similarity matrix for 114k songs (that would be over 50 GB). We compute similarity **per query** instead.

**Books: Goodbooks-10k.** Ten thousand popular Goodreads books. The original version only has titles, authors and user-created tags (noisy shelves like "to-read" and "favorites"). The **extended** version adds real descriptions and genres, which makes it far better for embeddings. Fall back to the original if the extended file ever disappears.

> **Tip for the facilitators:** run **movies** as the main live-coding example for both sessions (it has the best story: TF-IDF vs embeddings). Let participants pick **songs or books** for the hands-on extension in Session 2.

### 4.3 Loading each dataset in Colab

#### Movies (TMDB 5000, via Hugging Face)

```python
import pandas as pd
from datasets import load_dataset

movies = load_dataset("AiresPucrs/tmdb-5000-movies", split="train").to_pandas()
print(movies.shape)
movies[["title", "overview", "genres", "keywords"]].head()
```

Useful columns: `title`, `overview`, `genres`, `keywords`, `tagline`, `vote_average`, `vote_count`, `release_date`.
`genres` and `keywords` are stored as JSON-style text like `[{"id": 28, "name": "Action"}, ...]` and need parsing (see the starter code).

Backup source: the original files `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` are on Kaggle (free account needed). Organisers should keep a copy in the repo's `data/` folder in case of connectivity problems.

#### Songs (Spotify Tracks Dataset, via Hugging Face)

```python
import pandas as pd

URL = "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv"
songs = pd.read_csv(URL, index_col=0)
print(songs.shape)
songs.head()
```

Columns include `track_id`, `artists`, `album_name`, `track_name`, `popularity`, `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `track_genre`.

The same track can appear several times under different genres, so **deduplicate** (done in the starter code).

#### Books (Goodbooks-10k extended)

```python
import pandas as pd
from ast import literal_eval

URL = "https://raw.githubusercontent.com/malcolmosh/goodbooks-10k/master/books_enriched.csv"
books = pd.read_csv(URL, index_col=[0], converters={"genres": literal_eval})
print(books.shape)
print(books.columns.tolist())   # check the exact column names before building features
books.head()
```

Expect columns such as `title`, `authors`, `description` and `genres`. Print the column list first and adjust the starter code if names differ.

**Fallback: original Goodbooks-10k (tags only)**

```python
base = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/"
books = pd.read_csv(base + "books.csv")
tags = pd.read_csv(base + "tags.csv")
book_tags = pd.read_csv(base + "book_tags.csv")
```

Join `book_tags` to `tags` on `tag_id`, then to `books` on `goodreads_book_id`. Filter out noise tags such as `to-read`, `favorites`, `currently-reading` and `owned` before building text, and keep only the top 10 to 15 tags per book by `count`.

#### Optional: MovieLens latest-small

```python
import urllib.request, zipfile, io, pandas as pd
z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(
    "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip").read()))
ml_movies = pd.read_csv(z.open("ml-latest-small/movies.csv"))
```

This only has genres, so it is weak for content-based recommendations. It is better kept for a stretch goal on collaborative filtering (using the ratings file).

---

## 5. Repository structure

```
recsys-workshop/
├── README.md
├── notebooks/
│   ├── session1_starter.ipynb      # scaffolded, with TODOs
│   ├── session1_solution.ipynb
│   ├── session2_starter.ipynb
│   └── session2_solution.ipynb
├── data/                            # backup copies of datasets (small files only)
├── embeddings/                      # precomputed .npy files for fast-forward
└── slides/
```

Keep **checkpoint notebooks** for the end of each hands-on block so anyone who falls behind can catch up.

---

## 6. Session 1: Foundations and TF-IDF baseline

**Goal:** understand the theory and finish with a working recommender built on TF-IDF + cosine similarity.
**Deliverable:** a notebook where `recommend("The Dark Knight")` returns 5 sensible movies.

### Cosine similarity in one minute

For two vectors **a** and **b**:

```
cosine_similarity(a, b) = (a . b) / (|a| x |b|)
```

- `1.0` = pointing in the same direction (very similar)
- `0.0` = unrelated
- It compares **direction, not length**, so a long description and a short one about the same topic still match.

---

## 7. Session 2: Embeddings, evaluation and app

**Goal:** upgrade the baseline with semantic embeddings, compare approaches, and ship an interactive app.
**Deliverable:** a shareable recommender app each participant can publish to GitHub and put on their CV.

### The TF-IDF vs embeddings moment

TF-IDF matches **words**. Embeddings match **meaning**. Try a movie whose overview uses unusual vocabulary and compare the two result lists side by side. Participants should see embeddings finding thematically related films that share almost no words.

---

## 8. Starter code

Copy these cells into the notebooks. They are written to run on the free tier. Organisers: please **dry-run everything the day before**, since hosted datasets and library versions can change.

### 8.1 Shared helper (works for every dataset)

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def recommend(query, vectors, df, title_col="title", k=5):
    """Return the k most similar items to `query`.
    `vectors` can be a sparse TF-IDF matrix or a dense NumPy array."""
    titles = df[title_col].str.lower()
    matches = df.index[titles == query.lower()]
    if len(matches) == 0:
        close = df.loc[titles.str.contains(query.lower(), regex=False), title_col].head(5).tolist()
        raise ValueError(f"'{query}' not found. Did you mean: {close}")
    i = df.index.get_loc(matches[0])
    sims = cosine_similarity(vectors[i:i+1], vectors).ravel()   # one row only, never the full matrix
    top = np.argsort(-sims)[1:k+1]                              # skip index 0 (the item itself)
    out = df.iloc[top].copy()
    out["similarity"] = sims[top].round(3)
    return out
```

### 8.2 Movies: clean, TF-IDF, embeddings

```python
import json, pandas as pd
from datasets import load_dataset

movies = load_dataset("AiresPucrs/tmdb-5000-movies", split="train").to_pandas()

def names(cell):
    try:
        return [d["name"] for d in json.loads(cell)]
    except Exception:
        return []

movies["overview"] = movies["overview"].fillna("")
movies["genre_list"] = movies["genres"].apply(names)
movies["keyword_list"] = movies["keywords"].apply(names)

movies["soup"] = (
    movies["overview"]
    + " Genres: " + movies["genre_list"].apply(", ".join)
    + ". Keywords: " + movies["keyword_list"].apply(", ".join)
)

movies = movies.drop_duplicates("title").reset_index(drop=True)
movies = movies[movies["soup"].str.len() > 30].reset_index(drop=True)
```

**TF-IDF baseline (Session 1):**

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=20000)
X = tfidf.fit_transform(movies["soup"])
print(X.shape)

recommend("The Dark Knight", X, movies)[["title", "similarity"]]
```

**Sentence embeddings (Session 2):**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")      # small and CPU-friendly (about 90 MB)
emb = model.encode(
    movies["soup"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
)
print(emb.shape)   # (n_movies, 384)

recommend("The Dark Knight", emb, movies)[["title", "similarity"]]
```

On free-tier CPU, expect this encoding step to take a few minutes for about 4,800 movies. It is much faster with a GPU. **Save the result to Drive or share a precomputed file** (see section 3.5).

### 8.3 Songs: audio-feature vectors

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

URL = "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv"
songs = pd.read_csv(URL, index_col=0)

# 1. Clean: one row per track, keep the more popular ones so results feel familiar
songs = songs.drop_duplicates("track_id")
songs = songs.drop_duplicates(["track_name", "artists"])
songs = songs.dropna(subset=["track_name", "artists"])
songs = songs.sort_values("popularity", ascending=False).head(30000).reset_index(drop=True)

# 2. A readable label to search by
songs["label"] = songs["track_name"] + " - " + songs["artists"]

# 3. Build vectors from numeric audio features
features = ["danceability", "energy", "loudness", "speechiness",
            "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
S = StandardScaler().fit_transform(songs[features])     # scaling matters: tempo and loudness would dominate otherwise

recommend(songs["label"].iloc[0], S, songs, title_col="label")[["label", "track_genre", "similarity"]]
```

Notes:
- Limiting to the 30,000 most popular tracks keeps memory low and makes recommendations feel familiar. It also means each query compares 1 vector against 30k, which takes milliseconds.
- Participants can try adding genre as an extra feature (one-hot encode `track_genre` with `pd.get_dummies` and concatenate) and see how recommendations change.
- Because the dataset has no lyrics, sentence embeddings are **not** a good fit here. Explain this honestly: choosing the right representation for the data is a core ML skill.

### 8.4 Books: description and genres

```python
import pandas as pd
from ast import literal_eval

URL = "https://raw.githubusercontent.com/malcolmosh/goodbooks-10k/master/books_enriched.csv"
books = pd.read_csv(URL, index_col=[0], converters={"genres": literal_eval})
print(books.columns.tolist())      # confirm the column names first

books["description"] = books["description"].fillna("")
books["soup"] = (
    books["title"] + ". " + books["authors"].astype(str) + ". "
    + books["description"] + " Genres: " + books["genres"].apply(lambda g: ", ".join(g))
)
books = books.drop_duplicates("title").reset_index(drop=True)
books = books[books["soup"].str.len() > 40].reset_index(drop=True)

# Reuse exactly the same pipeline as movies:
from sklearn.feature_extraction.text import TfidfVectorizer
Xb = TfidfVectorizer(stop_words="english", max_features=20000).fit_transform(books["soup"])
recommend("The Hobbit", Xb, books)[["title", "similarity"]]
```

Embedding about 10,000 book descriptions on CPU takes longer than the movies (several minutes). Precompute it before the session and share the `.npy` file.

### 8.5 Gradio app

```python
import gradio as gr

def app_fn(choice):
    result = recommend(choice, emb, movies)          # swap in X / S and the right dataframe as needed
    return result[["title", "similarity"]]

demo = gr.Interface(
    fn=app_fn,
    inputs=gr.Dropdown(choices=sorted(movies["title"].tolist()),
                       label="Pick or type a movie you like"),
    outputs=gr.Dataframe(label="Top 5 recommendations"),
    title="Movie Recommender",
    description="Type a movie you like and get 5 similar ones.",
)
demo.launch(share=True)     # prints a temporary public link
```

The dropdown is searchable, so users can type to filter. The `share=True` link is temporary and only works while the notebook is running.

### 8.6 Optional: visualise the embedding space

```python
!pip install -q umap-learn
import umap, matplotlib.pyplot as plt

coords = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(emb)
plt.figure(figsize=(8, 6))
plt.scatter(coords[:, 0], coords[:, 1], s=3, alpha=0.5)
plt.title("Movie embeddings projected to 2D")
plt.show()
```
---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: sentence_transformers` | Run `!pip install -q sentence-transformers gradio`, restart the session if asked, then re-run from the top |
| Notebook says "Runtime disconnected" | Click Reconnect and re-run all cells (**Runtime > Run all**) |
| Dataset download fails or is slow | Use the backup copy in `data/`, or ask an organiser for the shared file |
| `KeyError` on a column name | Run `df.columns.tolist()` and adjust the column names in the code |
| `ValueError: 'X' not found` | Check spelling or use the suggestions the function prints |
| Out of memory | Restart the runtime, avoid full similarity matrices, subsample the data |
| Embedding step is very slow | Use the precomputed `.npy` file, or reduce the dataset size |
| Gradio link does not appear | Re-run the launch cell, and make sure `share=True` is set |
| Recommendations look strange | Print the `soup` text of both items and check what the model actually sees |

---

## 10. Stretch goals

- **Hybrid score:** combine TF-IDF and embedding similarity (for example `0.4 * tfidf + 0.6 * embedding`).
- **Multi-item queries:** average the vectors of 3 favourite movies and recommend from that.
- **Filters:** restrict results by genre, year or minimum rating.
- **Popularity bias:** discuss why popular items dominate, and try re-ranking by `vote_count`.
- **Collaborative filtering:** use MovieLens ratings and compare the two approaches.
- **Evaluate:** write 10 "obviously similar" pairs (for example sequels) and measure how often each method ranks them in the top 5.
- **Deploy permanently:** host the app on Hugging Face Spaces (free) instead of a temporary Gradio link.
- **Better text models:** try `all-mpnet-base-v2` (better quality, slower).

---

## 13. Credits and licenses

- **TMDB 5000 Movies:** data from [The Movie Database (TMDB)](https://www.themoviedb.org/). This project uses TMDB data but is not endorsed or certified by TMDB. Hugging Face mirror used: `AiresPucrs/tmdb-5000-movies`.
- **Spotify Tracks Dataset:** `maharshipandya/spotify-tracks-dataset` on Hugging Face (collected via the Spotify Web API; dataset card lists a BSD license).
- **Goodbooks-10k:** by Zygmunt Zajac ([zygmuntz/goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k)); extended version with descriptions from [malcolmosh/goodbooks-10k-extended](https://github.com/malcolmosh/goodbooks-10k-extended).
- **MovieLens:** GroupLens Research, University of Minnesota.
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`.


---

*Built by GDG on Campus. Questions? Ask a mentor or open an issue in this repository.*
