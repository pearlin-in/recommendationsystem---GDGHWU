# Precomputed embeddings (fast-forward)

Encoding thousands of texts on a free Colab CPU takes several minutes. These files let anyone skip that step.

| File | Rows | Made from |
|---|---|---|
| `movies_minilm.npy` | 4,800 movies × 384 | `movies["soup"]` built in `session2_*.ipynb` |
| `books_minilm.npy` | 9,964 books × 384 | `books["soup"]` built in `session2_*.ipynb` (Part 9, option B) |

- Model: `sentence-transformers/all-MiniLM-L6-v2` with `normalize_embeddings=True`.
- Stored as `float16` to halve the file size. The notebook converts them back to `float32`.
- The row order matches the cleaned DataFrame in the notebook: movies sorted by `vote_count`, books by `ratings_count`, then duplicate titles dropped.

You don't load these by hand. `load_or_compute_embeddings()` in Session 2 first looks in your cache (Google Drive if enabled), then here, and computes the embeddings only if neither works.
It also re-encodes a few rows to check that a file still matches the data. If a dataset changes upstream, it recomputes instead of returning wrong results.

**Regenerating (organisers):** run `notebooks/session2_solution.ipynb` with no `cache/` folder. It writes `cache/movies_minilm.npy` and `cache/books_minilm.npy`. Copy them into this folder and commit.
