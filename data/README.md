# Backup copies of the datasets

The notebooks always try the original online source first. If it fails (slow Wi-Fi at the venue, a mirror that disappeared), they load these copies from this repo's GitHub URL instead. You don't need to download anything by hand.

| File | Original source | Size | Loaded with |
|---|---|---|---|
| `tmdb_5000_movies.parquet` | Hugging Face `AiresPucrs/tmdb-5000-movies` (exact copy) | 14 MB | `pd.read_parquet(...)` |
| `spotify_tracks.csv.gz` | Hugging Face `maharshipandya/spotify-tracks-dataset` `dataset.csv` (gzipped) | 8 MB | `pd.read_csv(..., index_col=0)` |
| `books_enriched.csv.gz` | GitHub `malcolmosh/goodbooks-10k` `books_enriched.csv` (gzipped) | 5 MB | `pd.read_csv(..., index_col=[0], converters=...)` |

pandas reads `.gz` files directly, so the loading code is the same as for the original CSVs.

Licenses and credits for each dataset are listed in the main [README](../README.md#11-credits-and-licenses).
