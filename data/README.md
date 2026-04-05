# Data

`companies.csv` contains ~1,000 synthetic company records with columns:

- `id` — unique identifier
- `company_name` — display name
- `country` — country of operation
- `long_offering` — rich text bio (100-400 words)

Ingest this into a vector database, embed the `long_offering` field, and use it as the
retrieval corpus.

Do not commit proprietary data.
