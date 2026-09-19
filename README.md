# Candidate generation for Avito services

This repository produces an `answer.csv` with up to 50 service listings for
each query in the supplied benchmark. It uses only the three supplied Parquet
files and local, open-source Python packages; it does not call an external API.

## Reproduce

Python 3.12 is recommended. Put `train.parquet`, `benchmark_items.parquet` and
`benchmark_queries.parquet` in a data directory, then run:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python solution.py --data-dir /path/to/data --output answer.csv
```

The script fixes all tie handling and uses no randomness at inference time.
For a local proxy evaluation, run:

```bash
.venv/bin/python solution.py --data-dir /path/to/data --validate --validation-count 500
```

## Approach

The retrieval index consists of three sparse TF-IDF matrices built over the
benchmark corpus: title word unigrams/bigrams, title character 3–5-grams and
word unigrams/bigrams from a compact combination of title, item parameters and
description. Character grams tolerate spelling and inflection differences;
the body index catches services described outside the title. The query text
is scored against all three, so the index can retrieve listings absent from
training.

The training clicks provide aggregate priors for the search location and the
listing's microcategory. We store distributions, not query-specific answer
lists. For unseen query wording, nearby historical queries in character TF-IDF
space provide a weak microcategory prior. The final score blends lexical
similarity with these priors. A large location multiplier reflects the strong
location match in observed clicks.

Only the following fields influence retrieval: `search_query`,
`search_location_id`, `item_title_raw`, `item_description_raw`,
`item_infm_params_text`, `item_location_id` and `item_microcat_id`. The corpus
`item_id` values are preserved as strings. Rating, price and contact settings
are not used, because candidate generation prioritizes recall over ordering.

## Validation and error analysis

The `--validate` mode samples historical clicked items that are also present
in the benchmark corpus and removes those query/item interactions from the
history priors. It reports a click-level Recall@50 proxy and results by seen
query text and location match. This proxy is not the hidden benchmark metric:
the hidden benchmark can contain multiple relevant items per query and a
different query mix.

With random seed 42 and 500 held-out interactions, the selected weights gave
**0.768 click-level Recall@50**. The held-out breakdown was 0.797 for seen
queries with the same location, 0.815 for unseen queries with the same
location, 0.676 for seen queries with a different location and 0.412 for
unseen queries with a different location. The final configuration weights
location 12.0, microcategory 0.2 and body text 1.0.

Observed error types include descriptions where the service is absent from the
title, spelling or case variation between query and listing, and listings in
a nearby rather than identical location. The body index, character grams and
location priors respectively address those cases. The code comments document
the feature construction and held-out split.
