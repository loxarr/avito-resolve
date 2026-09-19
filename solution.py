"""Candidate generation for Avito services.

The solution uses only local Parquet files and open-source Python packages.
Run ``python solution.py --data-dir /path/to/dataset`` to write answer.csv.
``--validate`` evaluates held-out historical clicks whose items are in the
benchmark corpus before producing the answer.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def normalize(text: str) -> str:
    """Use the same small normalization for historical and target queries."""
    return re.sub(r"\s+", " ", str(text).lower().replace("ё", "е")).strip()


def top_indices(values: np.ndarray, count: int) -> np.ndarray:
    if count >= len(values):
        return np.argsort(-values)
    indices = np.argpartition(values, -count)[-count:]
    return indices[np.argsort(-values[indices])]


class Retriever:
    def __init__(self, items: pd.DataFrame, train: pd.DataFrame):
        self.items = items.reset_index(drop=True)
        self.n = len(items)
        self.ids = items.item_id.to_numpy()
        self.id_to_pos = {x: k for k, x in enumerate(self.ids)}
        self.location = items.item_location_id.to_numpy()
        self.microcat = items.item_microcat_id.to_numpy()

        title = [normalize(x) for x in items.item_title_raw.fillna("").tolist()]
        # Descriptions and parameter strings are long and repetitive. A prefix
        # covers the concrete service while keeping the sparse index compact.
        content = [
            f"{t} {t} {normalize(p[:300])} {normalize(d[:900])}"
            for t, p, d in zip(title,
                               items.item_infm_params_text.fillna("").tolist(),
                               items.item_description_raw.fillna("").tolist())
        ]
        self.word = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=220_000,
            sublinear_tf=True, dtype=np.float32,
        )
        self.char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=3,
            max_features=300_000, sublinear_tf=True, dtype=np.float32,
        )
        self.body = TfidfVectorizer(
            ngram_range=(1, 2), min_df=3, max_features=250_000,
            sublinear_tf=True, dtype=np.float32,
        )
        print("Indexing titles (words)...", flush=True)
        self.word_matrix = self.word.fit_transform(title)
        print("Indexing titles (characters)...", flush=True)
        self.char_matrix = self.char.fit_transform(title)
        print("Indexing descriptions and attributes...", flush=True)
        self.body_matrix = self.body.fit_transform(content)
        del title, content

        # Historical clicked locations and service types are query priors,
        # never answer lists. They transfer to previously unseen corpus items.
        self.loc_prior: dict[int, list[tuple[int, float]]] = {}
        for loc, group in train.groupby("search_location_id", sort=False):
            counts = group.item_location_id.value_counts().head(12)
            self.loc_prior[int(loc)] = list(zip(counts.index.astype(int),
                                                (counts / counts.sum()).astype(float)))
        train = train.copy()
        train["query_norm"] = [normalize(x) for x in train.search_query.tolist()]
        self.cat_prior: dict[str, list[tuple[int, float]]] = {}
        for query, group in train.groupby("query_norm", sort=False):
            counts = group.item_microcat_id.value_counts().head(8)
            self.cat_prior[query] = list(zip(counts.index.astype(int),
                                              (counts / counts.sum()).astype(float)))
        # Character neighbors infer a service type for previously unseen
        # phrasings (e.g. different Russian case endings or word order).
        self.query_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2,
            sublinear_tf=True, dtype=np.float32,
        )
        self.known_queries = list(self.cat_prior)
        self.query_matrix = self.query_vectorizer.fit_transform(self.known_queries)

    def category_probabilities(self, query: str) -> list[tuple[int, float]]:
        if query in self.cat_prior:
            return self.cat_prior[query]
        similarities = (self.query_vectorizer.transform([query])
                        @ self.query_matrix.T).toarray().ravel()
        neighbors = top_indices(similarities, 8)
        totals: dict[int, float] = defaultdict(float)
        for neighbor in neighbors:
            similarity = float(similarities[neighbor])
            if similarity < 0.35:
                continue
            for category, probability in self.cat_prior[self.known_queries[neighbor]]:
                totals[category] += similarity ** 4 * probability
        total = sum(totals.values())
        return [(category, value / total) for category, value in totals.items()] if total else []

    def features(self, query: str, location_id: int) -> tuple[np.ndarray, ...]:
        """Return item-level lexical and structured scores for one request."""
        q = normalize(query)
        word = (self.word.transform([q]) @ self.word_matrix.T).toarray().ravel()
        char = (self.char.transform([q]) @ self.char_matrix.T).toarray().ravel()
        body = (self.body.transform([q]) @ self.body_matrix.T).toarray().ravel()
        exact_loc = (self.location == location_id).astype(np.float32)
        loc_prior = np.zeros(self.n, dtype=np.float32)
        for target_loc, probability in self.loc_prior.get(int(location_id), []):
            loc_prior[self.location == target_loc] = probability
        cat_prior = np.zeros(self.n, dtype=np.float32)
        for target_cat, probability in self.category_probabilities(q):
            cat_prior[self.microcat == target_cat] = probability
        return word, char, body, exact_loc, loc_prior, cat_prior

    @staticmethod
    def score(features: tuple[np.ndarray, ...], loc_weight: float = 1.0,
              cat_weight: float = 0.0, body_weight: float = 0.65) -> np.ndarray:
        word, char, body, exact_loc, loc_prior, cat_prior = features
        lexical = 1.5 * word + char + body_weight * body
        location = np.maximum(exact_loc, loc_prior)
        return lexical * (1 + loc_weight * location) + cat_weight * cat_prior * np.maximum(0.1, lexical)


def validate(retriever: Retriever, valid: pd.DataFrame) -> None:
    """A reproducible proxy test using observed clicks inside the search corpus.

    This is a click-level proxy: the hidden benchmark may have a different
    distribution and multiple relevant items per query.
    """
    weights = [(a, b, c) for a in (2.0, 4.0, 8.0, 12.0)
               for b in (0.0, 0.2, 0.5) for c in (0.65, 1.0)]
    hits = np.zeros(len(weights), dtype=np.float64)
    subgroup_hits = defaultdict(lambda: np.zeros(len(weights), dtype=np.float64))
    subgroup_count = Counter()
    for k, row in enumerate(valid.itertuples(index=False)):
        feats = retriever.features(row.search_query, row.search_location_id)
        target = retriever.id_to_pos[row.item_id]
        group = ("seen" if normalize(row.search_query) in retriever.cat_prior else "unseen",
                 "same_loc" if row.search_location_id == row.item_location_id else "other_loc")
        subgroup_count[group] += 1
        for m, (loc_w, cat_w, body_w) in enumerate(weights):
            scores = retriever.score(feats, loc_w, cat_w, body_w)
            found = np.count_nonzero(scores > scores[target]) < 50
            hits[m] += found
            subgroup_hits[group][m] += found
        if (k + 1) % 200 == 0:
            print(f"Validated {k + 1}/{len(valid)}", flush=True)
    for idx in np.argsort(-hits)[:12]:
        print(f"Recall@50 proxy={hits[idx]/len(valid):.4f} parameters={weights[idx]}")
    best = int(np.argmax(hits))
    for group, count in subgroup_count.items():
        print(f"  {group}: {count} rows; recall={subgroup_hits[group][best]/count:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("answer.csv"))
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validation-count", type=int, default=1200)
    parser.add_argument("--location-weight", type=float, default=12.0)
    parser.add_argument("--category-weight", type=float, default=0.2)
    parser.add_argument("--body-weight", type=float, default=1.0)
    args = parser.parse_args()

    item_cols = ["item_id", "item_title_raw", "item_infm_params_text",
                 "item_description_raw", "item_location_id", "item_microcat_id"]
    train_cols = ["search_query", "search_location_id", "item_id",
                  "item_location_id", "item_microcat_id"]
    print("Loading parquet files...", flush=True)
    items = pd.read_parquet(args.data_dir / "benchmark_items.parquet", columns=item_cols)
    train = pd.read_parquet(args.data_dir / "train.parquet", columns=train_cols)
    queries = pd.read_parquet(args.data_dir / "benchmark_queries.parquet")
    if args.validate:
        # Remove the held-out query/item interactions before estimating query
        # service and location priors; otherwise validation would leak labels.
        eligible = train.loc[train.item_id.isin(items.item_id)].drop_duplicates(
            ["search_query", "search_location_id", "item_id"]
        )
        valid = eligible.sample(n=min(args.validation_count, len(eligible)), random_state=42)
        keys = pd.MultiIndex.from_frame(valid[["search_query", "search_location_id", "item_id"]])
        train_keys = pd.MultiIndex.from_frame(train[["search_query", "search_location_id", "item_id"]])
        history = train.loc[~train_keys.isin(keys)]
        retriever = Retriever(items, history)
        validate(retriever, valid)
        return

    retriever = Retriever(items, train)

    answers = []
    for k, row in enumerate(queries.itertuples(index=False)):
        feats = retriever.features(row.search_query, row.search_location_id)
        scores = retriever.score(feats, args.location_weight,
                                 args.category_weight, args.body_weight)
        selected = retriever.ids[top_indices(scores, 50)]
        answers.append(" ".join(selected))
        if (k + 1) % 250 == 0:
            print(f"Predicted {k + 1}/{len(queries)}", flush=True)
    result = pd.DataFrame({"query_id": queries.query_id, "answer": answers})
    # Fail loudly if any ID conversion or serialization mistake would silently
    # lower the benchmark score.
    assert result.query_id.is_unique and result.query_id.str.fullmatch(r"[A-Za-z0-9]{16}").all()
    valid_ids = set(retriever.ids)
    for answer in result.answer:
        ids = answer.split(" ")
        assert len(ids) <= 50 and len(ids) == len(set(ids))
        assert all(re.fullmatch(r"[0-9a-f]{16}", x) and x in valid_ids for x in ids)
    result.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Saved {len(queries)} answers to {args.output}", flush=True)


if __name__ == "__main__":
    main()
