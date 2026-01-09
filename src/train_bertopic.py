import json
import pandas as pd
from tqdm import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora

def topic_diversity(topics, topk=10):
    top_words = [word for topic in topics for word in topic[:topk]]
    unique_words = set(top_words)
    return len(unique_words) / len(top_words)

def run_bertopic():
    DATA_PATH = Path("data/processed/docs_tokens.jsonl")
    OUT_METRICS = Path("experiments/bertopic_metrics.csv")
    OUT_TOPICS = Path("experiments/bertopic_topics.csv")
    OUT_PLOT = Path("experiments/bertopic_coherence_td.png")

    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)

    # ===== 1. Load Data =====
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
        docs = [" ".join(d["tokens"]) for d in data if len(d["tokens"]) > 0]
        tokens = [d["tokens"] for d in data if len(d["tokens"]) > 0]
    print(f"Loaded {len(docs)} documents")

    # ===== 2. Setup embedding model =====
    embedder = SentenceTransformer("indobenchmark/indobert-base-p1")

    # ===== 3. Prepare dictionary for CoherenceModel =====
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(text) for text in tokens]

    metrics = []
    topic_records = []

    # ===== 4. Loop training BERTopic =====
    for k in tqdm(range(5, 51, 5), desc="Training BERTopic (K=5–50)"):

        topic_model = BERTopic(
            embedding_model=embedder,
            nr_topics=k,
            verbose=False,
            calculate_probabilities=False,
            low_memory=True
        )

        topics, _ = topic_model.fit_transform(docs)

        # Extract top words
        topic_words = topic_model.get_topics()
        topics_for_eval = []
        for tid, words in topic_words.items():
            topics_for_eval.append([(w, _) for w, _ in words])

        # Topic Diversity
        td = topic_diversity([[w for w, _ in words] for words in topic_words.values()])

        # Coherence
        cm = CoherenceModel(
            topics=[[w for w, _ in words] for words in topic_words.values()],
            texts=tokens,
            dictionary=dictionary,
            coherence="c_v",
            processes=1
        )
        coherence = cm.get_coherence()

        metrics.append({"num_topics": k, "coherence": coherence, "topic_diversity": td})

        # Simpan semua topik
        for tid, words in topic_words.items():
            word_list = ", ".join([w for w, _ in words])
            topic_records.append({
                "num_topics": k,
                "topic_id": tid,
                "words": word_list
            })

    # ===== 5. Save results =====
    df_metrics = pd.DataFrame(metrics)
    df_topics = pd.DataFrame(topic_records)
    df_metrics.to_csv(OUT_METRICS, index=False)
    df_topics.to_csv(OUT_TOPICS, index=False)
    print("\n✅ Metrics saved to:", OUT_METRICS)
    print("✅ Topics saved to:", OUT_TOPICS)

    # ===== 6. Plot =====
    plt.figure(figsize=(9,6))
    plt.plot(df_metrics["num_topics"], df_metrics["coherence"], label="Coherence (C_v)", marker="o")
    plt.plot(df_metrics["num_topics"], df_metrics["topic_diversity"], label="Topic Diversity", marker="s")
    plt.xlabel("Jumlah Topik (K)")
    plt.ylabel("Nilai")
    plt.title("Evaluasi BERTopic pada Terjemahan Al-Qur'an")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=300)
    plt.show()

    print("\n✅ Plot saved to:", OUT_PLOT)

if __name__ == "__main__":
    run_bertopic()
