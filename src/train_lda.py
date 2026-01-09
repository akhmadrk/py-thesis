import json
import pandas as pd
from tqdm import tqdm
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def topic_diversity(topics, topk=10):
    # Correct unpacking
    top_words = [word for _, words in topics for word, _ in words[:topk]]
    unique_words = set(top_words)
    return len(unique_words) / len(top_words)

def run_lda():
    DATA_PATH = Path("data/processed/docs_tokens.jsonl")
    OUT_METRICS = Path("experiments/lda_metrics.csv")
    OUT_TOPICS = Path("experiments/lda_topics.csv")
    OUT_PLOT = Path("experiments/lda_coherence_td.png")

    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)

    # ====== 1. LOAD DATA ======
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        docs = [json.loads(line)["tokens"] for line in f if len(json.loads(line)["tokens"]) > 0]

    print(f"Loaded {len(docs)} documents")

    # ====== 2. CORPUS & DICTIONARY ======
    dictionary = corpora.Dictionary(docs)
    dictionary.filter_extremes(no_below=2, no_above=0.5)
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    metrics = []
    topics_records = []

    for k in tqdm(range(5, 51, 5), desc="Training LDA (K=5–50)"):
        lda_model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            passes=10,
            random_state=42,
            alpha="auto",
            eta="auto",
        )

        # Coherence (non-parallel, avoid multiprocessing issue)
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=docs,
            dictionary=dictionary,
            coherence="c_v",
            processes=1  # 👈 disable multiprocessing
        )
        coherence = coherence_model.get_coherence()

        # Topic diversity
        td = topic_diversity(lda_model.show_topics(num_topics=k, num_words=10, formatted=False))

        metrics.append({"num_topics": k, "coherence": coherence, "topic_diversity": td})

        for topic_id, words in lda_model.show_topics(num_topics=k, num_words=15, formatted=False):
            topics_records.append({
                "num_topics": k,
                "topic_id": topic_id,
                "words": ", ".join([w for w, _ in words])
            })

    # ====== 5. SAVE RESULTS ======
    df_metrics = pd.DataFrame(metrics)
    df_topics = pd.DataFrame(topics_records)

    df_metrics.to_csv(OUT_METRICS, index=False)
    df_topics.to_csv(OUT_TOPICS, index=False)

    print("\n✅ Metrics saved to:", OUT_METRICS)
    print("✅ Topics saved to:", OUT_TOPICS)

    # ====== 6. PLOT ======
    plt.figure(figsize=(9,6))
    plt.plot(df_metrics["num_topics"], df_metrics["coherence"], label="Coherence (C_v)", marker="o")
    plt.plot(df_metrics["num_topics"], df_metrics["topic_diversity"], label="Topic Diversity", marker="s")
    plt.xlabel("Jumlah Topik (K)")
    plt.ylabel("Nilai")
    plt.title("Evaluasi LDA pada Terjemahan Al-Qur'an")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=300)
    plt.show()

    print("\n✅ Plot saved to:", OUT_PLOT)


if __name__ == "__main__":
    run_lda()
