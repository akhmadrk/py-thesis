import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation


# ===============================
# Topic Diversity
# ===============================
def topic_diversity(topics, topk=10):
    top_words = [word for topic in topics for word in topic[:topk]]
    unique_words = set(top_words)
    return len(unique_words) / len(top_words)


# ===============================
# Main Experiment
# ===============================
def run_combined_ctm():

    DATA_PATH = Path("data/processed/docs_tokens.jsonl")

    OUT_DIR = Path("experiments/combined_ctm")
    OUT_METRICS = OUT_DIR / "ctm_metrics.csv"
    OUT_TOPICS = OUT_DIR / "ctm_topics.csv"
    OUT_PLOT = OUT_DIR / "ctm_coherence_td.png"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ===== 1. Load Data =====
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    docs = [" ".join(d["tokens"]) for d in data if len(d["tokens"]) > 0]
    tokens = [d["tokens"] for d in data if len(d["tokens"]) > 0]

    print(f"Loaded {len(docs)} documents")

    # ===== 2. Dictionary for Coherence =====
    dictionary = Dictionary(tokens)

    metrics = []
    topic_records = []
    
    # --- Data preparation (BoW + BERT embedding)
    tp = TopicModelDataPreparation(
        contextualized_model="indobenchmark/indobert-base-p1"
    )
    
    dataset = tp.fit(
        text_for_contextual=docs,
        text_for_bow=docs
    )


    # ===== 3. Loop Number of Topics =====
    for k in tqdm(range(5, 51, 5), desc="Training Combined CTM (K=5–50)"):        

        # --- Combined CTM
        ctm = CombinedTM(
            bow_size=len(tp.vocab),
            contextual_size=768,
            n_components=k,
            num_epochs=5,
            batch_size=64,
            dropout=0.2
        )

        ctm.fit(dataset)

        # ===== 4. Extract Topics =====
        topic_words = ctm.get_topic_lists(10)

        # Topic Diversity
        td = topic_diversity(topic_words)

        # Coherence
        cm = CoherenceModel(
            topics=topic_words,
            texts=tokens,
            dictionary=dictionary,
            coherence="c_v",
            processes=1
        )
        coherence = cm.get_coherence()

        metrics.append({
            "num_topics": k,
            "coherence": coherence,
            "topic_diversity": td
        })

        # Save topic words
        for tid, words in enumerate(topic_words):
            topic_records.append({
                "num_topics": k,
                "topic_id": tid,
                "words": ", ".join(words)
            })

    # ===== 5. Save Results =====
    df_metrics = pd.DataFrame(metrics)
    df_topics = pd.DataFrame(topic_records)

    df_metrics.to_csv(OUT_METRICS, index=False)
    df_topics.to_csv(OUT_TOPICS, index=False)

    print("✅ Metrics saved to:", OUT_METRICS)
    print("✅ Topics saved to:", OUT_TOPICS)

    # ===== 6. Plot =====
    plt.figure(figsize=(9, 6))
    plt.plot(df_metrics["num_topics"], df_metrics["coherence"],
             marker="o", label="Coherence (C_v)")
    plt.plot(df_metrics["num_topics"], df_metrics["topic_diversity"],
             marker="s", label="Topic Diversity")

    plt.xlabel("Jumlah Topik (K)")
    plt.ylabel("Nilai")
    plt.title("Evaluasi Combined CTM pada Terjemahan Al-Qur'an")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=300)
    plt.show()

    print("✅ Plot saved to:", OUT_PLOT)


if __name__ == "__main__":
    run_combined_ctm()
