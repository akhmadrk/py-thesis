import pandas as pd, matplotlib.pyplot as plt

lda = pd.read_csv("experiments/lda_metrics.csv")
btm = pd.read_csv("experiments/btm_metrics.csv")
bertopic = pd.read_csv("experiments/bertopic_metrics.csv")

plt.figure(figsize=(9,6))
plt.plot(lda.num_topics, lda.coherence, label="LDA – Coherence", marker="o")
plt.plot(btm.num_topics, btm.coherence, label="BTM – Coherence", marker="s")
plt.plot(bertopic.num_topics, bertopic.coherence, label="BERTopic – Coherence", marker="^")

plt.xlabel("Jumlah Topik (K)")
plt.ylabel("Coherence (C_v)")
plt.title("Perbandingan Coherence Score Antar Model")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("experiments/all_model_coherence.png", dpi=300)
plt.show()