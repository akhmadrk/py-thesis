import pandas as pd
import json
import re
from pathlib import Path
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()


# --- Inisialisasi stopword ---
stop_words = set(stopwords.words("indonesian"))
sastrawi_stop = set(StopWordRemoverFactory().get_stop_words())
combined_stop = stop_words.union(sastrawi_stop)



# --- File paths ---
INPUT_CSV = Path("data/raw/quran_terjemahan.csv")
DOCS_JSONL = Path("data/processed/docs.jsonl")
TOKENS_JSONL = Path("data/processed/docs_tokens.jsonl")

# --- Setup stopwords ---
stop_words = set(stopwords.words("indonesian"))
sastrawi_stop = set(StopWordRemoverFactory().get_stop_words())
combined_stop = stop_words.union(sastrawi_stop)

# --- Clean function ---
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # 1️⃣ Normalisasi apostrof ke bentuk tunggal
    text = (
        text.replace("’", "'")
            .replace("‘", "'")
            .replace("`", "'")
            .replace("ʼ", "'")
            .replace("ʾ", "'")
            .replace("ʿ", "'")
    )

    # 2️⃣ Lowercase dan hilangkan spasi ganda
    text = text.lower().strip()

    # 3️⃣ Bersihkan karakter non huruf tapi biarkan tanda - dan '
    text = re.sub(r"[^a-z'\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # 4️⃣ Normalisasi transliterasi Qur’an & Rahman, dll.
    #    (gunakan regex agar mencakup variasi tanda hubung / spasi)
    text = re.sub(r"al[\-\s']*qur['’`]*an", "alquran", text)
    text = re.sub(r"ar[\-\s']*rahm[ai][mn]", "arrahman", text)

    # 5️⃣ Terapkan mapping normalize_map untuk bentuk sisa
    for k, v in normalize_map.items():
        text = text.replace(k, v)

    # 6️⃣ Hilangkan spasi ganda akhir
    text = re.sub(r"\s+", " ", text).strip()

    return text

# --- Kamus normalisasi transliterasi umum ---
normalize_map = {
    "alqur'an": "alquran",
    "al-qur'an": "alquran",
    "qur'an": "quran",
    "ar-rahman": "arrahman",
    "ar-rahim": "arrahim",
    "ar-rahmaan": "arrahman",
    "ar-rahim": "arrahim",
}


# --- Stopword kustom umum (penyambung + kata kosong) ---
custom_stopwords = {
    "dan", "yang", "di", "ke", "pada", "dari", "itu", "karena", "sebagai",
    "oleh", "maka", "agar", "dengan", "jika", "adalah", "tidak", "akan",
    "untuk", "saja", "atau", "ia", "kami", "kamu", "mereka", "nya", "pun",
    "lah", "hanya", "sesungguhnya", "sebuah", "ini", "itu", "tersebut"
}

# --- Stopword khas teks Al-Qur’an ---
quranic_stopwords = {
    "alif", "lam", "mim", "ha", "ya", "sin", "tha", "kaf", "nun", "qaf", "ra",
    "sad", "ain", "ta", "sa", "hamim", "yasins", "thasin", "thasim", "thaha",
    "yaasiin", "thaa", "haa", "rahmaan", "rahim"
}

# Gabungkan semua stopword
combined_stop = stop_words.union(sastrawi_stop).union(custom_stopwords).union(quranic_stopwords)


def tokenize_text(text: str):
    # 1️⃣ normalisasi apostrof
    text = (
        text.lower()
            .replace("’", "'")
            .replace("‘", "'")
            .replace("`", "'")
            .replace("ʼ", "'")
    )

    # 2️⃣ normalisasi Qur'an-like
    text = re.sub(r"al[\-\s']*qur['’`]*an", "alquran", text)

    # 3️⃣ ambil token (izinkan tanda hubung/apostrof di tengah)
    tokens = re.findall(r"[a-z]+(?:['\-][a-z]+)*", text)

    # 4️⃣ mapping transliterasi
    tokens = [normalize_map.get(t, t) for t in tokens]

    # 5️⃣ gabung prefiks arab “al” + kata berikutnya
    merged = []
    skip = False
    for i, t in enumerate(tokens):
        if skip:
            skip = False
            continue
        if t == "al" and i + 1 < len(tokens):
            merged.append(f"al-{tokens[i+1]}")
            skip = True
        else:
            merged.append(t)

    # 6️⃣ buang stopword umum & lafadz muqatha’at
    filtered_tokens = [t for t in merged if t not in combined_stop and len(t) > 2]

    # 7️⃣ stemming (AMAN)
    stemmed_tokens = [stemmer.stem(t) for t in filtered_tokens]

    # 8️⃣ filter ulang hasil stemming
    final_tokens = [
        t for t in stemmed_tokens
        if t not in combined_stop and len(t) > 2
    ]

    return final_tokens

# --- Main process ---
def main():
    df = pd.read_csv(INPUT_CSV)
    tqdm.pandas(desc="Cleaning text")

    # Pastikan kolom sesuai
    df = df.rename(columns={
        "surah": "surah",
        "ayat": "ayat",
        "terjemahan": "text"
    })

    with open(DOCS_JSONL, "w", encoding="utf-8") as f_out1, \
         open(TOKENS_JSONL, "w", encoding="utf-8") as f_out2:

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing ayat"):
            text_clean = clean_text(row["text"])
            tokens = tokenize_text(text_clean)

            doc = {
                "id": f"{row['surah']}:{row['ayat']}",
                "surah": int(row["surah"]),
                "ayah": int(row["ayat"]),
                "text": text_clean,
            }
            if len(tokens) == 0:
              continue

            doc_tokens = {
                "id": doc["id"],
                "tokens": tokens,
            }

            f_out1.write(json.dumps(doc, ensure_ascii=False) + "\n")
            f_out2.write(json.dumps(doc_tokens, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(df)} ayat to {DOCS_JSONL} and {TOKENS_JSONL}")

if __name__ == "__main__":
    main()
