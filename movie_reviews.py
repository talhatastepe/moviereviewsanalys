import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

nltk.download("stopwords", quiet=True)

# Veri setlerini yükledim
try:
    df_letterboxd = pd.read_csv(
        "C:/Users/ACER/Downloads/moviereviewsdataset/letterboxd-reviews.csv",
        encoding="ISO-8859-1",
    )
    print(f"Letterboxd veri seti yüklendi. Boyut: {df_letterboxd.shape}")
except Exception as e:
    print(f"Letterboxd veri seti yüklenirken hata: {e}")
    df_letterboxd = pd.DataFrame()

try:
    df_metacritic = pd.read_csv(
        "C:/Users/ACER/Downloads/moviereviewsdataset/metacritic-reviews.csv",
        encoding="ISO-8859-1",
        on_bad_lines="skip",
    )
    print(f"Metacritic veri seti yüklendi. Boyut: {df_metacritic.shape}")
except Exception as e:
    print(f"Metacritic veri seti yüklenirken hata: {e}")
    df_metacritic = pd.DataFrame()

# Sütun isimlerini checkledim
if not df_letterboxd.empty:
    print("\nLetterboxd sütunları:", df_letterboxd.columns.tolist())
if not df_metacritic.empty:
    print("Metacritic sütunları:", df_metacritic.columns.tolist())


# Temizleme işlemi gerçekleştirme
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", "", text)  # HTML tag'leri kaldır
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Harf dışı karakterleri kaldır
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens)


def topic_label(text):
    if not isinstance(text, str):
        return "unknown"

    text = text.lower()

    if any(word in text for word in ["story", "plot", "script", "writing"]):
        return "story"
    elif any(word in text for word in ["acting", "actor", "performance", "cast"]):
        return "acting"
    elif any(word in text for word in ["cinematography", "visual", "scenes", "shots"]):
        return "visuals"
    elif any(word in text for word in ["music", "soundtrack", "score", "sound"]):
        return "music"
    elif any(word in text for word in ["direction", "director"]):
        return "direction"
    else:
        return "general"


# Letterboxd-Rating değerlerini temizleyip sayısal değerlere dönüştüren fonksiyon
def extract_letterboxd_rating(rating_str):
    if pd.isna(rating_str) or not isinstance(rating_str, str):
        return 0.0

    # Letterboxddda değerlendirmeler yıldız üzerinden olduğundan yıldız sayısını tahmin etme - her bir yıldız için için 1 puan
    star_count = rating_str.count("â??")

    # Yarım yıldız için 0.5 ekleme
    if "Â½" in rating_str:
        return star_count + 0.5

    # Eğer sayısal bir değer varsa doğrudan dönüştürme işi;
    try:
        return float(rating_str)
    except:
        return float(star_count)


# Metacritic - Rating değerlerini sayısal değerlere dönüştüren fonksiyon
def extract_metacritic_rating(rating):
    if pd.isna(rating):
        return 0.0

    # Metacritic puanları genellikle 0-100 arasında;
    try:
        rating_value = float(rating)
        # 100 üzerinden olan puanları 5 üzerine dönüştür (uyumlu olması için)
        if rating_value > 10:  # Muhtemelen 100 üzerinden
            return rating_value / 20  # 100 -> 5 olacak şekilde
        return rating_value
    except:
        return 0.0


# Duygu etiketleme fonksiyonu (5 üzerinden puanlama için)
def sentiment_label(rating):
    if pd.isna(rating):
        return "neutral"

    try:
        rating = float(rating)
        if rating >= 3.5:  # 3.5 ve üzeri pozitif
            return "positive"
        elif rating >= 1.5:  # 1.5-3.5 arası nötr
            return "neutral"
        elif rating > 0:  # 0-1.5 arası negatif
            return "negative"
        else:  # Rating yoksa
            return "unknown"
    except:
        return "unknown"


# Letterboxd veri setini işle
if not df_letterboxd.empty:
    # Review sütununu temizleme işi
    review_col_letterboxd = "Review" if "Review" in df_letterboxd.columns else None
    if review_col_letterboxd:
        df_letterboxd["cleaned_review"] = df_letterboxd[review_col_letterboxd].apply(
            clean_text
        )

    # Rating sütununu temizleme işi
    rating_col_letterboxd = "Rating" if "Rating" in df_letterboxd.columns else None
    if rating_col_letterboxd:
        df_letterboxd["numeric_rating"] = df_letterboxd[rating_col_letterboxd].apply(
            extract_letterboxd_rating
        )
        df_letterboxd["sentiment"] = df_letterboxd["numeric_rating"].apply(
            sentiment_label
        )

    # sourcee sütunu ekle
    df_letterboxd["source"] = "letterboxd"

    print("\nLetterboxd veri seti işlendi.")
    print(df_letterboxd[["numeric_rating", "sentiment", "cleaned_review"]].head())

    # Duygu dağılımını gösterme;;
    sentiment_counts_letterboxd = df_letterboxd["sentiment"].value_counts()
    print("\nLetterboxd Duygu Dağılımı:")
    print(sentiment_counts_letterboxd)

# Metacritic veri setini işleme işi;
if not df_metacritic.empty:
    # metacriticin içeriğini kontrol etme
    print("\nMetacritic örnek veriler:")
    print(df_metacritic.head())

    # Review sütununu belirle summary olmayabilir idk??
    review_col_metacritic = "summary" if "summary" in df_metacritic.columns else None

    if review_col_metacritic:
        # Review sütununu temizle
        df_metacritic["cleaned_review"] = df_metacritic[review_col_metacritic].apply(
            clean_text
        )

        if "User rating" in df_metacritic.columns:
            rating_col_metacritic = "User rating"
        elif "Rating" in df_metacritic.columns:
            rating_col_metacritic = "Rating"
        else:
            rating_col_metacritic = None

        if rating_col_metacritic:
            df_metacritic["numeric_rating"] = df_metacritic[
                rating_col_metacritic
            ].apply(extract_metacritic_rating)
            df_metacritic["sentiment"] = df_metacritic["numeric_rating"].apply(
                sentiment_label
            )

        # sourcee sütunu ekle
        df_metacritic["source"] = "metacritic"

        print("\nMetacritic veri seti işlendi.")
        print(df_metacritic[["numeric_rating", "sentiment", "cleaned_review"]].head())

        # Duygu dağılımını göster
        sentiment_counts_metacritic = df_metacritic["sentiment"].value_counts()
        print("\nMetacritic Duygu Dağılımı:")
        print(sentiment_counts_metacritic)
    else:
        print("\nMetacritic veri setinde review sütunu bulunamadı.")

df_letterboxd["topic"] = df_letterboxd["cleaned_review"].apply(topic_label)
df_metacritic["topic"] = df_metacritic["cleaned_review"].apply(topic_label)

# İki veri setini birleştirme
columns_to_keep = ["cleaned_review", "numeric_rating", "sentiment", "topic", "source"]

combined_reviews = pd.DataFrame()

if not df_letterboxd.empty and "cleaned_review" in df_letterboxd.columns:
    letterboxd_subset = df_letterboxd[columns_to_keep].copy()
    combined_reviews = pd.concat([combined_reviews, letterboxd_subset])

if not df_metacritic.empty and "cleaned_review" in df_metacritic.columns:
    metacritic_subset = df_metacritic[columns_to_keep].copy()
    combined_reviews = pd.concat([combined_reviews, metacritic_subset])

print("\nBirleştirilmiş veri seti boyutu:", combined_reviews.shape)

# ====== YENİ CLASSIFICATION BÖLÜMÜ ======

# Veri temizleme ve filtreleme
print("\n" + "=" * 50)
print("CLASSIFICATION MODELLERI EĞİTİLİYOR")
print("=" * 50)

# Boş yorumları ve bilinmeyen etiketleri kaldır
filtered_data = combined_reviews[
    (combined_reviews["cleaned_review"].str.len() > 10)  # En az 10 karakter
    & (combined_reviews["sentiment"] != "unknown")
    & (combined_reviews["topic"] != "unknown")
].copy()

print(f"Filtrelenmiş veri boyutu: {filtered_data.shape}")
print(f"Duygu dağılımı:\n{filtered_data['sentiment'].value_counts()}")
print(f"Konu dağılımı:\n{filtered_data['topic'].value_counts()}")

# 1. SENTIMENT CLASSIFICATION
print("\n" + "-" * 30)
print("1. SENTIMENT CLASSIFICATION")
print("-" * 30)

if len(filtered_data) > 50:  # Yeterli veri varsa
    # TF-IDF Vektörizasyon
    tfidf_sentiment = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # Unigram ve bigram
        min_df=2,
        max_df=0.95,
    )

    X_sentiment = tfidf_sentiment.fit_transform(filtered_data["cleaned_review"])
    y_sentiment = filtered_data["sentiment"]

    # Train-Test Split
    X_train_sent, X_test_sent, y_train_sent, y_test_sent = train_test_split(
        X_sentiment, y_sentiment, test_size=0.2, random_state=42, stratify=y_sentiment
    )

    # Birden fazla model dene
    sentiment_models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    best_sentiment_model = None
    best_sentiment_score = 0

    for name, model in sentiment_models.items():
        print(f"\n{name} eğitiliyor...")
        model.fit(X_train_sent, y_train_sent)
        y_pred_sent = model.predict(X_test_sent)
        accuracy = accuracy_score(y_test_sent, y_pred_sent)

        print(f"{name} Doğruluk: {accuracy:.4f}")
        print(
            f"\nClassification Report:\n{classification_report(y_test_sent, y_pred_sent)}"
        )

        if accuracy > best_sentiment_score:
            best_sentiment_score = accuracy
            best_sentiment_model = model
            best_sentiment_name = name

    print(
        f"\nEn iyi Sentiment Model: {best_sentiment_name} (Doğruluk: {best_sentiment_score:.4f})"
    )

    # Confusion Matrix görselleştirme
    y_pred_best_sent = best_sentiment_model.predict(X_test_sent)
    cm_sentiment = confusion_matrix(y_test_sent, y_pred_best_sent)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_sentiment,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=best_sentiment_model.classes_,
        yticklabels=best_sentiment_model.classes_,
    )
    plt.title(f"Sentiment Classification Confusion Matrix\n{best_sentiment_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("sentiment_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Model kaydet
    joblib.dump(best_sentiment_model, "best_sentiment_model.pkl")
    joblib.dump(tfidf_sentiment, "tfidf_sentiment_vectorizer.pkl")
    print("Sentiment modeli kaydedildi: best_sentiment_model.pkl")

# 2. TOPIC CLASSIFICATION
print("\n" + "-" * 30)
print("2. TOPIC CLASSIFICATION")
print("-" * 30)

if len(filtered_data) > 50:
    # TF-IDF Vektörizasyon (Topic için)
    tfidf_topic = TfidfVectorizer(
        max_features=3000, ngram_range=(1, 2), min_df=2, max_df=0.95
    )

    X_topic = tfidf_topic.fit_transform(filtered_data["cleaned_review"])
    y_topic = filtered_data["topic"]

    # Train-Test Split
    X_train_topic, X_test_topic, y_train_topic, y_test_topic = train_test_split(
        X_topic, y_topic, test_size=0.2, random_state=42, stratify=y_topic
    )

    # Topic Classification modelleri
    topic_models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    best_topic_model = None
    best_topic_score = 0

    for name, model in topic_models.items():
        print(f"\n{name} eğitiliyor...")
        model.fit(X_train_topic, y_train_topic)
        y_pred_topic = model.predict(X_test_topic)
        accuracy = accuracy_score(y_test_topic, y_pred_topic)

        print(f"{name} Doğruluk: {accuracy:.4f}")
        print(
            f"\nClassification Report:\n{classification_report(y_test_topic, y_pred_topic)}"
        )

        if accuracy > best_topic_score:
            best_topic_score = accuracy
            best_topic_model = model
            best_topic_name = name

    print(f"\nEn iyi Topic Model: {best_topic_name} (Doğruluk: {best_topic_score:.4f})")

    # Confusion Matrix görselleştirme
    y_pred_best_topic = best_topic_model.predict(X_test_topic)
    cm_topic = confusion_matrix(y_test_topic, y_pred_best_topic)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_topic,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=best_topic_model.classes_,
        yticklabels=best_topic_model.classes_,
    )
    plt.title(f"Topic Classification Confusion Matrix\n{best_topic_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.savefig("topic_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Model kaydet
    joblib.dump(best_topic_model, "best_topic_model.pkl")
    joblib.dump(tfidf_topic, "tfidf_topic_vectorizer.pkl")
    print("Topic modeli kaydedildi: best_topic_model.pkl")

# 3. CLASSIFICATION PREDICTION FUNCTION
print("\n" + "-" * 30)
print("3. PREDICTION FUNCTIONS")
print("-" * 30)


def predict_sentiment_and_topic(review_text):
    """
    Yeni bir yorum için sentiment ve topic tahmini yapar
    """
    try:
        # Metni temizle
        cleaned_text = clean_text(review_text)

        if len(cleaned_text) < 5:
            return "Çok kısa metin", "unknown"

        # Sentiment tahmini
        try:
            sentiment_model = joblib.load("best_sentiment_model.pkl")
            sentiment_vectorizer = joblib.load("tfidf_sentiment_vectorizer.pkl")

            text_vectorized_sent = sentiment_vectorizer.transform([cleaned_text])
            sentiment_pred = sentiment_model.predict(text_vectorized_sent)[0]
            sentiment_proba = sentiment_model.predict_proba(text_vectorized_sent)[
                0
            ].max()
        except:
            sentiment_pred = "model_error"
            sentiment_proba = 0

        # Topic tahmini
        try:
            topic_model = joblib.load("best_topic_model.pkl")
            topic_vectorizer = joblib.load("tfidf_topic_vectorizer.pkl")

            text_vectorized_topic = topic_vectorizer.transform([cleaned_text])
            topic_pred = topic_model.predict(text_vectorized_topic)[0]
            topic_proba = topic_model.predict_proba(text_vectorized_topic)[0].max()
        except:
            topic_pred = "model_error"
            topic_proba = 0

        return {
            "sentiment": sentiment_pred,
            "sentiment_confidence": f"{sentiment_proba:.2f}",
            "topic": topic_pred,
            "topic_confidence": f"{topic_proba:.2f}",
            "cleaned_text": cleaned_text,
        }

    except Exception as e:
        return f"Hata: {str(e)}"


# Test tahminleri
test_reviews = [
    "This movie has amazing cinematography and beautiful visual effects",
    "The acting was terrible and the story made no sense",
    "Great soundtrack and music, really enhanced the emotional scenes",
    "The director did an excellent job with character development",
]

print("\nTest Tahminleri:")
for i, review in enumerate(test_reviews, 1):
    result = predict_sentiment_and_topic(review)
    print(f"\n{i}. '{review}'")
    if isinstance(result, dict):
        print(
            f"   Sentiment: {result['sentiment']} (güven: {result['sentiment_confidence']})"
        )
        print(f"   Topic: {result['topic']} (güven: {result['topic_confidence']})")
    else:
        print(f"   Sonuç: {result}")

# ====== MEVCUT GÖRSELLEŞTIRME BÖLÜMÜ ======

# Görselleştirme - Konu dağılımı
plt.figure(figsize=(10, 6))
sns.countplot(x="topic", data=combined_reviews, hue="source")
plt.title("Film Yorumlarının Konu Dağılımı (Kaynaklara Göre)")
plt.xlabel("Konu")
plt.ylabel("Yorum Sayısı")
plt.xticks(rotation=45)
plt.savefig("topic_distribution.png", dpi=300, bbox_inches="tight")
plt.show()
print("Konu dağılımı grafiği 'topic_distribution.png' olarak kaydedildi.")

# Veri seti istatistikleri
if not combined_reviews.empty:
    print("\nBirleştirilmiş veri seti özeti:")
    print(f"Toplam yorum sayısı: {len(combined_reviews)}")
    print(f"Kaynak dağılımı:\n{combined_reviews['source'].value_counts()}")
    print(f"Duygu dağılımı:\n{combined_reviews['sentiment'].value_counts()}")

    # Boş olmayanları say
    non_empty_reviews = (
        combined_reviews["cleaned_review"].apply(lambda x: len(x) > 0).sum()
    )
    print(f"Boş olmayan yorum sayısı: {non_empty_reviews}")

    # Ortalama puanlar
    avg_rating = combined_reviews["numeric_rating"].mean()
    print(f"Ortalama rating: {avg_rating:.2f}/5")

    try:
        # Görselleştirme - Duygu dağılımı
        plt.figure(figsize=(10, 6))
        sns.countplot(x="sentiment", data=combined_reviews, hue="source")
        plt.title("Film Yorumlarının Duygu Dağılımı (Kaynaklara Göre)")
        plt.xlabel("Duygu")
        plt.ylabel("Yorum Sayısı")
        plt.savefig("sentiment_distribution.png", dpi=300, bbox_inches="tight")
        plt.show()
        print("Duygu dağılımı grafiği 'sentiment_distribution.png' olarak kaydedildi.")

        # Görselleştirme-Rating dağılımı;;;
        plt.figure(figsize=(10, 6))
        sns.histplot(data=combined_reviews, x="numeric_rating", hue="source", bins=10)
        plt.title("Film Puanlarının Dağılımı (Kaynaklara Göre)")
        plt.xlabel("Puan (5 üzerinden)")
        plt.ylabel("Yorum Sayısı")
        plt.savefig("rating_distribution.png", dpi=300, bbox_inches="tight")
        plt.show()
        print("Rating dağılımı grafiği 'rating_distribution.png' olarak kaydedildi.")
    except Exception as e:
        print(f"Görselleştirme hatası: {e}")

# Veri setini kaydet
if not combined_reviews.empty:
    combined_reviews.to_csv("combined_movie_reviews.csv", index=False)
    print("\nBirleştirilmiş veri seti 'combined_movie_reviews.csv' olarak kaydedildi.")

# ====== MEVCUT JSON EXPORT BÖLÜMÜ ======


# Letterboxd'den film bilgileri çıkarma
def extract_letterboxd_movies(df):
    movies = {}
    for _, row in df.iterrows():
        name = row["Movie name"]
        year = row["Release Year"]
        rating = row["Rating"]
        review = row["Review"]

        # Eğer film daha önce eklenmemişse, kaydet
        if name not in movies:
            movies[name] = {
                "movie_name": name,
                "release_year": year,
                "ratings": [],
                "reviews": [],
            }

        # Puanı sayısala dönüştürüp eklemek için basit kontrol
        try:
            rating_float = float(rating)
        except:
            rating_float = None

        if rating_float is not None:
            movies[name]["ratings"].append(rating_float)

        if isinstance(review, str) and review.strip() != "":
            movies[name]["reviews"].append(review.strip())

    # Ortalama puan hesapla
    for movie in movies.values():
        if movie["ratings"]:
            movie["average_rating"] = sum(movie["ratings"]) / len(movie["ratings"])
        else:
            movie["average_rating"] = None
        del movie["ratings"]  # Artık gerek yok

    return list(movies.values())


# Metacritic'den film bilgileri çıkarma
def extract_metacritic_movies(df):
    movies = {}
    for _, row in df.iterrows():
        name = row["Movie name"]
        release_date = row["Release Date"]
        rating = (
            row["User rating"]
            if "User rating" in df.columns
            else row.get("Rating", None)
        )
        summary = row["summary"] if "summary" in df.columns else ""

        # Yıl çekme
        year = None
        if isinstance(release_date, str):
            year = release_date.split("-")[0]

        if name not in movies:
            movies[name] = {
                "movie_name": name,
                "release_year": year,
                "ratings": [],
                "summaries": [],
            }

        try:
            rating_float = float(rating)
        except:
            rating_float = None

        if rating_float is not None:
            movies[name]["ratings"].append(rating_float)

        if isinstance(summary, str) and summary.strip() != "":
            movies[name]["summaries"].append(summary.strip())

    # Ortalama puan hesapla
    for movie in movies.values():
        if movie["ratings"]:
            movie["average_rating"] = sum(movie["ratings"]) / len(movie["ratings"])
        else:
            movie["average_rating"] = None
        del movie["ratings"]

    return list(movies.values())


letterboxd_movies = extract_letterboxd_movies(df_letterboxd)
metacritic_movies = extract_metacritic_movies(df_metacritic)


# na değerlerini null olarak dönüştürme işi bazı veriler nan olarak gözüküyor uymuyor;
class NaNHandlingEncoder(json.JSONEncoder):
    def default(self, obj):
        # NumPy ve Python'un float NaN değerlerini kontrol et
        if isinstance(obj, float) and (np.isnan(obj) or obj != obj):
            return None

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


# Letterboxd filmlerini kaydet
with open("letterboxd_movies.json", "w", encoding="utf-8") as f:
    json.dump(
        letterboxd_movies, f, indent=4, ensure_ascii=False, cls=NaNHandlingEncoder
    )

# Metacritic filmlerini kaydet
with open("metacritic_movies.json", "w", encoding="utf-8") as f:
    json.dump(
        metacritic_movies, f, indent=4, ensure_ascii=False, cls=NaNHandlingEncoder
    )

print("Filmler JSON dosyalarına kaydedildi.")

# ====== MEVCUT CHATBOT BÖLÜMÜ ======

import json

# JSON dosyasını oku
with open("metacritic_movies.json", "r", encoding="utf-8") as f:
    movies = json.load(f)

# Film adlarını küçük harfe çevirerek arama kolaylığı sağlamaa
movie_lookup = {m["movie_name"].lower(): m for m in movies}


# Basit chatbot fonksiyonu
def ask_about_movie(user_input):
    query = user_input.lower().strip()

    # Doğrudan film adını arar
    if query in movie_lookup:
        movie = movie_lookup[query]
        print(
            f"\n🎬 Movie: {movie['movie_name']} ({movie.get('release_year', 'Unknown')})"
        )
        print(f"⭐ Average User Rating: {movie.get('average_rating', 'N/A')}")
        print("📝 Summaries:")
        for summary in movie.get("summaries", []):
            print(f"- {summary}")
    else:
        print("Bu film veritabanında bulunamadı. Lütfen tam ismiyle tekrar deneyin.")


user_input = input("Bir film ismi yazın:")
ask_about_movie(user_input)

# ====== MODEL PERFORMANCE SUMMARY ======

print("\n" + "=" * 50)
print("MODEL PERFORMANS ÖZETİ")
print("=" * 50)
