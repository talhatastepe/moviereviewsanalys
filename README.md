Film Yorum Analizi ve Makine Öğrenmesi ile Sınıflandırma
🎬 Proje Tanımı
Bu proje, film yorumlarını doğal dil işleme (NLP) ve makine öğrenimi teknikleri kullanarak otomatik olarak duygu (olumlu, olumsuz, nötr) ve konu (oyunculuk, senaryo, görsellik, müzik, yönetmenlik) bazında sınıflandırmayı amaçlamaktadır. Temel hedefimiz, büyük veri kümelerindeki yorumları hızlı ve etkili bir şekilde analiz ederek, filmler hakkındaki genel algıyı ve yorumların hangi spesifik alanlara odaklandığını anlamaktır.

Bu tür bir analiz, hem film yapımcıları için geri bildirim sağlama, hem izleyiciler için film seçimi yapma, hem de pazarlama stratejilerini belirleme açısından büyük önem taşımaktadır.

✨ Temel Özellikler
Veri Toplama ve Ön İşleme: Letterboxd ve Metacritic platformlarından film yorumları ve derecelendirmeleri alınarak kapsamlı bir temizleme ve normalizasyon işleminden geçirilir.
Duygu Analizi (Sentiment Analysis): Film yorumlarının pozitif, negatif veya nötr olduğunu belirlemek için makine öğrenmesi modelleri kullanılır.
Konu Sınıflandırması (Topic Classification): Yorumların hangi film unsuruna (örneğin, senaryo, oyunculuk, görsellik, müzik, yönetmenlik) odaklandığını belirlemek için modeller eğitilir.
Model Eğitimi ve Değerlendirme: TF-IDF vektörizasyonu ile metinler sayısal formata dönüştürülür ve Multinomial Naive Bayes, Logistic Regression, RandomForestClassifier gibi farklı makine öğrenimi algoritmaları eğitilir ve performansları karşılaştırılır.
Görselleştirme: Duygu, konu ve rating dağılımları gibi önemli istatistikler grafiklerle görselleştirilir.
Model Kaydı: Eğitilen en iyi duygu ve konu sınıflandırma modelleri ile TF-IDF vektörleyicileri .pkl formatında kaydedilir.
JSON Dışa Aktarımı: Film bilgileri ve yorum özetleri JSON formatında dışa aktarılır.
Basit Chatbot Uygulaması: Kullanıcılardan film adı alarak ilgili filmin ortalama puanını ve yorum özetlerini sunan basit bir etkileşimli arayüz.
🚀 Kurulum ve Kullanım
Bu projeyi yerel ortamınızda çalıştırmak için aşağıdaki adımları izleyin:

Ön Gereksinimler
Python 3.x
pip (Python paket yöneticisi)
Bağımlılıkların Yüklenmesi
Proje bağımlılıklarını yüklemek için aşağıdaki komutu çalıştırın:

Bash

pip install pandas scikit-learn nltk matplotlib seaborn numpy jupyterlab
Eğer nltk stopwords indirmediyseniz, kodunuzdaki nltk.download("stopwords") komutu bunu otomatik olarak yapacaktır.

Veri Setleri
Proje, yerel diskinizdeki CSV dosyalarını kullanır. moviereviewsdataset klasöründeki letterboxd-reviews.csv ve metacritic-reviews.csv dosyalarını belirtilen yola yerleştirmeniz gerekmektedir:

C:/Users/ACER/Downloads/moviereviewsdataset/letterboxd-reviews.csv
C:/Users/ACER/Downloads/moviereviewsdataset/metacritic-reviews.csv
Not: Dosya yollarını kendi sisteminize göre ayarlamanız gerekebilir. (df_letterboxd = pd.read_csv("path/to/your/letterboxd-reviews.csv", ...) ve benzer şekilde df_metacritic için.)

Projeyi Çalıştırma
Ana komut dosyası tüm veri işleme, model eğitimi, değerlendirme, görselleştirme ve dışa aktarma adımlarını otomatik olarak çalıştırır.

Proje klasörüne gidin.

Ana Python betiğini çalıştırın:

Bash

python your_project_name.py # veya dosyanızın adı main.py olabilir
(Eğer kodu bir Jupyter Notebook/Lab ortamında çalıştırıyorsanız, hücreleri sırayla çalıştırmanız yeterlidir.)

Komut dosyası tamamlandığında şunları göreceksiniz:

Konsol çıktıları ile veri yükleme, işleme, model eğitimi ve performans raporları.
Proje dizininde oluşturulan görselleştirmeler (.png dosyaları).
Kaydedilen modeller ve vektörleyiciler (.pkl dosyaları).
Birleştirilmiş temizlenmiş yorumlar (combined_movie_reviews.csv).
JSON formatında dışa aktarılmış film verileri (letterboxd_movies.json, metacritic_movies.json).
Basit chatbot etkileşimi için bir kullanıcı girdisi istemi.
📊 Proje Çıktıları ve Analizler
Projenin çıktıları arasında şunlar bulunmaktadır:

sentiment_confusion_matrix.png: Duygu sınıflandırma modelinin performansını gösteren karışıklık matrisi.
topic_confusion_matrix.png: Konu sınıflandırma modelinin performansını gösteren karışıklık matrisi.
topic_distribution.png: Film yorumlarının konulara göre dağılımını gösteren grafik.
sentiment_distribution.png: Film yorumlarının duyguya göre dağılımını gösteren grafik.
rating_distribution.png: Film puanlarının dağılımını gösteren grafik.
best_sentiment_model.pkl: Eğitilmiş en iyi duygu sınıflandırma modeli.
tfidf_sentiment_vectorizer.pkl: Duygu sınıflandırması için kullanılan TF-IDF vektörleyici.
best_topic_model.pkl: Eğitilmiş en iyi konu sınıflandırma modeli.
tfidf_topic_vectorizer.pkl: Konu sınıflandırması için kullanılan TF-IDF vektörleyici.
combined_movie_reviews.csv: Temizlenmiş ve birleştirilmiş tüm film yorumları.
letterboxd_movies.json: Letterboxd'dan çıkarılan film bilgileri.
metacritic_movies.json: Metacritic'den çıkarılan film bilgileri.
