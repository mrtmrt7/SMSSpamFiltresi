import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix
# Verimizi yüklüyoruz
df = pd.read_csv("TurkishSMSCollection.csv", sep=';')


# Verilerimizi temizliyoruz
df["Message"] = df["Message"].str.lower()
df["Message"] = df["Message"].str.translate(str.maketrans('', '', string.punctuation))
df["Message"] = df["Message"].str.replace(r'\s+', ' ', regex=True)

# Stopwords temizleme (Türkçedeki anlamsız kelimeleri çıkarma, örn: ve, veya)
import nltk
nltk.download('stopwords')
turkce_stopwords = set(stopwords.words('turkish'))
df["Message"] = df["Message"].apply(lambda x: ' '.join([
    word for word in x.split() if word not in turkce_stopwords
]))

# Veri dağılımını görselleştiriyoruz
plt.figure(figsize=(8, 6))
renk = ["red", "darkblue"]
sns.countplot(x="Group", data=df, hue="Group", palette=renk, dodge=False, legend=False)
plt.title("Veri Dağılımı")
plt.xlabel("Grup")
plt.ylabel("Mesaj Sayısı")
grup_sayilari = df["Group"].value_counts()
for i, sayi in enumerate(grup_sayilari):
    plt.text(i, sayi + 1, str(sayi), ha='center', va='bottom', fontsize=12, color="black")
plt.show()

# Eğitim ve test verisine bölüyoruz (%80 eğitim, %20 test)
X = df["Message"]
y = df["Group"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# TF-IDF ile metinleri sayısallaştırıyoruz
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

def modeli_degerlendir(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    dogruluk = accuracy_score(y_test, y_pred)
    kesinlik = precision_score(y_test, y_pred, average='weighted')
    duyarlilik = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return dogruluk, kesinlik, duyarlilik, f1

# Değerlendireceğimiz algoritmaların listesi
modeller = {
    "KNN": KNeighborsClassifier(),
    "LR": LogisticRegression(),
    "NB": MultinomialNB(),
    "DT": DecisionTreeClassifier(),
    "SVM": SVC(),
    "YSA": MLPClassifier(max_iter=500)
}

sonuclar = []

# Modelleri eğitme ve değerlendirme kısmımız
for isim, model in modeller.items():
    dogruluk, kesinlik, duyarlilik, f1 = modeli_degerlendir(model, X_train_tfidf, y_train, X_test_tfidf, y_test)
    sonuclar.append({
        "Model": isim,
        "Accuracy": dogruluk,
        "Precision": kesinlik,
        "Recall": duyarlilik,
        "F1 Skoru": f1
    })
    # Confusion Matrix'i oluşturduk
    y_pred = model.predict(X_test_tfidf)
    cm = confusion_matrix(y_test, y_pred)

    # Confusion Matrix'i görselleştirdik
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title(f"{isim} - Confusion Matrix")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.show()
# Sonuçları bir DataFrame olarak gösteriyoruz
sonuclar_df = pd.DataFrame(sonuclar)

# Tüm modellerin doğruluk metriklerini karşılaştırma tablosu
print(sonuclar_df)

# Grafiksel olarak karşılaştırma
renkler = ['red', 'darkblue', 'green', 'orange']
sonuclar_df.set_index("Model", inplace=True)
sonuclar_df.plot(kind='bar', figsize=(14, 8), color=renkler)
plt.title("Modellerin Performans Karşılaştırması")
plt.ylabel("Değer")
plt.xlabel("Model")
plt.xticks(rotation=45)
plt.legend(loc="lower right")
plt.show()

# Kelime Bulutu (WordCloud) oluşturma
cleaned_text = ' '.join(df["Message"])
wordcloud = WordCloud(width=1280, height=720, background_color='white').generate(cleaned_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Kelime Bulutu", fontsize=15)
plt.show()



# Rastgele 2 mesaj seçip tahmin yapma
rastgele_mesajlar = random.sample(list(df["Message"]), 2)

# Mesajları TF-IDF'e dönüştürme
rastgele_mesajlar_tfidf = tfidf.transform(rastgele_mesajlar)

# Modelin tahminleri
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
tahminler = model.predict(rastgele_mesajlar_tfidf)

# Tahmin sonuçlarını gösteriyoruz
for mesaj, tahmin in zip(rastgele_mesajlar, tahminler):
    if tahmin == 1:
        print(f"Rastgele Seçilen Mesaj: '{mesaj}' : Spam")
    else:
        print(f"Rastgele Seçilen Mesaj: '{mesaj}' : Normal")
print(df.head(5))