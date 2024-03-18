import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy import stats
import numpy as np
from scipy.stats import f_oneway, ttest_ind
from sklearn.cluster import KMeans

veri = pd.read_csv("musteri.csv")
musteri_sayisi = len(veri)
ortalama_yas = veri["Doğum Tarihi"].apply(lambda x: 2023 - int(x.split("-")[0])).mean()
cinsiyet_dağılımı = veri["Cinsiyet"].value_counts()
veri.dropna(inplace=True)

en_cok_satan_urunler = veri["Ürünler"].str.split(",").explode().str.strip().value_counts().head(5)
en_yuksek_siparis_sayisi = veri.nlargest(5, "Sipariş Sayısı")
veri["Üyelik Tarihi"] = pd.to_datetime(veri["Üyelik Tarihi"])
uyelik_tarihi_analiz = veri.groupby(veri["Üyelik Tarihi"].dt.year)["Müşteri ID"].count()

erkekler_harcama = veri[veri["Cinsiyet"] == "Erkek"]["Toplam Harcama"]
kadinlar_harcama = veri[veri["Cinsiyet"] == "Kadın"]["Toplam Harcama"]
t_statistik, p_deger = stats.ttest_ind(erkekler_harcama, kadinlar_harcama)

X = veri[["Sipariş Sayısı"]]
y = veri["Toplam Harcama"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_lineer = LinearRegression()
model_lineer.fit(X_train, y_train)
y_pred_lineer = model_lineer.predict(X_test)
rmse_lineer = mean_squared_error(y_test, y_pred_lineer, squared=False)

model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)

plt.figure(figsize=(8, 6))
plt.hist(veri["Doğum Tarihi"].apply(lambda x: 2023 - int(x.split("-")[0])), bins=10, color='skyblue')
plt.title("Müşteri Yaş Dağılımı")
plt.xlabel("Yaş")
plt.ylabel("Müşteri Sayısı")
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(data=veri, x="Cinsiyet", palette="Set2")
plt.title("Müşteri Cinsiyet Dağılımı")
plt.xlabel("Cinsiyet")
plt.ylabel("Müşteri Sayısı")
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=veri, x="Sipariş Sayısı", y="Toplam Harcama", hue="Cinsiyet", palette="Set1")
plt.title("Sipariş Sayısı ile Toplam Harcama Karşılaştırması")
plt.xlabel("Sipariş Sayısı")
plt.ylabel("Toplam Harcama")
plt.legend(title="Cinsiyet")
plt.show()

print(f"Müşteri Sayısı: {musteri_sayisi}")
print(f"Ortalama Yaş: {ortalama_yas:.2f}")
print("Cinsiyet Dağılımı:\n", cinsiyet_dağılımı)
print("En Çok Satan Ürünler:\n", en_cok_satan_urunler)
print("En Yüksek Sipariş Sayısına Sahip Müşteriler:\n", en_yuksek_siparis_sayisi)
print(f"Harcama Cinsiyet Testi - T İstatistiği: {t_statistik:.2f}")
print(f"Harcama Cinsiyet Testi - p-Değeri: {p_deger:.4f}")
print("RMSE (Root Mean Squared Error) Değeri (Lineer Regresyon):", rmse_lineer)
print("RMSE (Root Mean Squared Error) Değeri (Random Forest Regresyon):", rmse_rf)

korelasyon_matrisi = veri.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(korelasyon_matrisi, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Değişkenler Arası Korelasyon Matrisi")
plt.show()

kategori_grupları = veri["Ürün Kategorisi"].unique()
kategori_anova = {kategori: veri[veri["Ürün Kategorisi"] == kategori]["Toplam Harcama"] for kategori in kategori_grupları}
f_statistik, p_deger = f_oneway(*kategori_anova.values())

print("ANOVA İstatistiği (F):", f_statistik)
print("p-Değeri:", p_deger)

veri["Yaş"].fillna(veri["Yaş"].mean(), inplace=True)

veri = pd.get_dummies(veri, columns=["Ürün Kategorisi"], drop_first=True)

kategori_grupları = veri["Ürün Kategorisi"].unique()
kategori_testleri = {kategori: veri[veri["Ürün Kategorisi"] == kategori]["Toplam Harcama"] for kategori in kategori_grupları}

kategori_1 = "Kategori A"
kategori_2 = "Kategori B"

t_statistik, p_deger = ttest_ind(kategori_testleri[kategori_1], kategori_testleri[kategori_2])

print(f"{kategori_1} ile {kategori_2} arasındaki Harcama Farkı Testi:")
print("T İstatistiği:", t_statistik)
print("p-Değeri:", p_deger)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Gerçek Değerler')
plt.plot(X_test, y_pred_lineer, color='red', linewidth=2, label='Lineer Regresyon Tahmini')
plt.plot(X_test, y_pred_rf, color='green', linewidth=2, label='Random Forest Regresyon Tahmini')
plt.title("Toplam Harcama Tahminleri")
plt.xlabel("Sipariş Sayısı")
plt.ylabel("Toplam Harcama")
plt.legend()
plt.show()

X_segmentasyon = veri[["Sipariş Sayısı", "Toplam Harcama"]]
kmeans = KMeans(n_clusters=3, random_state=42)
veri["Segment"] = kmeans.fit_predict(X_segmentasyon)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=veri, x="Sipariş Sayısı", y="Toplam Harcama", hue="Segment", palette="viridis")
plt.title("Müşteri Segmentasyonu")
plt.xlabel("Sipariş Sayısı")
plt.ylabel("Toplam Harcama")
plt.show()
