# ml-dl-ogreniyom

```markdown
# ml-dl-ogreniyom

**Proje Açıklaması**: Bu Python betiği, bir müşteri veri seti üzerinde veri analizi, görselleştirme, hipotez testleri ve makine öğrenimi tahminleri yapmak için kullanılan bir çalışmadır.

## İçindekiler

1. [Kurulum](#kurulum)
2. [Kullanım](#kullanım)
3. [Proje Yapısı](#proje-yapısı)
4. [Bağımlılıklar](#bağımlılıklar)
5. [Lisans](#lisans)

## Kurulum

Proje kurulumu için aşağıdaki adımları takip edebilirsiniz:

1. Bu projeyi klonlayın veya ZIP olarak indirin.

2. Python 3.7 veya daha yeni bir sürümünün bilgisayarınızda yüklü olduğundan emin olun.

3. Gerekli bağımlılıkları yüklemek için aşağıdaki komutu kullanın:

```shell
pip install -r requirements.txt
```

4. Projeyi çalıştırmak için aşağıdaki komutu kullanabilirsiniz:

```shell
python formusteri.py
```

## Kullanım

Bu proje, bir müşteri veri seti üzerinde çeşitli analizleri yapmak ve sonuçları görselleştirmek için kullanılır. Proje başlıca şunları içerir:

### Adım 1: Veri Hazırlığı
- Veri setini yükler.
- Eksik verileri temizler.
- Temel veri istatistiklerini hesaplar.

### Adım 2: Veri Keşfi ve İleri Analiz
- En çok satan ürünleri bulur.
- En yüksek sipariş sayısına sahip müşterileri belirler.
- Müşteri üyelik tarihlerini analiz eder.

### Adım 3: Hipotez Testleri
- Cinsiyetlere göre harcama farkını test eder.

### Adım 4: Makine Öğrenimi Tahminleri (Random Forest)
- Toplam harcamayı sipariş sayısıyla tahmin eder.

### Adım 5: Veri Görselleştirme
- Müşteri yaş dağılımını görselleştirir.
- Müşteri cinsiyet dağılımını görselleştirir.
- Sipariş sayısı ile toplam harcamayı karşılaştırır.

### Adım 6: Raporlama
- Temel projeyi raporlar.

### Adım 7: İleri Düzey İstatistik Analizi
- Korelasyon analizi yapar.
- Varyans analizi (ANOVA) yapar.

### Adım 8: Veri Temizleme ve Dönüştürme İşlemleri
- Eksik verileri yönetir.
- Kategorik değişkenleri kodlar.

### Adım 9: Diğer Hipotez Testleri veya A/B Testleri
- Ürün kategorilerine göre harcama farklarını test eder.

## Proje Yapısı

Proje, aşağıdaki dosya ve klasörlerden oluşur:

- `formusteri.py`: Projenin ana Python betiği.
- `musteri.csv`: Kullanılan müşteri veri seti.
- `requirements.txt`: Bağımlılıkları içeren dosya.
- `README.md`: Proje açıklamaları ve kullanım kılavuzu.

## Bağımlılıklar

Bu projenin çalışması için aşağıdaki Python kütüphanelerinin yüklü olması gerekmektedir:

- pandas
- matplotlib
- seaborn
- scikit-learn
- scipy
- numpy

Gerekli bağımlılıkları yüklemek için yukarıdaki "Kurulum" bölümünü takip edebilirsiniz.

