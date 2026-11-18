# Çok Kademeli Tedarik Zincirinde Kamçı Etkisini Önleyen Sipariş Politikası
### Tablo tabanlı Çok Ajanlı Pekiştirmeli Öğrenme (Q-öğrenme & SARSA) — derin öğrenme yok

## 1. Problem
Müşteri → **Ana Depo** → **Fabrika** → **Hammadde Tedarikçisi** → dış kaynak.
Teslimat gecikmeleri (lead time) ve talep belirsizliği yüzünden müşteri talebindeki küçük
artışlar yukarı kademelere doğru büyür (**kamçı etkisi / bullwhip effect**): tedarikçi bazen
devasa sipariş verir, bazen hiç vermez; zincir bir yanda stok fazlası, öbür yanda yok satma yaşar.

## 2. Veri: sentetik simülasyon
Gerçek veri seti olmadığından, literatürdeki "Bira Oyunu" (Beer Game) yaklaşımı gibi
ortamı simülasyonla üretiyoruz (`supply_chain_env.py`). Her bölüm = 52 hafta.

| Bileşen | Model |
|---|---|
| Müşteri talebi | Poisson( taban 10 + mevsimsellik ±2.5 (26 hf) + hafif trend ), %70 olasılıkla 8–37. haftalar arasında kalıcı **%40–80 talep sıçraması** |
| Teslimat gecikmesi | Her bağlantı için 2 rejimli Markov zinciri: **Normal** {1,2,3} hf, **Kriz** {3,4,5} hf (krize giriş %4, çıkış %25 / hafta) |
| Bilgi gecikmesi | Bir kademenin siparişini üst kademe 1 hafta sonra görür |
| Karşılanamayan talep | Backorder olarak birikir, sonraki haftalarda karşılanır |

Eğitim, doğrulama ve test bölümleri **farklı rastgele tohumlarla** üretilir; test sonuçları
ajanın hiç görmediği 300 senaryo üzerindedir.

## 3. MDP tasarımı
Her kademe bir ajandır (**Çok Ajanlı RL, Independent Learners**). Her ajanın kendi Q-tablosu var.

**Durum** (ayrıklaştırılmış, 8×3×2×2 = **96 durum**, `agents.py › discretize`):

| Özellik | Tanım | Aralık sayısı |
|---|---|---|
| Stok pozisyonu açığı | `eldeki stok − backorder + yoldaki siparişler` − `tahmin × (gecikme tahmini+1)` | 8 |
| Talep trendi | (son talep − üstel düzeltilmiş tahmin) / tahmin → düşüş / sabit / artış | 3 |
| Tedarikçi gecikme tahmini | gözlenen teslimat sürelerinin üstel ortalaması → normal / gecikmeli | 2 |
| Backorder var mı | evet / hayır | 2 |

**Eylem** (7 ayrık eylem): `sipariş = talep tahmini + Δ`, Δ ∈ {−8, −4, −2, 0, +2, +4, +8}.
Bu "tahmin + düzeltme" kurgusu eylem uzayını küçük tutar ve tablo yöntemlerini mümkün kılar.

**Ödül** (kademe *i*, hafta *t*):

```
r_i = p_i·gönderilen − h_i·stok − b_i·backorder − λ·|o_t − o_{t−1}|
       (gelir)          (tutma)     (yok satma)      (kamçı/dalgalanma cezası)
```
p = (4, 3, 2), h = (0.5, 0.4, 0.3), b = (2.0, 1.6, 1.2), λ = 0.25. Yok satma cezası her kademede stok tutma maliyetinin **4 katı**dır.

Çok ajanlı işbirliği için **hibrit ödül** kullanılır: `r̃_i = r_i + 0.5 · Σ_{j≠i} r_j`.
Böylece her kademe kendi kârının yanında zincirin geri kalanını da önemser
(`--reward local|team|hybrid` ile değiştirilebilir).

## 4. Yöntemler (derin öğrenme yok)
* **Q-öğrenme** (off-policy): `Q(s,a) ← Q + α[r + γ·max_a' Q(s',a') − Q]`
* **SARSA** (on-policy): `Q(s,a) ← Q + α[r + γ·Q(s',a') − Q]`
* γ = 0.95, ε-açgözlü keşif (1.0 → 0.02 doğrusal), α = 0.1 / (1 + 0.002·ziyaret sayısı), α ≥ 0.01
* 6000 eğitim bölümü (≈ 312 bin adım/ajan), CPU'da ≈ 1.5 dk
* **Doğrulama tabanlı kontrol noktası:** bağımsız ajanlar birbirini sürekli değiştirdiği için
  ortam her ajan açısından durağan değildir ve politika salınım yapar. Her 100 bölümde
  açgözlü politika 40 doğrulama senaryosunda ölçülür, en iyi Q-tabloları saklanır.

**Karşılaştırma (baz) politikaları:**
* *Pass-through* — gelen talebi aynen iletir (stok yönetimi yok).
* *Order-up-to (base-stock)* — endüstride yaygın; `sipariş = S − stok pozisyonu`.
* *Yumuşatılmış order-up-to* — Sterman'ın çapalama-ayarlama sezgiseli; açığın %30'unu kapatır.

## 5. Sonuçlar

Base-stock kuralında müşteri talebindeki dalgalanma tedarikçiye ulaşana kadar **32 kat** büyüyor;
RL ajanları bunu **2,2–2,6 katta** tutuyor (≈%92 azalma). Sonuçlar 300 görülmemiş test senaryosunda,
RL için **5 farklı eğitim tohumunun ortalaması ± %95 güven aralığı** olarak verilmiştir
(`experiments.py`, ayrıntılar `results/deneyler.json`).

| Politika | Zincir kârı | Müşteri dolum oranı | Kamçı oranı Depo | Fabrika | Tedarikçi | Stok tutma | Yok satma | Dalgalanma |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Q-öğrenme (5 tohum)** | **3341 ± 68** | **%94,6 ± 0,9** | 1,40 | 1,91 | 2,60 | 2117 | **535** | 189 |
| **SARSA (5 tohum)** | 3270 ± 113 | %94,0 ± 2,0 | 1,46 | **1,55** | **2,23** | 2157 | 538 | 150 |
| Yumuşatılmış order-up-to | 2729 | %84,3 | 1,71 | 3,78 | 7,85 | 1451 | 1592 | 131 |
| Pass-through | 2414 | %76,6 | 0,98 | 0,94 | 0,90 | 627 | 2371 | 145 |
| Order-up-to (base-stock) | 1890 | %74,8 | 3,90 | 11,99 | **31,92** | 1184 | 2155 | 554 |

*Kamçı oranı = Var(kademenin siparişleri) / Var(müşteri talebi); 1 = hiç büyütme yok.
Maliyetler 52 haftalık bölüm başına zincir toplamıdır.*

**Ödül paylaşımı deneyi** (local / team / hybrid, 3 tohum): kâr 3229–3380 aralığında ve güven
aralıkları örtüşüyor; ödül paylaşım biçimi sonucu anlamlı biçimde değiştirmiyor.

## 6. Neyi sağlıyor, neyi daha iyi yapıyor

RL ajanları, klasik kuralların birbirinin yerine seçmek zorunda kaldığı iki hedefi aynı anda
tutturuyor: düşük kamçı etkisi ve yüksek müşteri hizmeti.

| Karşılaştırma | RL'nin kazancı (Q-öğrenme, 5 tohum) | Nedeni |
|---|---|---|
| Base-stock'a göre | Kâr **+%77** (1890 → 3341), dolum %74,8 → %94,6, tedarikçi kamçı oranı 31,9 → 2,6 (−%92), yok satma cezası −%75 | Base-stock her stok açığını tek seferde kapatır, tahmindeki küçük değişimi (gecikme + 1) katı büyütür. Ajan açığı kademeli kapatmayı öğreniyor. |
| Yumuşatılmış order-up-to'ya göre | Kâr **+%22** (2729 → 3341), dolum +10 puan, tedarikçi kamçı oranı 7,9 → 2,6 | Sezgisel kural sabit bir katsayıyla (%30) düzeltir; ajan düzeltmeyi talep trendine, gecikme krizine ve backorder durumuna göre seçer. |
| Pass-through'a göre | Kâr **+%38**, dolum %76,6 → %94,6 | Pass-through büyütme yapmaz ama stok yönetmediği için talep sıçramasında yok satmaya düşer. |

* **Derin öğrenme yok:** ajan başına 96 × 7 = 672 sayılık bir tablo; politika doğrudan okunabilir.
* **Ucuz eğitim:** tek tohum ≈ 50 sn, dizüstü bilgisayarda, GPU gerekmez.
* **Belirsizliğe dayanıklı:** görülmemiş talep şokları ve gecikme krizlerinde 5 tohumda tutarlı.
* **Bedeli:** RL daha fazla güvenlik stoğu tutuyor; stok tutma maliyeti base-stock'tan ≈%79 yüksek
  (1184 → 2117). Yok satma cezası stok maliyetinin 4 katı olduğu için bu kârlı bir takas;
  maliyet oranı değişirse ajan yeniden eğitilmelidir.

Grafikler `results/` klasöründe (`main.py`'nin tek tohumlu koşusundan, tohum 0):
1. `1_ogrenme_egrisi.png` — eğitim ve doğrulama kârı
2. `2_kamci_orani.png` — kademe bazında kamçı oranı (log ölçek)
3. `3_ornek_bolum.png` — aynı senaryoda base-stock vs Q-öğrenme sipariş ve net stok yörüngeleri
4. `4_maliyet_kirilimi.png` — stok / yok satma / dalgalanma maliyetleri

## 7. Sınırlılıklar
* **Sentetik veri:** talep ve gecikme parametreleri gerçek bir şirketten kalibre edilmedi.
* **Baz politikalar ayarlanmadı:** güvenlik stoğu (1 haftalık talep) ve katsayı (0,3) sabit;
  en iyi ayarlanmış sezgisel kuralla kıyas daha adil olur.
* **Durağan olmayan çok ajanlı öğrenme:** uzun eğitimde politika salınıyor; doğrulama tabanlı
  kontrol noktası bunu çözüyor ama yakınsama garantisi yok.
* **Kaba durum uzayı:** 96 durum, aynı açık aralığındaki farklı durumları ayırt edemiyor.
* **Tek ürün, doğrusal zincir:** kapasite kısıtı, çoklu ürün ve birden fazla tedarikçi yok.

## 8. Kullanılabilecek gerçek veri setleri
Hazır bir "kamçı etkisi" veri seti yok; gerçek veri simülatörün talep ve gecikme süreçlerini beslemek için kullanılır.

| Veri seti | Projede neyi besler | Erişim |
|---|---|---|
| [DataCo Smart Supply Chain](https://data.mendeley.com/datasets/8gx2fvg2k6/5) | Talep + gerçek/planlanan sevkiyat günü → gecikme dağılımı (**en uygun aday**) | Mendeley/Kaggle, CC BY 4.0 |
| [M5 Forecasting (Walmart)](https://www.kaggle.com/c/m5-forecasting-accuracy/data) | Hiyerarşik talep (ürün → mağaza → eyalet) | Kaggle |
| [Store Item Demand Forecasting](https://www.kaggle.com/c/demand-forecasting-kernels-only) | Temiz günlük talep, 10 mağaza × 50 ürün | Kaggle |
| [Online Retail II (UCI)](https://archive.ics.uci.edu/dataset/502/online+retail+ii) | Toptan satış faturalarından haftalık talep | UCI, CC BY 4.0, hesap gerekmez |
| [Olist Brezilya E-Ticaret](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) | Tahmini ve gerçek teslim tarihleri → gecikme | Kaggle |
| [OR-Gym InvManagement](https://github.com/hubbs5/or-gym) | Veri değil; literatürdeki çok kademeli envanter ortamıyla kıyas | GitHub, MIT |

## 9. Sonraki adımlar
- [ ] DataCo'dan haftalık talep ve ampirik gecikme dağılımı çıkaran yükleyici (`DemandGenerator`, `LeadTimeProcess` yerine)
- [ ] Baz politikaların parametrelerini ızgara aramasıyla ayarla, RL'yi en iyi sezgisele karşı kıyasla
- [ ] λ'yı 0–1 aralığında tarayıp maliyet–kamçı etkisi Pareto eğrisini çiz
- [ ] Ardışık eğitim: diğer ajanlar sabitken bir ajanı eğit, sırayla dön
- [ ] Tile coding ile doğrusal fonksiyon yaklaşımı (hâlâ derin öğrenme değil)
- [ ] Senaryo bazlı analiz: yalnız talep şoku / yalnız gecikme krizi / ikisi birlikte

## 10. Çalıştırma
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python main.py                      # tek tohum, tam deney + grafikler (~2 dk)
.venv/bin/python main.py --episodes 1000      # hızlı deneme
.venv/bin/python main.py --reward team        # ödül paylaşımı seçimi
.venv/bin/python experiments.py               # 5 tohum + ödül ablasyonu, paralel (~5 dk)
```
Her iki betik de `results/` içindeki dosyaların üzerine yazar.

## 11. Dosyalar
| Dosya | İçerik |
|---|---|
| `supply_chain_env.py` | Simülasyon ortamı, sentetik talep ve gecikme süreçleri |
| `agents.py` | Durum ayrıklaştırma, Q-öğrenme/SARSA ajanı, baz politikalar |
| `main.py` | Eğitim, kontrol noktası, test, grafikler |
| `experiments.py` | Çoklu tohum ve ödül paylaşımı deneyleri |
| `results/` | `sonuclar.json` (tek tohum), `deneyler.json` (5 tohum), Q-tabloları, grafikler, loglar |

## 12. Ekstra
| `Rehber İçerik` | https://drive.google.com/file/d/1WUMEBiarsVhhcx5s0gTYxx6oEWXXx0dJ/view |
