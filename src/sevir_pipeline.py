"""
sevir_pipeline.py  --  Oraj Tespiti (TUBITAK 2209-A)

Duzeltilen hatalar:
  1) VIS/IR indeks karisikligi  -> her ornek kendi kanalinin file_index'i ile okunur
  2) Mevsimsel confound          -> her iki sinif da ayni zaman penceresine kisitlanir
  3) Train/test sizintisi        -> bolme event 'id' bazinda yapilir ve diske kaydedilir
  4) IR normalizasyonu           -> SEVIR olcekleme (santigrat x 100) dogru uygulanir
  5) Sessiz veri bozulmasi       -> gecersiz satirlar indeks kurulurken elenir

Kullanim:
    from sevir_pipeline import *
    idx = build_index()                 # ne kadar veri var, once bunu calistirin
    split = make_split(idx)
    train_gen, val_gen = make_generators(idx, split)
    model = build_model('resnet')
    train(model, train_gen, val_gen, name='resnet')
"""

import os
import json
import numpy as np
import pandas as pd
import h5py
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, applications

# --------------------------------------------------------------------------
# Konfigurasyon
# --------------------------------------------------------------------------

DATA_DIR = 'data/sevir'
CATALOG_PATH = os.path.join(DATA_DIR, 'CATALOG.csv')
SPLIT_PATH = 'split_event_level.json'

# SEVIR resmi olcekleme katsayilari (MIT-AI-Accelerator/neurips-2020-sevir)
#   ir069 / ir107 -> santigrat derece
#   vis           -> reflektans
SEVIR_SCALE = {'vis': 1 / 10000.0, 'ir069': 1 / 100.0, 'ir107': 1 / 100.0}

# SEVIR kanal cozunurlukleri  (N, H, W, T=49)
CHANNEL_DIM = {'vis': (768, 768), 'ir069': (192, 192), 'ir107': (192, 192)}

# HER IKI SINIF ICIN ORTAK zaman penceresi. Mevsimsel kisayolu bu engelliyor.
TIME_START = '2018-06-01'
TIME_END = '2018-07-01'

N_FRAMES = 49  # SEVIR her olayda 49 zaman adimi tutar


# --------------------------------------------------------------------------
# 1. Indeks kurulumu
# --------------------------------------------------------------------------

def build_index(data_dir=DATA_DIR, channel='ir107',
                time_start=TIME_START, time_end=TIME_END,
                validate=True, verbose=True):
    """Egitilebilir orneklerin listesini kurar.

    Tek kanal uzerinden calisir: her satir kendi dosyasinin kendi file_index'ini
    tasir, boylece VIS/IR indeks karisikligi yapisal olarak imkansiz hale gelir.
    """
    cat = pd.read_csv(CATALOG_PATH, parse_dates=['time_utc'], low_memory=False)
    n0 = len(cat)

    cat = cat[cat.img_type == channel].copy()

    # Diskte gercekten olan dosyalar
    on_disk = {f for f in cat['file_name'].unique()
               if os.path.exists(os.path.join(data_dir, f))}
    cat = cat[cat['file_name'].isin(on_disk)]
    n_disk = len(cat)

    # ORTAK zaman penceresi -- kritik adim
    cat = cat[(cat.time_utc >= time_start) & (cat.time_utc < time_end)]
    n_time = len(cat)

    cat['label'] = cat['file_name'].str.contains('STORMEVENTS').astype(int)
    cat = cat.drop_duplicates(subset=['id']).reset_index(drop=True)

    # Gecersiz file_index'leri simdi ele -- egitim sirasinda sessizce sifir
    # doldurmak yerine burada temizliyoruz
    if validate and len(cat):
        keep = np.ones(len(cat), dtype=bool)
        for fname, grp in cat.groupby('file_name'):
            with h5py.File(os.path.join(data_dir, fname), 'r') as hf:
                n_events = hf[channel].shape[0]
            bad = grp.index[grp['file_index'] >= n_events]
            keep[bad] = False
        n_bad = (~keep).sum()
        cat = cat[keep].reset_index(drop=True)
        if verbose and n_bad:
            print(f"  {n_bad} satir gecersiz file_index nedeniyle elendi")

    idx = cat[['id', 'file_name', 'file_index', 'time_utc', 'label']].copy()

    if verbose:
        n1, n0_ = int((idx.label == 1).sum()), int((idx.label == 0).sum())
        print(f"Katalog satiri            : {n0}")
        print(f"  {channel} + diskte      : {n_disk}")
        print(f"  {time_start[:7]} penceresi: {n_time}")
        print(f"  tekil olay              : {len(idx)}")
        print(f"    Oraj  (1)             : {n1}")
        print(f"    Normal(0)             : {n0_}")
        if len(idx):
            print(f"    oraj orani            : %{100 * n1 / len(idx):.1f}")
        print(f"  49 kare ile potansiyel goruntu: {len(idx) * N_FRAMES:,}")
    return idx


# --------------------------------------------------------------------------
# 2. Event-bazli bolme (tek kaynak, diske yazilir)
# --------------------------------------------------------------------------

def make_split(index, test_size=0.2, seed=42, path=SPLIT_PATH, verbose=True):
    """Bolmeyi olay id'si uzerinden yapar ve JSON olarak kaydeder.

    Egitim ve degerlendirme AYNI dosyayi okur; bolmelerin ayrismasi imkansiz.
    """
    from sklearn.model_selection import train_test_split

    train_ids, val_ids = train_test_split(
        index['id'].values,
        test_size=test_size,
        random_state=seed,
        stratify=index['label'].values,
    )
    split = {'train': sorted(train_ids.tolist()),
             'val': sorted(val_ids.tolist()),
             'seed': seed, 'test_size': test_size}
    with open(path, 'w') as f:
        json.dump(split, f)
    if verbose:
        print(f"Bolme kaydedildi -> {path}  (train={len(train_ids)}, val={len(val_ids)})")
    return split


def load_split(path=SPLIT_PATH):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# 3. Normalizasyon
# --------------------------------------------------------------------------

def normalize(raw, channel):
    """Ham SEVIR degerlerini 0-1 araligina getirir."""
    x = raw.astype('float32') * SEVIR_SCALE[channel]
    if channel.startswith('ir'):
        # x artik santigrat. Derin konveksiyon tepeleri cok soguk (-80..-90 C),
        # sicak zemin +30 C civari. Firtina = parlak olsun diye ters ceviriyoruz.
        x = (30.0 - x) / 120.0
    return np.clip(x, 0.0, 1.0)


# --------------------------------------------------------------------------
# 4. Veri jeneratoru
# --------------------------------------------------------------------------

class SevirSequence(tf.keras.utils.Sequence):
    """Her ornek (olay, zaman adimi) ciftidir.

    frames_per_event kadar ESIT ARALIKLI kare alinir -- rastgele degil, boylece
    dogrulama seti tekrar edilebilir olur. Ayni olayin kareleri korelasyonlu
    oldugu icin bu gercek ornek sayisini artirmaz, veri cogaltma sayilir.
    """

    def __init__(self, index, ids, channel='ir107', data_dir=DATA_DIR,
                 batch_size=16, dim=None, frames_per_event=5,
                 augment=False, shuffle=True, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.channel = channel
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dim = dim or CHANNEL_DIM[channel]
        self.augment = augment
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self._handles = {}
        self.n_failed = 0

        sub = index[index['id'].isin(set(ids))].reset_index(drop=True)
        ts = np.linspace(4, N_FRAMES - 5, frames_per_event).astype(int)
        self.samples = [(r.file_name, int(r.file_index), int(t), int(r.label))
                        for r in sub.itertuples() for t in ts]
        self.indexes = np.arange(len(self.samples))
        self.on_epoch_end()

    def _h5(self, fname):
        if fname not in self._handles:
            self._handles[fname] = h5py.File(os.path.join(self.data_dir, fname), 'r')
        return self._handles[fname]

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, i):
        sel = self.indexes[i * self.batch_size:(i + 1) * self.batch_size]
        X = np.zeros((len(sel), *self.dim, 1), dtype='float32')
        y = np.zeros(len(sel), dtype='int32')

        for j, k in enumerate(sel):
            fname, fidx, t, label = self.samples[k]
            try:
                ds = self._h5(fname)[self.channel]
                frame = ds[fidx, :, :, t]           # SEVIR duzeni: (N, H, W, T)
                frame = normalize(frame, self.channel)
                if frame.shape != tuple(self.dim):
                    frame = cv2.resize(frame, self.dim[::-1],
                                       interpolation=cv2.INTER_LINEAR)
                if self.augment:
                    frame = np.rot90(frame, self.rng.integers(0, 4))
                    if self.rng.random() > 0.5:
                        frame = np.fliplr(frame)
                    if self.rng.random() > 0.5:
                        frame = np.flipud(frame)
                X[j, :, :, 0] = frame
                y[j] = label
            except Exception as e:
                # Sessizce y=0 atamiyoruz -- bu sahte negatif uretirdi.
                self.n_failed += 1
                if self.n_failed <= 5:
                    print(f"UYARI okuma hatasi {fname}[{fidx},t={t}]: {e}")
                y[j] = label   # etiket yine de dogru kalsin
        return X, y

    def get_labels(self):
        """predict() ile ayni sirada etiketler (shuffle=False iken gecerli)."""
        return np.array([s[3] for s in self.samples])[self.indexes]

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    def __del__(self):
        for h in getattr(self, '_handles', {}).values():
            try:
                h.close()
            except Exception:
                pass


def make_generators(index, split, channel='ir107', batch_size=16,
                    frames_per_event=5, dim=(192, 192)):
    train_gen = SevirSequence(index, split['train'], channel=channel, dim=dim,
                              batch_size=batch_size, frames_per_event=frames_per_event,
                              augment=True, shuffle=True)
    val_gen = SevirSequence(index, split['val'], channel=channel, dim=dim,
                            batch_size=batch_size, frames_per_event=frames_per_event,
                            augment=False, shuffle=False)
    print(f"Egitim orneği: {len(train_gen.samples)} | Dogrulama: {len(val_gen.samples)}")
    return train_gen, val_gen


# --------------------------------------------------------------------------
# 5. Modeller (ResNet50 + DenseNet121 -- karsilastirma icin ikisi de var)
# --------------------------------------------------------------------------

def build_model(arch='resnet', input_shape=(192, 192, 1), freeze_backbone=True):
    """arch: 'resnet' | 'densenet'"""
    backbones = {'resnet': applications.ResNet50,
                 'densenet': applications.DenseNet121}
    if arch not in backbones:
        raise ValueError("arch 'resnet' veya 'densenet' olmali")

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(3, 3, padding='same', kernel_initializer='he_normal',
                      name='gray_to_rgb')(inputs)

    base = backbones[arch](include_top=False, weights='imagenet',
                           input_shape=(input_shape[0], input_shape[1], 3))
    base.trainable = not freeze_backbone
    x = base(x, training=not freeze_backbone)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    return models.Model(inputs, outputs, name=f"{arch}_oraj")


def get_backbone(model):
    """Govde modelini surumden bagimsiz sekilde bulur."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise ValueError('Govde (backbone) katmani bulunamadi')


# --------------------------------------------------------------------------
# 6. Egitim (iki asamali: once bas, sonra ince ayar)
# --------------------------------------------------------------------------

def class_weights(index, split):
    tr = index[index['id'].isin(set(split['train']))]
    n1, n0 = int((tr.label == 1).sum()), int((tr.label == 0).sum())
    total = n0 + n1
    w = {0: total / (2.0 * max(n0, 1)), 1: total / (2.0 * max(n1, 1))}
    print(f"Class weights -> Normal:{w[0]:.2f}  Oraj:{w[1]:.2f}")
    return w


def _callbacks(name):
    return [
        # Keras 3'te ModelCheckpoint .h5 kabul etmiyor -> .keras
        tf.keras.callbacks.ModelCheckpoint(f"{name}_best.keras",
                                           monitor='val_auc', mode='max',
                                           save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max',
                                         patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                             patience=2, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.CSVLogger(f"{name}_history.csv", append=True),
    ]


def _compile(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')],
    )


def train_stable(model, train_gen, val_gen, name, cw=None,
                 epochs_head=8, epochs_ft=25, unfreeze_from='conv4'):
    """ONERILEN EGITIM FONKSIYONU.

    Asama 1: govde tamamen donuk, yalnizca yeni bas egitilir.
    Asama 2: govdenin SADECE ust bloklari acilir, BatchNorm katmanlari
             donuk tutulur ve cok dusuk lr ile ince ayar yapilir.

    BatchNorm'un donuk tutulmasi kritik: aksi halde kucuk veride model
    ilk devirde cokuyor (val_loss patlamasi, tum orneklere 'oraj' demesi).

    unfreeze_from: ResNet50 icin 'conv4' veya 'conv5';
                   DenseNet121 icin 'conv4' veya 'conv5' blok adlari.
    """
    base = get_backbone(model)

    print("=== Asama 1: govde donuk ===")
    base.trainable = False
    _compile(model, 1e-3)
    model.fit(train_gen, validation_data=val_gen, epochs=epochs_head,
              class_weight=cw, callbacks=_callbacks(name))

    print(f"=== Asama 2: '{unfreeze_from}' ve sonrasi acik, BN donuk ===")
    base.trainable = True
    hit = False
    for l in base.layers:
        if l.name.startswith(unfreeze_from):
            hit = True
        l.trainable = hit and not isinstance(l, tf.keras.layers.BatchNormalization)
    print(f"egitilebilir katman: {sum(l.trainable for l in base.layers)} / {len(base.layers)}")

    _compile(model, 2e-5)
    model.fit(train_gen, validation_data=val_gen, epochs=epochs_ft,
              class_weight=cw, callbacks=_callbacks(name))

    model.save(f"{name}_final.keras")
    print(f"Kaydedildi: {name}_final.keras")
    return model


def train(model, train_gen, val_gen, name='resnet', cw=None,
          epochs_head=6, epochs_finetune=8):
    """Asama 1: govde donuk, sadece bas egitilir (kucuk veride asiri ogrenmeyi onler).
       Asama 2: govde acilir, cok dusuk lr ile ince ayar."""
    base = get_backbone(model)

    print("\n=== Asama 1: govde donuk ===")
    base.trainable = False
    _compile(model, 1e-3)
    h1 = model.fit(train_gen, validation_data=val_gen, epochs=epochs_head,
                   class_weight=cw, callbacks=_callbacks(name))

    print("\n=== Asama 2: ince ayar ===")
    base.trainable = True
    _compile(model, 1e-5)
    h2 = model.fit(train_gen, validation_data=val_gen, epochs=epochs_finetune,
                   class_weight=cw, callbacks=_callbacks(name))

    model.save(f"{name}_final.keras")
    print(f"Kaydedildi: {name}_final.keras")
    return h1, h2


# --------------------------------------------------------------------------
# 7. Degerlendirme -- AYNI bolmeyi diskten okur
# --------------------------------------------------------------------------

def evaluate(model_path, index, split=None, channel='ir107',
             batch_size=16, dim=(192, 192), frames_per_event=5):
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 roc_auc_score)
    import matplotlib.pyplot as plt

    split = split or load_split()
    gen = SevirSequence(index, split['val'], channel=channel, dim=dim,
                        batch_size=batch_size, frames_per_event=frames_per_event,
                        augment=False, shuffle=False)

    model = tf.keras.models.load_model(model_path)
    y_prob = model.predict(gen, verbose=1).ravel()
    y_true = gen.get_labels()[:len(y_prob)]
    y_pred = (y_prob > 0.5).astype(int)

    print("\n--- Siniflandirma Raporu ---")
    print(classification_report(y_true, y_pred,
                                target_names=['Normal (0)', 'Oraj (1)'], digits=3))
    try:
        print(f"ROC AUC: {roc_auc_score(y_true, y_prob):.4f}")
    except ValueError:
        pass

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, str(v), ha='center', va='center')
    ax.set_xticks([0, 1], ['Normal', 'Oraj'])
    ax.set_yticks([0, 1], ['Normal', 'Oraj'])
    ax.set_xlabel('Tahmin'); ax.set_ylabel('Gercek'); ax.set_title('Confusion Matrix')
    fig.colorbar(im); fig.tight_layout()
    etiket = os.path.basename(model_path).replace('.keras', '')
    fig.savefig(f'confusion_matrix_{etiket}.png', dpi=150, bbox_inches='tight')
    plt.show()
    return y_true, y_prob


def plot_roc(sonuclar, kayit='roc.png'):
    """ROC egrisi cizer.

    sonuclar: {'ResNet50': (y_true, y_prob), 'DenseNet121': (...)} sozlugu
              veya tek bir (y_true, y_prob) demeti.
    """
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    if isinstance(sonuclar, tuple):
        sonuclar = {'Model': sonuclar}

    plt.figure(figsize=(5.5, 5.5))
    for ad, (yt, yp) in sonuclar.items():
        fpr, tpr, _ = roc_curve(yt, yp[:len(yt)])
        plt.plot(fpr, tpr, lw=2, label=f'{ad} (AUC = {auc(fpr, tpr):.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Rastgele (AUC = 0,500)')
    plt.xlabel('Yanlış Alarm Oranı (FPR)')
    plt.ylabel('Doğru Tespit Oranı (TPR)')
    plt.title('ROC Eğrisi')
    plt.legend(loc='lower right')
    plt.grid(alpha=.3)
    plt.savefig(kayit, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Kaydedildi: {kayit}')


def yanlis_alarm_gorsel(index, split, y_true, y_prob, n=6, esik=0.8,
                        channel='ir107', dim=(192, 192), frames_per_event=5,
                        kayit='yanlis_alarmlar.png'):
    """'Normal' etiketli ama model tarafindan yuksek guvenle 'oraj' denen kareler.

    Bu goruntulerde gercekten konvektif yapi gorunuyorsa, yanlis alarmlarin
    bir kismi modelin hatasi degil etiket gurultusudur.
    """
    import matplotlib.pyplot as plt

    gen = SevirSequence(index, split['val'], channel=channel, dim=dim,
                        batch_size=16, frames_per_event=frames_per_event,
                        augment=False, shuffle=False)
    X = np.concatenate([gen[i][0] for i in range(len(gen))])

    yp = y_prob[:len(y_true)]
    fp = np.where((y_true == 0) & (yp > esik))[0][:n]
    if len(fp) == 0:
        print(f'Esik {esik} uzerinde yanlis alarm bulunamadi.')
        return

    fig, axes = plt.subplots(1, len(fp), figsize=(2.3 * len(fp), 3))
    for ax, k in zip(np.atleast_1d(axes), fp):
        ax.imshow(X[k, :, :, 0], cmap='inferno', vmin=0, vmax=1)
        ax.set_title(f'p(oraj) = {yp[k]:.2f}', fontsize=9)
        ax.axis('off')
    fig.suptitle('"Normal" etiketli, model yüksek olasılıkla "oraj" diyor', fontsize=11)
    fig.tight_layout()
    fig.savefig(kayit, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Kaydedildi: {kayit}')


def zamansal_dagilim(index, kayit='zamansal_dagilim.png'):
    """Siniflarin yil ici zaman dagilimini karsilastirir.

    Iki histogram ortusuyorsa mevsimsel confound kontrol altinda demektir.
    Bu, rapora konacak bir KANIT figurudur.
    """
    import matplotlib.pyplot as plt

    gun = pd.to_datetime(index.time_utc).dt.dayofyear
    plt.figure(figsize=(9, 3.2))
    for lab, ad, renk in [(1, 'Oraj', '#fc9272'), (0, 'Normal', '#9ecae1')]:
        plt.hist(gun[index.label == lab], bins=30, alpha=.6, label=ad, color=renk)
    plt.xlabel('Yılın günü')
    plt.ylabel('Olay sayısı')
    plt.title('Sınıfların zamansal dağılımı (örtüşme = mevsimsel yanlılık yok)')
    plt.legend()
    plt.grid(alpha=.3)
    plt.savefig(kayit, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Kaydedildi: {kayit}')


def plot_history(csv_path):
    import matplotlib.pyplot as plt
    h = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, (a, b, title) in zip(axes, [('accuracy', 'val_accuracy', 'Dogruluk'),
                                        ('loss', 'val_loss', 'Kayip')]):
        ax.plot(h[a], marker='o', label='Egitim')
        ax.plot(h[b], marker='o', label='Dogrulama')
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.legend(); ax.grid(True)
    fig.tight_layout()
    fig.savefig(csv_path.replace('.csv', '_plot.png'), dpi=150)
    plt.show()


# --------------------------------------------------------------------------
# 8. Teshis -- "verim yeterli mi" sorusunu olcerek cevaplar
# --------------------------------------------------------------------------

def diagnose(index, channel='ir107', data_dir=DATA_DIR, n=6):
    """Deger araliklarini ve ornek kareleri gosterir.

    Normalizasyonun dogru oldugunu gozle dogrulamak icin: firtina karelerinde
    parlak (soguk tepe) bolgeler gorunmeli, histogram 0-1 arasina yayilmali.
    """
    import matplotlib.pyplot as plt

    print(f"Toplam olay: {len(index)}  |  Oraj: {int(index.label.sum())}  "
          f"Normal: {int((index.label == 0).sum())}")

    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 5))
    for row, lab in enumerate([1, 0]):
        sub = index[index.label == lab]
        if sub.empty:
            continue
        pick = sub.sample(min(n, len(sub)), random_state=0)
        for col, r in enumerate(pick.itertuples()):
            with h5py.File(os.path.join(data_dir, r.file_name), 'r') as hf:
                raw = hf[channel][int(r.file_index), :, :, N_FRAMES // 2]
            img = normalize(raw, channel)
            if col == 0:
                print(f"  label={lab}  ham aralik: {raw.min()} .. {raw.max()}"
                      f"   -> normalize: {img.min():.2f} .. {img.max():.2f}")
            ax = axes[row, col]
            ax.imshow(img, cmap='inferno', vmin=0, vmax=1)
            ax.set_title(f"{'ORAJ' if lab else 'NORMAL'}\n"
                         f"{pd.Timestamp(r.time_utc):%d/%m %H:%M}", fontsize=8)
            ax.axis('off')
    fig.suptitle('Ust: Oraj  |  Alt: Normal   (parlak = soguk bulut tepesi)')
    fig.tight_layout()
    fig.savefig('veri_ornekleri.png', dpi=150)
    plt.show()
