"""
ek_calismalar.py  --  TUBITAK 2209-A / Oraj Tespiti

  A) IP-2: Optik akis ile bulut hareketlerinin cikarilmasi ve analizi
  B) Kullanici dostu arayuz (Gradio demosu)

Kullanim (Colab):
    from sevir_pipeline import *
    from ek_calismalar import *

    idx = build_index(time_start='2018-05-01', time_end='2018-07-01')

    optik_akis_gorsel(idx)         # Sekil: t, t+5dk, akis alani
    df = optik_akis_analizi(idx)   # Nicel karsilastirma + kutu grafigi

    arayuz_baslat('resnet_best.keras', idx)   # Gradio demo

NOT: Bu kod calistirilarak dogrulanmamistir; ilk calistirmada hata
cikarsa mesaji paylasin.
"""

import os
import h5py
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from sevir_pipeline import (DATA_DIR, N_FRAMES, normalize, load_split)


# ==========================================================================
# A) OPTIK AKIS  --  IP-2
# ==========================================================================

def _kare(row, t, data_dir=DATA_DIR, channel='ir107'):
    """Tek bir olayin t. zaman adimindaki normalize edilmis karesi."""
    with h5py.File(os.path.join(data_dir, row.file_name), 'r') as hf:
        raw = hf[channel][int(row.file_index), :, :, int(t)]
    return normalize(raw, channel)


def _akis(a, b):
    """Iki ardisik kare arasindaki Farneback optik akisi.

    Donen: flow (H,W,2), buyukluk (H,W)
    """
    ai = (np.clip(a, 0, 1) * 255).astype('uint8')
    bi = (np.clip(b, 0, 1) * 255).astype('uint8')
    flow = cv2.calcOpticalFlowFarneback(
        ai, bi, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return flow, mag


def _oznitelik(flow, mag, frame):
    """Akis alanindan ozet oznitelikler.

    ort_hiz      : ortalama hareket buyuklugu (piksel/kare)
    p95_hiz      : 95. yuzdelik hiz (en hizli bolgeler)
    bulut_hiz    : yalnizca soguk (bulutlu) bolgelerdeki ortalama hiz
    yayilma      : ortalama pozitif diverjans -> orsu yayilmasi
    """
    div = np.gradient(flow[..., 0], axis=1) + np.gradient(flow[..., 1], axis=0)
    bulut = frame > 0.5                      # normalize edilmis: yuksek = soguk
    return {
        'ort_hiz': float(mag.mean()),
        'p95_hiz': float(np.percentile(mag, 95)),
        'bulut_hiz': float(mag[bulut].mean()) if bulut.any() else 0.0,
        'yayilma': float(div[div > 0].mean()) if (div > 0).any() else 0.0,
        'bulut_orani': float(bulut.mean()),
    }


def optik_akis_gorsel(index, t=24, data_dir=DATA_DIR, seed=0,
                      kayit='optik_akis.png'):
    """Bir oraj ve bir normal olay icin niteliksel karsilastirma sekli."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))

    for satir, (lab, ad) in enumerate([(1, 'ORAJ'), (0, 'NORMAL')]):
        alt = index[index.label == lab]
        if alt.empty:
            continue
        r = alt.sample(1, random_state=seed).iloc[0]
        a = _kare(r, t, data_dir)
        b = _kare(r, t + 1, data_dir)
        flow, mag = _akis(a, b)

        axes[satir, 0].imshow(a, cmap='inferno', vmin=0, vmax=1)
        axes[satir, 0].set_title(f'{ad} — t (IR)', fontsize=10)

        axes[satir, 1].imshow(b, cmap='inferno', vmin=0, vmax=1)
        axes[satir, 1].set_title(f'{ad} — t + 5 dk', fontsize=10)

        im = axes[satir, 2].imshow(mag, cmap='viridis')
        axes[satir, 2].set_title(f'Optik akış büyüklüğü\nort={mag.mean():.2f} px/kare',
                                 fontsize=10)
        plt.colorbar(im, ax=axes[satir, 2], fraction=0.046)

        # akis vektorleri (seyreltilmis)
        adim = max(1, a.shape[0] // 20)
        Y, X = np.mgrid[0:a.shape[0]:adim, 0:a.shape[1]:adim]
        axes[satir, 2].quiver(X, Y,
                              flow[::adim, ::adim, 0], flow[::adim, ::adim, 1],
                              color='white', scale=40, width=0.003)

        for ax in axes[satir]:
            ax.axis('off')

    fig.suptitle('Ardışık uydu görüntülerinden optik akış ile bulut hareketinin çıkarılması',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(kayit, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Kaydedildi: {kayit}')


def optik_akis_analizi(index, n=25, t_list=(12, 24, 36), data_dir=DATA_DIR,
                       seed=0, kayit='optik_akis_dagilim.png'):
    """Oraj ve normal olaylarin akis ozniteliklerini nicel olarak karsilastirir.

    Her siniftan n olay, her olaydan len(t_list) zaman adimi ornekler.
    Donen: pandas DataFrame (olay basina ortalama oznitelikler)
    """
    kayitlar = []
    for lab in (1, 0):
        alt = index[index.label == lab]
        if alt.empty:
            continue
        sec = alt.sample(min(n, len(alt)), random_state=seed)
        for r in sec.itertuples():
            ozs = []
            for t in t_list:
                if t + 1 >= N_FRAMES:
                    continue
                try:
                    a = _kare(r, t, data_dir)
                    b = _kare(r, t + 1, data_dir)
                    flow, mag = _akis(a, b)
                    ozs.append(_oznitelik(flow, mag, a))
                except Exception as e:
                    print(f'atlandi ({r.id}, t={t}): {e}')
            if ozs:
                ort = {k: float(np.mean([o[k] for o in ozs])) for k in ozs[0]}
                ort['label'] = lab
                ort['id'] = r.id
                kayitlar.append(ort)

    df = pd.DataFrame(kayitlar)
    if df.empty:
        print('Hic ornek islenemedi.')
        return df

    # --- ozet tablo
    ozet = df.groupby('label')[['ort_hiz', 'p95_hiz', 'bulut_hiz',
                                'yayilma', 'bulut_orani']].agg(['mean', 'std'])
    print('\n--- Optik akış öznitelikleri (0 = Normal, 1 = Oraj) ---')
    print(ozet.round(3).to_string())

    # --- istatistiksel test
    try:
        from scipy import stats
        print('\n--- Mann-Whitney U testi (sınıflar arası fark) ---')
        for k in ['ort_hiz', 'p95_hiz', 'bulut_hiz', 'yayilma', 'bulut_orani']:
            u, p = stats.mannwhitneyu(df[df.label == 1][k],
                                      df[df.label == 0][k],
                                      alternative='two-sided')
            im = '  anlamlı (p<0,05)' if p < 0.05 else '  anlamlı değil'
            print(f'{k:14s}  p = {p:.4f}{im}')
    except ImportError:
        pass

    # --- kutu grafigi
    kolonlar = ['ort_hiz', 'p95_hiz', 'bulut_hiz', 'yayilma']
    basliklar = ['Ortalama hız', '95. yüzdelik hız',
                 'Bulutlu bölge hızı', 'Yayılma (diverjans)']
    fig, axes = plt.subplots(1, len(kolonlar), figsize=(4 * len(kolonlar), 4))
    for ax, k, bas in zip(axes, kolonlar, basliklar):
        veri = [df[df.label == 0][k].values, df[df.label == 1][k].values]
        bp = ax.boxplot(veri, labels=['Normal', 'Oraj'], patch_artist=True)
        for patch, renk in zip(bp['boxes'], ['#9ecae1', '#fc9272']):
            patch.set_facecolor(renk)
        ax.set_title(bas, fontsize=11)
        ax.grid(alpha=.3)
    fig.suptitle('Optik akış özniteliklerinin sınıflara göre dağılımı', fontsize=13)
    fig.tight_layout()
    fig.savefig(kayit, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'\nKaydedildi: {kayit}')
    return df


# ==========================================================================
# B) ARAYUZ  --  Gradio demosu
# ==========================================================================

def arayuz_baslat(model_path, index, data_dir=DATA_DIR, channel='ir107',
                  share=True):
    """Egitilmis model icin basit bir web arayuzu baslatir.

    Colab'de calistirildiginda paylasilabilir bir baglanti uretir.
    Once:  !pip install -q gradio
    """
    import gradio as gr
    import tensorflow as tf
    from matplotlib import cm

    model = tf.keras.models.load_model(model_path)
    val_ids = set(load_split()['val'])
    sub = index[index['id'].isin(val_ids)].reset_index(drop=True)
    print(f'Doğrulama kümesinden {len(sub)} olay yüklendi.')

    def _renklendir(frame):
        return (cm.inferno(np.clip(frame, 0, 1))[..., :3] * 255).astype('uint8')

    def tahmin(olay_no, zaman):
        r = sub.iloc[int(olay_no)]
        frame = _kare(r, int(zaman), data_dir, channel)
        p = float(model.predict(frame[None, ..., None], verbose=0).ravel()[0])

        gercek = 'ORAJ' if r.label == 1 else 'NORMAL'
        tahmin_s = 'ORAJ' if p > 0.5 else 'NORMAL'
        durum = '✓ Doğru' if tahmin_s == gercek else '✗ Hatalı'

        bilgi = (
            f"**Olay kimliği:** {r.id}\n\n"
            f"**Tarih (UTC):** {pd.Timestamp(r.time_utc):%d.%m.%Y %H:%M}\n\n"
            f"**Zaman adımı:** {int(zaman)} / {N_FRAMES - 1} "
            f"(olay başlangıcından ~{int(zaman) * 5} dk sonra)\n\n"
            f"**Gerçek etiket:** {gercek}\n\n"
            f"**Model tahmini:** {tahmin_s} — {durum}"
        )
        return _renklendir(frame), {'Oraj': p, 'Normal': 1 - p}, bilgi

    def rastgele():
        return int(np.random.randint(len(sub))), int(np.random.randint(N_FRAMES))

    with gr.Blocks(title='Uydu Görüntülerinden Oraj Tespiti') as demo:
        gr.Markdown(
            '# Uydu Görüntülerinden Oraj Tespiti\n'
            'TÜBİTAK 2209-A — Termal kızılötesi (IR 10,7 µm) uydu görüntülerinden '
            'derin öğrenme ile oraj tespiti.\n\n'
            'Aşağıdaki örnekler modelin **eğitim sırasında hiç görmediği** '
            'doğrulama kümesinden seçilmektedir.'
        )
        with gr.Row():
            with gr.Column(scale=1):
                s_olay = gr.Slider(0, max(len(sub) - 1, 1), value=0, step=1,
                                   label='Olay numarası')
                s_zaman = gr.Slider(0, N_FRAMES - 1, value=24, step=1,
                                    label='Zaman adımı (5 dk aralıklı)')
                with gr.Row():
                    b_tahmin = gr.Button('Tahmin Et', variant='primary')
                    b_rastgele = gr.Button('Rastgele Örnek')
                c_olasilik = gr.Label(label='Model çıktısı', num_top_classes=2)
            with gr.Column(scale=1):
                c_gorsel = gr.Image(label='Termal kızılötesi görüntü '
                                          '(parlak = soğuk bulut tepesi)')
                c_bilgi = gr.Markdown()

        b_tahmin.click(tahmin, [s_olay, s_zaman],
                       [c_gorsel, c_olasilik, c_bilgi])
        b_rastgele.click(rastgele, None, [s_olay, s_zaman]) \
                  .then(tahmin, [s_olay, s_zaman],
                        [c_gorsel, c_olasilik, c_bilgi])
        demo.load(tahmin, [s_olay, s_zaman], [c_gorsel, c_olasilik, c_bilgi])

    demo.launch(share=share, debug=False)
    return demo
