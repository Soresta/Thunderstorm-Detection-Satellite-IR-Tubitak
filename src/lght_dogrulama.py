"""
lght_dogrulama.py  --  TUBITAK 2209-A / Oraj Tespiti

AMAC: Mevcut etiketleri DEGISTIRMEK degil, DOGRULAMAK.

SEVIR'in yildirim (lght) kanalini kullanarak, STORMEVENTS/RANDOMEVENTS
ayriminin meteorolojik olarak ne kadar tutarli oldugunu olcer.
Meteorolojik tanim geregi oraj, yildirim/gok gurultusu ureten konvektif
firtinadir; dolayisiyla yildirim sayimi etiket kalitesi icin dogrudan
bir olcuttur.

Mevcut model sonuclarini, bolmeyi ve grafikleri ETKILEMEZ.

Kullanim (Colab):
    from sevir_pipeline import *
    from lght_dogrulama import *

    idx = build_index(time_start='2018-05-01', time_end='2018-07-01')

    lght_indir()                      # yildirim dosyalarini indir
    lght_kesif()                      # ADIM 1: formati incele  <-- ONCE BUNU
    df = lght_dogrulama(idx, n=60)    # ADIM 2: nicel karsilastirma

NOT: Bu kod calistirilarak dogrulanmamistir. lght dosyalarinin ic yapisi
hakkinda savunmaci (defensive) varsayimlar kullanilmistir; lght_kesif()
ciktisina gore _sayim_cikar() fonksiyonu uyarlanmasi gerekebilir.
"""

import os
import glob
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sevir_pipeline import DATA_DIR, CATALOG_PATH


# ==========================================================================
# 0) Indirme
# ==========================================================================

def lght_indir(data_dir=DATA_DIR, yil=2018, aylar=('0501_0601', '0601_0701')):
    """SEVIR yildirim dosyalarini indirir.

    Yildirim verisi goruntu degil olay basina sonme listesi oldugundan
    dosyalar goruntu kanallarina kiyasla cok kucuktur.
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    from tqdm import tqdm

    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    for ay in aylar:
        key = f'data/lght/{yil}/SEVIR_LGHT_ALLEVENTS_{yil}_{ay}.h5'
        local = os.path.join(data_dir, key.replace('data/', '', 1))
        if os.path.exists(local):
            print('mevcut:', local)
            continue
        os.makedirs(os.path.dirname(local), exist_ok=True)
        try:
            size = s3.head_object(Bucket='sevir', Key=key)['ContentLength']
            print(f'{key}  ({size/1e6:.0f} MB)')
            with tqdm(total=size, unit='B', unit_scale=True) as bar:
                s3.download_file('sevir', key, local, Callback=bar.update)
        except Exception as e:
            print(f'INDIRILEMEDI {key}: {e}')
            print('  -> Dosya adi farkli olabilir. Katalogdaki lght satirlarina bakin:')
            print("     cat = pd.read_csv(CATALOG_PATH, low_memory=False)")
            print("     print(cat[cat.img_type=='lght'].file_name.unique()[:10])")


def lght_dosyalari(data_dir=DATA_DIR):
    return sorted(glob.glob(os.path.join(data_dir, 'lght', '**', '*.h5'),
                            recursive=True))


# ==========================================================================
# 1) Kesif  --  ONCE BUNU CALISTIRIN
# ==========================================================================

def lght_kesif(data_dir=DATA_DIR, n_anahtar=5):
    """lght dosyasinin ic yapisini yazdirir.

    Ciktiya bakarak _sayim_cikar() fonksiyonunun dogru calisip
    calismadigini teyit edin.
    """
    dosyalar = lght_dosyalari(data_dir)
    if not dosyalar:
        print('lght dosyasi bulunamadi. Once lght_indir() calistirin.')
        return

    yol = dosyalar[0]
    print(f'Dosya: {yol}')
    print(f'Boyut: {os.path.getsize(yol)/1e6:.1f} MB\n')

    with h5py.File(yol, 'r') as hf:
        anahtarlar = list(hf.keys())
        print(f'Toplam anahtar (olay) sayisi: {len(anahtarlar)}')
        print(f'Ilk anahtarlar: {anahtarlar[:n_anahtar]}\n')

        for k in anahtarlar[:n_anahtar]:
            obj = hf[k]
            if isinstance(obj, h5py.Dataset):
                print(f'  {k}: shape={obj.shape}  dtype={obj.dtype}')
                if obj.size and obj.ndim >= 1:
                    ornek = obj[:3]
                    print(f'      ilk satirlar:\n{ornek}')
            else:
                print(f'  {k}: GRUP -> {list(obj.keys())[:5]}')
    print('\nBeklenen yapi: her anahtar bir olay kimligi, degeri (N, 5) '
          'boyutunda sonme listesi (N = sonme sayisi).')
    print('Farkli ise _sayim_cikar() fonksiyonunu buna gore uyarlayin.')


# ==========================================================================
# 2) Sayim
# ==========================================================================

def _sayim_cikar(obj):
    """Bir olayin veri nesnesinden sonme (flash) sayisini cikarir.

    Savunmaci: farkli yapilarda da makul bir sayi dondurmeye calisir.
    """
    if obj is None:
        return None
    try:
        if isinstance(obj, h5py.Group):
            # Grup ise ilk dataset'i al
            alt = [obj[k] for k in obj.keys() if isinstance(obj[k], h5py.Dataset)]
            if not alt:
                return None
            obj = alt[0]
        shape = obj.shape
        if len(shape) == 0:
            return int(obj[()])
        # (N, ozellik) -> N sonme;  (N,) -> N
        return int(shape[0])
    except Exception:
        return None


def _lght_haritasi(data_dir=DATA_DIR):
    """{olay_id: (dosya_yolu, anahtar)} haritasi kurar."""
    harita = {}
    for yol in lght_dosyalari(data_dir):
        try:
            with h5py.File(yol, 'r') as hf:
                for k in hf.keys():
                    harita[k] = yol
        except Exception as e:
            print(f'okunamadi {yol}: {e}')
    return harita


def lght_sayimlari(index, data_dir=DATA_DIR, n=None, seed=0, verbose=True):
    """Index'teki olaylar icin yildirim sonme sayilarini toplar.

    Donen: DataFrame(id, label, sonme)
    """
    harita = _lght_haritasi(data_dir)
    if not harita:
        print('Yildirim verisi bulunamadi.')
        return pd.DataFrame()

    if verbose:
        print(f'Yildirim dosyalarinda {len(harita)} olay kimligi bulundu.')

    alt = index
    if n:
        parcalar = []
        for lab in (1, 0):
            g = index[index.label == lab]
            parcalar.append(g.sample(min(n, len(g)), random_state=seed))
        alt = pd.concat(parcalar)

    kayitlar, eslesmeyen = [], 0
    acik = {}
    try:
        for r in alt.itertuples():
            yol = harita.get(str(r.id))
            if yol is None:
                eslesmeyen += 1
                continue
            if yol not in acik:
                acik[yol] = h5py.File(yol, 'r')
            sayi = _sayim_cikar(acik[yol].get(str(r.id)))
            if sayi is not None:
                kayitlar.append({'id': r.id, 'label': int(r.label), 'sonme': sayi})
    finally:
        for h in acik.values():
            h.close()

    df = pd.DataFrame(kayitlar)
    if verbose:
        print(f'Eslesen olay: {len(df)}  |  Eslesmeyen: {eslesmeyen}')
    return df


# ==========================================================================
# 3) Dogrulama analizi
# ==========================================================================

def lght_dogrulama(index, n=60, esikler=(1, 10, 50, 100), data_dir=DATA_DIR,
                   seed=0, kayit='lght_dogrulama.png'):
    """Etiket kalitesini yildirim verisiyle olcer.

    Mevcut etiketleri DEGISTIRMEZ; yalnizca tutarliliklarini raporlar.
    """
    df = lght_sayimlari(index, data_dir=data_dir, n=n, seed=seed)
    if df.empty:
        return df

    oraj = df[df.label == 1]['sonme']
    normal = df[df.label == 0]['sonme']

    print('\n--- Yıldırım sönme sayısı özeti ---')
    ozet = df.groupby('label')['sonme'].agg(
        ['count', 'mean', 'median', 'std', 'min', 'max'])
    ozet.index = ['Normal (0)', 'Oraj (1)']
    print(ozet.round(1).to_string())

    print('\n--- Etiket tutarlılığı (farklı eşiklere göre) ---')
    satirlar = []
    for e in esikler:
        oraj_akt = float((oraj >= e).mean() * 100)
        norm_akt = float((normal >= e).mean() * 100)
        satirlar.append({
            'esik': e,
            'oraj_yildirimli_%': round(oraj_akt, 1),
            'normal_yildirimli_%': round(norm_akt, 1),
        })
        print(f'  >= {e:4d} sönme:  Oraj sınıfının %{oraj_akt:.1f}\'i  |  '
              f'Normal sınıfının %{norm_akt:.1f}\'i yıldırım içeriyor')
    tutarlilik = pd.DataFrame(satirlar)

    try:
        from scipy import stats
        u, p = stats.mannwhitneyu(oraj, normal, alternative='greater')
        print(f'\nMann-Whitney U (Oraj > Normal):  p = {p:.2e}')
    except ImportError:
        pass

    # --- gorsel
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    kutu = [normal.values, oraj.values]
    bp = ax.boxplot(kutu, labels=['Normal', 'Oraj'], patch_artist=True,
                    showfliers=False)
    for patch, renk in zip(bp['boxes'], ['#9ecae1', '#fc9272']):
        patch.set_facecolor(renk)
    ax.set_ylabel('Yıldırım sönme sayısı')
    ax.set_title('Sınıflara göre yıldırım aktivitesi')
    ax.grid(alpha=.3)

    ax = axes[1]
    x = np.arange(len(tutarlilik))
    ax.bar(x - 0.2, tutarlilik['oraj_yildirimli_%'], 0.4,
           label='Oraj sınıfı', color='#fc9272')
    ax.bar(x + 0.2, tutarlilik['normal_yildirimli_%'], 0.4,
           label='Normal sınıfı', color='#9ecae1')
    ax.set_xticks(x, [f'≥ {e}' for e in tutarlilik['esik']])
    ax.set_xlabel('Yıldırım eşiği')
    ax.set_ylabel('Sınıfın yüzdesi (%)')
    ax.set_title('Eşiğe göre yıldırım içeren olay oranı')
    ax.legend()
    ax.grid(alpha=.3, axis='y')

    fig.suptitle('Etiket kalitesinin yıldırım verisiyle doğrulanması', fontsize=13)
    fig.tight_layout()
    fig.savefig(kayit, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'\nKaydedildi: {kayit}')

    return df
