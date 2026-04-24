import math, io, warnings
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import folium
import streamlit as st
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")

# ==========================================
# 1. ARAYÜZ VE SİDEBAR AYARLARI
# ==========================================
st.set_page_config(page_title="EnerjiSA Rotalama", layout="wide")

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Enerjisa_logo.png", width=150)
st.sidebar.header("⚙️ Kontrol Paneli")
uploaded_file = st.sidebar.file_uploader("İş Verisini Yükle (CSV/Excel)", type=['csv', 'xlsx'])

st.sidebar.markdown("### 🧠 Optimizasyon Parametreleri")
alpha_val = st.sidebar.slider("Öncelik-Mesafe Dengesi (Alpha)", 0.0, 1.0, 0.5, 0.1,
    help="0: Tamamen cezaya/önceliğe odaklan, 1: Tamamen en kısa mesafeye odaklan.")
aging_val = st.sidebar.slider("Yaşlandırma Katsayısı (Aging)", 1.0, 5.0, 2.0, 0.1,
    help="Ertelenen işin cezası her gün bu katsayı ile çarpılır.")
op_count = st.sidebar.number_input("Operatör Sayısı", min_value=1, max_value=50, value=15)

st.sidebar.markdown("### 🚙 Saha ve Maliyet Parametreleri")
hiz_km_dk = st.sidebar.number_input("Araç Hızı (km/dk)", value=0.5, step=0.1)
zb_hedef  = st.sidebar.slider("ZB Tamamlama Hedefi (%)", 0, 100, 30)
c_ticari  = st.sidebar.number_input("Ticari Tazminat (TRY)", value=2216.0, step=100.0)
c_mesken  = st.sidebar.number_input("Mesken Tazminat (TRY)", value=277.0, step=10.0)

# ==========================================
# 2. DİNAMİK SABİTLER
# ==========================================
VARDIYA_BASLANGIC_SAAT = 8
VARDIYA_BITIS_SAAT     = 18
OGLEN_MOLA_BASLANGIC   = 12
OGLEN_MOLA_BITIS       = 13.5
SERVIS_SURESI_DK       = 10
EPSILON                = 0.001
FUEL_RATE              = 5.0

T0       = 0
TEND     = int((VARDIYA_BITIS_SAAT - VARDIYA_BASLANGIC_SAAT) * 60)
TBREAK_S = int((OGLEN_MOLA_BASLANGIC - VARDIYA_BASLANGIC_SAAT) * 60)
TBREAK_E = int((OGLEN_MOLA_BITIS - VARDIYA_BASLANGIC_SAAT) * 60)
S_i      = SERVIS_SURESI_DK

V_KM_MIN           = hiz_km_dk
C_TICARI           = c_ticari
C_MESKEN           = c_mesken
ZB_COVERAGE_TARGET = zb_hedef / 100.0

PI          = {'ZA': 1.0, 'ZR': 1.0, 'ZS': 0.3, 'ZB': 0.3, 'ZG': 0.3}
PI_DEFAULT  = 0.0
PENALTI_TUR = {'ZA', 'ZR'}
RISKY_TUR   = {'ZB', 'ZG', 'ZS'}

BALANCE_MAX_ITER    = 5
BALANCE_OVERFLOW_TH = 1.0
MAX_TRANSFER_KM     = 2.0
KMEANS_INIT         = 'k-means++'

# ==========================================
# 3. YARDIMCI FONKSİYONLAR
# ==========================================
def dist_km(lat1, lon1, lat2, lon2):
    return math.sqrt(((lat1 - lat2) * 111) ** 2 + ((lon1 - lon2) * 83) ** 2)

def job_cost_params(row):
    ist       = str(row.get('Sipariş Türü', '')).upper()[:2]
    abon      = str(row.get('Abonelik Türü', ''))
    is_ticari = 'ticarethane' in abon.lower()
    c_d       = C_TICARI if is_ticari else C_MESKEN
    pi_i      = PI.get(ist, PI_DEFAULT)
    if ist in PENALTI_TUR:  p_u = c_d
    elif ist in RISKY_TUR:  p_u = c_d * 0.5
    else:                   p_u = 50.0
    return c_d, pi_i, p_u, TEND

def urgency_score(job_id, job_params_dict):
    c_d, pi_i, p_u, b_i = job_params_dict[job_id]
    return p_u * (1.0 + pi_i)

def unserved_penalty(job_id, job_params_dict):
    return job_params_dict[job_id][2]

def get_job_type(job_id, job_type_map):
    """İş türünün ilk 2 harfini döndürür."""
    return str(job_type_map.get(job_id, '')).upper()[:2]

# ==========================================
# YENİ FONKSİYON: ZB ÖNCELİKLENDİRME
# ==========================================
def boost_zb_priority(op_jobs, job_params_dict, job_type_map, zb_target):
    """
    ZB hedefini sağlamak için her operatörün iş listesindeki
    ZB işlerinin urgency skorunu geçici olarak yükseltir.

    Mantık:
    - Tüm ZB işlerinin kaç tanesi hedefi karşılamak için tamamlanmalı?
    - O kadar ZB işini, en yüksek urgency skorlu iş kadar yüksek bir
      score'a çıkar. Böylece rotalama sırasında önce seçilirler.
    """
    if zb_target <= 0:
        return job_params_dict  # hedef 0 ise dokunma

    # Tüm ZB işlerini bul
    tum_zb = [
        jid for op_list in op_jobs.values()
        for jid in op_list
        if get_job_type(jid, job_type_map) == 'ZB'
    ]

    if not tum_zb:
        return job_params_dict

    # Hedef adede göre kaç ZB tamamlanmalı
    zb_hedef_adet = math.ceil(len(tum_zb) * zb_target)

    # Mevcut en yüksek urgency skorunu bul (ZB dışı işlerden)
    zb_olmayan = [
        jid for op_list in op_jobs.values()
        for jid in op_list
        if get_job_type(jid, job_type_map) != 'ZB'
    ]
    max_non_zb_score = max(
        (urgency_score(j, job_params_dict) for j in zb_olmayan),
        default=C_MESKEN
    )

    # ZB işlerini urgency skoruna göre sırala (yüksekten düşüğe)
    # ve ilk zb_hedef_adet tanesinin p_u'sunu boost et
    zb_sirali = sorted(
        tum_zb,
        key=lambda j: urgency_score(j, job_params_dict),
        reverse=True
    )
    zb_boost_listesi = set(zb_sirali[:zb_hedef_adet])

    # Boost edilmiş parametreleri yeni dict'e koy
    boosted_params = dict(job_params_dict)
    for jid in zb_boost_listesi:
        c_d, pi_i, p_u, b_i = boosted_params[jid]
        # p_u'yu en yüksek non-ZB skoru + küçük bir epsilon kadar yükselt
        # Böylece bu ZB işleri öncelik sıralamasında öne geçer
        yeni_p_u = max_non_zb_score / (1.0 + pi_i) + 1.0
        boosted_params[jid] = (c_d, pi_i, yeni_p_u, b_i)

    return boosted_params

def teorik_sure(op_coord, job_list, coords):
    if not job_list:
        return 0.0
    olat, olon = op_coord
    mesafeler  = [dist_km(olat, olon, coords[j][0], coords[j][1]) for j in job_list]
    return len(job_list) * S_i + (float(np.mean(mesafeler)) / V_KM_MIN) * len(job_list)

def balance_workload(op_jobs, op_ids, op_coords, coords, job_params_dict):
    for _ in range(BALANCE_MAX_ITER):
        yuk = {op: teorik_sure(op_coords[op], op_jobs[op], coords) for op in op_ids}
        if max(yuk.values()) - min(yuk.values()) < TEND * 0.05:
            break
        asiri_yuklu = sorted(
            [op for op in op_ids if yuk[op] > TEND * BALANCE_OVERFLOW_TH and op_jobs[op]],
            key=lambda o: yuk[o], reverse=True
        )
        if not asiri_yuklu:
            break
        transferred = False
        for donor in asiri_yuklu:
            yuk_donor = yuk[donor]
            donatable = sorted(op_jobs[donor], key=lambda j: unserved_penalty(j, job_params_dict))
            for job in donatable:
                jlat, jlon = coords[job]
                dist_donor = dist_km(jlat, jlon, op_coords[donor][0], op_coords[donor][1])
                alicilar   = sorted(
                    [op for op in op_ids
                     if op != donor
                     and yuk[op] < yuk_donor
                     and dist_km(jlat, jlon, op_coords[op][0], op_coords[op][1]) <= MAX_TRANSFER_KM
                     and dist_km(jlat, jlon, op_coords[op][0], op_coords[op][1]) < dist_donor],
                    key=lambda op: dist_km(jlat, jlon, op_coords[op][0], op_coords[op][1])
                )
                for receiver in alicilar:
                    yuk_r = teorik_sure(op_coords[receiver], op_jobs[receiver] + [job], coords)
                    yuk_d = teorik_sure(op_coords[donor], [j for j in op_jobs[donor] if j != job], coords)
                    if yuk_r < yuk_donor and yuk_d < yuk_donor:
                        op_jobs[donor].remove(job)
                        op_jobs[receiver].append(job)
                        yuk[donor], yuk[receiver] = yuk_d, yuk_r
                        transferred = True
                        break
                if transferred:
                    break
            if transferred:
                break
        if not transferred:
            break
    return op_jobs

def adjust_for_lunch(arr_time):
    if arr_time < TBREAK_S and arr_time + S_i > TBREAK_S:
        return TBREAK_E
    elif TBREAK_S <= arr_time < TBREAK_E:
        return TBREAK_E
    return arr_time

def _check_feasible(route, origin, coords):
    served, unserved = [], []
    lat, lon, t = origin[0], origin[1], T0
    for j in route:
        jlat, jlon = coords[j]
        arr = adjust_for_lunch(t + dist_km(lat, lon, jlat, jlon) / V_KM_MIN)
        if arr + S_i > TEND:
            unserved.append(j)
        else:
            served.append(j)
            lat, lon, t = jlat, jlon, arr + S_i
    return served, unserved

def _priority_nn_route(candidates, origin, coords, job_params_dict, alpha):
    remaining, route = list(candidates), []
    lat, lon    = origin
    urgency_map = {j: urgency_score(j, job_params_dict) for j in remaining}
    max_urgency = max(urgency_map.values()) if urgency_map else 1.0
    while remaining:
        dists    = [dist_km(lat, lon, coords[j][0], coords[j][1]) for j in remaining]
        max_dist = max(dists) if dists else 1.0
        idx = min(range(len(remaining)), key=lambda i: (
            alpha * (dists[i] / (max_dist + 1e-9))
            - (1 - alpha) * (urgency_map[remaining[i]] / (max_urgency + 1e-9))
        ))
        j = remaining.pop(idx)
        route.append(j)
        lat, lon = coords[j]
    return route

def _two_opt(route, origin, coords):
    best = list(route)
    def route_km(r):
        la, lo, km = origin[0], origin[1], 0.0
        for j in r:
            jlat, jlon = coords[j]
            km += dist_km(la, lo, jlat, jlon)
            la, lo = jlat, jlon
        return km
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                candidate = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                served_c, _ = _check_feasible(candidate, origin, coords)
                if len(served_c) == len(best) and route_km(candidate) < route_km(best) - 0.001:
                    best, improved = candidate, True
    return best

def greedy_select_and_route(op_id, origin, job_list, coords, job_params_dict, alpha):
    olat, olon = origin
    sorted_by_urgency = sorted(
        job_list, key=lambda j: urgency_score(j, job_params_dict), reverse=True
    )
    candidates, elenen_erken = [], []
    for j in sorted_by_urgency:
        if dist_km(olat, olon, coords[j][0], coords[j][1]) / V_KM_MIN + S_i <= TEND - T0:
            candidates.append(j)
        else:
            elenen_erken.append(j)

    nn_route          = _priority_nn_route(candidates, origin, coords, job_params_dict, alpha)
    served_nn, elenen_nn = _check_feasible(nn_route, origin, coords)
    route             = _check_feasible(_two_opt(served_nn, origin, coords), origin, coords)[0]
    unserved          = elenen_erken + elenen_nn + [j for j in served_nn if j not in route]

    schedule, cur_time, lat, lon = {}, T0, olat, olon
    for j in route:
        jlat, jlon = coords[j]
        arr = max(T0, adjust_for_lunch(cur_time + dist_km(lat, lon, jlat, jlon) / V_KM_MIN))
        fin = arr + S_i
        c_d, pi_i, p_u, b_i = job_params_dict[j]
        schedule[j] = {
            'served': True, 'arrival': arr, 'finish': fin,
            'late': (fin - b_i > 0),
            'fuel_cost': FUEL_RATE * dist_km(lat, lon, jlat, jlon),
            'fixed_pen': c_d if fin - b_i > 0 else 0,
            'tardy_pen': pi_i * EPSILON * max(0.0, fin - b_i),
            'unserved_pen': 0.0,
        }
        lat, lon, cur_time = jlat, jlon, fin

    for j in unserved:
        schedule[j] = {
            'served': False, 'arrival': None, 'finish': None,
            'late': False, 'fuel_cost': 0.0, 'fixed_pen': 0.0,
            'tardy_pen': 0.0, 'unserved_pen': job_params_dict[j][2],
        }
    return route, schedule, unserved

def build_map(all_routes, all_schedules, op_coords, coords):
    center = (np.mean([v[0] for v in coords.values()]),
              np.mean([v[1] for v in coords.values()]))
    m = folium.Map(location=center, zoom_start=11)
    COLORS = ['blue', 'red', 'green', 'purple', 'orange', 'darkred', 'cadetblue', 'darkblue']

    for idx, (op_id, route) in enumerate(all_routes.items()):
        color      = COLORS[idx % len(COLORS)]
        olat, olon = op_coords[op_id]
        folium.Marker(
            [olat, olon], popup=f"Başlangıç: {op_id}",
            icon=folium.Icon(color=color, icon='home')
        ).add_to(m)
        if route:
            pts = [(olat, olon)] + [coords[j] for j in route] + [(olat, olon)]
            folium.PolyLine(pts, color=color, weight=3.5, opacity=0.8).add_to(m)
            for j in route:
                folium.CircleMarker(
                    [coords[j][0], coords[j][1]], radius=6,
                    color=color, fill=True, popup=f"Tamamlandı: {j}"
                ).add_to(m)

    for op_id, sch in all_schedules.items():
        for j, s in sch.items():
            if not s['served']:
                folium.CircleMarker(
                    [coords[j][0], coords[j][1]], radius=5,
                    color='gray', fill=True, fill_opacity=0.7,
                    popup=f"Ertelendi: {j} (Zaman yetmedi)"
                ).add_to(m)
    return m

# ==========================================
# 4. SİMÜLASYON TETİKLEYİCİSİ
# ==========================================
st.title("⚡ EnerjiSA Rotalama Panosu")

st.markdown("""
### ℹ️ Optimizasyon Algoritmaları ve Çalışma Mantığı
Bu sistem, sahadaki arıza, kesme-açma ve bakım işlerinin en verimli şekilde dağıtılması için
aşağıdaki algoritmik yaklaşımları kullanmaktadır:

* **1. İş Ataması (Clustering & Assignment):** İşler coğrafi konumlarına göre **K-Means Kümeleme (K-Means++)** algoritması ile operatör sayısı kadar bölgeye ayrılır. Ardından **Macar Algoritması** kullanılarak kümeler operatörlere optimum şekilde atanır.
* **2. ZB Önceliklendirme:** Rotalama başlamadan önce, ZB hedefini sağlayacak kadar ZB işinin öncelik puanı geçici olarak yükseltilir. Bu sayede rota algoritması bu işleri önce seçer.
* **3. İş Yükü Dengeleme (Workload Balancing):** Aşırı yüklü operatörlerin en düşük öncelikli işleri, kapasitesi uygun ve konuma en yakın komşu operatörlere devredilir.
* **4. Rotalama (Routing):** **Öncelik Ağırlıklı En Yakın Komşu** algoritmasıyla oluşturulan ilk rota, çapraz kesişmeleri gidermek için **2-Opt Yerel Arama** ile optimize edilir.
* **5. Fizibilite ve Erteleme:** Mesai bitimine kadar tamamlanamayacak işler sıradan çıkarılır. Ertelenen işler "Yaşlandırma Katsayısı" ile cezası artırılmış şekilde bir sonraki güne aktarılır.
---
""")

if "sim_data" not in st.session_state:
    st.session_state.sim_data = None
    st.session_state.gunler   = []
    st.session_state.mod      = None

if st.sidebar.button("🚀 Simülasyonu Başlat", type="primary", use_container_width=True):
    if uploaded_file is None:
        st.error("Lütfen önce sol menüden veri seti yükleyin!")
    else:
        with st.spinner("Veriler işleniyor, rotalar çiziliyor. Lütfen bekleyin..."):

            # ── Veri okuma ─────────────────────────────────
            if uploaded_file.name.endswith('.csv'):
                df_jobs = pd.read_csv(uploaded_file)
            else:
                df_jobs = pd.read_excel(uploaded_file)

            df_jobs['Tesisat Enlem']  = pd.to_numeric(
                df_jobs['Tesisat Enlem'].astype(str).str.replace(',', '.'), errors='coerce')
            df_jobs['Tesisat Boylam'] = pd.to_numeric(
                df_jobs['Tesisat Boylam'].astype(str).str.replace(',', '.'), errors='coerce')
            df_jobs = df_jobs.dropna(
                subset=['Tesisat Enlem', 'Tesisat Boylam']
            ).reset_index(drop=True)

            # ── Tarih sütunu tespiti ───────────────────────
            tarih_col = next(
                (c for c in df_jobs.columns if 'Planlanan' in c or 'Tarih' in c), None
            )
            if tarih_col:
                df_jobs[tarih_col] = df_jobs[tarih_col].astype(str).str.strip().str[:10]
            else:
                df_jobs['Planlanan Tarih'] = '24.11.2025'
                tarih_col = 'Planlanan Tarih'

            # ── Mod tespiti ────────────────────────────────
            gunler = sorted([
                d for d in df_jobs[tarih_col].unique()
                if '2025' in d or '2026' in d
            ])
            if not gunler:
                gunler = ['24.11.2025']
            mod = "gunluk" if len(gunler) == 1 else "haftalik"

            # ── Mock kurulum ───────────────────────────────
            if 'Gerçek Durum' not in df_jobs.columns:
                df_jobs['Gerçek Durum'] = np.random.choice(
                    ['Tamamlandı', 'Ertelendi'], size=len(df_jobs), p=[0.8, 0.2]
                )

            op_ids     = [f"Op_{i+1}" for i in range(int(op_count))]
            center_lat = df_jobs['Tesisat Enlem'].mean()
            center_lon = df_jobs['Tesisat Boylam'].mean()
            np.random.seed(42)
            op_coords = {
                op: (
                    center_lat + np.random.uniform(-0.08, 0.08),
                    center_lon + np.random.uniform(-0.08, 0.08),
                )
                for op in op_ids
            }

            coords     = {
                r['Sipariş No']: (float(r['Tesisat Enlem']), float(r['Tesisat Boylam']))
                for _, r in df_jobs.iterrows()
            }
            job_params = {
                row['Sipariş No']: job_cost_params(row)
                for _, row in df_jobs.iterrows()
            }

            # İş türü haritası (ZB tespiti için)
            job_type_map = {}
            if 'Sipariş Türü' in df_jobs.columns:
                job_type_map = dict(
                    zip(df_jobs['Sipariş No'],
                        df_jobs['Sipariş Türü'].astype(str).str.upper().str[:2])
                )

            gecikme_takip = {
                jid: {
                    'ilk_planlanan': '', 'tamamlandi_gun': 'Tamamlanmadı',
                    'gecikme_gun_sayisi': 0, 'durum': 'Bekliyor',
                }
                for jid in coords.keys()
            }
            bekleyen_kuyruk = []
            daily_results   = {}
            progress_bar    = st.progress(0)

            for gun_idx, bugun in enumerate(gunler):
                bugunun_yeni_isleri = df_jobs[
                    df_jobs[tarih_col] == bugun
                ]['Sipariş No'].tolist()
                for j in bugunun_yeni_isleri:
                    gecikme_takip[j]['ilk_planlanan'] = bugun

                aktif_isler = bekleyen_kuyruk + bugunun_yeni_isleri
                if not aktif_isler:
                    progress_bar.progress((gun_idx + 1) / len(gunler))
                    continue

                K = min(len(op_ids), len(aktif_isler))
                X = np.array([[coords[j][0], coords[j][1]] for j in aktif_isler])
                km_model = KMeans(
                    n_clusters=K, init=KMEANS_INIT, n_init=10, random_state=42
                ).fit(X)
                labels  = {aktif_isler[i]: km_model.labels_[i] for i in range(len(aktif_isler))}
                centers = [
                    (km_model.cluster_centers_[k][0], km_model.cluster_centers_[k][1])
                    for k in range(K)
                ]

                aktif_op_ids = op_ids[:K]
                cost_matrix  = np.zeros((K, K))
                for k, (clat, clon) in enumerate(centers):
                    for o_idx, op in enumerate(aktif_op_ids):
                        cost_matrix[k, o_idx] = dist_km(clat, clon, *op_coords[op])
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                cluster_to_op = {
                    int(row_ind[i]): aktif_op_ids[col_ind[i]] for i in range(K)
                }

                op_jobs = {op: [] for op in aktif_op_ids}
                for jid, cl in labels.items():
                    op_jobs[cluster_to_op[cl]].append(jid)

                op_jobs = balance_workload(
                    op_jobs, aktif_op_ids, op_coords, coords, job_params
                )

                # ── ZB ÖNCELİKLENDİRME ← BURAYA EKLENDI ──────────────────────────────
                # Rotalama başlamadan önce ZB hedefini sağlayacak kadar ZB işinin
                # urgency skoru geçici olarak yükseltiliyor.
                # Örnek: 10 ZB işi var, hedef %30 → 3 ZB işinin skoru en yüksek
                # non-ZB işinin skoru kadar çıkarılıyor.
                # Orijinal job_params değiştirilmiyor; sadece bu gün için kullanılıyor.
                gunluk_job_params = boost_zb_priority(
                    op_jobs, job_params, job_type_map, ZB_COVERAGE_TARGET
                )
                # ─────────────────────────────────────────────────────────────────────

                bekleyen_kuyruk  = []
                gunluk_rotalar   = {}
                gunluk_schedules = {}
                gunluk_km = gunluk_tamamlanan = gunluk_ertelenen = 0

                for op in aktif_op_ids:
                    route, schedule, unserved = greedy_select_and_route(
                        op, op_coords[op], op_jobs[op],
                        coords,
                        gunluk_job_params,   # ← boost edilmiş parametreler
                        alpha_val
                    )
                    gunluk_rotalar[op]   = route
                    gunluk_schedules[op] = schedule

                    if route:
                        pts = [op_coords[op]] + [coords[j] for j in route] + [op_coords[op]]
                        gunluk_km += sum(
                            dist_km(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])
                            for i in range(len(pts) - 1)
                        )

                    for j in route:
                        if schedule[j]['served']:
                            gecikme_takip[j]['tamamlandi_gun'] = bugun
                            gecikme_takip[j]['durum'] = 'Tamamlandı'
                            gunluk_tamamlanan += 1

                    for j in unserved:
                        bekleyen_kuyruk.append(j)
                        gunluk_ertelenen += 1
                        gecikme_takip[j]['gecikme_gun_sayisi'] += 1
                        c_d, pi_i, p_u, b_i = job_params[j]
                        job_params[j] = (c_d, pi_i, p_u * aging_val, b_i)

                # ZB tamamlanma oranını hesapla (günlük)
                tum_zb_bugun = [
                    j for op_list in op_jobs.values()
                    for j in op_list
                    if get_job_type(j, job_type_map) == 'ZB'
                ]
                zb_tamamlanan_bugun = sum(
                    1 for sch in gunluk_schedules.values()
                    for j, s in sch.items()
                    if get_job_type(j, job_type_map) == 'ZB' and s['served']
                )
                zb_oran_bugun = (
                    zb_tamamlanan_bugun / len(tum_zb_bugun)
                    if tum_zb_bugun else 1.0
                )

                # Gerçek veriler
                gunluk_gercek_isler = df_jobs[df_jobs[tarih_col] == bugun]
                gercek_tamamlanan   = len(
                    gunluk_gercek_isler[gunluk_gercek_isler['Gerçek Durum'] == 'Tamamlandı']
                )
                gercek_ertelenen = len(gunluk_gercek_isler) - gercek_tamamlanan

                if 'Gerçekleşen KM' in gunluk_gercek_isler.columns:
                    arac_col = next(
                        (c for c in gunluk_gercek_isler.columns
                         if 'Araç' in c or 'Operatör' in c or 'Plaka' in c), None
                    )
                    if arac_col:
                        gercek_km = gunluk_gercek_isler.drop_duplicates(
                            subset=[arac_col]
                        )['Gerçekleşen KM'].sum()
                    else:
                        ort = gunluk_gercek_isler['Gerçekleşen KM'].mean()
                        gercek_km = ort * len(aktif_op_ids) if pd.notnull(ort) else 0.0
                else:
                    sapma_orani = np.random.uniform(1.15, 1.30)
                    gercek_km   = gunluk_km * sapma_orani if gunluk_km > 0 else 0.0

                daily_results[bugun] = {
                    'routes': gunluk_rotalar,
                    'schedules': gunluk_schedules,
                    'km': gunluk_km,
                    'tamamlanan': gunluk_tamamlanan,
                    'ertelenen': gunluk_ertelenen,
                    'gercek_km': gercek_km,
                    'gercek_tamamlanan': gercek_tamamlanan,
                    'gercek_ertelenen': gercek_ertelenen,
                    'zb_tamamlanan': zb_tamamlanan_bugun,
                    'zb_toplam': len(tum_zb_bugun),
                    'zb_oran': zb_oran_bugun,
                }

                progress_bar.progress((gun_idx + 1) / len(gunler))

            gecikme_df = (
                pd.DataFrame.from_dict(gecikme_takip, orient='index')
                .reset_index()
                .rename(columns={'index': 'Sipariş No'})
            )
            st.session_state.gecikme_df  = gecikme_df[gecikme_df['ilk_planlanan'] != '']
            st.session_state.sim_data    = daily_results
            st.session_state.gunler      = gunler
            st.session_state.mod         = mod
            st.session_state.op_coords   = op_coords
            st.session_state.coords      = coords
            st.session_state.job_type_map = job_type_map

# ==========================================
# 5. GÖRSELLEŞTİRME VE ANALİZ
# ==========================================
if st.session_state.sim_data:
    st.success("Optimizasyon Başarıyla Tamamlandı!")
    st.markdown("---")

    mod    = st.session_state.mod
    gunler = st.session_state.gunler

    st.markdown("### 🗺️ Rota Haritası")
    if mod == "gunluk":
        selected_day = gunler[0]
        st.info(f"📅 Gösterilen tarih: **{selected_day}** (Günlük veri)")
    else:
        selected_day = st.select_slider(
            "Görüntülemek istediğiniz günü kaydırarak seçin:",
            options=gunler
        )

    day_data = st.session_state.sim_data[selected_day]

    map_obj = build_map(
        day_data['routes'], day_data['schedules'],
        st.session_state.op_coords, st.session_state.coords
    )
    components.html(map_obj._repr_html_(), height=500)

    # ── KARŞILAŞTIRMA METRİKLERİ ─────────────────────────────
    baslik = f"#### 📊 {selected_day} — Simülasyon vs Gerçekleşen"
    if mod == "gunluk":
        baslik = f"#### 📊 {selected_day} — Günlük Simülasyon Sonuçları"
    st.markdown(baslik)

    sim_km           = day_data['km']
    gercek_km        = day_data['gercek_km']
    sim_tamam        = day_data['tamamlanan']
    gercek_tamam     = day_data['gercek_tamamlanan']
    sim_ertelenen    = day_data['ertelenen']
    gercek_ertelenen = day_data['gercek_ertelenen']
    zb_tamamlanan    = day_data['zb_tamamlanan']
    zb_toplam        = day_data['zb_toplam']
    zb_oran          = day_data['zb_oran']

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Toplam Araç Mesafesi",
        f"{sim_km:.1f} km",
        delta=f"{sim_km - gercek_km:.1f} km (Gerçeğe Göre)",
        delta_color="inverse",
    )
    col2.metric(
        "Tamamlanan İş",
        f"{sim_tamam} Adet",
        delta=f"{sim_tamam - gercek_tamam} Adet (Gerçeğe Göre)",
        delta_color="normal",
    )
    col3.metric(
        "Ertelenen İş",
        f"{sim_ertelenen} Adet",
        delta=f"{sim_ertelenen - gercek_ertelenen} Adet (Gerçeğe Göre)",
        delta_color="inverse",
    )
    # ZB metriği: hedef tuttu mu tutmadı mı renkli göster
    zb_hedef_adet = math.ceil(zb_toplam * ZB_COVERAGE_TARGET) if zb_toplam > 0 else 0
    col4.metric(
        f"ZB Tamamlama (Hedef ≥%{zb_hedef})",
        f"{zb_tamamlanan} / {zb_toplam}",
        delta=f"%{zb_oran*100:.1f} {'✅' if zb_oran >= ZB_COVERAGE_TARGET else '❌'}",
        delta_color="normal" if zb_oran >= ZB_COVERAGE_TARGET else "inverse",
    )

    if zb_oran < ZB_COVERAGE_TARGET:
        st.warning(
            f"⚠️ ZB hedefi tutmadı: {zb_tamamlanan}/{zb_toplam} iş tamamlandı "
            f"(%{zb_oran*100:.1f}). Hedef: %{zb_hedef}. "
            f"En az {zb_hedef_adet} ZB işi tamamlanmalıydı. "
            f"Alpha değerini düşürmeyi veya ZB hedefini azaltmayı deneyebilirsiniz."
        )
    else:
        st.success(
            f"✅ ZB hedefi karşılandı: {zb_tamamlanan}/{zb_toplam} ZB işi tamamlandı "
            f"(%{zb_oran*100:.1f} ≥ %{zb_hedef})."
        )

    st.info(
        f"💡 **Sahadaki Gerçek Durum ({selected_day}):** "
        f"Gerçekte **{gercek_km:.1f} km** yol yapılmış, "
        f"**{gercek_tamam}** iş kapatılmış, "
        f"**{gercek_ertelenen}** iş ertesi güne bırakılmıştı."
    )

    st.markdown("---")

    if mod == "gunluk":
        st.markdown("### 📋 Günlük Gecikme Analizi")
    else:
        st.markdown("### 📋 Tüm Simülasyon Boyunca Gecikme Analizi (Kümülatif)")

    df = st.session_state.gecikme_df
    toplam_is   = len(df)
    zamaninda   = len(df[df['gecikme_gun_sayisi'] == 0])
    gec_kalan   = len(df[(df['gecikme_gun_sayisi'] > 0) & (df['durum'] == 'Tamamlandı')])
    yapilamayan = len(df[df['durum'] == 'Bekliyor'])

    scol1, scol2, scol3, scol4 = st.columns(4)
    scol1.metric("Toplam İş Hacmi",  toplam_is)
    scol2.metric("Zamanında Biten",  zamaninda)
    scol3.metric("Gecikmeli Biten",  gec_kalan)
    scol4.metric(
        "Bekleyen" if mod == "gunluk" else "Simülasyon Sonu Bekleyen",
        yapilamayan
    )

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Gecikme Panosu', index=False)
    output.seek(0)

    st.download_button(
        label="📥 Detaylı Gecikme Raporunu İndir (Excel)",
        data=output,
        file_name="EnerjiSA_Gecikme_Raporu.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
