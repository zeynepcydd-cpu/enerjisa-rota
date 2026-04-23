import math, io, time, warnings
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import folium
import streamlit as st
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# STREAMLIT SAYFA AYARLARI
# ─────────────────────────────────────────────
st.set_page_config(page_title="EnerjiSA Optimizasyon Paneli", layout="wide")

# ─────────────────────────────────────────────
# SABİTLER  (sidebar'dan override edilenler hariç)
# ─────────────────────────────────────────────
VARDIYA_BASLANGIC_SAAT = 8
OGLEN_MOLA_BASLANGIC   = 12
OGLEN_MOLA_BITIS       = 13.5      # 13:30

EPSILON     = 0.001
T0          = 0
FUEL_RATE   = 5.0                  # TL/km

PI          = {'ZA': 1.0, 'ZR': 1.0, 'ZS': 0.3, 'ZB': 0.3, 'ZG': 0.3}
PI_DEFAULT  = 0.0
PENALTI_TUR = {'ZA', 'ZR'}
RISKY_TUR   = {'ZB', 'ZG', 'ZS'}

SKIP_STATUS = {
    'IPTL', 'İPTAL', 'CANCELLED', 'OK', 'TAMAMLANDI',
    'CLOSED', 'KIPT', 'IPTL ODME', 'KOK'
}

KMEANS_INIT     = 'k-means++'
KMEANS_N_INIT   = 15
KMEANS_MAX_ITER = 500

BALANCE_MAX_ITER    = 5
BALANCE_OVERFLOW_TH = 1.0
MAX_TRANSFER_KM     = 2.0
MIN_JOBS_PER_OP     = 30

# ─────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────
def dist_km(lat1, lon1, lat2, lon2):
    return math.sqrt(((lat1 - lat2) * 111) ** 2 + ((lon1 - lon2) * 83) ** 2)

def dk_to_saat(dk):
    if dk is None:
        return "—"
    h = int(dk // 60) + VARDIYA_BASLANGIC_SAAT
    return f"{h:02d}:{int(dk % 60):02d}"

def job_cost_params(row, c_ticari, c_mesken, t_bitis):
    ist      = str(row.get('Sipariş Türü', '')).upper()[:2]
    abon     = str(row.get('Abonelik Türü', ''))
    is_ticari = 'ticarethane' in abon.lower()
    c_d      = c_ticari if is_ticari else c_mesken
    pi_i     = PI.get(ist, PI_DEFAULT)
    if ist in PENALTI_TUR:
        p_u = c_d
    elif ist in RISKY_TUR:
        p_u = c_d * 0.5
    else:
        p_u = 50.0
    return c_d, pi_i, p_u, t_bitis

def urgency_score(job_id, job_params_dict):
    c_d, pi_i, p_u, b_i = job_params_dict[job_id]
    return p_u * (1.0 + pi_i)

def unserved_penalty(job_id, job_params_dict):
    c_d, pi_i, p_u, b_i = job_params_dict[job_id]
    return p_u

# ─────────────────────────────────────────────
# ÖĞLE MOLASI DÜZELTMESİ
# ─────────────────────────────────────────────
def adjust_for_lunch(arr_time, servis_suresi, t_mola_s, t_mola_e):
    """Varış zamanını öğle molasına göre kaydırır."""
    if arr_time < t_mola_s:
        if arr_time + servis_suresi > t_mola_s:
            return t_mola_e
    elif t_mola_s <= arr_time < t_mola_e:
        return t_mola_e
    return arr_time

# ─────────────────────────────────────────────
# K-MEANS KÜMELEME
# ─────────────────────────────────────────────
def kmeans_clustering(job_ids, coords, K):
    X = np.array([[coords[j][0], coords[j][1]] for j in job_ids])
    km = KMeans(
        n_clusters=K,
        init=KMEANS_INIT,
        n_init=KMEANS_N_INIT,
        max_iter=KMEANS_MAX_ITER,
        random_state=42
    )
    km.fit(X)
    labels  = {job_ids[i]: int(km.labels_[i]) for i in range(len(job_ids))}
    centers = [(km.cluster_centers_[k][0], km.cluster_centers_[k][1]) for k in range(K)]
    return labels, centers

def assign_clusters_to_operators(centers, op_ids, op_coords):
    K     = len(centers)
    n_ops = len(op_ids)
    cost  = np.zeros((K, n_ops))
    for k, (clat, clon) in enumerate(centers):
        for o_idx, op in enumerate(op_ids):
            cost[k, o_idx] = dist_km(clat, clon, *op_coords[op])
    row_ind, col_ind = linear_sum_assignment(cost)
    cluster_to_op = {int(row_ind[i]): op_ids[col_ind[i]] for i in range(len(row_ind))}
    return cluster_to_op

# ─────────────────────────────────────────────
# İŞ YÜKÜ ANALİZİ
# ─────────────────────────────────────────────
def teorik_sure(op_coord, job_list, coords, hiz, servis_dk):
    if not job_list:
        return 0.0
    olat, olon  = op_coord
    mesafeler   = [dist_km(olat, olon, coords[j][0], coords[j][1]) for j in job_list]
    ort_m       = float(np.mean(mesafeler))
    return len(job_list) * servis_dk + (ort_m / hiz) * len(job_list)

def compute_workload(op_id, origin, job_list, coords, job_params_dict, hiz, servis_dk, t_bitis):
    if not job_list:
        return {}
    vardiya_dk   = t_bitis - T0
    n            = len(job_list)
    olat, olon   = origin

    toplam_servis    = n * servis_dk
    mesafeler        = [dist_km(olat, olon, coords[j][0], coords[j][1]) for j in job_list]
    ort_mesafe_km    = float(np.mean(mesafeler))
    max_mesafe_km    = float(np.max(mesafeler))
    toplam_mesafe_km = float(np.sum(mesafeler))
    teorik_seyahat   = (ort_mesafe_km / hiz) * n
    teorik_toplam    = toplam_servis + teorik_seyahat
    kapasite_oran    = teorik_toplam / vardiya_dk
    fazla_yuk_dk     = max(0.0, teorik_toplam - vardiya_dk)
    teorik_max_is    = int(vardiya_dk / (servis_dk + ort_mesafe_km / hiz))

    skorlar    = [urgency_score(j, job_params_dict) for j in job_list]
    from_c_tic = job_params_dict[job_list[0]][0]  # referans
    yuksek     = sum(1 for s in skorlar if s >= from_c_tic * 1.5)
    orta       = sum(1 for s in skorlar if from_c_tic * 0.5 <= s < from_c_tic * 1.5)
    dusuk      = sum(1 for s in skorlar if s < from_c_tic * 0.5)

    return {
        'n_is': n,
        'vardiya_dk': vardiya_dk,
        'toplam_servis_dk': toplam_servis,
        'teorik_seyahat_dk': round(teorik_seyahat, 1),
        'teorik_toplam_dk': round(teorik_toplam, 1),
        'kapasite_oran': round(kapasite_oran, 3),
        'fazla_yuk_dk': round(fazla_yuk_dk, 1),
        'teorik_max_is': teorik_max_is,
        'ort_mesafe_km': round(ort_mesafe_km, 2),
        'max_mesafe_km': round(max_mesafe_km, 2),
        'toplam_mesafe_km': round(toplam_mesafe_km, 2),
        'oncelik_yuksek': yuksek,
        'oncelik_orta': orta,
        'oncelik_dusuk': dusuk,
    }

# ─────────────────────────────────────────────
# İŞ YÜKÜ DENGELEME
# ─────────────────────────────────────────────
def balance_workload(op_jobs, op_ids, op_coords, coords, job_params_dict, hiz, servis_dk, t_bitis):
    toplam_transfer = 0
    for iteration in range(BALANCE_MAX_ITER):
        yuk = {
            op: teorik_sure(op_coords[op], op_jobs[op], coords, hiz, servis_dk)
            for op in op_ids
        }
        if max(yuk.values()) - min(yuk.values()) < t_bitis * 0.05:
            break

        asiri_yuklu = sorted(
            [op for op in op_ids if yuk[op] > t_bitis * BALANCE_OVERFLOW_TH and op_jobs[op]],
            key=lambda o: yuk[o], reverse=True
        )
        if not asiri_yuklu:
            break

        iter_transfer = 0
        for donor in asiri_yuklu:
            yuk_donor  = yuk[donor]
            donatable  = sorted(op_jobs[donor], key=lambda j: unserved_penalty(j, job_params_dict))
            transferred = False
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
                    yuk_r = teorik_sure(op_coords[receiver], op_jobs[receiver] + [job], coords, hiz, servis_dk)
                    yuk_d = teorik_sure(op_coords[donor], [j for j in op_jobs[donor] if j != job], coords, hiz, servis_dk)
                    if yuk_r < yuk_donor and yuk_d < yuk_donor:
                        op_jobs[donor].remove(job)
                        op_jobs[receiver].append(job)
                        yuk[donor]    = yuk_d
                        yuk[receiver] = yuk_r
                        iter_transfer    += 1
                        toplam_transfer  += 1
                        transferred       = True
                        break
                if transferred:
                    break
        if iter_transfer == 0:
            break

    return op_jobs, toplam_transfer

def floor_balance(op_jobs, op_ids, op_coords, coords, job_params_dict, hiz, servis_dk):
    transfer = 0
    for _ in range(200):
        az_yuklu = sorted(
            [op for op in op_ids if len(op_jobs[op]) < MIN_JOBS_PER_OP],
            key=lambda o: len(op_jobs[o])
        )
        if not az_yuklu:
            break
        receiver  = az_yuklu[0]
        rlat, rlon = op_coords[receiver]
        best_job = best_donor = None
        best_dist = float('inf')
        for donor in op_ids:
            if donor == receiver or len(op_jobs[donor]) <= MIN_JOBS_PER_OP:
                continue
            for job in op_jobs[donor]:
                d = dist_km(coords[job][0], coords[job][1], rlat, rlon)
                if d < best_dist:
                    best_dist = d
                    best_job  = job
                    best_donor = donor
        if best_job is None:
            break
        op_jobs[best_donor].remove(best_job)
        op_jobs[receiver].append(best_job)
        transfer += 1
    return op_jobs, transfer

# ─────────────────────────────────────────────
# ROTALAMA ALGORİTMALARI
# ─────────────────────────────────────────────
def _priority_nn_route(candidates, origin, coords, job_params_dict, alpha):
    """
    Öncelik-ağırlıklı en-yakın-komşu rotalama.
    alpha=0 → tamamen öncelik bazlı
    alpha=1 → tamamen mesafe bazlı
    """
    remaining    = list(candidates)
    route        = []
    lat, lon     = origin
    urgency_map  = {j: urgency_score(j, job_params_dict) for j in remaining}
    max_urgency  = max(urgency_map.values()) if urgency_map else 1.0

    while remaining:
        dists    = [dist_km(lat, lon, coords[j][0], coords[j][1]) for j in remaining]
        max_dist = max(dists) if dists else 1.0
        idx = min(
            range(len(remaining)),
            key=lambda i: (
                alpha * (dists[i] / (max_dist + 1e-9))
                - (1 - alpha) * (urgency_map[remaining[i]] / (max_urgency + 1e-9))
            )
        )
        j = remaining.pop(idx)
        route.append(j)
        lat, lon = coords[j]
    return route

# DÜZELTME: yapılamayan işte konum/zaman GÜNCELLENMİYOR
def _check_feasible(route, origin, coords, hiz, servis_dk, t_bitis, t_mola_s, t_mola_e):
    served   = []
    unserved = []
    lat, lon = origin
    t        = T0
    for j in route:
        jlat, jlon = coords[j]
        travel     = dist_km(lat, lon, jlat, jlon) / hiz
        arr        = adjust_for_lunch(t + travel, servis_dk, t_mola_s, t_mola_e)
        if arr + servis_dk > t_bitis:
            unserved.append(j)
            # ← Hatalı işte konum ve zaman ilerletilmiyor
        else:
            served.append(j)
            lat, lon = jlat, jlon   # yalnızca servis edilen iş için güncelle
            t        = arr + servis_dk
    return served, unserved

def _two_opt(route, origin, coords, hiz, servis_dk, t_bitis, t_mola_s, t_mola_e):
    best = list(route)

    def route_km(r):
        la, lo = origin
        km = 0.0
        for j in r:
            jlat, jlon = coords[j]
            km += dist_km(la, lo, jlat, jlon)
            la, lo = jlat, jjon = jlat, jlon
        return km

    # 2-opt: rota uzunluğunu minimize ederken fizibiliteyi koru
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                candidate = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                served_c, _ = _check_feasible(
                    candidate, origin, coords, hiz, servis_dk, t_bitis, t_mola_s, t_mola_e
                )
                def _km(r):
                    la, lo = origin
                    km = 0.0
                    for jj in r:
                        jlat, jlon = coords[jj]
                        km += dist_km(la, lo, jlat, jlon)
                        la, lo = jlat, jlon
                    return km
                if len(served_c) == len(best) and _km(candidate) < _km(best) - 0.001:
                    best     = candidate
                    improved = True
    return best

def greedy_select_and_route(op_id, origin, job_list, coords, job_params_dict,
                             alpha, hiz, servis_dk, t_bitis, t_mola_s, t_mola_e):
    olat, olon = origin

    # Yüksek öncelikli işleri başa al
    sorted_by_urgency = sorted(
        job_list,
        key=lambda j: urgency_score(j, job_params_dict),
        reverse=True
    )

    # Tek yolculukla bile sığmayanları erken ele
    candidates   = []
    elenen_erken = []
    t_kalan      = t_bitis - T0
    for j in sorted_by_urgency:
        jlat, jlon = coords[j]
        tek_yol_dk = dist_km(olat, olon, jlat, jlon) / hiz
        if tek_yol_dk + servis_dk <= t_kalan:
            candidates.append(j)
        else:
            elenen_erken.append(j)

    nn_route = _priority_nn_route(candidates, origin, coords, job_params_dict, alpha)
    served_nn, elenen_nn = _check_feasible(
        nn_route, origin, coords, hiz, servis_dk, t_bitis, t_mola_s, t_mola_e
    )

    route = _two_opt(served_nn, origin, coords, hiz, servis_dk, t_bitis, t_mola_s, t_mola_e)
    final_served, elenen_2opt = _check_feasible(
        route, origin, coords, hiz, servis_dk, t_bitis, t_mola_s, t_mola_e
    )
    route    = final_served
    unserved = elenen_erken + elenen_nn + elenen_2opt

    # Zaman çizelgesi oluştur
    schedule = {}
    lat, lon = olat, olon
    cur_time = T0
    for j in route:
        jlat, jlon  = coords[j]
        travel      = dist_km(lat, lon, jlat, jlon) / hiz
        arr         = adjust_for_lunch(cur_time + travel, servis_dk, t_mola_s, t_mola_e)
        arr         = max(arr, T0)
        fin         = arr + servis_dk
        c_d, pi_i, p_u, b_i = job_params_dict[j]
        tardiness   = max(0.0, fin - b_i)
        travel_km   = dist_km(lat, lon, jlat, jlon)
        fuel_cost   = FUEL_RATE * travel_km
        fixed_pen   = c_d if tardiness > 0.0 else 0.0
        tardy_pen   = pi_i * EPSILON * tardiness
        schedule[j] = {
            'served': True, 'arrival': arr, 'finish': fin,
            'tardiness': tardiness, 'late': tardiness > 0.0,
            'fuel_cost': fuel_cost, 'fixed_pen': fixed_pen,
            'tardy_pen': tardy_pen, 'unserved_pen': 0.0,
        }
        lat, lon = jlat, jlon
        cur_time = fin

    for j in unserved:
        c_d, pi_i, p_u, b_i = job_params_dict[j]
        schedule[j] = {
            'served': False, 'arrival': None, 'finish': None,
            'tardiness': 0.0, 'late': False,
            'fuel_cost': 0.0, 'fixed_pen': 0.0,
            'tardy_pen': 0.0, 'unserved_pen': p_u,
        }
    return route, schedule, unserved

# ─────────────────────────────────────────────
# FOLİUM HARİTASI
# ─────────────────────────────────────────────
def build_folium_map(all_routes, all_schedules, op_coords, coords, job_meta):
    all_lats = [v[0] for v in coords.values()]    + [v[0] for v in op_coords.values()]
    all_lons = [v[1] for v in coords.values()]    + [v[1] for v in op_coords.values()]
    center   = (np.mean(all_lats), np.mean(all_lons))
    m        = folium.Map(location=center, zoom_start=11, tiles='OpenStreetMap')

    COLORS = [
        'blue', 'red', 'green', 'purple', 'orange', 'darkred',
        'cadetblue', 'darkgreen', 'darkblue', 'darkpurple',
        'black', 'gray', 'lightred', 'beige', 'pink'
    ]

    for idx, (op_id, route) in enumerate(all_routes.items()):
        color   = COLORS[idx % len(COLORS)]
        olat, olon = op_coords[op_id]

        folium.Marker(
            [olat, olon],
            popup=f"<b>Operatör {op_id}</b>",
            icon=folium.Icon(color=color, icon='home', prefix='fa')
        ).add_to(m)

        if not route:
            continue

        points = [(olat, olon)] + [coords[j] for j in route] + [(olat, olon)]
        folium.PolyLine(points, color=color, weight=2.5, opacity=0.8).add_to(m)

        sch = all_schedules[op_id]
        for order, j in enumerate(route):
            s       = sch[j]
            jlat, jlon = coords[j]
            tip     = job_meta[j]['tip']
            popup_txt = (
                f"<b>#{order+1} — {j}</b><br>"
                f"Tür: {tip}<br>"
                f"Varış: {dk_to_saat(s['arrival'])}<br>"
                f"Bitiş: {dk_to_saat(s['finish'])}<br>"
                f"{'⚠️ Geç' if s['late'] else '✅ Zamanında'}<br>"
                f"Maliyet: {s['fuel_cost']+s['fixed_pen']+s['tardy_pen']:.1f} TRY"
            )
            folium.CircleMarker(
                [jlat, jlon], radius=7, color=color, fill=True,
                fill_opacity=0.85,
                popup=folium.Popup(popup_txt, max_width=220)
            ).add_to(m)

        # Yapılamayanları gri göster
        for j, s in sch.items():
            if not s['served']:
                jlat, jlon = coords[j]
                folium.CircleMarker(
                    [jlat, jlon], radius=5, color='gray', fill=True, fill_opacity=0.4,
                    popup=folium.Popup(
                        f"<b>{j}</b><br>Yapılamadı<br>Ceza: {s['unserved_pen']:.0f} TRY",
                        max_width=160)
                ).add_to(m)
    return m

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Model Ayarları")
    method = st.selectbox("Atama Algoritması", [
        "K-Means (Bölge Bazlı)",
        "En Yakın Operatör (Mesafe Bazlı)"
    ])
    alpha       = st.slider("Öncelik (0) ↔ Mesafe (1) Dengesi", 0.0, 1.0, 0.5, 0.05)
    zb_target   = st.slider("ZB Hizmet Hedefi (%)", 0, 100, 30)

    st.divider()
    st.subheader("🕒 Zaman & Kapasite")
    v_bitis_saat = st.slider("Vardiya Bitiş Saati", 16.0, 22.0, 18.0, 0.5)
    servis_dk    = st.number_input("Servis Süresi (dk)", 5, 60, 10)
    hiz_ayar     = st.slider("Hız (km/dk)", 0.1, 1.5, 0.5, 0.05)

    # Türetilen zaman sabitleri
    TEND     = int((v_bitis_saat - VARDIYA_BASLANGIC_SAAT) * 60)
    TBREAK_S = int((OGLEN_MOLA_BASLANGIC - VARDIYA_BASLANGIC_SAAT) * 60)
    TBREAK_E = int((OGLEN_MOLA_BITIS - VARDIYA_BASLANGIC_SAAT) * 60)

    st.divider()
    st.subheader("💰 Maliyetler")
    c_tic = st.number_input("Ticari Ceza (₺)", value=2216.0)
    c_mes = st.number_input("Mesken Cezası (₺)", value=277.0)

    st.divider()
    uploaded_file = st.file_uploader("📂 Excel Dosyasını Yükle", type=["xlsx"])
    run           = st.button("🚀 Rotalamayı Başlat", use_container_width=True)

# ─────────────────────────────────────────────
# ANA PROGRAM
# ─────────────────────────────────────────────
if uploaded_file and run:
    with st.spinner("Hesaplanıyor..."):
        try:
            # ── Veri okuma ──────────────────────────────
            xls    = pd.ExcelFile(uploaded_file)
            df_j   = pd.read_excel(xls, sheet_name=0)
            df_o   = pd.read_excel(xls, sheet_name=1)

            df_o.columns = [c.strip().lower() for c in df_o.columns]
            lat_col  = next(c for c in df_o.columns if 'lat' in c)
            lon_col  = next(c for c in df_o.columns if 'lon' in c)
            user_col = df_o.columns[0]

            op_ids    = df_o[user_col].astype(str).tolist()
            op_coords = {str(r[user_col]): (float(r[lat_col]), float(r[lon_col]))
                         for _, r in df_o.iterrows()}

            # İş verisini temizle
            df_j = df_j.dropna(subset=['Tesisat Enlem', 'Tesisat Boylam'])
            df_j['Tesisat Enlem']  = pd.to_numeric(df_j['Tesisat Enlem'],  errors='coerce')
            df_j['Tesisat Boylam'] = pd.to_numeric(df_j['Tesisat Boylam'], errors='coerce')
            df_j = df_j.dropna(subset=['Tesisat Enlem', 'Tesisat Boylam'])
            if 'Sipariş Durumu' in df_j.columns:
                df_j = df_j[
                    ~df_j['Sipariş Durumu'].astype(str).str.strip().str.upper().isin(SKIP_STATUS)
                ]
            df_j = df_j.reset_index(drop=True)

            job_ids    = df_j['Sipariş No'].tolist()
            coords     = {r['Sipariş No']: (float(r['Tesisat Enlem']), float(r['Tesisat Boylam']))
                          for _, r in df_j.iterrows()}
            job_params = {r['Sipariş No']: job_cost_params(r, c_tic, c_mes, TEND)
                          for _, r in df_j.iterrows()}
            job_meta   = {r['Sipariş No']: {
                              'tip': str(r.get('Sipariş Türü', '??'))[:2].upper(),
                              'segment': 'Ticari' if 'ticarethane' in
                                         str(r.get('Abonelik Türü', '')).lower() else 'Mesken'
                          } for _, r in df_j.iterrows()}

            st.info(f"✅ Veri yüklendi: **{len(job_ids)} iş**, **{len(op_ids)} operatör**")

            # ── FAZ 1: İŞ ATAMA ─────────────────────────
            op_jobs = {op: [] for op in op_ids}

            if "K-Means" in method:
                with st.status("Faz 1/3: K-Means kümeleme…"):
                    labels, centers = kmeans_clustering(job_ids, coords, len(op_ids))
                    cluster_to_op   = assign_clusters_to_operators(centers, op_ids, op_coords)
                    for jid, cl in labels.items():
                        op_jobs[cluster_to_op[cl]].append(jid)
            else:
                with st.status("Faz 1/3: En yakın operatör atama…"):
                    for jid in job_ids:
                        jlat, jlon = coords[jid]
                        best_op    = min(
                            op_ids,
                            key=lambda o: dist_km(jlat, jlon, op_coords[o][0], op_coords[o][1])
                        )
                        op_jobs[best_op].append(jid)
                    centers = []

            # ── FAZ 2: DENGELEME ────────────────────────
            with st.status("Faz 2/3: İş yükü dengeleme…"):
                op_jobs, n_transfer = balance_workload(
                    op_jobs, op_ids, op_coords, coords, job_params,
                    hiz_ayar, servis_dk, TEND
                )
                op_jobs, n_floor = floor_balance(
                    op_jobs, op_ids, op_coords, coords, job_params,
                    hiz_ayar, servis_dk
                )

            # ── FAZ 3: ROTALAMA ─────────────────────────
            all_routes    = {}
            all_schedules = {}
            with st.status("Faz 3/3: Rotalama hesaplanıyor…"):
                for op_id in op_ids:
                    route, schedule, unserved = greedy_select_and_route(
                        op_id, op_coords[op_id], op_jobs[op_id],
                        coords, job_params,
                        alpha, hiz_ayar, servis_dk, TEND, TBREAK_S, TBREAK_E
                    )
                    all_routes[op_id]    = route
                    all_schedules[op_id] = schedule

            # ── SONUÇ GÖSTERİMİ ─────────────────────────
            tab1, tab2, tab3 = st.tabs(["🗺️ İnteraktif Harita", "📊 Performans Analizi", "📋 Detaylı Tablo"])

            with tab1:
                m = build_folium_map(all_routes, all_schedules, op_coords, coords, job_meta)
                components.html(m._repr_html_(), height=620)

            with tab2:
                st.info(f"Strateji: **{method}** | Alpha: {alpha} | "
                        f"Vardiya: {VARDIYA_BASLANGIC_SAAT}:00–{v_bitis_saat:.0f}:00")

                # Özet metrikler
                total_done     = sum(len(r) for r in all_routes.values())
                total_unserved = len(job_ids) - total_done
                total_cost     = sum(
                    s['fuel_cost'] + s['fixed_pen'] + s['tardy_pen'] + s['unserved_pen']
                    for sch in all_schedules.values() for s in sch.values()
                )

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Tamamlanan İş", f"{total_done} / {len(job_ids)}")
                col2.metric("Yapılamayan",   str(total_unserved))
                col3.metric("Genel Hizmet Oranı",
                            f"%{total_done/max(len(job_ids),1)*100:.1f}")
                col4.metric("Toplam Maliyet", f"{total_cost:,.0f} ₺")

                # ZB analizi
                zb_jobs = [j for j in job_ids if job_meta[j]['tip'] == 'ZB']
                zb_done = sum(1 for sch in all_schedules.values()
                              for j, s in sch.items()
                              if job_meta[j]['tip'] == 'ZB' and s['served'])
                zb_rate = (zb_done / len(zb_jobs) * 100) if zb_jobs else 100.0
                st.metric("ZB Hizmet Oranı",
                          f"%{zb_rate:.1f}",
                          delta=f"{zb_rate - zb_target:.1f}% hedef farkı")
                if zb_rate < zb_target:
                    st.warning(f"⚠️ ZB hedefinin (%{zb_target}) altında kalındı!")
                else:
                    st.success(f"✅ ZB hedefi (%{zb_target}) karşılandı.")

                # Operatör bazlı özet tablo
                rows = []
                for op_id in op_ids:
                    route = all_routes[op_id]
                    sch   = all_schedules[op_id]
                    srv   = [j for j, s in sch.items() if s['served']]
                    unsrv = [j for j, s in sch.items() if not s['served']]
                    late  = sum(1 for j in srv if sch[j]['late'])
                    olat, olon = op_coords[op_id]
                    pts   = [(olat, olon)] + [coords[j] for j in route] + [(olat, olon)]
                    km    = sum(dist_km(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])
                                for i in range(len(pts) - 1))
                    op_cost = sum(
                        sch[j]['fuel_cost'] + sch[j]['fixed_pen'] + sch[j]['tardy_pen']
                        for j in srv
                    ) + sum(sch[j]['unserved_pen'] for j in unsrv)
                    rows.append({
                        'Operatör': op_id,
                        'Tamamlanan': len(srv),
                        'Yapılamayan': len(unsrv),
                        'Geç': late,
                        'Mesafe (km)': round(km, 1),
                        'Maliyet (₺)': round(op_cost, 1),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            with tab3:
                servis_rows = []
                elenen_rows = []
                for op_id in op_ids:
                    sch = all_schedules[op_id]
                    for j, s in sch.items():
                        meta = job_meta[j]
                        base = {'Operatör': op_id, 'Sipariş No': j,
                                'İş Tipi': meta['tip'], 'Segment': meta['segment']}
                        if s['served']:
                            servis_rows.append({
                                **base,
                                'Varış': dk_to_saat(s['arrival']),
                                'Bitiş': dk_to_saat(s['finish']),
                                'Gecikme (dk)': round(s['tardiness'], 2),
                                'Yakıt (₺)': round(s['fuel_cost'], 2),
                                'Sabit Ceza (₺)': round(s['fixed_pen'], 2),
                                'Dk Ceza (₺)': round(s['tardy_pen'], 4),
                                'Toplam (₺)': round(
                                    s['fuel_cost'] + s['fixed_pen'] + s['tardy_pen'], 2),
                            })
                        else:
                            elenen_rows.append({
                                **base,
                                'Atanamama Ceza (₺)': round(s['unserved_pen'], 2),
                            })

                st.subheader("✅ Servis Edilenler")
                st.dataframe(pd.DataFrame(servis_rows), use_container_width=True, hide_index=True)
                st.subheader("❌ Elenmiş İşler")
                st.dataframe(pd.DataFrame(elenen_rows), use_container_width=True, hide_index=True)

                # Excel indirme
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                    pd.DataFrame(servis_rows).to_excel(
                        writer, sheet_name='Servis Edilenler', index=False)
                    pd.DataFrame(elenen_rows).to_excel(
                        writer, sheet_name='Elenmiş İşler', index=False)
                st.download_button(
                    "⬇️ Sonuçları Excel Olarak İndir",
                    data=buf.getvalue(),
                    file_name="energisa_sonuc.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        except Exception as e:
            st.error(f"🚨 Uygulama hatası: {e}")
            st.exception(e)

elif not uploaded_file:
    st.info("👈 Sol panelden Excel dosyasını yükleyin ve **Rotalamayı Başlat** butonuna tıklayın.")
