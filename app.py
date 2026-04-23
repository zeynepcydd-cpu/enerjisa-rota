import math, io, time, warnings
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import folium
import streamlit as st
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")

# --- STREAMLIT SAYFA AYARLARI ---
st.set_page_config(page_title="EnerjiSA Optimizasyon Paneli", layout="wide")

# --- SABİTLER ---
VARDIYA_BASLANGIC_SAAT = 8 
OGLEN_MOLA_BASLANGIC = 12 
OGLEN_MOLA_BITIS = 13.5 
EPSILON = 0.001
T0 = 0

PI = {'ZA': 1.0, 'ZR': 1.0, 'ZS': 0.3, 'ZB': 0.3, 'ZG': 0.3}
PENALTI_TUR = {'ZA', 'ZR'}
RISKY_TUR = {'ZB', 'ZG', 'ZS'}
SKIP_STATUS = {'IPTL', 'İPTAL', 'CANCELLED', 'OK', 'TAMAMLANDI', 'CLOSED', 'KIPT', 'IPTL ODME', 'KOK'}

# --- YARDIMCI FONKSİYONLAR ---
def dist_km(lat1, lon1, lat2, lon2):
    return math.sqrt(((lat1 - lat2) * 111) ** 2 + ((lon1 - lon2) * 83) ** 2)

def dk_to_saat(dk):
    if dk is None: return "-"
    h = int(dk // 60) + 8
    return f"{h:02d}:{int(dk % 60):02d}"

def job_cost_params(row, c_ticari, c_mesken, t_bitis):
    ist = str(row.get('Sipariş Türü', '')).upper()[:2]
    abon = str(row.get('Abonelik Türü', ''))
    is_ticari = 'ticarethane' in abon.lower()
    c_d = c_ticari if is_ticari else c_mesken
    pi_i = PI.get(ist, 0.0)
    p_u = c_d if ist in PENALTI_TUR else (c_d * 0.5 if ist in RISKY_TUR else 50.0)
    return c_d, pi_i, p_u, t_bitis

def urgency_score(job_id, job_params_dict):
    c_d, pi_i, p_u, b_i = job_params_dict[job_id]
    return p_u * (1.0 + pi_i)

def adjust_for_lunch(arr_time, servis_suresi, t_mola_s, t_mola_e):
    if t_mola_s <= arr_time < t_mola_e: return t_mola_e
    if arr_time < t_mola_s and arr_time + servis_suresi > t_mola_s: return t_mola_e
    return arr_time

# --- ROTALAMA ALGORİTMALARI ---
def _priority_nn_route(candidates, origin, coords, job_params_dict, alpha, hiz):
    remaining = list(candidates)
    route, lat, lon = [], origin[0], origin[1]
    if not remaining: return []
    urgency_map = {j: urgency_score(j, job_params_dict) for j in remaining}
    max_urg = max(urgency_map.values()) if urgency_map else 1.0
    while remaining:
        dists = [dist_km(lat, lon, coords[j][0], coords[j][1]) for j in remaining]
        max_d = max(dists) if dists else 1.0
        idx = min(range(len(remaining)), key=lambda i: (alpha * (dists[i]/(max_d + 1e-9)) - (1-alpha) * (urgency_map[remaining[i]]/(max_urg + 1e-9))))
        j = remaining.pop(idx)
        route.append(j); lat, lon = coords[j]
    return route

def _check_feasible(route, origin, coords, hiz, servis_suresi, t_bitis, t_mola_s, t_mola_e):
    served, lat, lon, t = [], origin[0], origin[1], T0
    for j in route:
        travel_time = dist_km(lat, lon, coords[j][0], coords[j][1]) / hiz
        arr = adjust_for_lunch(t + travel_time, servis_suresi, t_mola_s, t_mola_e)
        if arr + servis_suresi <= t_bitis:
            served.append(j); lat, lon = coords[j]; t = arr + servis_suresi
    return served, [j for j in route if j not in served]

def build_folium_map(all_routes, op_coords, coords, job_meta):
    all_lats = [v[0] for v in coords.values()] + [v[0] for v in op_coords.values()]
    all_lons = [v[1] for v in coords.values()] + [v[1] for v in op_coords.values()]
    m = folium.Map(location=[np.mean(all_lats), np.mean(all_lons)], zoom_start=11)
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'darkred', 'cadetblue', 'black']
    for idx, (op, route) in enumerate(all_routes.items()):
        color = colors[idx % len(colors)]
        folium.Marker(op_coords[op], popup=f"Op: {op}", icon=folium.Icon(color=color, icon='user', prefix='fa')).add_to(m)
        if route:
            points = [op_coords[op]] + [coords[j] for j in route]
            folium.PolyLine(points, color=color, weight=3, opacity=0.7).add_to(m)
            for j in route:
                folium.CircleMarker(coords[j], radius=5, color=color, fill=True, popup=f"İş: {j} ({job_meta[j]['tip']})").add_to(m)
    return m

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Model Ayarları")
    method = st.selectbox("Atama Algoritması", ["K-Means (Bölge Bazlı)", "En Yakın Operatör (Mesafe Bazlı)"])
    alpha = st.slider("Öncelik (0) ↔ Mesafe (1) Dengesi", 0.0, 1.0, 0.5)
    zb_target = st.slider("ZB Hizmet Hedefi (%)", 0, 100, 30)
    
    st.divider()
    st.subheader("🕒 Zaman & Kapasite")
    v_bitis_saat = st.slider("Vardiya Bitiş Saati", 16.0, 22.0, 18.0, 0.5)
    t_bitis_dk = int((v_bitis_saat - 8) * 60)
    servis_dk = st.number_input("Servis Süresi (dk)", 5, 60, 10)
    hiz_ayar = st.slider("Hız (km/dk)", 0.1, 1.5, 0.5)

    st.divider()
    st.subheader("💰 Maliyetler")
    c_tic = st.number_input("Ticari Ceza (₺)", value=2216.0)
    c_mes = st.number_input("Mesken Cezası (₺)", value=277.0)
    
    uploaded_file = st.file_uploader("Excel Dosyasını Yükle", type=["xlsx"])
    run = st.button("🚀 Rotalamayı Başlat", use_container_width=True)

# --- ANA PROGRAM ---
if uploaded_file and run:
    with st.spinner("Hesaplanıyor..."):
        try:
            xls = pd.ExcelFile(uploaded_file)
            df_j = pd.read_excel(xls, sheet_name=0)
            df_o = pd.read_excel(xls, sheet_name=1)
            
            # Sütun İsimlerini Esnekleştirme (KeyError: 'latitude' Çözümü)
            df_o.columns = [c.strip().lower() for c in df_o.columns]
            lat_col = next(c for c in df_o.columns if 'lat' in c)
            lon_col = next(c for c in df_o.columns if 'lon' in c)
            user_col = df_o.columns[0]

            op_ids = df_o[user_col].astype(str).tolist()
            op_coords = {str(r[user_col]): (r[lat_col], r[lon_col]) for _, r in df_o.iterrows()}
            
            df_j = df_j.dropna(subset=['Tesisat Enlem', 'Tesisat Boylam'])
            if 'Sipariş Durumu' in df_j.columns:
                df_j = df_j[~df_j['Sipariş Durumu'].astype(str).str.strip().str.upper().isin(SKIP_STATUS)]
            
            job_ids = df_j['Sipariş No'].tolist()
            coords = {r['Sipariş No']: (r['Tesisat Enlem'], r['Tesisat Boylam']) for _, r in df_j.iterrows()}
            job_params = {r['Sipariş No']: job_cost_params(r, c_tic, c_mes, t_bitis_dk) for _, r in df_j.iterrows()}
            job_meta = {r['Sipariş No']: {'tip': str(r.get('Sipariş Türü', '??'))[:2]} for _, r in df_j.iterrows()}

            op_jobs = {op: [] for op in op_ids}

            # 1. FAZ: İŞ ATAMA
            if "K-Means" in method:
                X = np.array([[coords[j][0], coords[j][1]] for j in job_ids])
                km = KMeans(n_clusters=len(op_ids), random_state=42).fit(X)
                # Kümeleri en yakın operatöre ata (Macar Algoritması basitleştirilmiş)
                centers = km.cluster_centers_
                cost_matrix = np.array([[dist_km(c[0], c[1], op_coords[op][0], op_coords[op][1]) for op in op_ids] for c in centers])
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                cluster_to_op = {r: op_ids[c] for r, c in zip(row_ind, col_ind)}
                for i, jid in enumerate(job_ids):
                    op_jobs[cluster_to_op[km.labels_[i]]].append(jid)
            else:
                for jid in job_ids:
                    j_lat, j_lon = coords[jid]
                    best_op = min(op_ids, key=lambda o: dist_km(j_lat, j_lon, op_coords[o][0], op_coords[o][1]))
                    op_jobs[best_op].append(jid)

            # 2. FAZ: ROTALAMA
            all_routes = {}
            for op in op_ids:
                raw_route = _priority_nn_route(op_jobs[op], op_coords[op], coords, job_params, alpha, hiz_ayar)
                served_route, _ = _check_feasible(raw_route, op_coords[op], coords, hiz_ayar, servis_dk, t_bitis_dk, TBREAK_S, TBREAK_E)
                all_routes[op] = served_route

            # SONUÇ GÖSTERİMİ
            t1, t2 = st.tabs(["🗺️ İnteraktif Harita", "📊 Performans Analizi"])
            with t1:
                m = build_folium_map(all_routes, op_coords, coords, job_meta)
                components.html(m._repr_html_(), height=600)
            with t2:
                st.info(f"Seçilen Strateji: {method} | Alpha: {alpha}")
                total_done = sum(len(r) for r in all_routes.values())
                st.metric("Toplam Tamamlanan İş", f"{total_done} / {len(job_ids)}")
                
                # Basit bir ZB analizi
                zb_jobs = [j for j in job_ids if job_meta[j]['tip'] == 'ZB']
                zb_done = sum(1 for r in all_routes.values() for j in r if job_meta[j]['tip'] == 'ZB')
                zb_rate = (zb_done / len(zb_jobs) * 100) if zb_jobs else 100
                st.metric("ZB Hizmet Oranı", f"%{zb_rate:.1f}", delta=f"{zb_rate - zb_target:.1f}% Hedef Farkı")
                
                if zb_rate < zb_target:
                    st.warning(f"Dikkat: ZB hedefinin (%{zb_target}) altında kalındı!")

        except Exception as e:
            st.error(f"Uygulama hatası: {e}")
