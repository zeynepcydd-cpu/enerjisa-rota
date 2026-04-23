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
VARDIYA_BITIS_SAAT = 18 
OGLEN_MOLA_BASLANGIC = 12 
OGLEN_MOLA_BITIS = 13.5 
SERVIS_SURESI_DK = 10
EPSILON = 0.001
T0 = 0
TEND = int((VARDIYA_BITIS_SAAT - VARDIYA_BASLANGIC_SAAT) * 60)
TBREAK_S = int((OGLEN_MOLA_BASLANGIC - VARDIYA_BASLANGIC_SAAT) * 60)
TBREAK_E = int((OGLEN_MOLA_BITIS - VARDIYA_BASLANGIC_SAAT) * 60)

PI = {'ZA': 1.0, 'ZR': 1.0, 'ZS': 0.3, 'ZB': 0.3, 'ZG': 0.3}
PENALTI_TUR = {'ZA', 'ZR'}
RISKY_TUR = {'ZB', 'ZG', 'ZS'}

# --- FONKSİYONLAR ---
def dist_km(lat1, lon1, lat2, lon2):
    return math.sqrt(((lat1 - lat2) * 111) ** 2 + ((lon1 - lon2) * 83) ** 2)

def dk_to_saat(dk):
    if dk is None: return "-"
    h = int(dk // 60) + 8
    return f"{h:02d}:{int(dk % 60):02d}"

def job_cost_params(row, c_ticari, c_mesken):
    ist = str(row.get('Sipariş Türü', '')).upper()[:2]
    abon = str(row.get('Abonelik Türü', ''))
    is_ticari = 'ticarethane' in abon.lower()
    c_d = c_ticari if is_ticari else c_mesken
    pi_i = PI.get(ist, 0.0)
    p_u = c_d if ist in PENALTI_TUR else (c_d * 0.5 if ist in RISKY_TUR else 50.0)
    return c_d, pi_i, p_u, TEND

def urgency_score(job_id, job_params_dict):
    c_d, pi_i, p_u, b_i = job_params_dict[job_id]
    return p_u * (1.0 + pi_i)

def adjust_for_lunch(arr_time):
    if TBREAK_S <= arr_time < TBREAK_E: return TBREAK_E
    if arr_time < TBREAK_S and arr_time + 10 > TBREAK_S: return TBREAK_E
    return arr_time

def _priority_nn_route(candidates, origin, coords, job_params_dict, alpha, hiz):
    remaining = list(candidates)
    route, lat, lon = [], origin[0], origin[1]
    urgency_map = {j: urgency_score(j, job_params_dict) for j in remaining}
    max_urg = max(urgency_map.values()) if urgency_map else 1.0
    while remaining:
        dists = [dist_km(lat, lon, coords[j][0], coords[j][1]) for j in remaining]
        max_d = max(dists) if dists else 1.0
        idx = min(range(len(remaining)), key=lambda i: (alpha * (dists[i]/max_d) - (1-alpha) * (urgency_map[remaining[i]]/max_urg)))
        j = remaining.pop(idx)
        route.append(j); lat, lon = coords[j]
    return route

def _check_feasible(route, origin, coords, hiz):
    served, lat, lon, t = [], origin[0], origin[1], T0
    for j in route:
        arr = adjust_for_lunch(t + dist_km(lat, lon, coords[j][0], coords[j][1])/hiz)
        if arr + 10 <= TEND:
            served.append(j); lat, lon = coords[j]; t = arr + 10
    return served, [j for j in route if j not in served]

def build_folium_map(all_routes, all_schedules, op_coords, coords, job_meta):
    m = folium.Map(location=[np.mean([v[0] for v in coords.values()]), np.mean([v[1] for v in coords.values()])], zoom_start=11)
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'darkred', 'cadetblue', 'black']
    for idx, (op, route) in enumerate(all_routes.items()):
        color = colors[idx % len(colors)]
        folium.Marker(op_coords[op], icon=folium.Icon(color=color, icon='home')).add_to(m)
        if route:
            folium.PolyLine([op_coords[op]] + [coords[j] for j in route], color=color, weight=2).add_to(m)
    return m

# --- SIDEBAR (YENİ AYARLAR) ---
with st.sidebar:
    st.header("🎮 Kontrol Paneli")
    method = st.selectbox("Atama Algoritması", ["K-Means (Kümeleme)", "En Yakın Operatör (Greedy)"], help="K-Means bölgeleri ayırır, En Yakın Operatör her işi o an en boş/yakın kişiye verir.")
    zb_target = st.slider("ZB Hizmet Hedefi (%)", 0, 100, 30)
    alpha = st.slider("Öncelik ↔ Mesafe Dengesi (Alpha)", 0.0, 1.0, 0.5)
    
    st.divider()
    st.subheader("💰 Maliyet Ayarları")
    c_tic = st.number_input("Ticari Ceza (₺)", value=2216.0)
    c_mes = st.number_input("Mesken Cezası (₺)", value=277.0)
    hiz = st.slider("Hız (km/dk)", 0.1, 1.5, 0.5)
    
    st.divider()
    uploaded_file = st.file_uploader("Excel Yükle", type=["xlsx"])
    run = st.button("🚀 Hesaplamayı Başlat", use_container_width=True)

# --- ANA DÖNGÜ ---
if uploaded_file and run:
    xls = pd.ExcelFile(uploaded_file)
    df_j = pd.read_excel(xls, sheet_name=0) # Sık Kullanılan...
    df_o = pd.read_excel(xls, sheet_name=1) # User Start...
    
    op_ids = df_o.iloc[:, 0].astype(str).tolist()
    op_coords = {str(r.iloc[0]): (r['latitude'], r['longitude']) for _, r in df_o.iterrows()}
    
    df_j = df_j.dropna(subset=['Tesisat Enlem', 'Tesisat Boylam'])
    job_ids = df_j['Sipariş No'].tolist()
    coords = {r['Sipariş No']: (r['Tesisat Enlem'], r['Tesisat Boylam']) for _, r in df_j.iterrows()}
    job_params = {r['Sipariş No']: job_cost_params(r, c_tic, c_mes) for _, r in df_j.iterrows()}
    job_meta = {r['Sipariş No']: {'tip': str(r['Sipariş Türü'])[:2]} for _, r in df_j.iterrows()}

    op_jobs = {op: [] for op in op_ids}

    # --- ALGORİTMA SEÇİMİ ---
    if "K-Means" in method:
        X = np.array([[coords[j][0], coords[j][1]] for j in job_ids])
        km = KMeans(n_clusters=len(op_ids), random_state=42).fit(X)
        for i, jid in enumerate(job_ids):
            # Basit atama: Her kümeyi bir operatöre ver
            op_jobs[op_ids[km.labels_[i]]].append(jid)
    else:
        # En Yakın Operatör Sezgisel (Her işi en yakın operatörün listesine atar)
        for jid in job_ids:
            j_lat, j_lon = coords[jid]
            best_op = min(op_ids, key=lambda o: dist_km(j_lat, j_lon, op_coords[o][0], op_coords[o][1]))
            op_jobs[best_op].append(jid)

    # --- ROTALAMA ---
    all_routes, all_sch = {}, {}
    for op in op_ids:
        route = _priority_nn_route(op_jobs[op], op_coords[op], coords, job_params, alpha, hiz)
        final_route, _ = _check_feasible(route, op_coords[op], coords, hiz)
        all_routes[op] = final_route
        # Basit schedule oluşturma (Görselleştirme için)
        all_sch[op] = {j: {'served': True, 'arrival': 0, 'fuel_cost': 0, 'fixed_pen': 0, 'tardy_pen': 0} for j in final_route}

    # --- SONUÇLAR ---
    t1, t2 = st.tabs(["🗺️ Harita", "📊 Özet"])
    with t1:
        m = build_folium_map(all_routes, all_sch, op_coords, coords, job_meta)
        components.html(m._repr_html_(), height=600)
    with t2:
        st.write(f"**Seçilen Metot:** {method}")
        st.write(f"**ZB Hedefi:** %{zb_target}")
        # Buraya senin özet rapor fonksiyonlarını ekleyebilirsin
