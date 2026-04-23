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
st.set_page_config(page_title="EnerjiSA Dinamik Rotalama", layout="wide")

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Enerjisa_logo.png", width=150)
st.sidebar.header("⚙️ Kontrol Paneli")
uploaded_file = st.sidebar.file_uploader("İş Verisini Yükle (CSV/Excel)", type=['csv', 'xlsx'])

st.sidebar.markdown("### 🧠 Optimizasyon Parametreleri")
alpha_val = st.sidebar.slider("Öncelik-Mesafe Dengesi (Alpha)", 0.0, 1.0, 0.5, 0.1, help="0: Tamamen cezaya/önceliğe odaklan, 1: Tamamen en kısa mesafeye odaklan.")
aging_val = st.sidebar.slider("Yaşlandırma Katsayısı (Aging)", 1.0, 5.0, 2.0, 0.1, help="Ertelenen işin cezası her gün bu katsayı ile çarpılır.")
op_count = st.sidebar.number_input("Operatör Sayısı", min_value=1, max_value=50, value=15)

st.sidebar.markdown("### 🚙 Saha ve Maliyet Parametreleri")
hiz_km_dk = st.sidebar.number_input("Araç Hızı (km/dk)", value=0.5, step=0.1)
zb_hedef = st.sidebar.slider("ZB Tamamlama Hedefi (%)", 0, 100, 30)
c_ticari = st.sidebar.number_input("Ticari Tazminat (TRY)", value=2216.0, step=100.0)
c_mesken = st.sidebar.number_input("Mesken Tazminat (TRY)", value=277.0, step=10.0)

# ==========================================
# 2. DİNAMİK SABİTLER (Arayüzden beslenir)
# ==========================================
VARDIYA_BASLANGIC_SAAT = 8
VARDIYA_BITIS_SAAT = 18
OGLEN_MOLA_BASLANGIC = 12
OGLEN_MOLA_BITIS = 13.5  
SERVIS_SURESI_DK = 10
EPSILON = 0.001
FUEL_RATE = 5.0

T0 = 0
TEND = int((VARDIYA_BITIS_SAAT - VARDIYA_BASLANGIC_SAAT) * 60)
TBREAK_S = int((OGLEN_MOLA_BASLANGIC - VARDIYA_BASLANGIC_SAAT) * 60)
TBREAK_E = int((OGLEN_MOLA_BITIS - VARDIYA_BASLANGIC_SAAT) * 60)
S_i = SERVIS_SURESI_DK

V_KM_MIN = hiz_km_dk
C_TICARI = c_ticari
C_MESKEN = c_mesken
ZB_COVERAGE_TARGET = zb_hedef / 100.0

PI = {'ZA': 1.0, 'ZR': 1.0, 'ZS': 0.3, 'ZB': 0.3, 'ZG': 0.3}
PI_DEFAULT = 0.0
PENALTI_TUR = {'ZA', 'ZR'}
RISKY_TUR = {'ZB', 'ZG', 'ZS'}
BALANCE_MAX_ITER = 5
BALANCE_OVERFLOW_TH = 1.0
MAX_TRANSFER_KM = 2.0
KMEANS_INIT = 'k-means++'

# ==========================================
# 3. YARDIMCI FONKSİYONLAR
# ==========================================
def dist_km(lat1, lon1, lat2, lon2):
    return math.sqrt(((lat1 - lat2) * 111) ** 2 + ((lon1 - lon2) * 83) ** 2)

def job_cost_params(row):
    ist = str(row.get('Sipariş Türü', '')).upper()[:2]
    abon = str(row.get('Abonelik Türü', ''))
    is_ticari = 'ticarethane' in abon.lower()
    c_d = C_TICARI if is_ticari else C_MESKEN
    pi_i = PI.get(ist, PI_DEFAULT)
    if ist in PENALTI_TUR: p_u = c_d
    elif ist in RISKY_TUR: p_u = c_d * 0.5
    else: p_u = 50.0
    return c_d, pi_i, p_u, TEND

def urgency_score(job_id, job_params_dict):
    c_d, pi_i, p_u, b_i = job_params_dict[job_id]
    return p_u * (1.0 + pi_i)

def unserved_penalty(job_id, job_params_dict):
    return job_params_dict[job_id][2]

def teorik_sure(op_coord, job_list, coords):
    if not job_list: return 0.0
    olat, olon = op_coord
    mesafeler = [dist_km(olat, olon, coords[j][0], coords[j][1]) for j in job_list]
    return len(job_list) * S_i + (float(np.mean(mesafeler)) / V_KM_MIN) * len(job_list)

def balance_workload(op_jobs, op_ids, op_coords, coords, job_params_dict):
    for _ in range(BALANCE_MAX_ITER):
        yuk = {op: teorik_sure(op_coords[op], op_jobs[op], coords) for op in op_ids}
        if max(yuk.values()) - min(yuk.values()) < TEND * 0.05: break
        asiri_yuklu = sorted([op for op in op_ids if yuk[op] > TEND * BALANCE_OVERFLOW_TH and op_jobs[op]], key=lambda o: yuk[o], reverse=True)
        if not asiri_yuklu: break
        transferred = False
        for donor in asiri_yuklu:
            yuk_donor = yuk[donor]
            donatable = sorted(op_jobs[donor], key=lambda j: unserved_penalty(j, job_params_dict))
            for job in donatable:
                jlat, jlon = coords[job]
                dist_donor = dist_km(jlat, jlon, op_coords[donor][0], op_coords[donor][1])
                alicilar = sorted([op for op in op_ids if op != donor and yuk[op] < yuk_donor and dist_km(jlat, jlon, op_coords[op][0], op_coords[op][1]) <= MAX_TRANSFER_KM and dist_km(jlat, jlon, op_coords[op][0], op_coords[op][1]) < dist_donor], key=lambda op: dist_km(jlat, jlon, op_coords[op][0], op_coords[op][1]))
                for receiver in alicilar:
                    yuk_r = teorik_sure(op_coords[receiver], op_jobs[receiver] + [job], coords)
                    yuk_d = teorik_sure(op_coords[donor], [j for j in op_jobs[donor] if j != job], coords)
                    if yuk_r < yuk_donor and yuk_d < yuk_donor:
                        op_jobs[donor].remove(job)
                        op_jobs[receiver].append(job)
                        yuk[donor], yuk[receiver] = yuk_d, yuk_r
                        transferred = True
                        break
                if transferred: break
            if transferred: break
        if not transferred: break
    return op_jobs

def adjust_for_lunch(arr_time):
    if arr_time < TBREAK_S and arr_time + S_i > TBREAK_S: return TBREAK_E
    elif TBREAK_S <= arr_time < TBREAK_E: return TBREAK_E
    return arr_time

def _check_feasible(route, origin, coords):
    served, unserved = [], []
    lat, lon, t = origin[0], origin[1], T0
    for j in route:
        jlat, jlon = coords[j]
        arr = adjust_for_lunch(t + dist_km(lat, lon, jlat, jlon) / V_KM_MIN)
        if arr + S_i > TEND: unserved.append(j)
        else:
            served.append(j)
            lat, lon, t = jlat, jlon, arr + S_i
    return served, unserved

def _priority_nn_route(candidates, origin, coords, job_params_dict, alpha):
    remaining, route = list(candidates), []
    lat, lon = origin
    urgency_map = {j: urgency_score(j, job_params_dict) for j in remaining}
    max_urgency = max(urgency_map.values()) if urgency_map else 1.0

    while remaining:
        dists = [dist_km(lat, lon, coords[j][0], coords[j][1]) for j in remaining]
        max_dist = max(dists) if dists else 1.0
        idx = min(range(len(remaining)), key=lambda i: (alpha * (dists[i] / (max_dist + 1e-9)) - (1 - alpha) * (urgency_map[remaining[i]] / (max_urgency + 1e-9))))
        j = remaining.pop(idx)
        route.append(j)
        lat, lon = coords[j]
    return route

def _two_opt(route, origin, coords):
    best = list(route)
    def route_km(r):
        lat, lon, km = origin[0], origin[1], 0.0
        for j in r:
            jlat, jlon = coords[j]
            km += dist_km(lat, lon, jlat, jlon)
            lat, lon = jlat, jlon
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
    sorted_by_urgency = sorted(job_list, key=lambda j: urgency_score(j, job_params_dict), reverse=True)
    candidates, elenen_erken = [], []
    for j in sorted_by_urgency:
        if dist_km(olat, olon, coords[j][0], coords[j][1]) / V_KM_MIN + S_i <= TEND - T0: candidates.append(j)
        else: elenen_erken.append(j)

    nn_route = _priority_nn_route(candidates, origin, coords, job_params_dict, alpha)
    served_nn, elenen_nn = _check_feasible(nn_route, origin, coords)
    route = _check_feasible(_two_opt(served_nn, origin, coords), origin, coords)[0]
    unserved = elenen_erken + elenen_nn + [j for j in served_nn if j not in route]

    schedule, cur_time, lat, lon = {}, T0, olat, olon
    for j in route:
        jlat, jlon = coords[j]
        arr = max(T0, adjust_for_lunch(cur_time + dist_km(lat, lon, jlat, jlon) / V_KM_MIN))
        fin = arr + S_i
        c_d, pi_i, p_u, b_i = job_params_dict[j]
        schedule[j] = {'served': True, 'arrival': arr, 'finish': fin, 'late': (fin - b_i > 0), 'fuel_cost': FUEL_RATE * dist_km(lat, lon, jlat, jlon), 'fixed_pen': c_d if fin - b_i > 0 else 0, 'tardy_pen': pi_i * EPSILON * max(0.0, fin - b_i), 'unserved_pen': 0.0}
        lat, lon, cur_time = jlat, jlon, fin

    for j in unserved:
        schedule[j] = {'served': False, 'arrival': None, 'finish': None, 'late': False, 'fuel_cost': 0.0, 'fixed_pen': 0.0, 'tardy_pen': 0.0, 'unserved_pen': job_params_dict[j][2]}
    return route, schedule, unserved

def build_map(all_routes, all_schedules, op_coords, coords):
    center = (np.mean([v[0] for v in coords.values()]), np.mean([v[1] for v in coords.values()]))
    m = folium.Map(location=center, zoom_start=11)
    COLORS = ['blue', 'red', 'green', 'purple', 'orange', 'darkred', 'cadetblue', 'darkblue']
    
    # Çizgiler ve Tamamlanan İşler
    for idx, (op_id, route) in enumerate(all_routes.items()):
        color = COLORS[idx % len(COLORS)]
        olat, olon = op_coords[op_id]
        folium.Marker([olat, olon], popup=f"Başlangıç: {op_id}", icon=folium.Icon(color=color, icon='home')).add_to(m)
        if route:
            pts = [(olat, olon)] + [coords[j] for j in route] + [(olat, olon)]
            folium.PolyLine(pts, color=color, weight=3.5, opacity=0.8).add_to(m)
            for j in route:
                folium.CircleMarker([coords[j][0], coords[j][1]], radius=6, color=color, fill=True, popup=f"Tamamlandı: {j}").add_to(m)
    
    # Ertelenen Gri İşler (Unserved)
    for op_id, sch in all_schedules.items():
        for j, s in sch.items():
            if not s['served']:
                folium.CircleMarker(
                    [coords[j][0], coords[j][1]], radius=5, color='gray', fill=True, fill_opacity=0.7,
                    popup=f"Ertellendi: {j} (Zaman yetmedi)"
                ).add_to(m)
    return m

# ==========================================
# 4. SİMÜLASYON TETİKLEYİCİSİ
# ==========================================
st.title("⚡ EnerjiSA Dinamik Rotalama Panosu")

# Session state hazırlıkları
if "sim_data" not in st.session_state:
    st.session_state.sim_data = None
    st.session_state.gunler = []

if st.sidebar.button("🚀 Simülasyonu Başlat", type="primary", use_container_width=True):
    if uploaded_file is None:
        st.error("Lütfen önce sol menüden veri seti yükleyin!")
    else:
        with st.spinner("Veriler işleniyor, rotalar çiziliyor. Lütfen bekleyin..."):
            
            # Veri Okuma
            if uploaded_file.name.endswith('.csv'): df_jobs = pd.read_csv(uploaded_file)
            else: df_jobs = pd.read_excel(uploaded_file)
                
            df_jobs['Tesisat Enlem'] = pd.to_numeric(df_jobs['Tesisat Enlem'].astype(str).str.replace(',', '.'), errors='coerce')
            df_jobs['Tesisat Boylam'] = pd.to_numeric(df_jobs['Tesisat Boylam'].astype(str).str.replace(',', '.'), errors='coerce')
            df_jobs = df_jobs.dropna(subset=['Tesisat Enlem', 'Tesisat Boylam']).reset_index(drop=True)
            
            tarih_col = next((c for c in df_jobs.columns if 'Planlanan' in c or 'Tarih' in c), None)
            if tarih_col: df_jobs[tarih_col] = df_jobs[tarih_col].astype(str).str.strip().str[:10]
            else: 
                df_jobs['Planlanan Tarih'] = '24.11.2025'
                tarih_col = 'Planlanan Tarih'

            # --- GERÇEK VERİ KONTROLÜ VE MOCK ÜRETİMİ ---
            # Eğer veri setinde gerçek veriler yoksa, karşılaştırma yapabilmek için sahte veri üretir.
            # Kendi verinizde bu sütunlar varsa aşağıdaki isimleri ona göre düzenleyebilirsiniz.
            if 'Gerçek Durum' not in df_jobs.columns:
                df_jobs['Gerçek Durum'] = np.random.choice(['Tamamlandı', 'Ertelendi'], size=len(df_jobs), p=[0.8, 0.2])
            if 'Gerçekleşen KM' not in df_jobs.columns:
                df_jobs['Gerçekleşen KM'] = np.random.uniform(3.0, 10.0, size=len(df_jobs))

            # Mock Operatörler
            op_ids = [f"Op_{i+1}" for i in range(int(op_count))]
            center_lat, center_lon = df_jobs['Tesisat Enlem'].mean(), df_jobs['Tesisat Boylam'].mean()
            np.random.seed(42)
            op_coords = {op: (center_lat + np.random.uniform(-0.08, 0.08), center_lon + np.random.uniform(-0.08, 0.08)) for op in op_ids}

            coords = {r['Sipariş No']: (float(r['Tesisat Enlem']), float(r['Tesisat Boylam'])) for _, r in df_jobs.iterrows()}
            job_params = {row['Sipariş No']: job_cost_params(row) for _, row in df_jobs.iterrows()}
            
            gunler = sorted([d for d in df_jobs[tarih_col].unique() if '2025' in d or '2026' in d])
            if not gunler: gunler = ['24.11.2025']
            
            gecikme_takip = {jid: {'ilk_planlanan': '', 'tamamlandi_gun': 'Tamamlanmadı', 'gecikme_gun_sayisi': 0, 'durum': 'Bekliyor'} for jid in coords.keys()}
            bekleyen_kuyruk = []
            
            daily_results = {}
            progress_bar = st.progress(0)

            for gun_idx, bugun in enumerate(gunler):
                bugunun_yeni_isleri = df_jobs[df_jobs[tarih_col] == bugun]['Sipariş No'].tolist()
                for j in bugunun_yeni_isleri: gecikme_takip[j]['ilk_planlanan'] = bugun
                    
                aktif_isler = bekleyen_kuyruk + bugunun_yeni_isleri
                if not aktif_isler: 
                    progress_bar.progress((gun_idx + 1) / len(gunler))
                    continue

                K = min(len(op_ids), len(aktif_isler))
                X = np.array([[coords[j][0], coords[j][1]] for j in aktif_isler])
                km = KMeans(n_clusters=K, init=KMEANS_INIT, n_init=10, random_state=42).fit(X)
                labels = {aktif_isler[i]: km.labels_[i] for i in range(len(aktif_isler))}
                centers = [(km.cluster_centers_[k][0], km.cluster_centers_[k][1]) for k in range(K)]

                aktif_op_ids = op_ids[:K]
                cost_matrix = np.zeros((K, K))
                for k, (clat, clon) in enumerate(centers):
                    for o_idx, op in enumerate(aktif_op_ids):
                        cost_matrix[k, o_idx] = dist_km(clat, clon, *op_coords[op])
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                cluster_to_op = {int(row_ind[i]): aktif_op_ids[col_ind[i]] for i in range(K)}

                op_jobs = {op: [] for op in aktif_op_ids}
                for jid, cl in labels.items():
                    op_jobs[cluster_to_op[cl]].append(jid)
                    
                op_jobs = balance_workload(op_jobs, aktif_op_ids, op_coords, coords, job_params)

                bekleyen_kuyruk = []
                gunluk_rotalar = {}
                gunluk_schedules = {}
                gunluk_km = 0.0
                gunluk_tamamlanan = 0
                gunluk_ertelenen = 0
                
                for op in aktif_op_ids:
                    route, schedule, unserved = greedy_select_and_route(op, op_coords[op], op_jobs[op], coords, job_params, alpha_val)
                    gunluk_rotalar[op] = route
                    gunluk_schedules[op] = schedule
                    
                    if route:
                        pts = [op_coords[op]] + [coords[j] for j in route] + [op_coords[op]]
                        gunluk_km += sum(dist_km(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1]) for i in range(len(pts)-1))

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

                # --- GERÇEK VERİLERİ HESAPLAMA ---
                gunluk_gercek_isler = df_jobs[df_jobs[tarih_col] == bugun]
                gercek_tamamlanan = len(gunluk_gercek_isler[gunluk_gercek_isler['Gerçek Durum'] == 'Tamamlandı'])
                gercek_ertelenen = len(gunluk_gercek_isler) - gercek_tamamlanan
                gercek_km = gunluk_gercek_isler['Gerçekleşen KM'].sum()

                daily_results[bugun] = {
                    'routes': gunluk_rotalar,
                    'schedules': gunluk_schedules,
                    'km': gunluk_km,
                    'tamamlanan': gunluk_tamamlanan,
                    'ertelenen': gunluk_ertelenen,
                    'gercek_km': gercek_km,
                    'gercek_tamamlanan': gercek_tamamlanan,
                    'gercek_ertelenen': gercek_ertelenen
                }
                
                progress_bar.progress((gun_idx + 1) / len(gunler))
            
            # Verileri UI için Session'a kaydet
            gecikme_df = pd.DataFrame.from_dict(gecikme_takip, orient='index').reset_index().rename(columns={'index': 'Sipariş No'})
            st.session_state.gecikme_df = gecikme_df[gecikme_df['ilk_planlanan'] != '']
            st.session_state.sim_data = daily_results
            st.session_state.gunler = gunler
            st.session_state.op_coords = op_coords
            st.session_state.coords = coords

# ==========================================
# 5. GÖRSELLEŞTİRME VE ANALİZ
# ==========================================
if st.session_state.sim_data:
    st.success("Optimizasyon Başarıyla Tamamlandı!")
    st.markdown("---")
    
    # GÜN SEÇİCİ (SLIDER)
    st.markdown("### 🗺️ Günlük Rota Haritası")
    selected_day = st.select_slider("Görüntülemek istediğiniz günü kaydırarak seçin:", options=st.session_state.gunler)
    
    day_data = st.session_state.sim_data[selected_day]
    
    # Haritayı Çiz
    map_obj = build_map(day_data['routes'], day_data['schedules'], st.session_state.op_coords, st.session_state.coords)
    components.html(map_obj._repr_html_(), height=500)
    
    # GÜNLÜK ANALİZ: SİMÜLASYON VS GERÇEK
    st.markdown(f"#### 📊 {selected_day} - Simülasyon vs Gerçekleşen Karşılaştırması")
    
    # Fark Hesaplamaları
    sim_km = day_data['km']
    gercek_km = day_data['gercek_km']
    km_fark = sim_km - gercek_km 

    sim_tamam = day_data['tamamlanan']
    gercek_tamam = day_data['gercek_tamamlanan']
    tamam_fark = sim_tamam - gercek_tamam 

    sim_ertelenen = day_data['ertelenen']
    gercek_ertelenen = day_data['gercek_ertelenen']
    ertelenen_fark = sim_ertelenen - gercek_ertelenen

    col1, col2, col3 = st.columns(3)
    
    # delta_color="inverse" parametresi, değerin DÜŞMESİNİN iyi bir şey olduğunu belirtir (Yeşil ok aşağı gösterir).
    col1.metric(
        label="Toplam Araç Mesafesi (Simülasyon)", 
        value=f"{sim_km:.1f} km", 
        delta=f"{km_fark:.1f} km (Gerçeğe Göre)",
        delta_color="inverse" 
    )
    
    # delta_color="normal" parametresi, değerin ARTMASININ iyi bir şey olduğunu belirtir.
    col2.metric(
        label="Tamamlanan İş (Simülasyon)", 
        value=f"{sim_tamam} Adet", 
        delta=f"{tamam_fark} Adet (Gerçeğe Göre)",
        delta_color="normal"
    )
    
    col3.metric(
        label="Ertelenen İş (Simülasyon)", 
        value=f"{sim_ertelenen} Adet", 
        delta=f"{ertelenen_fark} Adet (Gerçeğe Göre)",
        delta_color="inverse"
    )

    # Özet Bilgi Kutusu
    st.info(f"💡 **Sahadaki Gerçek Durum ({selected_day}):** Gerçekte toplam **{gercek_km:.1f} km** yol yapılmış, **{gercek_tamam}** iş kapatılmış ve **{gercek_ertelenen}** iş ertesi güne bırakılmıştı.")
    
    st.markdown("---")
    
    # GENEL HAFTALIK ÖZET
    st.markdown("### 📋 Tüm Simülasyon Boyunca Gecikme Analizi (Kümülatif)")
    df = st.session_state.gecikme_df
    
    toplam_is = len(df)
    zamaninda = len(df[df['gecikme_gun_sayisi'] == 0])
    gec_kalan = len(df[(df['gecikme_gun_sayisi'] > 0) & (df['durum'] == 'Tamamlandı')])
    yapilamayan = len(df[df['durum'] == 'Bekliyor'])
    
    scol1, scol2, scol3, scol4 = st.columns(4)
    scol1.metric("Toplam İş Hacmi", toplam_is)
    scol2.metric("Zamanında Biten", zamaninda)
    scol3.metric("Gecikmeli Biten", gec_kalan)
    scol4.metric("Simülasyon Sonu Bekleyen", yapilamayan)
    
    # Excel Download Butonu
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Gecikme Panosu', index=False)
    output.seek(0)
    
    st.download_button(
        label="📥 Tüm Detaylı Gecikme Raporunu İndir (Excel)",
        data=output,
        file_name="EnerjiSA_Gecikme_Raporu.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
