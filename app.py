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
st.set_page_config(page_title="EnerjiSA Operatör Rotalama", layout="wide")

# --- SABİTLER VE PARAMETRELER ---
VARDIYA_BASLANGIC_SAAT = 8 
VARDIYA_BITIS_SAAT = 18 
OGLEN_MOLA_BASLANGIC = 12 
OGLEN_MOLA_BITIS = 13.5 

SERVIS_SURESI_DK = 10
C_TICARI = 2216.0
C_MESKEN = 277.0

T0 = 0
TEND = int((VARDIYA_BITIS_SAAT - VARDIYA_BASLANGIC_SAAT) * 60)
TBREAK_S = int((OGLEN_MOLA_BASLANGIC - VARDIYA_BASLANGIC_SAAT) * 60)
TBREAK_E = int((OGLEN_MOLA_BITIS - VARDIYA_BASLANGIC_SAAT) * 60)
S_i = SERVIS_SURESI_DK
EPSILON = 0.001

PI = {'ZA': 1.0, 'ZR': 1.0, 'ZS': 0.3, 'ZB': 0.3, 'ZG': 0.3}
PI_DEFAULT = 0.0
PENALTI_TUR = {'ZA', 'ZR'}
RISKY_TUR = {'ZB', 'ZG', 'ZS'}

ZB_COVERAGE_TARGET = 0.30
BALANCE_MAX_ITER = 5
BALANCE_OVERFLOW_TH = 1.0
MAX_TRANSFER_KM = 2.0
MIN_JOBS_PER_OP = 30

KMEANS_INIT = 'k-means++'
KMEANS_N_INIT = 15
KMEANS_MAX_ITER = 500
SKIP_STATUS = {'IPTL', 'İPTAL', 'CANCELLED', 'OK', 'TAMAMLANDI', 'CLOSED', 'KIPT', 'IPTL ODME', 'KOK'}

# --- FONKSİYONLAR ---
def dist_km(lat1, lon1, lat2, lon2):
    return math.sqrt(((lat1 - lat2) * 111) ** 2 + ((lon1 - lon2) * 83) ** 2)

def dk_to_saat(dk):
    if dk is None: return "-"
    h = int(dk // 60) + 8
    return f"{h:02d}:{int(dk % 60):02d}"

def job_cost_params(row):
    ist = str(row.get('Sipariş Türü', '')).upper()[:2]
    abon = str(row.get('Abonelik Türü', ''))
    is_ticari = 'ticarethane' in abon.lower()
    c_d = C_TICARI if is_ticari else C_MESKEN
    pi_i = PI.get(ist, PI_DEFAULT)
    if ist in PENALTI_TUR:
        p_u = c_d
    elif ist in RISKY_TUR:
        p_u = c_d * 0.5
    else:
        p_u = 50.0
    b_i = TEND
    return c_d, pi_i, p_u, b_i

def urgency_score(job_id, job_params_dict):
    c_d, pi_i, p_u, b_i = job_params_dict[job_id]
    return p_u * (1.0 + pi_i)

def unserved_penalty(job_id, job_params_dict):
    c_d, pi_i, p_u, b_i = job_params_dict[job_id]
    return p_u

def teorik_sure(op_coord, job_list, coords, hiz_km_dk):
    if not job_list: return 0.0
    olat, olon = op_coord
    mesafeler = [dist_km(olat, olon, coords[j][0], coords[j][1]) for j in job_list]
    ort_m = float(np.mean(mesafeler))
    return len(job_list) * S_i + (ort_m / hiz_km_dk) * len(job_list)

def compute_workload(op_id, origin, job_list, coords, job_params_dict, hiz_km_dk):
    if not job_list: return {}
    vardiya_dk = TEND - T0
    n = len(job_list)
    olat, olon = origin
    toplam_servis = n * S_i
    mesafeler = [dist_km(olat, olon, coords[j][0], coords[j][1]) for j in job_list]
    ort_mesafe_km = float(np.mean(mesafeler))
    teorik_seyahat = (ort_mesafe_km / hiz_km_dk) * n
    teorik_toplam = toplam_servis + teorik_seyahat
    kapasite_oran = teorik_toplam / vardiya_dk
    fazla_yuk_dk = max(0.0, teorik_toplam - vardiya_dk)
    teorik_max_is = int(vardiya_dk / (S_i + ort_mesafe_km / hiz_km_dk))
    return {
        'n_is': n, 'vardiya_dk': vardiya_dk, 'toplam_servis_dk': toplam_servis,
        'teorik_seyahat_dk': round(teorik_seyahat, 1), 'teorik_toplam_dk': round(teorik_toplam, 1),
        'kapasite_oran': round(kapasite_oran, 3), 'fazla_yuk_dk': round(fazla_yuk_dk, 1),
        'teorik_max_is': teorik_max_is, 'ort_mesafe_km': round(ort_mesafe_km, 2)
    }

def balance_workload(op_jobs, op_ids, op_coords, coords, job_params_dict, hiz_km_dk):
    for iteration in range(BALANCE_MAX_ITER):
        yuk = {op: teorik_sure(op_coords[op], op_jobs[op], coords, hiz_km_dk) for op in op_ids}
        asiri_yuklu = sorted([op for op in op_ids if yuk[op] > TEND * BALANCE_OVERFLOW_TH and op_jobs[op]], key=lambda o: yuk[o], reverse=True)
        if not asiri_yuklu: break
        transferred = False
        for donor in asiri_yuklu:
            yuk_donor = yuk[donor]
            donatable = sorted(op_jobs[donor], key=lambda j: unserved_penalty(j, job_params_dict))
            for job in donatable:
                jlat, jlon = coords[job]
                dist_donor = dist_km(jlat, jlon, op_coords[donor][0], op_coords[donor][1])
                alicilar = sorted(
                    [op for op in op_ids if op != donor and yuk[op] < yuk_donor and dist_km(jlat, jlon, op_coords[op][0], op_coords[op][1]) <= MAX_TRANSFER_KM and dist_km(jlat, jlon, op_coords[op][0], op_coords[op][1]) < dist_donor],
                    key=lambda op: dist_km(jlat, jlon, op_coords[op][0], op_coords[op][1])
                )
                for receiver in alicilar:
                    yuk_r = teorik_sure(op_coords[receiver], op_jobs[receiver] + [job], coords, hiz_km_dk)
                    yuk_d = teorik_sure(op_coords[donor], [j for j in op_jobs[donor] if j != job], coords, hiz_km_dk)
                    if yuk_r < yuk_donor and yuk_d < yuk_donor:
                        op_jobs[donor].remove(job)
                        op_jobs[receiver].append(job)
                        yuk[donor] = yuk_d
                        yuk[receiver] = yuk_r
                        transferred = True
                        break
                if transferred: break
            if transferred: break
        if not transferred: break
    return op_jobs

def adjust_for_lunch(arr_time):
    if arr_time < TBREAK_S:
        if arr_time + S_i > TBREAK_S: return TBREAK_E
    elif TBREAK_S <= arr_time < TBREAK_E:
        return TBREAK_E
    return arr_time

def _check_feasible(route, origin, coords, hiz_km_dk):
    served, unserved = [], []
    lat, lon = origin
    t = T0
    for j in route:
        jlat, jlon = coords[j]
        travel = dist_km(lat, lon, jlat, jlon) / hiz_km_dk
        arr = adjust_for_lunch(t + travel)
        if arr + S_i > TEND: unserved.append(j)
        else:
            served.append(j)
            lat, lon = jlat, jlon
            t = arr + S_i
    return served, unserved

def greedy_select_and_route(op_id, origin, job_list, coords, job_params_dict, hiz_km_dk, yakit_tl_km):
    olat, olon = origin
    sorted_by_penalty = sorted(job_list, key=lambda j: unserved_penalty(j, job_params_dict), reverse=True)
    candidates, elenen_erken = [], []
    t_kalan = TEND - T0
    
    for j in sorted_by_penalty:
        jlat, jlon = coords[j]
        tek_yol_dk = dist_km(olat, olon, jlat, jlon) / hiz_km_dk
        if tek_yol_dk + S_i <= t_kalan: candidates.append(j)
        else: elenen_erken.append(j)

    remaining, route = list(candidates), []
    lat, lon = origin
    while remaining:
        idx = min(range(len(remaining)), key=lambda i: dist_km(lat, lon, coords[remaining[i]][0], coords[remaining[i]][1]))
        j = remaining.pop(idx)
        route.append(j)
        lat, lon = coords[j]

    final_served, elenen_nn = _check_feasible(route, origin, coords, hiz_km_dk)
    unserved = elenen_erken + elenen_nn

    schedule = {}
    lat, lon = olat, olon
    cur_time = T0
    for j in final_served:
        jlat, jlon = coords[j]
        travel = dist_km(lat, lon, jlat, jlon) / hiz_km_dk
        arr = adjust_for_lunch(cur_time + travel)
        if arr < T0: arr = T0
        fin = arr + S_i

        c_d, pi_i, p_u, b_i = job_params_dict[j]
        tardiness = max(0.0, fin - b_i)
        late = tardiness > 0.0
        travel_km = dist_km(lat, lon, jlat, jlon)
        
        schedule[j] = {
            'served': True, 'arrival': arr, 'finish': fin, 'tardiness': tardiness,
            'late': late, 'fuel_cost': yakit_tl_km * travel_km,
            'fixed_pen': c_d if late else 0.0, 'tardy_pen': pi_i * EPSILON * tardiness,
            'unserved_pen': 0.0
        }
        lat, lon = jlat, jlon
        cur_time = fin

    for j in unserved:
        c_d, pi_i, p_u, b_i = job_params_dict[j]
        schedule[j] = {
            'served': False, 'arrival': None, 'finish': None, 'tardiness': 0.0,
            'late': False, 'fuel_cost': 0.0, 'fixed_pen': 0.0, 'tardy_pen': 0.0,
            'unserved_pen': p_u
        }
    return final_served, schedule, unserved

def build_folium_map(all_routes, all_schedules, op_coords, coords, job_meta):
    all_lats = [v[0] for v in coords.values()] + [v[0] for v in op_coords.values()]
    all_lons = [v[1] for v in coords.values()] + [v[1] for v in op_coords.values()]
    center = (np.mean(all_lats), np.mean(all_lons))
    m = folium.Map(location=center, zoom_start=11, tiles='OpenStreetMap')

    COLORS = ['blue', 'red', 'green', 'purple', 'orange', 'darkred', 'cadetblue', 'darkblue']

    for idx, (op_id, route) in enumerate(all_routes.items()):
        color = COLORS[idx % len(COLORS)]
        olat, olon = op_coords[op_id]
        folium.Marker([olat, olon], popup=f"<b>Operatör {op_id}</b><br>Başlangıç", icon=folium.Icon(color=color, icon='home', prefix='fa')).add_to(m)

        if route:
            points = [(olat, olon)] + [coords[j] for j in route] + [(olat, olon)]
            folium.PolyLine(points, color=color, weight=2.5, opacity=0.8).add_to(m)

        sch = all_schedules[op_id]
        for order, j in enumerate(route):
            s = sch[j]
            jlat, jlon = coords[j]
            tip = job_meta.get(j, {}).get('tip', '?')
            popup_txt = f"<b>#{order + 1} - {j}</b><br>Tür: {tip}<br>Varış: {dk_to_saat(s['arrival'])}<br>Maliyet: {s['fuel_cost'] + s['fixed_pen'] + s['tardy_pen']:.1f} ₺"
            folium.CircleMarker([jlat, jlon], radius=7, color=color, fill=True, fill_opacity=0.85, popup=folium.Popup(popup_txt, max_width=220)).add_to(m)

        for j, s in sch.items():
            if not s['served']:
                jlat, jlon = coords[j]
                folium.CircleMarker([jlat, jlon], radius=5, color='gray', fill=True, fill_opacity=0.4, popup=f"{j} - İptal").add_to(m)
    return m

# --- STREAMLIT ARAYÜZÜ ---
st.title("⚡ EnerjiSA Operatör Rotalama Sistemi")
st.markdown("Algoritma: K-Means Kümeleme + Greedy Optimizasyon Modeli")

with st.sidebar:
    st.header("⚙️ Model Parametreleri")
    hiz_km_dk = st.slider("Ortalama Hız (km/dk)", min_value=0.1, max_value=1.5, value=0.5, step=0.1)
    yakit_tl_km = st.number_input("Yakıt Maliyeti (TL/km)", value=5.0, step=0.5)
    st.divider()
    uploaded_file = st.file_uploader("Veri Setini Yükle (Excel)", type=["xlsx"])
    baslat_btn = st.button("🚀 Rotalamayı Başlat", use_container_width=True)

if uploaded_file and baslat_btn:
    with st.spinner("Matematiksel model çalışıyor, rotalar hesaplanıyor..."):
        try:
            xls = pd.ExcelFile(uploaded_file)
            data_sh = next(s for s in xls.sheet_names if any(x in s for x in ['Raporu', 'Sistem', 'Data', 'Sayfa', 'Sık']))
            start_sh = next(s for s in xls.sheet_names if any(x in s for x in ['Start', 'Position', 'Başlangıç', 'User']))

            df_jobs = pd.read_excel(xls, sheet_name=data_sh)
            df_ops = pd.read_excel(xls, sheet_name=start_sh)

            op_col = next((c for c in df_ops.columns if 'User' in c or 'Kullanıcı' in c), df_ops.columns[0])
            df_ops.rename(columns={op_col: 'User 1'}, inplace=True)
            df_ops = df_ops.dropna(subset=['latitude', 'longitude']).reset_index(drop=True)
            op_ids = df_ops['User 1'].astype(str).tolist()
            op_coords = {str(r['User 1']): (r['latitude'], r['longitude']) for _, r in df_ops.iterrows()}

            df_jobs = df_jobs[~df_jobs['Sipariş Durumu'].astype(str).str.strip().str.upper().isin(SKIP_STATUS)].copy()
            df_jobs = df_jobs.dropna(subset=['Tesisat Enlem', 'Tesisat Boylam']).reset_index(drop=True)
            
            K = len(op_ids)
            job_ids = df_jobs['Sipariş No'].tolist()
            coords = {r['Sipariş No']: (float(r['Tesisat Enlem']), float(r['Tesisat Boylam'])) for _, r in df_jobs.iterrows()}
            job_params = {row['Sipariş No']: job_cost_params(row) for _, row in df_jobs.iterrows()}
            
            job_meta = {row['Sipariş No']: {'tip': str(row.get('Sipariş Türü', '')).upper()[:2]} for _, row in df_jobs.iterrows()}

            # K-Means Kümeleme
            X = np.array([[coords[j][0], coords[j][1]] for j in job_ids])
            km = KMeans(n_clusters=K, init=KMEANS_INIT, n_init=KMEANS_N_INIT, random_state=42).fit(X)
            centers = [(km.cluster_centers_[k][0], km.cluster_centers_[k][1]) for k in range(K)]
            
            # Macar Algoritması Ataması
            cost = np.zeros((K, K))
            for k, (clat, clon) in enumerate(centers):
                for o_idx, op in enumerate(op_ids):
                    cost[k, o_idx] = dist_km(clat, clon, *op_coords[op])
            row_ind, col_ind = linear_sum_assignment(cost)
            cluster_to_op = {int(row_ind[i]): op_ids[col_ind[i]] for i in range(len(row_ind))}
            
            op_jobs = {op: [] for op in op_ids}
            for i, jid in enumerate(job_ids):
                op_jobs[cluster_to_op[int(km.labels_[i])]].append(jid)

            op_jobs = balance_workload(op_jobs, op_ids, op_coords, coords, job_params, hiz_km_dk)

            all_routes, all_schedules = {}, {}
            for op_id in op_ids:
                route, schedule, unserved = greedy_select_and_route(op_id, op_coords[op_id], op_jobs[op_id], coords, job_params, hiz_km_dk, yakit_tl_km)
                all_routes[op_id] = route
                all_schedules[op_id] = schedule

            # Sonuçları Görselleştirme
            tab1, tab2, tab3 = st.tabs(["🗺️ Operatör Haritası", "📊 Özet Rapor", "📁 Excel Çıktısı"])

            with tab1:
                st.subheader("İnteraktif Rota Haritası")
                m = build_folium_map(all_routes, all_schedules, op_coords, coords, job_meta)
                components.html(m.get_root().render(), height=600)

            with tab2:
                toplam_servis = sum(len([j for j, s in all_schedules[op].items() if s['served']]) for op in op_ids)
                toplam_iptal = sum(len([j for j, s in all_schedules[op].items() if not s['served']]) for op in op_ids)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Toplam Tamamlanan İş", toplam_servis)
                col2.metric("Toplam Elenen İş", toplam_iptal)
                col3.metric("Aktif Operatör Sayısı", K)

                st.divider()
                st.write("Operatör Bazlı İş Yükü Dağılımı")
                ozet_data = []
                for op in op_ids:
                    servis = len([j for j, s in all_schedules[op].items() if s['served']])
                    maliyet = sum(s['fuel_cost'] + s['fixed_pen'] for j, s in all_schedules[op].items() if s['served'])
                    ozet_data.append({"Operatör": op, "Tamamlanan": servis, "Maliyet (₺)": round(maliyet, 2)})
                st.dataframe(pd.DataFrame(ozet_data), use_container_width=True)

            with tab3:
                st.subheader("Veri Çıktılarını İndir")
                
                rows_servis = []
                for op_id in op_ids:
                    for j, s in all_schedules[op_id].items():
                        if s['served']:
                            rows_servis.append({
                                'Operatör': op_id, 'Sipariş No': j, 'Varış': dk_to_saat(s['arrival']),
                                'Maliyet': round(s['fuel_cost'] + s['fixed_pen'], 2)
                            })
                df_out = pd.DataFrame(rows_servis)
                st.dataframe(df_out)
                
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df_out.to_excel(writer, index=False)
                
                st.download_button(label="📥 Raporu Excel Olarak İndir", data=buffer.getvalue(), file_name="EnerjiSA_Rota_Sonuc.xlsx", mime="application/vnd.ms-excel")
                
        except Exception as e:
            st.error(f"Bir hata oluştu: {str(e)}")
elif not uploaded_file:
    st.info("👈 Lütfen sol panelden EnerjiSA verilerini (Excel formatında) yükleyip başlat butonuna bas.")
