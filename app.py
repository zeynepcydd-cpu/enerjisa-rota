"""
EnerjiSA — Rotalama ve Karar Destek Sistemi
Streamlit Arayüzü  |  Algoritma: Dinamik Simülasyon v5
"""

import math, warnings, datetime, io
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import folium
import streamlit as st
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  SABİT PARAMETRELER
# ─────────────────────────────────────────────────────────────────────────────
SHIFT_S=8; SHIFT_E=18; BRK_S=12; BRK_E=13.5
S_i=15; V=0.5; FUEL=5.0
C_TIC=2216.0; C_MES=277.0
C_DK_T=5.0; C_DK_M=1.0
T0=0; TEND=int((SHIFT_E-SHIFT_S)*60)
TBRK_S=int((BRK_S-SHIFT_S)*60); TBRK_E=int((BRK_E-SHIFT_S)*60)
DUE_WIN={'ZC':180,'ZB':300,'ZA':360,'ZR':420,'ZD':540,
         'ZN':540,'ZW':540,'ZH':540,'ZS':540,'ZG':540}
SKIP_ST={'IPTL','İPTAL','CANCELLED','OK','TAMAMLANDI','CLOSED','KIPT','IPTL ODME','KOK'}
CAN_ST={'IPTL','BŞSZ','KIPT','IPTL ODME'}
P_MAP={'ZA':1.0,'ZR':1.0,'ZS':0.3,'ZB':0.3,'ZG':0.3}
PEN_T={'ZA','ZR'}; PEN_H={'ZB','ZG','ZS'}
_RGB=['#1F77B4','#D62728','#2CA02C','#9467BD','#FF7F0E','#8C564B',
      '#17BECF','#006400','#800000','#4B0082','#00688B','#8B4513',
      '#556B2F','#00CED1','#8B0000','#228B22','#4169E1','#DC143C',
      '#32CD32','#8A2BE2','#FF8C00','#20B2AA','#B8860B','#483D8B',
      '#A52A2A','#5F9EA0','#CD5C5C','#4682B4','#6B8E23','#C71585']

# ─────────────────────────────────────────────────────────────────────────────
#  YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────────────────────────────────────
def dist_km(a,b,c,d): return math.sqrt(((a-c)*111)**2+((b-d)*83)**2)
def dk2s(dk): h=int(dk//60)+SHIFT_S; return f'{h:02d}:{int(dk%60):02d}'
def s2dk(s):
    try:
        if isinstance(s,datetime.time): return (s.hour-SHIFT_S)*60+s.minute+s.second/60
        p=str(s).strip()[:8].split(':'); return (int(p[0])-SHIFT_S)*60+int(p[1])
    except: return None
def adj_brk(t):
    if t<TBRK_S and t+S_i>TBRK_S: return TBRK_E
    if TBRK_S<=t<TBRK_E: return TBRK_E
    return t
def cost_p(row):
    ist=str(row.get('Sipariş Türü','')).upper()[:2]
    tic='ticarethane' in str(row.get('Abonelik Türü','')).lower()
    c_d=C_TIC if tic else C_MES; c_dk=C_DK_T if tic else C_DK_M
    p_u=c_d if ist in PEN_T else c_d*0.5 if ist in PEN_H else 50.0
    return c_d,P_MAP.get(ist,0.0),p_u,TEND,c_dk
def get_pu(j,JP): return JP[j][2]
def teorik(origin,jobs,coords):
    if not jobs: return 0.0
    ds=[dist_km(origin[0],origin[1],coords[j][0],coords[j][1]) for j in jobs if j in coords]
    return len(jobs)*S_i+(float(np.mean(ds))/V if ds else 0)*len(jobs)
def due_date_of(jid,jt,jc,clamp=False):
    typ=jt.get(jid,'ZD'); win=DUE_WIN.get(typ,540)
    if typ=='ZC': return min(jc.get(jid,0)+win,TEND-30)
    if clamp:
        cre=jc.get(jid,0)
        if cre>win: return min(cre+30,TEND-10)
        return win
    return win

# ─────────────────────────────────────────────────────────────────────────────
#  ROTALAMA ALGORİTMALARI  (v5 ile aynı)
# ─────────────────────────────────────────────────────────────────────────────
def _nn(cands,origin,coords):
    rem=list(cands); route=[]; lat,lon=origin
    while rem:
        i=min(range(len(rem)),key=lambda i:dist_km(lat,lon,coords[rem[i]][0],coords[rem[i]][1]))
        j=rem.pop(i); route.append(j); lat,lon=coords[j]
    return route

def _feasible(route,origin,coords,due_map,t0=T0,tol=30):
    sv=[]; un=[]; lat,lon=origin; t=t0
    for j in route:
        if j not in coords: un.append(j); continue
        arr=adj_brk(t+dist_km(lat,lon,coords[j][0],coords[j][1])/V)
        if arr+S_i>TEND or arr+S_i>due_map.get(j,TEND)+tol: un.append(j)
        else: sv.append(j); lat,lon=coords[j]; t=arr+S_i
    return sv,un

def _rkm(route,origin,coords):
    lat,lon=origin; km=0.0
    for j in route:
        if j not in coords: continue
        km+=dist_km(lat,lon,coords[j][0],coords[j][1]); lat,lon=coords[j]
    return km

def _two_opt(route,origin,coords,due_map,t0):
    best=list(route); improved=True
    while improved:
        improved=False; bkm=_rkm(best,origin,coords)
        for i in range(len(best)-1):
            for j in range(i+2,len(best)):
                cand=best[:i]+best[i:j+1][::-1]+best[j+1:]
                sc,_=_feasible(cand,origin,coords,due_map,t0)
                if len(sc)==len(best) and _rkm(cand,origin,coords)<bkm-0.001:
                    best=cand; improved=True; break
            if improved: break
    return best

def _sched(route,origin,t0,coords,JP,due_map,new_set=None):
    ns=new_set or set(); sch={}; lat,lon=origin; cur=t0
    for j in route:
        if j not in coords: continue
        jlat,jlon=coords[j]; tr=dist_km(lat,lon,jlat,jlon)/V
        arr=adj_brk(cur+tr); arr=max(arr,cur); fin=arr+S_i
        c_d,pi,pu_,_,c_dk=JP.get(j,(277,0,50,600,0.2))
        tard=max(0.0,fin-due_map.get(j,TEND))
        fp=c_d*min(tard/60.0,1.0) if tard>0 else 0.0
        sch[j]={'served':True,'arrival':arr,'finish':fin,
                'fuel_cost':FUEL*dist_km(lat,lon,jlat,jlon),
                'fixed_pen':fp,'tardy_pen':c_dk*tard,
                'unserved_pen':0.0,'tardiness':tard,
                'due':due_map.get(j,TEND),'is_new':j in ns}
        lat,lon=jlat,jlon; cur=fin
    return sch

def route_op(op_id,origin,job_list,coords,JP,due_map,t0=T0,new_set=None,alpha=0.5):
    ns=new_set or set(); olat,olon=origin
    # alpha=0: tamamen öncelik bazlı  |  alpha=1: tamamen mesafe bazlı
    if alpha>=0.99:
        srt=sorted(job_list,key=lambda j:dist_km(olat,olon,coords[j][0],coords[j][1]) if j in coords else 9e9)
    elif alpha<=0.01:
        srt=sorted(job_list,key=lambda j:get_pu(j,JP),reverse=True)
    else:
        max_d=max((dist_km(olat,olon,coords[j][0],coords[j][1]) for j in job_list if j in coords),default=1.0)
        max_p=max((get_pu(j,JP) for j in job_list),default=1.0)
        srt=sorted(job_list,
                   key=lambda j:(alpha*dist_km(olat,olon,coords[j][0],coords[j][1])/max(max_d,1e-9)
                                 -(1-alpha)*get_pu(j,JP)/max(max_p,1e-9)) if j in coords else 9e9)
    cands=[j for j in srt if j in coords and dist_km(olat,olon,coords[j][0],coords[j][1])/V+S_i<=TEND-t0]
    elen=[j for j in job_list if j not in set(cands)]
    nn=_nn(cands,origin,coords); sv,en=_feasible(nn,origin,coords,due_map,t0)
    route=_two_opt(sv,origin,coords,due_map,t0); final,en2=_feasible(route,origin,coords,due_map,t0)
    unserved=elen+en+en2
    sch=_sched(final,origin,t0,coords,JP,due_map,ns)
    for j in unserved:
        c_d,pi,pu_,_,c_dk=JP.get(j,(277,0,50,600,0.2))
        sch[j]={'served':False,'arrival':None,'finish':None,'fuel_cost':0.0,
                'fixed_pen':0.0,'tardy_pen':0.0,'unserved_pen':pu_,
                'tardiness':0.0,'due':due_map.get(j,TEND),'is_new':j in ns}
    return final,sch,unserved

def reroute_commit(op_id,origin,t_start,remaining,new_jobs,cancelled,
                    coords,JP,due_map,commit_n=2,new_set=None):
    ns=new_set or set()
    active=[j for j in remaining+new_jobs if j not in cancelled and j in coords]
    if not active: return [],{}
    committed=[j for j in remaining[:commit_n] if j not in cancelled and j in coords]
    flexible=[j for j in active if j not in set(committed)]
    if committed:
        lat,lon=origin; cur=t_start
        for j in committed:
            tr=dist_km(lat,lon,coords[j][0],coords[j][1])/V
            arr=adj_brk(cur+tr); lat,lon=coords[j]; cur=arr+S_i
        fo,ft=(lat,lon),cur
    else: fo,ft=origin,t_start
    fr,fs,_=route_op(op_id,fo,flexible,coords,JP,due_map,ft,ns)
    cs=_sched(committed,origin,t_start,coords,JP,due_map,ns)
    for j in new_jobs:
        if j in fs: fs[j]['is_new']=True
    return committed+fr,{**cs,**fs}

def kmeans_cluster(job_ids,coords,op_ids,op_coords,n_init=10):
    K=len(op_ids)
    if not job_ids or K==0: return {},[]
    job_pts=np.array([coords[j] for j in job_ids])
    op_pts=np.array([op_coords[op] for op in op_ids])
    try: km=KMeans(n_clusters=K,init=op_pts,n_init=1,max_iter=500,random_state=42).fit(job_pts)
    except: km=KMeans(n_clusters=K,init='k-means++',n_init=n_init,max_iter=500,random_state=42).fit(job_pts)
    labels={job_ids[i]:int(km.labels_[i]) for i in range(len(job_ids))}
    centers=[(km.cluster_centers_[k][0],km.cluster_centers_[k][1]) for k in range(K)]
    cl={k:[] for k in range(K)}
    for jid,c in labels.items(): cl[c].append(jid)
    for k in range(K):
        if not cl[k]:
            big=max(cl,key=lambda x:len(cl[x]))
            bj=min(cl[big],key=lambda j:dist_km(coords[j][0],coords[j][1],centers[k][0],centers[k][1]))
            cl[big].remove(bj); cl[k].append(bj); labels[bj]=k
    return labels,centers

def macar_assign(centers,op_ids,op_coords):
    cost=np.array([[dist_km(centers[k][0],centers[k][1],op_coords[op][0],op_coords[op][1])
                    for op in op_ids] for k in range(len(centers))])
    ri,ci=linear_sum_assignment(cost)
    return {int(ri[i]):op_ids[ci[i]] for i in range(len(ri))}

def balance(op_jobs,op_ids,op_coords,coords,JP,bkm=2.0,iters=5,min_jobs=30):
    for _ in range(iters):
        yuk={op:teorik(op_coords[op],op_jobs[op],coords) for op in op_ids}
        asiri=[op for op in op_ids if yuk[op]>TEND and op_jobs[op]]
        if not asiri: break
        for donor in sorted(asiri,key=lambda o:yuk[o],reverse=True):
            for job in sorted(op_jobs[donor],key=lambda j:get_pu(j,JP)):
                if job not in coords: continue
                jlat,jlon=coords[job]
                dd=dist_km(jlat,jlon,op_coords[donor][0],op_coords[donor][1])
                rcvs=sorted([op for op in op_ids if op!=donor and yuk[op]<yuk[donor]
                             and dist_km(jlat,jlon,op_coords[op][0],op_coords[op][1])<=bkm
                             and dist_km(jlat,jlon,op_coords[op][0],op_coords[op][1])<dd],
                            key=lambda op:dist_km(jlat,jlon,op_coords[op][0],op_coords[op][1]))
                for rec in rcvs:
                    yd=teorik(op_coords[donor],[x for x in op_jobs[donor] if x!=job],coords)
                    yr=teorik(op_coords[rec],op_jobs[rec]+[job],coords)
                    if yr<yuk[donor] and yd<yuk[donor]:
                        op_jobs[donor].remove(job); op_jobs[rec].append(job)
                        yuk[donor]=yd; yuk[rec]=yr; break
    for _ in range(200):
        az=[op for op in op_ids if len(op_jobs[op])<min_jobs]
        if not az: break
        rec=sorted(az,key=lambda o:len(op_jobs[o]))[0]
        rlat,rlon=op_coords[rec]; bj=bd=None; bd_=float('inf')
        for donor in op_ids:
            if donor==rec or len(op_jobs[donor])<=min_jobs: continue
            for j in op_jobs[donor]:
                if j not in coords: continue
                d=dist_km(coords[j][0],coords[j][1],rlat,rlon)
                if d<bd_: bd_=d; bj=j; bd=donor
        if bj is None: break
        op_jobs[bd].remove(bj); op_jobs[rec].append(bj)
    return op_jobs

def op_state(route,schedule,t_sim,origin,coords):
    done=[]; pos=origin; cur=T0
    for j in route:
        s=schedule.get(j,{})
        if s.get('served') and s.get('finish') is not None and s['finish']<=t_sim:
            done.append(j); pos=coords.get(j,pos); cur=s['finish']
        else: break
    ds=set(done)
    return pos,cur,done,[j for j in route if j not in ds]

def _lost(dyn_s,op,done,rem,cancelled):
    ds=set(done); rs=set(rem)
    return [j for j,s in dyn_s[op].items()
            if j not in ds and j not in rs and not s.get('served') and j not in cancelled]

def assign_new(new_jobs,op_ids,states,coords,JP,lb_w=0.3,transfer_km=3.0):
    if not new_jobs: return {op:[] for op in op_ids}
    asgn={op:[] for op in op_ids}
    avg=max(1,np.mean([states[op]['rem_n'] for op in op_ids]))
    for j in sorted(new_jobs,key=lambda j:get_pu(j,JP) if j in JP else 0,reverse=True):
        if j not in coords: continue
        jlat,jlon=coords[j]
        if transfer_km>0:
            cands=[op for op in op_ids
                   if dist_km(states[op]['pos'][0],states[op]['pos'][1],jlat,jlon)<=transfer_km]
            if not cands: cands=op_ids
        else: cands=op_ids
        best=min(cands,key=lambda op:(
            dist_km(states[op]['pos'][0],states[op]['pos'][1],jlat,jlon)*
            (1+lb_w*states[op]['rem_n']/avg)))
        asgn[best].append(j); states[best]['rem_n']+=1
    return asgn

def simulate(cancel_ev,arrival_ev,op_ids,op_coords,op_jobs_init,
             coords,JP,due_map,st_r,st_s,
             n_thr=20,prox_km=0.3,commit_n=2,transfer_km=3.0,lb_w=0.3):
    dyn_r={op:list(r) for op,r in st_r.items()}
    dyn_s={op:{j:dict(s) for j,s in sch.items()} for op,sch in st_s.items()}
    cancelled=set(); new_pool=[]; new_assigned=set(); buf_n=0; n_reopt=0

    def _sts(t):
        sts={}
        for op in op_ids:
            pos,cur,done,rem=op_state(dyn_r[op],dyn_s[op],t,op_coords[op],coords)
            lost=_lost(dyn_s,op,done,rem,cancelled)
            sts[op]={'pos':pos,'t':cur,'done':done,'rem':rem+lost,'rem_n':len(rem)+len(lost)}
        return sts

    def _reopt_all(t):
        nonlocal n_reopt,new_assigned
        sts=_sts(t)
        arrived=[j for ta,j in new_pool if ta<=t and j not in new_assigned]
        asgn=assign_new(arrived,op_ids,sts,coords,JP,lb_w,transfer_km)
        for op in op_ids:
            for j in asgn[op]: new_assigned.add(j)
        for op in op_ids:
            st=sts[op]; ns_=set(asgn.get(op,[]))
            nr,ns=reroute_commit(op,st['pos'],st['t'],st['rem'],asgn.get(op,[]),
                                  cancelled,coords,JP,due_map,commit_n,ns_)
            ds={j:dyn_s[op][j] for j in st['done'] if j in dyn_s[op]}
            dyn_r[op]=st['done']+nr; dyn_s[op]={**ds,**ns}
        n_reopt+=1

    def _reroute1(op,t,extra=None):
        extra=extra or []
        pos,cur,done,rem=op_state(dyn_r[op],dyn_s[op],t,op_coords[op],coords)
        lost=_lost(dyn_s,op,done,rem,cancelled)
        full=[j for j in rem+lost if j not in cancelled]
        nr,ns=reroute_commit(op,pos,cur,full,[j for j in extra if j not in cancelled],
                              cancelled,coords,JP,due_map,commit_n,set(extra))
        ds={j:dyn_s[op][j] for j in done if j in dyn_s[op]}
        dyn_r[op]=done+nr; dyn_s[op]={**ds,**ns}

    all_ev=sorted(
        [(e['t'],'C',e['job'],e.get('op')) for e in cancel_ev]+
        [(e['t'],'N',e['job'],None) for e in arrival_ev],key=lambda x:x[0])

    for t_ev,typ,jid,op in all_ev:
        if typ=='C':
            if jid in cancelled: continue
            cancelled.add(jid)
            tgt=op if (op and op in dyn_r) else None
            if tgt: _reroute1(tgt,t_ev)
        else:
            new_pool.append((t_ev,jid)); buf_n+=1
            if prox_km>0 and jid in coords:
                sts=_sts(t_ev); jlat,jlon=coords[jid]
                near=min(op_ids,key=lambda o:dist_km(sts[o]['pos'][0],sts[o]['pos'][1],jlat,jlon))
                if dist_km(sts[near]['pos'][0],sts[near]['pos'][1],jlat,jlon)<=prox_km:
                    new_assigned.add(jid); _reroute1(near,t_ev,[jid]); buf_n=max(0,buf_n-1)
            if n_thr and buf_n>=n_thr: _reopt_all(t_ev); buf_n=0

    return dyn_r,dyn_s,new_assigned,cancelled,n_reopt

def compute_oracle(cancel_ev,arrival_ev,op_ids,op_coords,op_jobs_init,
                    coords,JP,due_map,jtype_map,jcre_map):
    can_set={e['job'] for e in cancel_ev}
    new_arr={e['job'] for e in arrival_ev if e['job'] in coords}
    oracle_ids=[j for op in op_ids for j in op_jobs_init[op] if j not in can_set]
    for j in new_arr:
        if j not in set(oracle_ids): oracle_ids.append(j)
    od=dict(due_map)
    for j in new_arr:
        if jcre_map.get(j,0)>0: od[j]=due_date_of(j,jtype_map,jcre_map,clamp=True)
    labels,centers=kmeans_cluster(oracle_ids,coords,op_ids,op_coords)
    c2o=macar_assign(centers,op_ids,op_coords)
    op_jobs={op:[] for op in op_ids}
    for jid,cl in labels.items(): op_jobs[c2o[cl]].append(jid)
    op2cl={op:cl for cl,op in c2o.items()}
    op_start={op:centers[op2cl[op]] for op in op_ids}
    op_jobs=balance(op_jobs,op_ids,op_start,coords,JP)
    or_r={}; or_s={}
    for op in op_ids:
        r,s,_=route_op(op,op_start[op],op_jobs[op],coords,JP,od)
        or_r[op]=r; or_s[op]=s
    return or_r,or_s,od,op_start

def total_cost(routes,schedules,op_ids,op_coords,coords,JP,due_map,rollover=0.1,due_exc=0.3):
    km=sum(_rkm(routes.get(op,[]),op_coords[op],coords) for op in op_ids)
    sp=sum(sum(s.get('fixed_pen',0)+s.get('tardy_pen',0) for s in sch.values())
           for sch in schedules.values())
    up=sum(JP.get(jid,(0,0,50))[2]*(1+rollover+(due_exc if due_map.get(jid,TEND)<TEND else 0))
           for sch in schedules.values() for jid,s in sch.items() if not s.get('served'))
    return km*FUEL+sp+up, km

def metrics(routes,schedules,op_ids,op_coords,coords,new_asgn,orig_ids):
    srv=sum(sum(1 for s in sch.values() if s.get('served')) for sch in schedules.values())
    uns=sum(sum(1 for s in sch.values() if not s.get('served')) for sch in schedules.values())
    km=sum(_rkm(routes.get(op,[]),op_coords[op],coords) for op in op_ids)
    fuel=sum(sum(s.get('fuel_cost',0) for s in sch.values()) for sch in schedules.values())
    tard=sum(sum(s.get('tardy_pen',0)+s.get('fixed_pen',0) for s in sch.values()) for sch in schedules.values())
    nsrv=sum(sum(1 for j,s in sch.items() if s.get('served') and j in new_asgn) for sch in schedules.values())
    late=sum(sum(1 for s in sch.values() if s.get('served') and s.get('tardiness',0)>0) for sch in schedules.values())
    return {'srv':srv,'uns':uns,'km':km,'fuel':fuel,'tard':tard,'nsrv':nsrv,'late':late,
            'kpj':km/max(srv,1),'srv_pct':srv/max(srv+uns,1)*100}

def reconstruct_historical(df_raw,due_map,JP,jtype_map):
    OP_COL='Siparişi Tamamlayan Kullanıcı 1'
    ok=df_raw[df_raw['Sipariş Durumu'].astype(str).str.strip().str.upper()=='OK'].copy()
    if OP_COL not in ok.columns: return {}
    ok['_t']=ok['Tamamlanma Saati'].apply(s2dk)
    for col in ['Tesisat Enlem','Tesisat Boylam']:
        ok[col]=pd.to_numeric(ok[col],errors='coerce')
    ok=ok.dropna(subset=['_t','Tesisat Enlem','Tesisat Boylam'])
    ok['_t']=ok['_t'].astype(float)
    if ok.empty: return {}
    historical={}
    for op,grp in ok.groupby(OP_COL):
        op=str(op); grp_s=grp.sort_values('_t')
        jobs=grp_s['Sipariş No'].tolist()
        pts=[(float(r['Tesisat Enlem']),float(r['Tesisat Boylam'])) for _,r in grp_s.iterrows()]
        times=grp_s['_t'].tolist()
        km=sum(dist_km(pts[i][0],pts[i][1],pts[i+1][0],pts[i+1][1]) for i in range(len(pts)-1))
        tardy_pen=0.0
        for jid,t_fin in zip(jobs,times):
            d=due_map.get(jid,TEND); tard=max(0.0,t_fin-d)
            c_d=JP.get(jid,(277,0,0,0,0))[0]; c_dk=JP.get(jid,(0,0,0,0,C_DK_M))[4]
            fp=c_d*min(tard/60.0,1.0) if tard>0 else 0.0
            tardy_pen+=fp+c_dk*tard
        historical[op]={'jobs':jobs,'positions':pts,'times':times,'km':km,
                        'n_served':len(jobs),'tardy_pen':tardy_pen,
                        'fuel':km*FUEL,'cost':km*FUEL+tardy_pen}
    return historical

# ─────────────────────────────────────────────────────────────────────────────
#  HARİTA
# ─────────────────────────────────────────────────────────────────────────────
def _jcol(tard,is_new,is_uns,rgb):
    if is_uns:  return '#9E9E9E'
    if tard>0:  return '#FF8C00'
    return rgb

def _op_popup(op,route,sch,op_coords,coords,JP,due_map):
    srv=[j for j in route if sch.get(j,{}).get('served')]
    uns=[j for j in route if not sch.get(j,{}).get('served')]
    km=_rkm(route,op_coords[op],coords)
    late=[j for j in srv if sch.get(j,{}).get('tardiness',0)>0]
    tp=sum(sch.get(j,{}).get('tardy_pen',0)+sch.get(j,{}).get('fixed_pen',0) for j in srv)
    up=sum(JP.get(j,(0,0,50))[2] for j in uns)
    return (f'<b>Op {op}</b><br>✓{len(srv)} servis | ✗{len(uns)} yapılamadı<br>'
            f'⚠{len(late)} gecikmeli | km:{km:.1f}<br>'
            f'Gecikme:{tp:.0f}₺ | Atanamama:{up:.0f}₺<br>'
            f'<b>Toplam:{km*FUEL+tp+up:.0f}₺</b>')

def make_map(routes,schedules,op_ids,op_coords,coords,df,
             cancelled=None,new_assigned=None,JP=None,due_map=None):
    cancelled=cancelled or set(); new_assigned=new_assigned or set()
    JP=JP or {}; due_map=due_map or {}
    lats=[v[0] for v in op_coords.values()]; lons=[v[1] for v in op_coords.values()]
    m=folium.Map(location=(float(np.mean(lats)),float(np.mean(lons))),zoom_start=12)
    legend="""<div style="position:fixed;bottom:30px;left:10px;z-index:1000;
        background:white;padding:10px;border-radius:8px;border:1px solid #ccc;font-size:12px;opacity:0.93">
    <b>İş Durumu</b><br>
    <span style="color:#1F77B4">●</span> Zamanında (operatör rengi)<br>
    <span style="color:#FF8C00">●</span> Gecikmeli<br>
    <span style="color:#1F77B4">◉</span> Yeni gelen iş<br>
    <span style="color:#9E9E9E">●</span> Yapılamadı<br>
    <span style="color:#CC0000">◉</span> İptal / BŞSZ</div>"""
    m.get_root().html.add_child(folium.Element(legend))
    jtm=dict(zip(df['Sipariş No'],df['Sipariş Türü'])) if 'Sipariş Türü' in df.columns else {}
    for idx,op in enumerate(op_ids):
        rgb=_RGB[idx%len(_RGB)]
        olat,olon=op_coords[op]
        route=routes.get(op,[]); sch=schedules.get(op,{})
        fg=folium.FeatureGroup(name=f'Op {op}',show=True)
        folium.CircleMarker([olat,olon],radius=12,color='black',fill=True,
            fill_color=rgb,fill_opacity=0.95,weight=2,
            popup=folium.Popup(_op_popup(op,route,sch,op_coords,coords,JP,due_map),max_width=260),
            tooltip=f'Op {op}').add_to(fg)
        cpts=[coords[j] for j in route if j in coords]
        if cpts: folium.PolyLine([(olat,olon)]+cpts,color=rgb,weight=2,opacity=0.7).add_to(fg)
        for seq,j in enumerate(route):
            s=sch.get(j,{}); co=coords.get(j)
            if not co: continue
            jlat,jlon=co; is_new=j in new_assigned; is_uns=not s.get('served',False)
            tard=s.get('tardiness',0) if s.get('served') else 0
            col=_jcol(tard,is_new,is_uns,rgb); rad=3 if is_uns else 5
            arr_s=f'<br>{dk2s(s["arrival"])}→{dk2s(s["finish"])}' if s.get('arrival') else '<br>YAPILAMADI'
            tard_s=f'<br>⚠{tard:.0f}dk' if tard>0 else ''
            popup=folium.Popup(
                f'<b>#{seq+1} {j}</b><br>Tür:{jtm.get(j,"?")}'+(' [YENİ]' if is_new else '')
                +arr_s+tard_s,max_width=200)
            folium.CircleMarker([jlat,jlon],radius=rad,color=col,fill=True,
                fill_color=col,fill_opacity=0.9,popup=popup).add_to(fg)
            if is_new and not is_uns:
                folium.CircleMarker([jlat,jlon],radius=2,color='white',fill=True,
                    fill_color='white',fill_opacity=1.0,weight=1).add_to(fg)
        fg.add_to(m)
    fg_c=folium.FeatureGroup(name='İptaller',show=True)
    for jid in sorted(cancelled):
        co=coords.get(jid)
        if not co: continue
        folium.CircleMarker([co[0],co[1]],radius=4,color='#CC0000',fill=True,
            fill_color='#CC0000',fill_opacity=0.85,
            popup=folium.Popup(f'✕ {jid} İPTAL',max_width=120)).add_to(fg_c)
    fg_c.add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    return m._repr_html_()

def make_historical_map(historical,op_coords_hint,df,due_map=None,JP=None):
    due_map=due_map or {}; JP=JP or {}
    if not historical: return None
    all_pts=[p for v in historical.values() for p in v['positions']]
    if not all_pts: return None
    lats=[p[0] for p in all_pts]; lons=[p[1] for p in all_pts]
    m=folium.Map(location=(float(np.mean(lats)),float(np.mean(lons))),zoom_start=12)
    legend="""<div style="position:fixed;bottom:30px;left:10px;z-index:1000;
        background:white;padding:10px;border-radius:8px;border:1px solid #ccc;font-size:12px;opacity:0.93">
    <b>Gerçek Operatör Rotaları</b><br>
    <span style="color:#1F77B4">●</span> Zamanında<br>
    <span style="color:#FF8C00">●</span> Gecikmeli</div>"""
    m.get_root().html.add_child(folium.Element(legend))
    jtm=dict(zip(df['Sipariş No'],df['Sipariş Türü'])) if 'Sipariş Türü' in df.columns else {}
    for idx,(op,h) in enumerate(historical.items()):
        rgb=_RGB[idx%len(_RGB)]
        fg=folium.FeatureGroup(name=f'Op {op}',show=True)
        pts=h['positions']
        if not pts: continue
        start=op_coords_hint.get(op,pts[0]) if op_coords_hint else pts[0]
        folium.CircleMarker([start[0],start[1]],radius=12,color='black',fill=True,
            fill_color=rgb,fill_opacity=0.95,weight=2,
            popup=folium.Popup(f'<b>Op {op}</b><br>✓{h["n_served"]} iş<br>'
                               f'km:{h["km"]:.1f} | {h["cost"]:.0f}₺',max_width=180),
            tooltip=f'Op {op}').add_to(fg)
        all_pts_r=[start]+pts
        folium.PolyLine(all_pts_r,color=rgb,weight=2,opacity=0.8).add_to(fg)
        for seq,(pt,t_fin,jid) in enumerate(zip(pts,h['times'],h['jobs'])):
            d=due_map.get(jid,TEND); tard=max(0.0,t_fin-d)
            col=rgb if tard<=0 else '#FF8C00'
            folium.CircleMarker([pt[0],pt[1]],radius=4,color=col,fill=True,fill_color=col,fill_opacity=0.9,
                popup=folium.Popup(
                    f'<b>#{seq+1} {jid}</b><br>Tür:{jtm.get(jid,"?")}<br>Bit:{dk2s(t_fin)}'
                    +(f'<br>⚠{tard:.0f}dk' if tard>0 else ''),max_width=170)).add_to(fg)
        fg.add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    return m._repr_html_()

# ─────────────────────────────────────────────────────────────────────────────
#  FORMAT TESPİT + HAFTALIK YARDIMCILAR
# ─────────────────────────────────────────────────────────────────────────────
SKIP_WK={'KOK'}

def detect_format(xls):
    sheets=xls.sheet_names
    if any(any(k in s for k in ['Start','Position','User','Başlangıç']) for s in sheets):
        return 'daily'
    df0=pd.read_excel(xls,sheet_name=sheets[0],nrows=0)
    if 'Yaratma Tarihi' in df0.columns:
        dft=pd.read_excel(xls,sheet_name=sheets[0],usecols=['Yaratma Tarihi'])
        return 'weekly' if dft['Yaratma Tarihi'].nunique()>1 else 'daily'
    return 'daily'

def build_job_pool(df_raw):
    df=df_raw[~df_raw['Sipariş Durumu'].astype(str).str.strip().str.upper().isin(SKIP_WK)].copy()
    for col in ['Tesisat Enlem','Tesisat Boylam']:
        df[col]=pd.to_numeric(df[col],errors='coerce')
    df=df.dropna(subset=['Tesisat Enlem','Tesisat Boylam']).reset_index(drop=True)
    coords={r['Sipariş No']:(float(r['Tesisat Enlem']),float(r['Tesisat Boylam'])) for _,r in df.iterrows()}
    JP={r['Sipariş No']:cost_p(r) for _,r in df.iterrows()}
    jtype_map=dict(zip(df['Sipariş No'],df['Sipariş Türü'].str[:2].str.upper()))
    jcre_map={}
    for _,row in df.iterrows():
        t=s2dk(row.get('Yaratma Saati','')); jcre_map[row['Sipariş No']]=t if (t and 0<=t<=TEND) else 0
    return df,coords,JP,jtype_map,jcre_map

def extract_day_ops(df_raw,day):
    op_col='Siparişi Tamamlayan Kullanıcı 1'
    if op_col not in df_raw.columns: return [],{}
    mask=(pd.to_datetime(df_raw['Yaratma Tarihi'],errors='coerce').dt.date==day)
    df_d=df_raw[mask].dropna(subset=['Tesisat Enlem','Tesisat Boylam'])
    cl={}
    for _,row in df_d.iterrows():
        op=str(row.get(op_col,'')).strip()
        if op in ('nan','None','','NaN'): continue
        cl.setdefault(op,[]).append((float(row['Tesisat Enlem']),float(row['Tesisat Boylam'])))
    return (list(cl.keys()),
            {op:(float(np.mean([c[0] for c in v])),float(np.mean([c[1] for c in v]))) for op,v in cl.items()})

def build_cancel_pool(df_raw,day,active_ids):
    mask=(pd.to_datetime(df_raw['Yaratma Tarihi'],errors='coerce').dt.date==day)
    evs=[]; seen=set()
    for _,row in df_raw[mask].iterrows():
        jid=str(row.get('Sipariş No','')); st=str(row.get('Sipariş Durumu','')).upper()
        if jid not in active_ids or st not in CAN_ST or jid in seen: continue
        t=s2dk(row.get('Tamamlanma Saati',''))
        if t and 0<=t<=TEND: evs.append({'t':t,'job':jid,'op':None}); seen.add(jid)
    return sorted(evs,key=lambda e:e['t'])

def sim_day_rolling(day,pool_ids,op_ids,op_coords,coords,JP,due_map,jtype_map,
                    cancel_ev,arrival_ev,p):
    if len(pool_ids)<len(op_ids):
        op_ids=op_ids[:max(1,len(pool_ids)//2+1)]
        op_coords={k:v for k,v in op_coords.items() if k in op_ids}
    if not op_ids or not pool_ids: return {},{},set(),set(),0,{},{},{},op_ids,{}
    labels,centers=kmeans_cluster(pool_ids,coords,op_ids,op_coords)
    c2o=macar_assign(centers,op_ids,op_coords)
    op_jobs={op:[] for op in op_ids}
    for jid,cl in labels.items(): op_jobs[c2o[cl]].append(jid)
    op2cl={op:cl for cl,op in c2o.items()}
    rs={op:centers[op2cl[op]] for op in op_ids}
    auto_min=max(1,len(pool_ids)//(len(op_ids)*3))
    op_jobs=balance(op_jobs,op_ids,rs,coords,JP,2.0,5,auto_min)
    st_r={}; st_s={}
    for op in op_ids:
        r,s,_=route_op(op,rs[op],op_jobs[op],coords,JP,due_map,alpha=p.get('alpha',0.5))
        st_r[op]=r; st_s[op]=s
    dyn_r,dyn_s,new_asgn,can,n_r=simulate(
        cancel_ev,arrival_ev,op_ids,rs,op_jobs,coords,JP,due_map,st_r,st_s,
        n_thr=p.get('n_thr',20),prox_km=p.get('prox_km',0.3),
        commit_n=p.get('commit_n',2),transfer_km=p.get('transfer_km',3.0))
    return dyn_r,dyn_s,can,new_asgn,n_r,st_r,st_s,rs,op_ids,op_jobs
def load_from_upload(uploaded_file):
    raw=uploaded_file.read()
    xls=pd.ExcelFile(io.BytesIO(raw))
    fmt=detect_format(xls)
    dsh=next(s for s in xls.sheet_names if any(x in s for x in ['Raporu','Sistem','Sık','Data','Sayfa']))
    df_raw=pd.read_excel(xls,sheet_name=dsh)
    ssh=next((s for s in xls.sheet_names if any(x in s for x in ['Start','Position','User','Başlangıç'])),None)
    op_ids=None; op_coords=None
    if ssh:
        df_ops=pd.read_excel(xls,sheet_name=ssh); df_ops.columns=df_ops.columns.str.strip()
        uc=next((c for c in df_ops.columns if 'User' in c),df_ops.columns[0])
        df_ops.rename(columns={uc:'User 1'},inplace=True); df_ops['User 1']=df_ops['User 1'].astype(str)
        df_ops=df_ops.dropna(subset=['latitude','longitude']).reset_index(drop=True)
        op_ids=df_ops['User 1'].tolist()
        op_coords={r['User 1']:(r['latitude'],r['longitude']) for _,r in df_ops.iterrows()}
    df=df_raw[~df_raw['Sipariş Durumu'].astype(str).str.strip().str.upper().isin(SKIP_ST)].copy()
    for col in ['Tesisat Enlem','Tesisat Boylam']:
        df[col]=pd.to_numeric(df[col],errors='coerce')
    df=df.dropna(subset=['Tesisat Enlem','Tesisat Boylam']).reset_index(drop=True)
    job_ids=df['Sipariş No'].tolist()
    coords={r['Sipariş No']:(float(r['Tesisat Enlem']),float(r['Tesisat Boylam'])) for _,r in df.iterrows()}
    JP={r['Sipariş No']:cost_p(r) for _,r in df.iterrows()}
    jtype_map=dict(zip(df['Sipariş No'],df['Sipariş Türü'].str[:2].str.upper()))
    jcre_map={}
    for _,row in df.iterrows():
        t=s2dk(row.get('Yaratma Saati','')); jcre_map[row['Sipariş No']]=t if (t and 0<=t<=TEND) else 0
    due_map={j:due_date_of(j,jtype_map,jcre_map) for j in job_ids}
    return df_raw,df,op_ids,op_coords,job_ids,coords,JP,due_map,jtype_map,jcre_map,fmt

def build_events(df_raw,orig_ids,op_jobs,coords,JP,due_map,jtype_map,jcre_map):
    is2op={j:op for op,jl in op_jobs.items() for j in jl}
    cev=[]; seen=set()
    for _,row in df_raw.iterrows():
        jid=str(row.get('Sipariş No','')); st=str(row.get('Sipariş Durumu','')).upper()
        if jid not in orig_ids or st not in CAN_ST or jid in seen: continue
        t=s2dk(row.get('Tamamlanma Saati',''))
        if t and 0<=t<=TEND: cev.append({'t':t,'job':jid,'op':is2op.get(jid)}); seen.add(jid)
    aev=[]
    for _,row in df_raw.iterrows():
        jid=str(row.get('Sipariş No',''))
        if jid in orig_ids: continue
        t=s2dk(row.get('Yaratma Saati',''))
        if t is None or not (0<=t<=TEND): continue
        lat=row.get('Tesisat Enlem'); lon=row.get('Tesisat Boylam')
        if pd.isna(lat) or pd.isna(lon): continue
        aev.append({'t':t,'job':jid})
        if jid not in JP: JP[jid]=cost_p(row)
        if jid not in coords: coords[jid]=(float(lat),float(lon))
        if jid not in jtype_map: jtype_map[jid]=str(row.get('Sipariş Türü','ZD')).upper()[:2]
        if jid not in jcre_map: jcre_map[jid]=t
        if jid not in due_map: due_map[jid]=due_date_of(jid,jtype_map,jcre_map,clamp=t>0)
    return sorted(cev,key=lambda e:e['t']),sorted(aev,key=lambda e:e['t'])

# ─────────────────────────────────────────────────────────────────────────────
#  STREAMLIT ARAYÜZÜ
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="EnerjiSA Rotalama Sistemi",layout="wide",page_icon="⚡")

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/4/41/Enerjisa_logo.png",
    width=150
)

st.sidebar.markdown("### 📂 Veri")
uploaded_file=st.sidebar.file_uploader("Excel dosyası yükle",type=['xlsx'])

st.sidebar.markdown("### ⚙️ Parametreler")
n_thr   =st.sidebar.number_input("Yeni iş tetikleyici N",1,100,20)
prox_km =st.sidebar.number_input("Yakınlık eşiği km (0=kapalı)",0.0,5.0,0.3,0.1)
commit_n=st.sidebar.number_input("Sabit ilk N iş",1,10,2)
transfer=st.sidebar.number_input("Maks. atama km (0=kısıtsız)",0.0,10.0,3.0,0.5)
alpha   =st.sidebar.slider("Öncelik-Mesafe Dengesi (Alpha)",0.0,1.0,0.5,0.05,
                            help="0 = tamamen öncelik bazlı, 1 = tamamen mesafe bazlı")
rollover=st.sidebar.number_input("Erteleme katsayısı",0.0,1.0,0.1,0.05)
due_exc =st.sidebar.number_input("Vade aşımı katsayısı",0.0,1.0,0.3,0.05)

calistir=st.sidebar.button("🚀 Modeli Çalıştır",type="primary",use_container_width=True)

st.title("⚡ EnerjiSA Rotalama ve Karar Destek Sistemi")
st.markdown("Statik plan, dinamik simülasyon ve gerçek operatör rotalarının karşılaştırmalı analizi.")

if not calistir:
    st.info("Sol menüden Excel dosyasını yükleyin ve **Modeli Çalıştır** butonuna basın.")
    st.stop()

if uploaded_file is None:
    st.error("Lütfen önce veri dosyasını yükleyin.")
    st.stop()

with st.spinner("Veri yükleniyor..."):
    try:
        df_raw,df,op_ids,op_coords,job_ids,coords,JP,due_map,jtype_map,jcre_map,fmt=load_from_upload(uploaded_file)
    except Exception as e:
        st.error(f"Veri yükleme hatası: {e}")
        st.stop()

st.sidebar.success(f"✓ {len(job_ids)} iş | Format: {fmt.upper()}")

# ══════════════════════════════════════════════════════
#  HAFTALIK VERİ AKIŞI
# ══════════════════════════════════════════════════════
if fmt=='weekly':
    import datetime as _dt

    with st.spinner("Haftalık iş havuzu hazırlanıyor..."):
        df_pool,pool_coords,pool_JP,pool_jtype,pool_jcre=build_job_pool(df_raw)
        all_ids=set(df_pool['Sipariş No'])
        coords.update(pool_coords); JP.update(pool_JP)
        jtype_map.update(pool_jtype); jcre_map.update(pool_jcre)
        due_map.update({j:due_date_of(j,jtype_map,jcre_map) for j in all_ids})
        df_raw['_date']=pd.to_datetime(df_raw['Yaratma Tarihi'],errors='coerce').dt.date
        all_days=sorted(df_raw['_date'].dropna().unique())
        work_days=[d for d in all_days if _dt.date.weekday(d)<5]
        sim_days=work_days[-5:] if len(work_days)>=5 else work_days
        backlog=set(df_raw[pd.to_datetime(df_raw['Yaratma Tarihi'],errors='coerce').dt.date
                           .apply(lambda d: d<sim_days[0] if d else False)]['Sipariş No'])&all_ids
        day_new={day:set(df_raw[pd.to_datetime(df_raw['Yaratma Tarihi'],errors='coerce').dt.date==day
                                ]['Sipariş No'])&all_ids for day in sim_days}

    p={'n_thr':int(n_thr),'prox_km':float(prox_km),'commit_n':int(commit_n),
       'transfer_km':float(transfer),'alpha':float(alpha),'lb_w':0.3}

    st.markdown(f"### 📅 Haftalık Simülasyon: {sim_days[0]} → {sim_days[-1]}")
    st.caption(f"Backlog: {len(backlog)} iş | Sim. günleri: {len(sim_days)}")

    progress=st.progress(0); status=st.empty()
    day_results=[]; all_left={}; carryover=set(backlog)

    for i,day in enumerate(sim_days):
        status.text(f"Gün {i+1}/{len(sim_days)}: {day} işleniyor...")
        progress.progress((i)/len(sim_days))

        today_new=day_new.get(day,set())
        today_pre=set(); arrival_ev_today=[]
        for jid in today_new:
            if jid not in all_ids: continue
            t_cre=jcre_map.get(jid,0)
            if t_cre<=0: today_pre.add(jid)
            else:
                arrival_ev_today.append({'t':t_cre,'job':jid})
                if jid not in due_map or due_map[jid]<t_cre:
                    due_map[jid]=due_date_of(jid,jtype_map,jcre_map,clamp=True)
        arrival_ev_today.sort(key=lambda e:e['t'])
        pool_ids=list((carryover|today_pre)&all_ids)
        day_op_ids,day_op_coords=extract_day_ops(df_raw,day)

        if not day_op_ids or not pool_ids:
            day_results.append({'day':day,'srv':0,'km':0,'cancelled':0,
                                 'n_carry':len(carryover),'n_pre':len(today_pre),
                                 'n_in':len(arrival_ev_today),'n_ops':0,
                                 'vpi':None,'carryover_n':len(carryover)})
            continue
        try:
            cans=build_cancel_pool(df_raw,day,set(pool_ids)|{e['job'] for e in arrival_ev_today})
            out=sim_day_rolling(day,pool_ids,list(day_op_ids),dict(day_op_coords),
                                coords,JP,due_map,jtype_map,cans,arrival_ev_today,p)
            dyn_r,dyn_s,cancelled,new_asgn,n_r,st_r,st_s,rs,used_ops,op_jobs=out
        except Exception as e:
            st.warning(f"  {day} hatası: {e}")
            day_results.append({'day':day,'srv':0,'km':0,'cancelled':0,
                                 'n_carry':len(carryover),'n_pre':len(today_pre),
                                 'n_in':len(arrival_ev_today),'n_ops':len(day_op_ids),
                                 'vpi':None,'carryover_n':0})
            continue

        dm=metrics(dyn_r,dyn_s,used_ops,rs,coords,new_asgn,set(pool_ids))

        # VPI günlük
        try:
            or_r,or_s,or_due,o_start=compute_oracle(
                cans,arrival_ev_today,used_ops,rs,op_jobs,coords,JP,due_map,jtype_map,jcre_map)
            f0c,_=total_cost(st_r,st_s,used_ops,rs,coords,JP,due_map,rollover,due_exc)
            fdc,_=total_cost(dyn_r,dyn_s,used_ops,rs,coords,JP,due_map,rollover,due_exc)
            fsc,_=total_cost(or_r,or_s,used_ops,o_start,coords,JP,or_due,rollover,due_exc)
            new_pen=sum(JP.get(e['job'],(0,0,50))[2]*(1+rollover+(due_exc if due_map.get(e['job'],TEND)<TEND else 0))
                        for e in arrival_ev_today if e['job'] in JP)
            f0_aug=f0c+new_pen
            def _p(c): return c/max(f0_aug,1)*100
            def _i(c): return (f0_aug-c)/max(f0_aug,1)*100
            vpi_row={'fs':_p(f0_aug),'fd':_p(fdc),'ft':_p(fsc),
                     'dyn_imp':_i(fdc),'vpi_imp':_i(fsc),
                     'eff':_i(fdc)/max(_i(fsc),0.001)*100}
        except Exception:
            vpi_row=None

        # Carryover
        carryover=set()
        for op_sch in dyn_s.values():
            for jid,s in op_sch.items():
                if not s.get('served') and jid not in cancelled: carryover.add(jid)
        for jid in pool_ids:
            all_dyn={j for sch in dyn_s.values() for j in sch}
            if jid not in all_dyn and jid not in cancelled: carryover.add(jid)

        for jid in carryover:
            typ=jtype_map.get(jid,'?'); all_left[typ]=all_left.get(typ,0)+1

        day_results.append({'day':day,'srv':dm['srv'],'km':dm['km'],
                             'cancelled':len(cancelled),'n_carry':len(set(pool_ids)-today_pre),
                             'n_pre':len(today_pre),'n_in':len(arrival_ev_today),
                             'n_ops':len(used_ops),'vpi':vpi_row,
                             'carryover_n':len(carryover),
                             'routes':{   # harita için
                                 'dyn_r':dyn_r,'dyn_s':dyn_s,
                                 'st_r':st_r,'st_s':st_s,
                                 'rs':rs,'op_ids':used_ops,
                                 'new_asgn':new_asgn,'cancelled':cancelled,
                             },
                             'op_metrics':{
                                 op:{
                                     'srv':sum(1 for s in dyn_s.get(op,{}).values() if s.get('served')),
                                     'km':_rkm(dyn_r.get(op,[]),rs[op],coords)
                                 } for op in used_ops
                             }})

    progress.progress(1.0); status.text("Tamamlandı.")

    # ── Haftalık çıktılar ──────────────────────────────────────────
    st.markdown("### 📋 Günlük İş Akışı")
    flow_rows=[]
    for r in day_results:
        bas=r['n_carry']+r['n_pre']; ekl=r['n_in']; ipt=r['cancelled']
        srv=r['srv']; son=bas+ekl-ipt-srv
        flow_rows.append({'Gün':str(r['day']),'Başlangıç':bas,'Eklenen':ekl,
                          'İptal':ipt,'Servis':srv,'Son':son,'km':round(r['km'],1),'Op':r['n_ops']})
    st.dataframe(pd.DataFrame(flow_rows),hide_index=True,use_container_width=True)

    vpi_rows=[r for r in day_results if r.get('vpi')]
    if vpi_rows:
        st.markdown("### 📈 VPI — Gün Gün")
        vpi_data=[]
        for r in vpi_rows:
            v=r['vpi']
            vpi_data.append({'Gün':str(r['day']),
                             'Fs (Statik %)':f"{v['fs']:.1f}%",
                             'Fd (Dinamik %)':f"{v['fd']:.1f}%",
                             'Ft (Tam Bilgi %)':f"{v['ft']:.1f}%",
                             'Dinamik İyileşme':f"+{v['dyn_imp']:.1f}%",
                             'VPI (Fs→Ft)':f"+{v['vpi_imp']:.1f}%",
                             'Yakalanma %':f"{v['eff']:.1f}%"})
        st.dataframe(pd.DataFrame(vpi_data),hide_index=True,use_container_width=True)
        avg_dyn=sum(r['vpi']['dyn_imp'] for r in vpi_rows)/len(vpi_rows)
        avg_vpi=sum(r['vpi']['vpi_imp'] for r in vpi_rows)/len(vpi_rows)
        avg_eff=sum(r['vpi']['eff'] for r in vpi_rows)/len(vpi_rows)
        col_a,col_b,col_c=st.columns(3)
        col_a.metric("Haftalık ort. dinamik iyileşme",f"%{avg_dyn:.1f}")
        col_b.metric("Haftalık ort. VPI",f"%{avg_vpi:.1f}")
        col_c.metric("Haftalık ort. yakalanma",f"%{avg_eff:.1f}")

    if all_left:
        st.markdown("### 📦 Hafta Sonu Kalan İşler (Tür Bazında)")
        left_df=pd.DataFrame([{'Tür':t,'Kalan':c,
                                'Oran':f"%{c/sum(all_left.values())*100:.1f}"}
                               for t,c in sorted(all_left.items(),key=lambda x:-x[1])])
        st.dataframe(left_df,hide_index=True,use_container_width=True)

    # ── Operatör başına dağılım histogramları ─────────────────────
    op_days=[r for r in day_results if r.get('op_metrics')]
    if op_days:
        st.markdown("### 📊 Operatör Başına Dağılım — Gün Gün")
        st.caption("Her operatörün günlük servis ettiği iş sayısı ve kat ettiği kilometre.")

        for r in op_days:
            om=r['op_metrics']
            if not om: continue
            day_label=str(r['day'])
            ops_sorted=sorted(om.keys(),key=lambda o:om[o]['srv'],reverse=True)
            srv_vals=[om[op]['srv'] for op in ops_sorted]
            km_vals =[round(om[op]['km'],1) for op in ops_sorted]
            op_labels=[str(op)[:8] for op in ops_sorted]

            with st.expander(f"📅 {day_label} — {len(ops_sorted)} operatör, toplam {r['srv']} servis, {r['km']:.1f} km",
                             expanded=False):
                df_bar=pd.DataFrame({
                    'Operatör': op_labels,
                    'Servis edilen iş': srv_vals,
                    'km': km_vals,
                })

                col_h1,col_h2=st.columns(2)

                with col_h1:
                    st.markdown("**Servis Edilen İş Sayısı**")
                    srv_chart=pd.DataFrame({'Servis':srv_vals},index=op_labels)
                    st.bar_chart(srv_chart,height=280,use_container_width=True)
                    avg_srv=sum(srv_vals)/max(len(srv_vals),1)
                    st.caption(f"Ort: {avg_srv:.1f} iş/op  |  Min: {min(srv_vals)}  |  Max: {max(srv_vals)}")

                with col_h2:
                    st.markdown("**Kat Edilen Kilometre**")
                    km_chart=pd.DataFrame({'km':km_vals},index=op_labels)
                    st.bar_chart(km_chart,height=280,use_container_width=True)
                    avg_km=sum(km_vals)/max(len(km_vals),1)
                    st.caption(f"Ort: {avg_km:.1f} km/op  |  Min: {min(km_vals)}  |  Max: {max(km_vals)}")

                st.dataframe(df_bar.set_index('Operatör'),use_container_width=True)

    # ── Haritalar (gün seçimli) ────────────────────────────────────
    route_days=[r for r in day_results if r.get('routes')]
    if route_days:
        st.markdown("### 🗺️ Haritalar")
        day_options=[str(r['day']) for r in route_days]
        chosen_day=st.selectbox("Görüntülenecek gün:",day_options,index=len(day_options)-1)
        rd=next(r for r in route_days if str(r['day'])==chosen_day)
        rv=rd['routes']
        hist_day=reconstruct_historical(df_raw,due_map,JP,jtype_map)
        map_tab1,map_tab2,map_tab3=st.tabs(["🗺️ Statik","🗺️ Dinamik","🗺️ Gerçek"])
        with map_tab1:
            st.caption(f"{chosen_day} — Statik Plan")
            with st.spinner("Harita oluşturuluyor..."):
                html=make_map(rv['st_r'],rv['st_s'],rv['op_ids'],rv['rs'],
                              coords,df_pool,JP=JP,due_map=due_map)
            components.html(html,height=550,scrolling=False)
        with map_tab2:
            st.caption(f"{chosen_day} — Dinamik Plan ({rd['n_ops']} op, {rd['srv']} servis, {rd['cancelled']} iptal)")
            with st.spinner("Harita oluşturuluyor..."):
                html=make_map(rv['dyn_r'],rv['dyn_s'],rv['op_ids'],rv['rs'],
                              coords,df_pool,cancelled=rv['cancelled'],
                              new_assigned=rv['new_asgn'],JP=JP,due_map=due_map)
            components.html(html,height=550,scrolling=False)
        with map_tab3:
            st.caption(f"{chosen_day} — Gerçek Operatör Rotaları")
            if hist_day:
                with st.spinner("Harita oluşturuluyor..."):
                    html=make_historical_map(hist_day,rv['rs'],df_pool,due_map=due_map,JP=JP)
                if html: components.html(html,height=550,scrolling=False)
            else:
                st.info("Bu veri setinde gerçek rota bilgisi (OK statüsü) bulunamadı.")

    st.stop()

# ══════════════════════════════════════════════════════
#  GÜNLÜK VERİ AKIŞI (mevcut kod devam ediyor)
# ══════════════════════════════════════════════════════

with st.spinner("Statik plan hesaplanıyor..."):
    labels,centers=kmeans_cluster(job_ids,coords,op_ids,op_coords)
    c2o=macar_assign(centers,op_ids,op_coords)
    op_jobs={op:[] for op in op_ids}
    for jid,cl in labels.items(): op_jobs[c2o[cl]].append(jid)
    op2cl={op:cl for cl,op in c2o.items()}
    op_start={op:centers[op2cl[op]] for op in op_ids}
    op_jobs=balance(op_jobs,op_ids,op_start,coords,JP)
    st_r={}; st_s={}
    for op in op_ids:
        r,s,_=route_op(op,op_start[op],op_jobs[op],coords,JP,due_map,alpha=float(alpha))
        st_r[op]=r; st_s[op]=s
    orig_ids=set(job_ids)
    cancel_ev,arrival_ev=build_events(df_raw,orig_ids,op_jobs,coords,JP,due_map,jtype_map,jcre_map)

with st.spinner(f"Dinamik simülasyon çalışıyor ({len(cancel_ev)} iptal, {len(arrival_ev)} yeni iş)..."):
    dyn_r,dyn_s,new_asgn,cancelled,n_reopt=simulate(
        cancel_ev,arrival_ev,op_ids,op_start,op_jobs,coords,JP,due_map,st_r,st_s,
        n_thr=int(n_thr),prox_km=float(prox_km),commit_n=int(commit_n),
        transfer_km=float(transfer))

with st.spinner("Oracle (VPI) hesaplanıyor..."):
    try:
        or_r,or_s,or_due,o_start=compute_oracle(
            cancel_ev,arrival_ev,op_ids,op_start,op_jobs,coords,JP,due_map,jtype_map,jcre_map)
        f0c,f0k=total_cost(st_r,st_s,op_ids,op_start,coords,JP,due_map,rollover,due_exc)
        fdc,fdk=total_cost(dyn_r,dyn_s,op_ids,op_start,coords,JP,due_map,rollover,due_exc)
        fsc,fsk=total_cost(or_r,or_s,op_ids,o_start,coords,JP,or_due,rollover,due_exc)
        new_pen=sum(JP.get(e['job'],(0,0,50))[2]*(1+rollover+(due_exc if due_map.get(e['job'],TEND)<TEND else 0))
                    for e in arrival_ev if e['job'] in JP)
        f0_aug=f0c+new_pen
        oracle_ok=True
    except Exception:
        oracle_ok=False

with st.spinner("Gerçek rotalar yeniden oluşturuluyor..."):
    historical=reconstruct_historical(df_raw,due_map,JP,jtype_map)

# ── Metrikler ──
st_m =metrics(st_r,st_s,op_ids,op_start,coords,set(),orig_ids)
dyn_m=metrics(dyn_r,dyn_s,op_ids,op_start,coords,new_asgn,orig_ids)

# ─────────────────────────────────────────────────────────────────────────────
#  SEKMELER
# ─────────────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5=st.tabs([
    "📊 Özet & VPI","🗺️ Statik Harita","🗺️ Dinamik Harita",
    "🗺️ Gerçek Harita","📋 Detay Tablolar"
])

with tab1:
    st.markdown("### Sonuç Özeti")
    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown("**Statik Plan**")
        st.metric("Servis edilen",f"{st_m['srv']:,}",
                  f"%{st_m['srv_pct']:.1f}")
        st.metric("Km (eve dönüş hariç)",f"{st_m['km']:.1f}")
        st.metric("Gecikme cezası",f"{st_m['tard']:,.0f} ₺")
    with c2:
        st.markdown("**Dinamik Plan**")
        st.metric("Servis edilen",f"{dyn_m['srv']:,}",
                  f"+{dyn_m['srv']-st_m['srv']} vs statik")
        st.metric("Km",f"{dyn_m['km']:.1f}",
                  f"{dyn_m['km']-st_m['km']:+.1f}")
        st.metric("Yeniden rotalama",f"{n_reopt}")
    with c3:
        st.markdown("**Gerçek Operatörler**")
        if historical:
            h_srv=sum(v['n_served'] for v in historical.values())
            h_km=sum(v['km'] for v in historical.values())
            h_cost=sum(v['cost'] for v in historical.values())
            st.metric("Tamamlanan",f"{h_srv:,}")
            st.metric("Km",f"{h_km:.1f}")
            st.metric("İşletme maliyeti",f"{h_cost:,.0f} ₺")
        else:
            st.info("Gerçek rota verisi bulunamadı.")

    st.markdown("---")
    if oracle_ok:
        st.markdown("### Tam Bilginin Değeri (VPI)")
        def pct(c): return c/max(f0_aug,1)*100
        def imp(c): return (f0_aug-c)/max(f0_aug,1)*100
        vpi_pct=imp(fsc); dyn_pct=imp(fdc); eff=dyn_pct/max(vpi_pct,0.001)*100
        f0s=st_m['srv']; fds=dyn_m['srv']
        fss=sum(sum(1 for s in sch.values() if s.get('served')) for sch in or_s.values())
        vpi_df=pd.DataFrame([
            {'Senaryo':'Fs — Statik Plan (referans %100)',
             'Maliyet %':f"{pct(f0_aug):.1f}%",'İyileşme':'—','Servis':f0s},
            {'Senaryo':'Fd — Dinamik Plan',
             'Maliyet %':f"{pct(fdc):.1f}%",'İyileşme':f"+{dyn_pct:.1f}%",'Servis':fds},
            {'Senaryo':'Ft — Tam Bilgi Planı (teorik en iyi)',
             'Maliyet %':f"{pct(fsc):.1f}%",'İyileşme':f"+{vpi_pct:.1f}%",'Servis':fss},
        ])
        st.dataframe(vpi_df,hide_index=True,use_container_width=True)
        col_a,col_b,col_c=st.columns(3)
        col_a.metric("VPI (Fs→Ft tam bilgi değeri)",f"%{vpi_pct:.1f}")
        col_b.metric("Dinamik kazanç (Fs→Fd)",f"%{dyn_pct:.1f}")
        col_c.metric("VPI yakalanma oranı",f"%{eff:.1f}")
        st.caption("Fs: gün başı statik plan + tüm yeni işler unserved (referans %100)  |  "
                   "Fd: dinamik modelimiz  |  Ft: tüm olaylar baştan bilinse elde edilecek teorik en iyi")

    # Tür bazında kapatma
    st.markdown("---")
    st.markdown("### İş Tipi Bazında Kapatma Yüzdeleri")
    st_t={}; dy_t={}
    for sch in st_s.values():
        for j,s in sch.items():
            typ=jtype_map.get(j,'?')
            st_t.setdefault(typ,{'srv':0,'tot':0})
            st_t[typ]['tot']+=1
            if s.get('served'): st_t[typ]['srv']+=1
    for sch in dyn_s.values():
        for j,s in sch.items():
            if j in cancelled: continue
            typ=jtype_map.get(j,'?')
            dy_t.setdefault(typ,{'srv':0,'tot':0})
            dy_t[typ]['tot']+=1
            if s.get('served'): dy_t[typ]['srv']+=1
    rows=[]
    for t in sorted(set(list(st_t)+list(dy_t))):
        s_=st_t.get(t,{'srv':0,'tot':0}); d_=dy_t.get(t,{'srv':0,'tot':0})
        rows.append({'Tür':t,
                     'Statik Servis':s_['srv'],'Statik %':f"{s_['srv']/max(s_['tot'],1)*100:.1f}%",
                     'Dinamik Servis':d_['srv'],'Dinamik %':f"{d_['srv']/max(d_['tot'],1)*100:.1f}%"})
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)

with tab2:
    st.markdown("### Statik Plan Haritası")
    st.caption(f"{st_m['srv']} iş servis edildi | {st_m['km']:.1f} km | {len(op_ids)} operatör")
    with st.spinner("Harita oluşturuluyor..."):
        html_st=make_map(st_r,st_s,op_ids,op_start,coords,df,JP=JP,due_map=due_map)
    components.html(html_st,height=600,scrolling=False)

with tab3:
    st.markdown("### Dinamik Plan Haritası")
    st.caption(f"{dyn_m['srv']} iş servis edildi | {dyn_m['nsrv']} yeni iş dahil | "
               f"{len(cancelled)} iptal | {n_reopt} yeniden rotalama")
    with st.spinner("Harita oluşturuluyor..."):
        html_dyn=make_map(dyn_r,dyn_s,op_ids,op_start,coords,df,
                          cancelled=cancelled,new_assigned=new_asgn,JP=JP,due_map=due_map)
    components.html(html_dyn,height=600,scrolling=False)

with tab4:
    st.markdown("### Gerçek Operatör Rotaları")
    if historical:
        h_srv=sum(v['n_served'] for v in historical.values())
        h_km=sum(v['km'] for v in historical.values())
        st.caption(f"{h_srv} iş tamamlandı | {h_km:.1f} km | {len(historical)} operatör")
        with st.spinner("Harita oluşturuluyor..."):
            html_hist=make_historical_map(historical,op_start,df,due_map=due_map,JP=JP)
        if html_hist:
            components.html(html_hist,height=600,scrolling=False)
    else:
        st.info("Veri setinde gerçek rota bilgisi (OK statüsü) bulunamadı.")

with tab5:
    st.markdown("### Statik vs Dinamik vs Gerçek — İşletme Karşılaştırması")
    st.caption("İşletme maliyeti = yakıt + gecikme cezası (unserved dahil değil)")
    if historical:
        h_srv=sum(v['n_served'] for v in historical.values())
        h_km=sum(v['km'] for v in historical.values())
        h_tard=sum(v['tardy_pen'] for v in historical.values())
        h_fuel=sum(v['fuel'] for v in historical.values())
        h_cost=h_fuel+h_tard
    comp_rows=[
        {'Senaryo':'Statik','Servis':st_m['srv'],'km':round(st_m['km'],1),
         'km/iş':round(st_m['kpj'],3),'Yakıt ₺':round(st_m['fuel'],0),
         'Gecikme ₺':round(st_m['tard'],0),
         'İşletme ₺':round(st_m['fuel']+st_m['tard'],0),
         '₺/iş':round((st_m['fuel']+st_m['tard'])/max(st_m['srv'],1),0)},
        {'Senaryo':'Dinamik','Servis':dyn_m['srv'],'km':round(dyn_m['km'],1),
         'km/iş':round(dyn_m['kpj'],3),'Yakıt ₺':round(dyn_m['fuel'],0),
         'Gecikme ₺':round(dyn_m['tard'],0),
         'İşletme ₺':round(dyn_m['fuel']+dyn_m['tard'],0),
         '₺/iş':round((dyn_m['fuel']+dyn_m['tard'])/max(dyn_m['srv'],1),0)},
    ]
    if historical:
        comp_rows.append(
            {'Senaryo':'Gerçek','Servis':h_srv,'km':round(h_km,1),
             'km/iş':round(h_km/max(h_srv,1),3),'Yakıt ₺':round(h_fuel,0),
             'Gecikme ₺':round(h_tard,0),'İşletme ₺':round(h_cost,0),
             '₺/iş':round(h_cost/max(h_srv,1),0)})
    st.dataframe(pd.DataFrame(comp_rows),hide_index=True,use_container_width=True)

    st.markdown("---")
    st.markdown("### En Yüksek Cezalı Yapılamayan İşler (Dinamik)")
    pen_rows=[]
    for sch in dyn_s.values():
        for jid,s in sch.items():
            if not s.get('served') and jid not in cancelled:
                pu_=JP.get(jid,(0,0,50))[2]
                pen_rows.append({'İş No':str(jid),'Tür':jtype_map.get(jid,'?'),
                                  'Vade':dk2s(due_map.get(jid,TEND)),
                                  'p_u ₺':round(pu_,0),
                                  'Toplam Ceza ₺':round(pu_*(1+rollover+(due_exc if due_map.get(jid,TEND)<TEND else 0)),0)})
    if pen_rows:
        top=sorted(pen_rows,key=lambda r:r['Toplam Ceza ₺'],reverse=True)[:20]
        st.dataframe(pd.DataFrame(top),hide_index=True,use_container_width=True)
