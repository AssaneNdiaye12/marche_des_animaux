import streamlit as st
import pandas as pd
import sqlite3, os, base64
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Marché des Animaux", page_icon="🐾", layout="wide")

# ── Config ────────────────────────────────────────────────────────────────────
CATS = {
    "🐕 Chiens":                  ("Inf_Chiens",             "CoinAfriqueSiteMap_Chiens"),
    "🐑 Moutons":                 ("Inf_Moutons",            "CoinAfriqueSiteMap_Moutons"),
    "🦆 Poules, Lapins, Pigeons": ("Inf_LapinsPoulesPigeons","CoinAfriqueSiteMap_PoulesLapinsPigeons"),
    "🐾 Autres Animaux":          ("Inf_AutresAni",          "CoinAfriqueSiteMap_Autres_Animaux"),
}
HIDE  = {'web_scraper_order','web_scraper_start_url','container_link','container','conrenaire','contenaire','_page_num','Lien_annonce'}
KOBO  = "https://ee-eu.kobotoolbox.org/x/oRhjimHa" 

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def conn(p): return sqlite3.connect(p, check_same_thread=False) if os.path.exists(p) else None

@st.cache_data
def load(tbl, _c): 
    try: return pd.read_sql_query(f"SELECT * FROM {tbl}", _c)
    except: return pd.DataFrame()

def prix(df):
    if 'Prix' not in df.columns: return pd.Series(dtype=float)
    return pd.to_numeric(df['Prix'].astype(str).str.replace(r'[^\d.]','',regex=True), errors='coerce').dropna()

def clean(df):
    if 'Details' in df.columns: df = df.rename(columns={'Details':'Nom'})
    for c in ['container_link','container','conrenaire','contenaire']:
        if c in df.columns: df = df.rename(columns={c:'Lien_annonce'})
    return df[[c for c in df.columns if c not in HIDE]]

def csv_dl(df, f): 
    b = base64.b64encode(df.to_csv(index=False).encode()).decode()
    return f'<a href="data:file/csv;base64,{b}" download="{f}">📥 Télécharger CSV</a>'

def kpis(df, p=None):
    p = p if p is not None else prix(df)
    for col, label, val in zip(st.columns(4),
        ["Annonces","Prix moyen","Médiane","Prix max"],
        [f"{len(df):,}", f"{p.mean():,.0f} CFA" if len(p) else "N/A",
         f"{p.median():,.0f} CFA" if len(p) else "N/A", f"{p.max():,.0f} CFA" if len(p) else "N/A"]):
        col.metric(label, val)

def traite(p, method):
    if "Wins" in method: return p.clip(p.quantile(.05), p.quantile(.95))
    Q1, Q3 = p.quantile(.25), p.quantile(.75)
    return p[(p >= Q1-1.5*(Q3-Q1)) & (p <= Q3+1.5*(Q3-Q1))]

def stat_row(p): return [f"{len(p):,}",f"{p.min():,.0f}",f"{p.quantile(.25):,.0f}",f"{p.median():,.0f}",f"{p.mean():,.0f}",f"{p.quantile(.75):,.0f}",f"{p.max():,.0f}",f"{p.std():,.0f}"]

# ── Layout ────────────────────────────────────────────────────────────────────
st.markdown("# 🐾 Marché des Animaux — Data & Analyse")
st.caption("Source : [CoinAfrique Sénégal](https://sn.coinafrique.com)"); st.markdown("---")

mode = st.sidebar.radio("Mode", ["📊 Données traitées","📥 Données non traitées","📈 Tableau de bord des données","💬 Commentaires"])
cat  = st.sidebar.selectbox("Catégorie", list(CATS))
tbl_t, tbl_b = CATS[cat]

# ── 1. Données traitées ───────────────────────────────────────────────────────
if mode == "📊 Données traitées":
    c = conn(st.sidebar.text_input("BD traitée", "data/SGBD_Coinafrique.db"))
    if not c: st.error("❌ BD introuvable"); st.stop()
    st.sidebar.success("✅ Connectée"); df = load(tbl_t, c)
    if df.empty: st.warning("Aucune donnée."); st.stop()
    st.subheader(cat); kpis(df); st.markdown("---")
    t1,t2,t3 = st.tabs(["📋 Données","📈 Graphiques","📥 Export"])
    with t1:
        if 'Adresse' in df.columns:
            locs = st.multiselect("Filtrer par localité", df['Adresse'].dropna().unique())
            if locs: df = df[df['Adresse'].isin(locs)]
        st.dataframe(df, use_container_width=True, height=480)
    with t2:
        if 'Adresse' in df.columns:
            st.plotly_chart(px.bar(df['Adresse'].value_counts().head(10), orientation='h', title="Top 10 localités", color_discrete_sequence=['#17a2b8']), use_container_width=True)
        p = prix(df)
        if len(p): st.plotly_chart(px.histogram(p, nbins=30, title="Distribution des prix", color_discrete_sequence=['#764ba2']), use_container_width=True)
    with t3:
        st.markdown(csv_dl(df, f"{tbl_t}.csv"), unsafe_allow_html=True)
        st.write(f"**{len(df)} lignes** | Colonnes : {', '.join(df.columns)}")

# ── 2. Données non traitées ───────────────────────────────────────────────────
elif mode == "📥 Données non traitées":
    c = conn(st.sidebar.text_input("BD non traitée", "data/SGBD_CoinafriqueN.db"))
    if not c: st.error("❌ BD introuvable"); st.stop()
    st.sidebar.success("✅ Connectée"); df = load(tbl_b, c)
    if df.empty: st.warning("Aucune donnée."); st.stop()
    df['_page_num'] = df['web_scraper_start_url'].str.extract(r'page=(\d+)',expand=False).astype(float).fillna(1).astype(int)
    pages = sorted(df['_page_num'].unique())
    pg    = st.sidebar.number_input("Page", 1, len(pages), 1, key="pg")
    url   = df[df['_page_num']==pages[pg-1]]['web_scraper_start_url'].iloc[0]
    df_pg = clean(df[df['_page_num']==pages[pg-1]].copy())
    st.subheader(cat); kpis(df)
    st.caption(f"Page {pg}/{len(pages)} — {len(df_pg)} annonces | 🔗 {url}"); st.markdown("---")
    c1,c2,c3,c4,c5 = st.columns(5)
    if c1.button("⏮️"): st.session_state['pg']=1;                    st.rerun()
    if c2.button("◀️"): st.session_state['pg']=max(1,pg-1);          st.rerun()
    c3.markdown(f"<div style='text-align:center;padding:8px'><b>{pg}/{len(pages)}</b></div>",unsafe_allow_html=True)
    if c4.button("▶️"): st.session_state['pg']=min(len(pages),pg+1); st.rerun()
    if c5.button("⏭️"): st.session_state['pg']=len(pages);           st.rerun()
    st.dataframe(df_pg, use_container_width=True, height=440); st.markdown("---")
    c1,c2 = st.columns(2)
    c1.markdown(csv_dl(df_pg, f"{tbl_b}_p{pg}.csv"), unsafe_allow_html=True); c1.caption(f"{len(df_pg)} annonces (page {pg})")
    df_all = clean(df.copy())
    c2.markdown(csv_dl(df_all, f"{tbl_b}_complet.csv"), unsafe_allow_html=True); c2.caption(f"{len(df_all)} annonces (tout)")
    if st.checkbox("📈 Visualisations"):
        p = prix(df)
        if len(p): st.plotly_chart(px.histogram(p, nbins=30, color_discrete_sequence=['#17a2b8'], title="Distribution des prix"), use_container_width=True)

# ── 3. Visualisations ─────────────────────────────────────────────────────────
elif mode == "📈 Tableau de bord des données":
    c = conn(st.sidebar.text_input("BD non traitée", "data/SGBD_CoinafriqueN.db"))
    meth = st.sidebar.selectbox("Outliers", ["Winsorization (5%-95%)","Filtre IQR"])
    if not c: st.error("❌ BD introuvable"); st.stop()
    df = load(tbl_b, c)
    if 'Details' in df.columns: df = df.rename(columns={'Details':'Nom'})
    pb = prix(df); pt = traite(pb, meth)
    st.subheader(f"📈 Tableau de bord des données — {cat}"); kpis(df, pb); st.markdown("---")
    t1,t2,t3 = st.tabs(["📊 Histogramme","📦 Boxplot","🔍 Comparaison"])
    with t1:
        c1,c2 = st.columns(2)
        for col,p,lab,clr in [(c1,pb,"🔴 Non traité","#e74c3c"),(c2,pt,f"🟢 {meth}","#27ae60")]:
            fig = px.histogram(p, nbins=40, color_discrete_sequence=[clr], opacity=.85, title=lab, labels={"value":"Prix (CFA)"})
            fig.add_vline(x=p.median(), line_dash="dash", line_color=clr, annotation_text=f"Médiane: {p.median():,.0f}")
            fig.update_layout(showlegend=False, plot_bgcolor="#fafafa", bargap=.05)
            col.plotly_chart(fig, use_container_width=True); col.caption(f"{len(p)} valeurs | max {p.max():,.0f} CFA")
    with t2:
        c1,c2 = st.columns(2)
        for col,p,lab,clr in [(c1,pb,"Non traité","#e74c3c"),(c2,pt,meth,"#27ae60")]:
            fig = go.Figure(go.Box(y=p, name=lab, marker_color=clr, boxmean="sd", boxpoints="outliers"))
            fig.update_layout(yaxis_title="Prix (CFA)", plot_bgcolor="#fafafa", showlegend=False, title=lab)
            col.plotly_chart(fig, use_container_width=True)
            col.caption(f"Q1={p.quantile(.25):,.0f} | Med={p.median():,.0f} | Q3={p.quantile(.75):,.0f} CFA")
    with t3:
        df_c = pd.DataFrame({"Prix":pd.concat([pb,pt],ignore_index=True),"Type":["Non traité"]*len(pb)+[meth]*len(pt)})
        st.plotly_chart(px.histogram(df_c,x="Prix",color="Type",nbins=40,barmode="overlay",opacity=.65,
            color_discrete_map={"Non traité":"#e74c3c",meth:"#27ae60"}), use_container_width=True)
        fig = go.Figure([go.Box(y=pb,name="Non traité",marker_color="#e74c3c",boxmean="sd"),
                         go.Box(y=pt,name=meth,marker_color="#27ae60",boxmean="sd")])
        fig.update_layout(yaxis_title="Prix (CFA)", plot_bgcolor="#fafafa"); st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pd.DataFrame({"Stat":["N","Min","Q1","Médiane","Moyenne","Q3","Max","σ"],
            "Non traité":stat_row(pb), meth:stat_row(pt)}).set_index("Stat"), use_container_width=True)

# ── 4. Commentaires ───────────────────────────────────────────────────────────
else:
    st.subheader("💬 Commentaires & Feedback")
    t1,t2,t3 = st.tabs(["📝 Feedback","🐛 Bug","⭐ Évaluation"])
    with t1:
        st.markdown(f'<div style="text-align:center;margin:20px 0"><a href="{KOBO}" target="_blank" style="background:#17a2b8;color:white;padding:12px 28px;border-radius:8px;text-decoration:none;font-weight:bold">🔗 Ouvrir le formulaire KoBoToolbox</a></div>', unsafe_allow_html=True)
        st.components.v1.iframe(src=KOBO, height=550, scrolling=True)
    with t2:
        titre = st.text_input("Titre du bug"); desc = st.text_area("Description", height=120)
        st.selectbox("Gravité", ["🟢 Mineur","🟡 Modéré","🟠 Important","🔴 Critique"])
        if st.button("🚨 Signaler", type="primary"):
            (st.success("✅ Bug signalé !") or st.balloons()) if (titre and desc) else st.error("Titre et description requis.")
    with t3:
        n = st.slider("Note", 1, 5, 4)
        st.markdown(f"<h3 style='text-align:center'>{'⭐'*n}{'☆'*(5-n)}</h3>", unsafe_allow_html=True)
        st.text_area("Commentaire (optionnel)", height=100)
        if st.button("📊 Soumettre", type="primary"): st.success(f"✅ Merci ! Note : {n}/5"); (st.balloons() if n>=4 else None)

st.markdown("---")
st.caption("🐾 Marché des Animaux · Streamlit · [CoinAfrique Sénégal](https://sn.coinafrique.com)")