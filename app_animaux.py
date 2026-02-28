import streamlit as st
import pandas as pd
import sqlite3, os, base64
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from requests import get
from bs4 import BeautifulSoup as bs
import time

st.set_page_config(page_title="Marché des Animaux", page_icon="🐾", layout="wide")

# ── Config ────────────────────────────────────────────────────────────────────
CATS = {
    "🐕 Chiens":                  ("chiens",                    "CoinAfriqueSiteMap_Chiens"),
    "🐑 Moutons":                 ("moutons",                   "CoinAfriqueSiteMap_Moutons"),
    "🦆 Poules, Lapins, Pigeons": ("poules-lapins-et-pigeons",  "CoinAfriqueSiteMap_PoulesLapinsPigeons"),
    "🐾 Autres Animaux":          ("autres-animaux",            "CoinAfriqueSiteMap_Autres_Animaux"),
}
HIDE  = {'web_scraper_order','web_scraper_start_url','container_link','container',
         'conrenaire','contenaire','_page_num','Lien_annonce'}
KOBO  = "https://ee-eu.kobotoolbox.org/x/oRhjimHa"
GFORM = "https://docs.google.com/forms/d/e/1FAIpQLSfpkmUCq2l-cUH6EgbWwheaIJu1uFUe1vZ74pJmLpyRVtzWlA/viewform?usp=publish-editor" 

# ═══════════════════════════════════════════════════════════════════════════════
# FONCTIONS — Scraping (depuis methodes.py)
# ═══════════════════════════════════════════════════════════════════════════════
def scraper_categorie(categorie, nb_pages=5, progress_cb=None):
    df_final   = pd.DataFrame()
    est_lapins = (categorie == 'poules-lapins-et-pigeons')

    for ind_page in range(1, nb_pages + 1):
        url = f'https://sn.coinafrique.com/categorie/{categorie}?page={ind_page}'
        try:
            res  = get(url, timeout=12)
            soup = bs(res.content, 'html.parser')
            containers = soup.find_all('div', 'col s6 m4 l3')
            data = []
            for container in containers:
                try:
                    Nom     = container.find('p', 'ad__card-description').a.text.strip()
                    Prix    = container.find('p', 'ad__card-price').text.replace('CFA','').strip()
                    Adresse = container.find('p', 'ad__card-location').span.text.strip()
                    Details = container.find('p', 'ad__card-description').a.get('title', 'N/A')
                    url_img = container.find('img')['src']
                    if est_lapins:
                        data.append({'Details': Details, 'Prix': Prix,
                                     'Adresse': Adresse, 'url_image': url_img})
                    else:
                        data.append({'Nom': Nom, 'Prix': Prix,
                                     'Adresse': Adresse, 'url_image': url_img})
                except:
                    pass
            if data:
                df_final = pd.concat([df_final, pd.DataFrame(data)],
                                     axis=0).reset_index(drop=True)
        except Exception as e:
            st.warning(f"Erreur page {ind_page} : {e}")

        if progress_cb:
            progress_cb(ind_page / nb_pages,
                        f"Page {ind_page}/{nb_pages} — {len(df_final)} annonces récupérées")
        time.sleep(0.4)

    return df_final


# ═══════════════════════════════════════════════════════════════════════════════
# FONCTIONS — Nettoyage & outliers (depuis methodes.py)
# ═══════════════════════════════════════════════════════════════════════════════
def nettoyer_prix(df):
    """
    Nettoie la colonne Prix :
      1. Remplace 'Prix sur demande' -> NaN
      2. Convertit en numérique
      3. Impute les NaN par médiane du groupe (Nom)
      4. Fallback : médiane globale
      5. Renomme Prix -> Prix (CFA)
    """
    df = df.copy()
    if 'Details' in df.columns:
        df = df.rename(columns={'Details': 'Nom'})
    nom_col = 'Nom' if 'Nom' in df.columns else df.columns[0]

    df['Prix_clean'] = df['Prix'].replace('Prix sur demande', np.nan)
    df['Prix_num'] = (
        df['Prix_clean']
        .astype(str)
        .str.replace(r'[^\d,\.]', '', regex=True)
        .str.replace(',', '.', regex=False)
        .str.replace(r'\s+', '', regex=True)
    )
    df['Prix_num'] = pd.to_numeric(df['Prix_num'], errors='coerce')

    median_by_nom = df.groupby(nom_col)['Prix_num'].median()
    df['Prix_num'] = df['Prix_num'].fillna(df[nom_col].map(median_by_nom))

    global_median  = df['Prix_num'].median()
    df['Prix_num'] = df['Prix_num'].fillna(global_median)

    df['Prix'] = df['Prix_num']
    df.rename(columns={'Prix': 'Prix (CFA)'}, inplace=True)
    df.drop(columns=['Prix_clean', 'Prix_num'], inplace=True, errors='ignore')
    return df


def winsoriser_prix(df, lower_pct=5, upper_pct=95):
    """
    Applique la Winsorisation (5%-95%) sur la colonne 'Prix (CFA)'.
    Retourne le DataFrame avec les prix corrigés et un dict de métadonnées.
    """
    df = df.copy()
    col = 'Prix (CFA)'
    if col not in df.columns:
        return df, {}

    serie = df[col].dropna()
    if serie.empty:
        return df, {}

    lo = np.percentile(serie, lower_pct)
    hi = np.percentile(serie, upper_pct)

    n_below = int((df[col] < lo).sum())
    n_above = int((df[col] > hi).sum())

    df[col] = df[col].clip(lo, hi)

    meta = {
        "borne_basse":  lo,
        "borne_haute":  hi,
        "n_remonte":    n_below,
        "n_abaisse":    n_above,
        "n_total":      len(serie),
    }
    return df, meta


def impute_outliers_winsorization(data):
    """Winsorization 5-95% sur les colonnes numériques asymétriques."""
    data = data.copy()
    for col in data.select_dtypes('number').columns:
        if not (-0.5 < data[col].skew() < 0.5):
            lower = np.percentile(data[col].dropna(), 5)
            upper = np.percentile(data[col].dropna(), 95)
            data[col] = data[col].clip(lower, upper)
    return data


def impute_outliers_iqr(data):
    """Filtre IQR sur les colonnes numériques symétriques."""
    data = data.copy()
    for col in data.select_dtypes('number').columns:
        if -0.15 < data[col].skew() < 0.15:
            Q1  = np.quantile(data[col].dropna(), 0.25)
            Q3  = np.quantile(data[col].dropna(), 0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            data[col] = np.where(data[col] < lower, lower,
                        np.where(data[col] > upper, upper, data[col]))
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS UI
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def conn(p):
    return sqlite3.connect(p, check_same_thread=False) if os.path.exists(p) else None

@st.cache_data
def load(tbl, _c):
    try:    return pd.read_sql_query(f"SELECT * FROM {tbl}", _c)
    except: return pd.DataFrame()

def prix_serie(df):
    col = 'Prix (CFA)' if 'Prix (CFA)' in df.columns else ('Prix' if 'Prix' in df.columns else None)
    if not col: return pd.Series(dtype=float)
    return pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.]', '', regex=True),
                         errors='coerce').dropna()

def clean_raw(df):
    if 'Details' in df.columns: df = df.rename(columns={'Details': 'Nom'})
    for c in ['container_link','container','conrenaire','contenaire']:
        if c in df.columns: df = df.rename(columns={c: 'Lien_annonce'})
    return df[[c for c in df.columns if c not in HIDE]]

def csv_dl(df, fname, label="📥 Télécharger CSV"):
    b = base64.b64encode(df.to_csv(index=False).encode()).decode()
    return (f'<a href="data:file/csv;base64,{b}" download="{fname}" '
            f'style="background:#28a745;color:white;padding:10px 22px;border-radius:8px;'
            f'text-decoration:none;font-weight:bold;display:inline-block">{label}</a>')

def kpis(df, p=None):
    p = p if p is not None else prix_serie(df)
    for col, label, val in zip(
        st.columns(4),
        ["Annonces", "Prix moyen", "Médiane", "Prix max"],
        [f"{len(df):,}",
         f"{p.mean():,.0f} CFA"   if len(p) else "N/A",
         f"{p.median():,.0f} CFA" if len(p) else "N/A",
         f"{p.max():,.0f} CFA"    if len(p) else "N/A"]
    ):
        col.metric(label, val)

def stat_row(p):
    return [f"{len(p):,}", f"{p.min():,.0f}", f"{p.quantile(.25):,.0f}",
            f"{p.median():,.0f}", f"{p.mean():,.0f}", f"{p.quantile(.75):,.0f}",
            f"{p.max():,.0f}", f"{p.std():,.0f}"]


# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🐾 Marché des Animaux — Data & Analyse")
st.caption("Source : [CoinAfrique Sénégal](https://sn.coinafrique.com)")
st.markdown("---")

mode         = st.sidebar.radio("Mode", [
    "📊 Données traitées",
    "📥 Données non traitées",
    "📈 Tableau de bord des données",
    "💬 Commentaires"
])
cat           = st.sidebar.selectbox("Catégorie", list(CATS))
slug, tbl_b   = CATS[cat]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DONNÉES TRAITÉES — Scraping live + nettoyage automatique + Winsorisation
# ═══════════════════════════════════════════════════════════════════════════════
if mode == "📊 Données traitées":

    st.subheader(f"📊 Données traitées — {cat}")
    st.info(
        "Les données sont scrappées **en direct** depuis CoinAfrique, puis nettoyées automatiquement : "
        "conversion numérique des prix, imputation des valeurs manquantes par médiane de groupe, "
        "et **correction des valeurs aberrantes par Winsorisation (5%–95%)**."
    )

    # ── Paramètres dans la sidebar ────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Paramètres de scraping")
    nb_pages = st.sidebar.slider("Nombre de pages", min_value=1, max_value=20, value=3, step=1)
    st.sidebar.caption(f"≈ {nb_pages * 20} annonces estimées")

    # ── Récapitulatif ─────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='padding:10px 18px;background:#f0f8ff;border-radius:8px;"
        f"border-left:4px solid #17a2b8;margin-bottom:16px'>"
        f"📂 <b>Catégorie :</b> {cat} &nbsp;│&nbsp; "
        f"📄 <b>Pages :</b> {nb_pages} &nbsp;│&nbsp; "
        f"🔗 sn.coinafrique.com/categorie/<b>{slug}</b>"
        f"</div>", unsafe_allow_html=True
    )

    # ── Boutons lancer / effacer ──────────────────────────────────────────────
    col_btn, col_reset = st.columns([3, 1])
    lancer = col_btn.button("🚀 Lancer le scraping", type="primary", use_container_width=True)
    key_df  = f"df_traite_{slug}"
    key_meta = f"wins_meta_{slug}"

    if col_reset.button("🗑️ Effacer", use_container_width=True):
        for k in [key_df, key_meta]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

    # ── Scraping avec barre de progression ───────────────────────────────────
    if lancer:
        progress_bar = st.progress(0, text="Démarrage du scraping...")
        status_txt   = st.empty()

        def update_progress(pct, msg):
            progress_bar.progress(pct, text=msg)
            status_txt.caption(msg)

        df_brut = scraper_categorie(slug, nb_pages, progress_cb=update_progress)
        progress_bar.empty()
        status_txt.empty()

        if df_brut.empty:
            st.error("❌ Aucune donnée récupérée. Vérifiez votre connexion ou réessayez.")
            st.stop()

        with st.spinner("🔧 Nettoyage des prix en cours..."):
            df_clean = nettoyer_prix(df_brut)

        with st.spinner("📐 Winsorisation des valeurs aberrantes (5%–95%)..."):
            df_wins, wins_meta = winsoriser_prix(df_clean)

        st.session_state[key_df]   = df_wins
        st.session_state[key_meta] = wins_meta
        st.success(f"✅ {len(df_wins)} annonces récupérées, nettoyées et corrigées !")
        st.rerun()

    # ── Affichage si données disponibles ─────────────────────────────────────
    if key_df in st.session_state:
        df   = st.session_state[key_df]
        meta = st.session_state.get(key_meta, {})

        st.markdown("---")

        # ── Bandeau Winsorisation ─────────────────────────────────────────────
        if meta:
            n_corr = meta['n_remonte'] + meta['n_abaisse']
            pct_corr = n_corr / meta['n_total'] * 100 if meta['n_total'] else 0
            st.markdown(
                f"<div style='padding:10px 18px;background:#fff8e1;border-radius:8px;"
                f"border-left:4px solid #f39c12;margin-bottom:12px'>"
                f"📐 <b>Winsorisation appliquée (5%–95%)</b> : "
                f"bornes [{meta['borne_basse']:,.0f} – {meta['borne_haute']:,.0f}] CFA &nbsp;│&nbsp; "
                f"<b>{meta['n_remonte']}</b> valeur(s) remontée(s) &nbsp;│&nbsp; "
                f"<b>{meta['n_abaisse']}</b> valeur(s) abaissée(s) &nbsp;│&nbsp; "
                f"<b>{n_corr}</b> correction(s) au total ({pct_corr:.1f}% des données)"
                f"</div>",
                unsafe_allow_html=True
            )

        kpis(df)
        st.markdown("---")

        t1, t2, t3 = st.tabs(["📋 Données", "📈 Graphiques", "📥 Export & Stats"])

        with t1:
            if 'Adresse' in df.columns:
                locs    = st.multiselect("Filtrer par localité",
                                         sorted(df['Adresse'].dropna().unique()))
                df_view = df[df['Adresse'].isin(locs)] if locs else df
            else:
                df_view = df
            st.dataframe(df_view, use_container_width=True, height=480)
            st.caption(f"{len(df_view)} annonces affichées sur {len(df)} au total")

        with t2:
            p = prix_serie(df)
            if 'Adresse' in df.columns:
                fig_loc = px.bar(
                    df['Adresse'].value_counts().head(10).reset_index(),
                    x='count', y='Adresse', orientation='h',
                    title="Top 10 localités",
                    color_discrete_sequence=['#17a2b8'],
                    labels={'count': "Nombre d'annonces", 'Adresse': ''}
                )
                fig_loc.update_layout(plot_bgcolor="#fafafa")
                st.plotly_chart(fig_loc, use_container_width=True)

            if len(p):
                fig_hist = px.histogram(
                    p, nbins=30, title="Distribution des Prix (CFA) — après Winsorisation",
                    color_discrete_sequence=['#764ba2'],
                    labels={"value": "Prix (CFA)", "count": "Nb annonces"}
                )
                fig_hist.add_vline(x=p.median(), line_dash="dash", line_color="#333",
                                   annotation_text=f"Médiane : {p.median():,.0f} CFA")
                if meta:
                    fig_hist.add_vline(x=meta['borne_basse'], line_dash="dot",
                                       line_color="#e74c3c",
                                       annotation_text=f"5% : {meta['borne_basse']:,.0f}")
                    fig_hist.add_vline(x=meta['borne_haute'], line_dash="dot",
                                       line_color="#e74c3c",
                                       annotation_text=f"95% : {meta['borne_haute']:,.0f}")
                fig_hist.update_layout(plot_bgcolor="#fafafa", bargap=0.05, showlegend=False)
                st.plotly_chart(fig_hist, use_container_width=True)

                if 'Nom' in df.columns and 'Prix (CFA)' in df.columns:
                    avg = (df.groupby('Nom')['Prix (CFA)'].mean()
                             .sort_values(ascending=False).head(12).reset_index())
                    fig_bar = px.bar(
                        avg, x='Prix (CFA)', y='Nom', orientation='h',
                        title="Prix moyen par race / type (Top 12) — après Winsorisation",
                        color_discrete_sequence=['#e67e22'],
                        labels={'Prix (CFA)': 'Prix moyen (CFA)', 'Nom': ''}
                    )
                    fig_bar.update_layout(plot_bgcolor="#fafafa")
                    st.plotly_chart(fig_bar, use_container_width=True)

        with t3:
            st.markdown("### 📥 Télécharger les données nettoyées & corrigées")
            st.markdown(
                csv_dl(df, f"{slug}_traite_winsorise.csv", "📥 Télécharger le CSV complet"),
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)
            st.write(f"**{len(df)} lignes** | Colonnes : {', '.join(df.columns)}")
            st.markdown("#### 📊 Statistiques descriptives")
            st.dataframe(df.describe(), use_container_width=True)

    else:
        st.markdown(
            "<div style='text-align:center;padding:60px 20px;color:#888;"
            "background:#fafafa;border-radius:12px;border:2px dashed #ddd;margin-top:20px'>"
            "<h3>👆 Configurez les paramètres dans la barre latérale</h3>"
            "<p>Sélectionnez une <b>catégorie</b> et le <b>nombre de pages</b>, "
            "puis cliquez sur <b>🚀 Lancer le scraping</b>.</p>"
            "</div>", unsafe_allow_html=True
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DONNÉES NON TRAITÉES
# ═══════════════════════════════════════════════════════════════════════════════
elif mode == "📥 Données non traitées":
    c = conn(st.sidebar.text_input("BD non traitée", "data/SGBD_CoinafriqueN.db"))
    if not c: st.error("❌ BD introuvable"); st.stop()
    st.sidebar.success("✅ Connectée")
    df = load(tbl_b, c)
    if df.empty: st.warning("Aucune donnée."); st.stop()

    df['_page_num'] = (df['web_scraper_start_url']
                       .str.extract(r'page=(\d+)', expand=False)
                       .astype(float).fillna(1).astype(int))
    pages = sorted(df['_page_num'].unique())
    pg    = st.sidebar.number_input("Page", 1, len(pages), 1, key="pg")
    url   = df[df['_page_num'] == pages[pg-1]]['web_scraper_start_url'].iloc[0]
    df_pg = clean_raw(df[df['_page_num'] == pages[pg-1]].copy())

    st.subheader(cat); kpis(df)
    st.caption(f"Page {pg}/{len(pages)} — {len(df_pg)} annonces | 🔗 {url}")
    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)
    if c1.button("⏮️"): st.session_state['pg'] = 1;                     st.rerun()
    if c2.button("◀️"): st.session_state['pg'] = max(1, pg-1);          st.rerun()
    c3.markdown(f"<div style='text-align:center;padding:8px'><b>{pg}/{len(pages)}</b></div>",
                unsafe_allow_html=True)
    if c4.button("▶️"): st.session_state['pg'] = min(len(pages), pg+1); st.rerun()
    if c5.button("⏭️"): st.session_state['pg'] = len(pages);            st.rerun()

    st.dataframe(df_pg, use_container_width=True, height=440)
    st.markdown("---")

    c1, c2 = st.columns(2)
    c1.markdown(csv_dl(df_pg, f"{tbl_b}_p{pg}.csv"), unsafe_allow_html=True)
    c1.caption(f"{len(df_pg)} annonces (page {pg})")
    df_all = clean_raw(df.copy())
    c2.markdown(csv_dl(df_all, f"{tbl_b}_complet.csv"), unsafe_allow_html=True)
    c2.caption(f"{len(df_all)} annonces (tout)")

    if st.checkbox("📈 Visualisations"):
        p = prix_serie(df)
        if len(p):
            st.plotly_chart(
                px.histogram(p, nbins=30, color_discrete_sequence=['#17a2b8'],
                             title="Distribution des prix"),
                use_container_width=True
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TABLEAU DE BORD
# ═══════════════════════════════════════════════════════════════════════════════
elif mode == "📈 Tableau de bord des données":
    c    = conn(st.sidebar.text_input("BD non traitée", "data/SGBD_CoinafriqueN.db"))
    meth = st.sidebar.selectbox("Méthode outliers", ["Winsorization (5%-95%)", "Filtre IQR"])
    if not c: st.error("❌ BD introuvable"); st.stop()

    df = load(tbl_b, c)
    if df.empty: st.warning("Aucune donnée."); st.stop()
    if 'Details' in df.columns: df = df.rename(columns={'Details': 'Nom'})

    # ── Série brute numérique ─────────────────────────────────────────────────
    pb = prix_serie(df)
    if pb.empty:
        st.error("❌ Aucune colonne Prix exploitable dans cette table."); st.stop()

    # ── Traitement des outliers DIRECTEMENT sur la série numérique ───────────
    def winsorize_serie(s):
        lo, hi = np.percentile(s, 5), np.percentile(s, 95)
        return s.clip(lo, hi)

    def iqr_serie(s):
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR = Q3 - Q1
        return s[(s >= Q1 - 1.5*IQR) & (s <= Q3 + 1.5*IQR)]

    pt = winsorize_serie(pb) if "Wins" in meth else iqr_serie(pb)

    # ── En-tête & KPIs comparatifs ────────────────────────────────────────────
    st.subheader(f"📈 Tableau de bord — {cat}")
    st.markdown("#### Comparaison des indicateurs clés")
    col_kpi1, col_kpi2 = st.columns(2)
    with col_kpi1:
        st.markdown("**🔴 Données brutes (non traitées)**")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("N valeurs", f"{len(pb):,}")
        k2.metric("Moyenne",   f"{pb.mean():,.0f} CFA")
        k3.metric("Médiane",   f"{pb.median():,.0f} CFA")
        k4.metric("Max",       f"{pb.max():,.0f} CFA")
    with col_kpi2:
        st.markdown(f"**🟢 Après {meth}**")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("N valeurs", f"{len(pt):,}",
                  delta=f"{len(pt)-len(pb):+,}" if len(pt) != len(pb) else None)
        k2.metric("Moyenne",   f"{pt.mean():,.0f} CFA",
                  delta=f"{pt.mean()-pb.mean():+,.0f}")
        k3.metric("Médiane",   f"{pt.median():,.0f} CFA",
                  delta=f"{pt.median()-pb.median():+,.0f}")
        k4.metric("Max",       f"{pt.max():,.0f} CFA",
                  delta=f"{pt.max()-pb.max():+,.0f}")

    st.markdown("---")
    t1, t2, t3 = st.tabs(["📊 Histogramme", "📦 Boxplot", "🔍 Comparaison & Stats"])

    with t1:
        c1, c2 = st.columns(2)
        for col, p, lab, clr in [(c1, pb, "🔴 Non traité", "#e74c3c"),
                                  (c2, pt, f"🟢 {meth}",   "#27ae60")]:
            fig = px.histogram(p, nbins=40, color_discrete_sequence=[clr],
                               opacity=.85, title=lab, labels={"value": "Prix (CFA)"})
            fig.add_vline(x=p.mean(), line_dash="dot", line_color="navy",
                          annotation_text=f"Moy: {p.mean():,.0f}",
                          annotation_position="top right")
            fig.add_vline(x=p.median(), line_dash="dash", line_color=clr,
                          annotation_text=f"Méd: {p.median():,.0f}",
                          annotation_position="top left")
            fig.update_layout(showlegend=False, plot_bgcolor="#fafafa", bargap=.05)
            col.plotly_chart(fig, use_container_width=True)
            col.caption(
                f"{len(p):,} valeurs | min {p.min():,.0f} → max {p.max():,.0f} CFA "
                f"| σ={p.std():,.0f}"
            )

    with t2:
        c1, c2 = st.columns(2)
        for col, p, lab, clr in [(c1, pb, "Non traité", "#e74c3c"),
                                  (c2, pt, meth,         "#27ae60")]:
            fig = go.Figure(go.Box(
                y=p, name=lab, marker_color=clr,
                boxmean="sd", boxpoints="outliers", jitter=0.3, pointpos=-1.8
            ))
            fig.update_layout(
                yaxis_title="Prix (CFA)", plot_bgcolor="#fafafa",
                showlegend=False, title=lab, yaxis=dict(tickformat=",.0f")
            )
            col.plotly_chart(fig, use_container_width=True)
            col.caption(
                f"Min={p.min():,.0f} | Q1={p.quantile(.25):,.0f} | "
                f"Méd={p.median():,.0f} | Q3={p.quantile(.75):,.0f} | "
                f"Max={p.max():,.0f} CFA"
            )

    with t3:
        df_comp = pd.DataFrame({
            "Prix (CFA)": pd.concat([pb, pt], ignore_index=True),
            "Type": ["Non traité"] * len(pb) + [meth] * len(pt)
        })
        fig_ov = px.histogram(
            df_comp, x="Prix (CFA)", color="Type", nbins=50,
            barmode="overlay", opacity=.65,
            color_discrete_map={"Non traité": "#e74c3c", meth: "#27ae60"},
            title="Superposition des distributions"
        )
        fig_ov.update_layout(plot_bgcolor="#fafafa", bargap=0.03,
                             xaxis=dict(tickformat=",.0f"))
        st.plotly_chart(fig_ov, use_container_width=True)

        fig_box = go.Figure([
            go.Box(y=pb, name="Non traité", marker_color="#e74c3c",
                   boxmean="sd", boxpoints="outliers"),
            go.Box(y=pt, name=meth,         marker_color="#27ae60",
                   boxmean="sd", boxpoints="outliers")
        ])
        fig_box.update_layout(
            yaxis_title="Prix (CFA)", plot_bgcolor="#fafafa",
            title="Boxplot comparatif", yaxis=dict(tickformat=",.0f")
        )
        st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("#### 📋 Statistiques comparées")
        st.dataframe(
            pd.DataFrame({
                "Statistique":   ["N valeurs","Min","Q1 (25%)","Médiane",
                                  "Moyenne","Q3 (75%)","Max","Écart-type"],
                "🔴 Non traité": stat_row(pb),
                f"🟢 {meth}":    stat_row(pt),
            }).set_index("Statistique"),
            use_container_width=True
        )

        if "Wins" in meth:
            lo5, hi95 = np.percentile(pb, 5), np.percentile(pb, 95)
            st.info(
                f"ℹ️ **Winsorization 5%-95%** : les prix < {lo5:,.0f} CFA ont été "
                f"remontés à {lo5:,.0f} CFA et les prix > {hi95:,.0f} CFA ont été "
                f"abaissés à {hi95:,.0f} CFA. Le max passe de **{pb.max():,.0f}** "
                f"à **{pt.max():,.0f} CFA** "
                f"(−{(1-pt.max()/pb.max())*100:.1f}%)."
            )
        else:
            Q1, Q3 = pb.quantile(.25), pb.quantile(.75)
            IQR = Q3 - Q1
            n_retires = len(pb) - len(pt)
            st.info(
                f"ℹ️ **Filtre IQR** : bornes acceptées [{Q1-1.5*IQR:,.0f} ; "
                f"{Q3+1.5*IQR:,.0f}] CFA. "
                f"**{n_retires} valeurs aberrantes supprimées** "
                f"({n_retires/len(pb)*100:.1f}% des données)."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. COMMENTAIRES
# ═══════════════════════════════════════════════════════════════════════════════
else:
    st.subheader("💬 Commentaires & Feedback")
    t1, t2, t3 = st.tabs(["📝 Feedback", "🐛 Bug", "⭐ Évaluation"])

    with t1:
        col_kobo, col_gform = st.columns(2)
        with col_kobo:
            st.markdown("#### 📋 KoBoToolbox")
            st.markdown(
                f'<div style="text-align:center;margin:12px 0">'
                f'<a href="{KOBO}" target="_blank" style="background:#17a2b8;color:white;'
                f'padding:12px 28px;border-radius:8px;text-decoration:none;font-weight:bold">'
                f'🔗 Ouvrir KoBoToolbox</a></div>',
                unsafe_allow_html=True
            )
        with col_gform:
            st.markdown("#### 📝 Google Forms")
            st.markdown(
                f'<div style="text-align:center;margin:12px 0">'
                f'<a href="{GFORM}" target="_blank" style="background:#ea4335;color:white;'
                f'padding:12px 28px;border-radius:8px;text-decoration:none;font-weight:bold">'
                f'🔗 Ouvrir Google Forms</a></div>',
                unsafe_allow_html=True
            )
        st.markdown("---")
        choix      = st.radio("Afficher le formulaire :", ["KoBoToolbox", "Google Forms"],
                              horizontal=True)
        iframe_url = KOBO if choix == "KoBoToolbox" else GFORM + "?embedded=true"
        st.components.v1.iframe(src=iframe_url, height=550, scrolling=True)

    with t2:
        titre = st.text_input("Titre du bug")
        desc  = st.text_area("Description", height=120)
        st.selectbox("Gravité", ["🟢 Mineur","🟡 Modéré","🟠 Important","🔴 Critique"])
        if st.button("🚨 Signaler", type="primary"):
            if titre and desc:
                st.success("✅ Bug signalé !")
                st.balloons()
            else:
                st.error("Titre et description sont requis.")

    with t3:
        n = st.slider("Note", 1, 5, 4)
        st.markdown(f"<h3 style='text-align:center'>{'⭐'*n}{'☆'*(5-n)}</h3>",
                    unsafe_allow_html=True)
        st.text_area("Commentaire (optionnel)", height=100)
        if st.button("📊 Soumettre", type="primary"):
            st.success(f"✅ Merci pour votre évaluation ! Note : {n}/5")
            if n >= 4: st.balloons()


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("🐾 Marché des Animaux · Streamlit · [CoinAfrique Sénégal](https://sn.coinafrique.com)")
