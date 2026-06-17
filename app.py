import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc,
                             confusion_matrix, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings("ignore")

# CONFIG
st.set_page_config(page_title="JO 2028 — Los Angeles", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #020617 0%, #0f172a 50%, #111827 100%);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: #020617;
    border-right: 1px solid rgba(255,255,255,0.05);
}
.block-container { max-width: 1400px; padding-top: 1.2rem; }
h1,h2,h3 { color: #f1f5f9 !important; font-weight: 700 !important; }
p, label, .stCaption { color: #94a3b8; }
.section {
    background: rgba(15,23,42,0.85);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 20px 22px 14px 22px;
    margin-bottom: 18px;
}
.kpi {
    background: rgba(15,23,42,0.9);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 14px 18px;
}
.kpi .label { color: #64748b; font-size: 0.85rem; font-weight: 600; }
.kpi .value { color: #f1f5f9; font-size: 1.8rem; font-weight: 800; }
.insight {
    background: rgba(30,58,138,0.18);
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 9px 14px;
    color: #bfdbfe;
    font-size: 0.89rem;
    margin-top: 8px;
}
.review {
    background: rgba(15,23,42,0.9);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
}
.review .author { color: #60a5fa; font-weight: 700; font-size: 0.92rem; }
.review .stars  { color: #f59e0b; font-size: 1rem; }
.review .body   { color: #cbd5e1; font-size: 0.90rem; margin-top: 3px; }
.year-hint { font-size: 0.78rem; color: #475569; margin-top: -4px; }
div[role="radiogroup"] {
    background: rgba(15,23,42,0.85);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 8px 12px;
    margin-bottom: 18px;
}
hr { border: none; border-top: 1px solid rgba(255,255,255,0.05); margin: 14px 0; }

/* Navigation sur une seule ligne */
div[role="radiogroup"] {
    display: flex !important;
    flex-wrap: nowrap !important;
    gap: 18px !important;
}
div[role="radiogroup"] label {
    white-space: nowrap !important;
}
</style>
""", unsafe_allow_html=True)


REMAP = {
    "URS": "RUS", "EUA": "GER", "EUN": "RUS", "TCH": "CZE",
    "YUG": "SRB", "GDR": "GER", "FRG": "GER", "SCG": "SRB",
    "ANZ": "AUS", "BOH": "CZE", "RU1": "RUS", "WIF": "JAM",
}


# CHARGEMENT 
@st.cache_resource
def load():
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(
        os.path.join(BASE_DIR, "data", "athlete_events.csv"),
        usecols=["Name", "Sex", "Age", "Height", "Weight", "Team", "NOC", "Games", "Year", "Season", "City", "Sport", "Event", "Medal"],
        low_memory=False
    )
    df = df.drop_duplicates()
    for col in ["Age", "Height", "Weight", "Year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["NOC"]  = df["NOC"].replace(REMAP)
    df["Team"] = df["NOC"]
    for col in ["Age", "Height", "Weight"]:
        med = df.groupby("Sport")[col].transform("median")
        df[col] = df[col].fillna(med).fillna(df[col].median())
    df = df[df["Age"].between(10, 75)]
    df = df[df["Height"].between(130, 230)]
    df = df[df["Weight"].between(30, 200)]
    df["Has_Medal"]   = df["Medal"].notna().astype(int)
    df["Medal_Score"] = df["Medal"].map({"Gold": 3, "Silver": 2, "Bronze": 1}).fillna(0)
    p = df.groupby("Name")["Year"].count().reset_index()
    p.columns = ["Name", "Participation_Count"]
    df = df.merge(p, on="Name", how="left")
    t = df.groupby("Name")["Medal_Score"].sum().reset_index()
    t.columns = ["Name", "Total_Medals_Athlete"]
    df = df.merge(t, on="Name", how="left")
    c = df.groupby("NOC")["Has_Medal"].sum().reset_index()
    c.columns = ["NOC", "Country_Medals"]
    df = df.merge(c, on="NOC", how="left")
    return df


@st.cache_data(show_spinner=False, persist="disk")
def get_athlete_rankings(_df):
    """Version rapide : on groupe seulement les lignes médaillées, donc beaucoup moins de données."""
    m = _df[_df["Has_Medal"] == 1]
    return (m.groupby(["Name", "NOC", "Sport", "Sex"], observed=True)
            .agg(Total=("Has_Medal", "sum"),
                 Or=("Medal",    lambda x: (x == "Gold").sum()),
                 Argent=("Medal",  lambda x: (x == "Silver").sum()),
                 Bronze=("Medal",  lambda x: (x == "Bronze").sum()),
                 Score=("Medal_Score", "sum"),
                 Participations=("Year", "nunique"),
                 Debut=("Year", "min"),
                 Fin=("Year", "max"))
            .reset_index())

@st.cache_data(show_spinner=False, persist="disk")
def get_new_gen(_df):
    """Précalcul nouvelles générations — fusion one-shot"""
    fy = _df.groupby("Name")["Year"].min().reset_index()
    fy.columns = ["Name", "First"]
    df_ng = _df[_df["Has_Medal"] == 1].merge(fy, on="Name", how="left")
    return df_ng[df_ng["First"] >= 2008]

@st.cache_data(show_spinner=False, persist="disk")
def get_cotes(_df):
    """Version rapide : cotes calculées seulement sur les médaillés récents."""
    df_rc = _df[(_df["Year"] >= 2012) & (_df["Has_Medal"] == 1)]
    return (df_rc.groupby(["Name", "NOC", "Sport"], observed=True)
            .agg(Med=("Has_Medal", "sum"),
                 Score=("Medal_Score", "sum"),
                 Parts=("Year", "nunique"),
                 Last=("Year", "max"))
            .reset_index())

@st.cache_data(show_spinner=False, persist="disk")
def get_momentum(_df):
    """Version rapide : momentum calculé seulement sur les lignes médaillées."""
    df_sum = _df[(_df["Season"] == "Summer") & (_df["Has_Medal"] == 1)]
    cym = df_sum.groupby(["NOC", "Year", "Sport"], observed=True)["Has_Medal"].sum().reset_index()
    r4 = sorted(cym["Year"].unique())[-4:]
    last_yr = r4[-1]
    w = {yr: i + 1 for i, yr in enumerate(r4)}
    return cym, r4, last_yr, w


# MODÈLES ML 
@st.cache_data(show_spinner=False, persist="disk")
def get_kmeans_data(_df):
    """
    Version rapide et cohérente avec les autres onglets.
    Les graphes restent identiques, mais le calcul est accéléré :
    - groupby optimisé
    - K-Means entraîné sur un échantillon stable
    - MiniBatchKMeans beaucoup plus rapide
    - cache disque pour éviter de recalculer à chaque lancement
    """
    CLUSTER_FEATURES = ["Age", "Height", "Weight", "Participation_Count", "Total_Medals_Athlete"]

    # 1) Une ligne par athlète / pays / sport / sexe
    df_cluster = (
        _df.groupby(["Name", "NOC", "Sport", "Sex"], observed=True, sort=False)
        .agg(
            Age=("Age", "median"),
            Height=("Height", "median"),
            Weight=("Weight", "median"),
            Participation_Count=("Participation_Count", "max"),
            Total_Medals_Athlete=("Total_Medals_Athlete", "max"),
        )
        .reset_index()
        .dropna(subset=CLUSTER_FEATURES)
    )

    # 2) Échantillon stable pour accélérer l'entraînement K-Means
    train_cluster = df_cluster
    if len(df_cluster) > 30000:
        train_cluster = df_cluster.sample(n=30000, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_cluster[CLUSTER_FEATURES])
    X_all_scaled = scaler.transform(df_cluster[CLUSTER_FEATURES])

    # 3) Méthode du coude rapide : même graphe, calcul plus léger
    inertias = []
    for k in range(2, 11):
        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=42,
            n_init=3,
            max_iter=60,
            batch_size=4096
        )
        km.fit(X_train_scaled)
        inertias.append(float(km.inertia_))

    # 4) Modèle final K=3 rapide
    kmeans = MiniBatchKMeans(
        n_clusters=3,
        random_state=42,
        n_init=3,
        max_iter=80,
        batch_size=4096
    )
    kmeans.fit(X_train_scaled)
    df_cluster["Cluster"] = kmeans.predict(X_all_scaled)

    score_by_cluster = (
        df_cluster.groupby("Cluster", observed=True)["Total_Medals_Athlete"]
        .mean()
        .sort_values()
    )
    mapping = {
        score_by_cluster.index[0]: "Débutants",
        score_by_cluster.index[1]: "Intermédiaires",
        score_by_cluster.index[2]: "Performants",
    }
    df_cluster["Profil"] = df_cluster["Cluster"].map(mapping)

    # Version légère pour le filtre pays
    df_cnoc = df_cluster[["NOC", "Profil"]].copy()

    return df_cluster, inertias, df_cnoc, CLUSTER_FEATURES



@st.cache_data(show_spinner=False)
def train_models(_df):
    FEATURES = ["Age", "Height", "Weight",
                "Participation_Count", "Total_Medals_Athlete", "Country_Medals"]
    df_model = _df[FEATURES + ["Has_Medal"]].dropna().copy()
    if len(df_model) > 60000:
        pos = df_model[df_model["Has_Medal"] == 1].sample(n=min(30000, (df_model["Has_Medal"] == 1).sum()), random_state=42)
        neg = df_model[df_model["Has_Medal"] == 0].sample(n=30000, random_state=42)
        df_model = pd.concat([pos, neg]).sample(frac=1, random_state=42)
    X = df_model[FEATURES]
    y = df_model["Has_Medal"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=50, max_depth=6,
                                min_samples_split=20, min_samples_leaf=8,
                                class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train_s, y_train)
    y_pred_lr = lr.predict(X_test_s)

    return rf, lr, scaler, X_test, X_test_s, y_test, y_pred_rf, y_pred_lr, FEATURES




df        = load()
df = df[df["Year"] <= 2016].copy()

ALL_YEARS = sorted(df["Year"].dropna().astype(int).unique().tolist())
SPORTS    = sorted(df["Sport"].dropna().unique().tolist())
COUNTRIES = sorted(df["NOC"].dropna().unique().tolist())
SEASONS   = sorted(df["Season"].dropna().unique().tolist())


# HELPERS
def kpi(label, value, color):
    st.markdown(
        f'<div class="kpi" style="border-left:3px solid {color};">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div></div>',
        unsafe_allow_html=True)

def insight(txt):
    st.markdown(f'<div class="insight">{txt}</div>', unsafe_allow_html=True)

def theme(fig, h=420):
    fig.update_layout(
        template="plotly_dark", height=h,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", size=13),
        margin=dict(l=20, r=20, t=50, b=20),
        title_x=0.01, title_font_size=14)
    return fig

def limit_year_axis(fig, start=1896, end=2016):
    """Force l'axe des années à s'arrêter visuellement en 2016."""
    ticks = [y for y in [1900, 1920, 1940, 1960, 1980, 2000, 2016] if start <= y <= end]
    fig.update_xaxes(range=[start, end], tickmode="array", tickvals=ticks)
    return fig

# SIDEBAR
st.sidebar.markdown("## Filtres")

season_sel = st.sidebar.selectbox("Saison", ["Toutes"] + SEASONS, key="season_global")

# Années disponibles selon la saison sélectionnée
avail = (
    sorted(df[df["Season"] == season_sel]["Year"].dropna().astype(int).unique().tolist())
    if season_sel != "Toutes" else ALL_YEARS
)

year_raw = st.sidebar.text_input("Année", value=str(avail[-1]), placeholder="Ex : 2016", key="year_global_txt")
st.sidebar.markdown(
    f'<div class="year-hint">Disponibles : {avail[0]} – {avail[-1]}</div>',
    unsafe_allow_html=True
)

year_sel = avail[-1]
if year_raw.strip().isdigit():
    candidate = int(year_raw.strip())
    if candidate in avail:
        year_sel = candidate
    else:
        st.sidebar.warning(f"Année non disponible pour {season_sel}. Choisis entre {avail[0]} et {avail[-1]}.")
elif year_raw.strip():
    st.sidebar.error("Saisir une année numérique")

sport_sel   = st.sidebar.selectbox("Sport",     ["Tous"] + SPORTS, key="sport_global")
sex_sel     = st.sidebar.selectbox("Sexe",       ["Tous", "M", "F"], key="sex_global")
country_sel = st.sidebar.selectbox("Pays (NOC)", ["Tous"] + COUNTRIES, key="country_global")

if st.sidebar.button("Appliquer les filtres", type="primary"):
    st.rerun()

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.caption(f"Edition active : JO {year_sel}")

df_f = df[df["Year"] == year_sel].copy()
if sport_sel   != "Tous":   df_f = df_f[df_f["Sport"]  == sport_sel]
if sex_sel     != "Tous":   df_f = df_f[df_f["Sex"]    == sex_sel]
if season_sel  != "Toutes": df_f = df_f[df_f["Season"] == season_sel]
if country_sel != "Tous":   df_f = df_f[df_f["NOC"]    == country_sel]


season_lbl = df[df["Year"] == year_sel]["Season"].iloc[0] \
             if not df[df["Year"] == year_sel].empty else ""
st.markdown(f"##  Jeux Olympiques {year_sel} · {season_lbl}")
st.caption("Application de data storytelling — performances olympiques historiques & prédictions JO 2028.")
st.markdown("<hr>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1: kpi("Athlètes",  f"{len(df_f):,}".replace(",", " "), "#3b82f6")
with c2: kpi("Pays",      str(df_f["NOC"].nunique()),          "#10b981")
with c3: kpi("Sports",    str(df_f["Sport"].nunique()),        "#f59e0b")
with c4: kpi("Médailles", str(int(df_f["Has_Medal"].sum())),   "#ef4444")
st.markdown("<br>", unsafe_allow_html=True)

PAGES = ["Analyse", "Carte", "Athlètes", "Timeline", "K-Means", "Prédiction 2028"]
page = st.radio("Navigation", PAGES, horizontal=True, label_visibility="collapsed", key="page")


if page == "Analyse":

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Top 10 pays médaillés — JO d'été vs JO d'hiver")

    col1, col2 = st.columns(2)
    for col, season, scale, label in [
        (col1, "Summer", "Blues",   "JO d'été"),
        (col2, "Winter", "Purples", "JO d'hiver")
    ]:
        with col:
            top = (df[df["Season"] == season]
                   .groupby("NOC")["Has_Medal"].sum()
                   .sort_values(ascending=False).head(10).reset_index())
            top.columns = ["Pays", "Médailles"]
            fig = px.bar(top, x="Pays", y="Médailles",
                         color="Médailles", color_continuous_scale=scale,
                         text="Médailles", title=f"Top 10 pays — {label}")
            fig.update_traces(textposition="outside",
                              textfont=dict(color="white", size=11))
            fig.update_xaxes(tickangle=45)
            fig.update_yaxes(range=[0, top["Médailles"].max() * 1.18])
            fig.update_layout(coloraxis_showscale=False, showlegend=False)
            st.plotly_chart(theme(fig, 400), use_container_width=True)

    top_ete = df[df["Season"]=="Summer"].groupby("NOC")["Has_Medal"].sum().idxmax()
    top_hiv = df[df["Season"]=="Winter"].groupby("NOC")["Has_Medal"].sum().idxmax()
    insight(f"Les JO d'été sont dominés par {top_ete}, les JO d'hiver par {top_hiv}. "
            "Le remapping URS→RUS et GDR→GER gonfle les scores historiques avant 1992 — "
            "choix assumé pour garder la continuité des nations.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Évolution des médailles dans le temps")

    df_tr = df
    if sport_sel   != "Tous": df_tr = df_tr[df_tr["Sport"] == sport_sel]
    if country_sel != "Tous": df_tr = df_tr[df_tr["NOC"]   == country_sel]

    col1, col2 = st.columns(2)
    for col, season, color, label in [
        (col1, "Summer", "#3b82f6", "JO d'été"),
        (col2, "Winter", "#a78bfa", "JO d'hiver")
    ]:
        with col:
            medals = (df_tr[df_tr["Season"] == season]
                      .groupby("Year")["Has_Medal"].sum().reset_index().sort_values("Year"))
            fig = px.line(medals, x="Year", y="Has_Medal", markers=True,
                          title=f"Évolution — {label}",
                          color_discrete_sequence=[color])
            fig.update_traces(line=dict(width=2.5), marker=dict(size=6))
            fig.update_layout(showlegend=False, yaxis_title="Médailles", xaxis_title="Année")
            fig.add_vline(x=2016, line_dash="dash", line_color="#ef4444",
                          annotation_text=" 2016", annotation_position="top right")
            limit_year_axis(fig)
            st.plotly_chart(theme(fig, 380), use_container_width=True)

    peak = (df_tr[df_tr["Season"] == "Summer"]
            .groupby("Year")["Has_Medal"].sum().idxmax())
    insight(f"Le nombre de médailles augmente régulièrement depuis 1896 avec l'ajout de nouvelles disciplines. "
            f"Le pic des JO d'été est atteint en {int(peak)}. "
            "Les creux correspondent aux guerres mondiales et aux boycotts (1980, 1984).")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Répartition des médailles par genre")

    col1, col2 = st.columns(2)
    for col, data, title in [
        (col1, df, "Toutes saisons"),
        (col2, df[df["Season"] == "Summer"], "JO d'été uniquement")
    ]:
        with col:
            gd = data.groupby("Sex")["Has_Medal"].sum().reset_index()
            gd["Sexe"] = gd["Sex"].map({"M": "Hommes", "F": "Femmes"})
            fig = px.pie(gd, names="Sexe", values="Has_Medal", hole=0.45,
                         color_discrete_sequence=["#3b82f6", "#ec4899"],
                         title=title)
            st.plotly_chart(theme(fig, 340), use_container_width=True)

    pct_f = (df["Sex"] == "F").mean() * 100
    insight(f"Les femmes représentent {pct_f:.1f}% des athlètes du dataset. "
            "La part féminine a fortement augmenté depuis 1990 avec l'ouverture progressive des JO aux disciplines féminines.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Heatmap des corrélations")
    df_corr = df[["Age", "Height", "Weight", "Has_Medal", "Medal_Score",
                  "Participation_Count", "Total_Medals_Athlete", "Country_Medals"]].copy()
    df_corr.columns = ["Âge", "Taille", "Poids", "A_médaille", "Score_médaille",
                        "Participations", "Médailles_athlète", "Médailles_pays"]
    corr = df_corr.corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, title="Corrélations entre variables")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=520,
        font=dict(color="#e2e8f0", size=13),
        coloraxis_showscale=True,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(tickangle=35, tickfont=dict(size=12)),
        yaxis=dict(tickfont=dict(size=12))
    )
    st.plotly_chart(fig, use_container_width=True)
    insight("Médailles_athlète et Médailles_pays sont les plus corrélées à A_médaille. "
            "L'âge, la taille et le poids ont un faible impact direct — "
            "ce sont les variables d'historique qui dominent la prédiction.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Score pondéré de médailles (Or×3 + Argent×2 + Bronze×1)")

    col1, col2 = st.columns(2)
    with col1:
        kpi_c = df.groupby("NOC")["Medal_Score"].sum().sort_values(ascending=False).head(10).reset_index()
        kpi_c.columns = ["Pays", "Score"]
        fig = px.bar(kpi_c, x="Score", y="Pays", orientation="h",
                     color="Score", color_continuous_scale="teal",
                     text="Score", title="Top 10 pays — score pondéré")
        fig.update_traces(textposition="inside", insidetextanchor="middle",
                          textfont=dict(color="white", size=11))
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(theme(fig, 380), use_container_width=True)

    with col2:
        kpi_s = df.groupby("Sport")["Medal_Score"].sum().sort_values(ascending=False).head(10).reset_index()
        kpi_s.columns = ["Sport", "Score"]
        fig = px.bar(kpi_s, x="Score", y="Sport", orientation="h",
                     color="Score", color_continuous_scale="Oranges",
                     text="Score", title="Top 10 sports — score pondéré")
        fig.update_traces(textposition="inside", insidetextanchor="middle",
                          textfont=dict(color="white", size=11))
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(theme(fig, 380), use_container_width=True)

    insight("L'athlétisme et la natation génèrent le plus de médailles car ce sont les disciplines "
            "avec le plus grand nombre d'épreuves. Les USA dominent le score pondéré historique.")
    st.markdown("</div>", unsafe_allow_html=True)


# ONGLET 2 - CARTE DES MÉDAILLES
if page == "Carte":
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Carte mondiale des médailles")

    f1, f2, f3 = st.columns(3)
    with f1:
        default_map_season = season_sel if season_sel in ["Summer", "Winter"] else "Summer"
        saison_map = st.selectbox(
            "Saison",
            ["Summer", "Winter"],
            index=["Summer", "Winter"].index(default_map_season),
            key="saison_map"
        )

  
    years_map = sorted(
        df[df["Season"] == saison_map]["Year"].dropna().astype(int).unique().tolist()
    )
    with f2:
        default_year = year_sel if year_sel in years_map else years_map[-1]
        yr_map = st.selectbox(
            "Année",
            years_map,
            index=years_map.index(default_year),
            key="yr_map"
        )
    with f3:
        type_med = st.selectbox("Médaille", ["Toutes", "Gold", "Silver", "Bronze"], key="type_med")

    df_map = df[df["Year"] == yr_map].copy()
    if saison_map != "Toutes": df_map = df_map[df_map["Season"] == saison_map]
    if type_med   != "Toutes": df_map = df_map[df_map["Medal"]  == type_med]

    NOC_TO_ISO = {
        "USA":"USA","CHN":"CHN","RUS":"RUS","GBR":"GBR","GER":"DEU","AUS":"AUS",
        "FRA":"FRA","ITA":"ITA","KOR":"KOR","JPN":"JPN","CAN":"CAN","NED":"NLD",
        "HUN":"HUN","CUB":"CUB","NOR":"NOR","SWE":"SWE","POL":"POL","ROU":"ROU",
        "FIN":"FIN","DEN":"DNK","ESP":"ESP","BRA":"BRA","KEN":"KEN","ETH":"ETH",
        "NZL":"NZL","BEL":"BEL","UKR":"UKR","CZE":"CZE","AUT":"AUT","SRB":"SRB",
        "SUI":"CHE","JAM":"JAM","ARG":"ARG","MEX":"MEX","RSA":"ZAF","IND":"IND",
        "TUR":"TUR","GRE":"GRC","AZE":"AZE","BLR":"BLR","IRN":"IRN","THA":"THA",
        "SVK":"SVK","GEO":"GEO","NGR":"NGA","EGY":"EGY","MAR":"MAR","COL":"COL",
        "CHI":"CHL","IRL":"IRL","POR":"PRT","CRO":"HRV","BUL":"BGR","LAT":"LVA",
        "LTU":"LTU","EST":"EST","TRI":"TTO","UZB":"UZB","KAZ":"KAZ","ALG":"DZA",
    }
    cmap = df_map.groupby("NOC")["Has_Medal"].sum().reset_index()
    cmap.columns = ["NOC", "Médailles"]
    cmap["ISO"] = cmap["NOC"].map(NOC_TO_ISO)
    cmap_iso = cmap.dropna(subset=["ISO"])

    if not cmap_iso.empty:
        fig = px.choropleth(cmap_iso, locations="ISO", locationmode="ISO-3",
                            color="Médailles", color_continuous_scale="YlOrRd",
                            hover_name="NOC",
                            title=f"Médailles — {yr_map} | {saison_map} | {type_med}")
        fig.update_layout(template="plotly_dark", height=500,
                          paper_bgcolor="rgba(0,0,0,0)",
                          geo=dict(showframe=False, showcoastlines=True,
                                   projection_type="natural earth",
                                   bgcolor="rgba(0,0,0,0)"),
                          margin=dict(l=0, r=0, t=45, b=0))
        st.plotly_chart(fig, use_container_width=True)
        top_map = cmap.sort_values("Médailles", ascending=False).iloc[0]
        insight(f"En {yr_map}, {top_map['NOC']} est le pays le plus médaillé "
                f"avec {int(top_map['Médailles'])} médailles.")
    else:
        st.info("Aucune donnée pour ces filtres.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Classement des pays")
    top_m = cmap.sort_values("Médailles", ascending=False).head(15)
    if not top_m.empty:
        fig = px.bar(top_m, x="Médailles", y="NOC", orientation="h",
                     text="Médailles", color="Médailles",
                     color_continuous_scale="YlOrRd",
                     title=f"Top 15 pays — {yr_map} | {type_med}")
        fig.update_traces(textposition="inside", insidetextanchor="middle",
                          textfont=dict(color="white", size=11))
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(theme(fig, 460), use_container_width=True)
        insight(f"Les 3 premiers pays concentrent "
                f"{int(top_m.head(3)['Médailles'].sum())} médailles sur "
                f"{int(top_m['Médailles'].sum())} dans ce classement.")
    st.markdown("</div>", unsafe_allow_html=True)


# ONGLET 3 — PERFORMANCES & ATHLÈTES

if page == "Athlètes":

    # Chargement uniquement quand cette page est ouverte
    with st.spinner("Chargement des classements athlètes..."):
        ath_rankings = get_athlete_rankings(df)
        new_gen_df = get_new_gen(df)
        cotes_df = get_cotes(df)

    # ── Athlètes les plus médaillés ──
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Athlètes les plus médaillés")

    a1, a2, a3 = st.columns(3)
    with a1: sp_a = st.selectbox("Sport", ["Tous"] + SPORTS, key="sp_a")
    with a2: co_a = st.selectbox("Pays",  ["Tous"] + COUNTRIES, key="co_a")
    with a3: n_a  = st.slider("Nombre", 5, 30, 15)

    # Filtrage sur le précalcul -
    rnk = ath_rankings
    if sp_a != "Tous": rnk = rnk[rnk["Sport"] == sp_a]
    if co_a != "Tous": rnk = rnk[rnk["NOC"]   == co_a]
    rnk = rnk.sort_values("Score", ascending=False).head(n_a)

    if not rnk.empty:
        fig = px.bar(rnk, x="Score", y="Name", orientation="h",
                     color="Total", color_continuous_scale="Viridis",
                     text="Total",
                     hover_data=["NOC", "Sport", "Or", "Argent", "Bronze", "Participations"],
                     title="Top athlètes — Score pondéré (Or=3, Argent=2, Bronze=1)")
        fig.update_traces(textposition="inside", insidetextanchor="middle",
                          textfont=dict(color="white", size=11))
        fig.update_layout(yaxis=dict(autorange="reversed"))
        fig.update_xaxes(range=[0, rnk["Score"].max() * 1.08])
        st.plotly_chart(theme(fig, 520), use_container_width=True)
        top_ath = rnk.iloc[0]
        insight(f"{top_ath['Name']} ({top_ath['NOC']}, {top_ath['Sport']}) "
                f"est l'athlète le plus titré avec {int(top_ath['Total'])} médailles "
                f"({int(top_ath['Or'])} or, {int(top_ath['Argent'])} argent, {int(top_ath['Bronze'])} bronze) "
                f"sur {int(top_ath['Participations'])} participations olympiques.")

        st.markdown("#### Classement")
        rank_colors = ["#f59e0b", "#94a3b8", "#c2794a", "#64748b", "#64748b"]
        rank_labels = ["1er", "2e", "3e", "4e", "5e"]
        pod = rnk.head(5).reset_index(drop=True)
        for i, r in pod.iterrows():
            col_r, col_n, col_p, col_s, col_or, col_ag, col_br, col_tot = st.columns([0.5,3,1,1.5,0.7,0.7,0.7,0.8])
            with col_r:
                st.markdown(f'<div style="color:{rank_colors[i]};font-weight:800;font-size:1rem;padding-top:6px">{rank_labels[i]}</div>', unsafe_allow_html=True)
            with col_n: st.markdown(f'<div style="color:#f1f5f9;font-weight:600;padding-top:6px">{r["Name"]}</div>', unsafe_allow_html=True)
            with col_p: st.markdown(f'<div style="color:#64748b;padding-top:6px">{r["NOC"]}</div>', unsafe_allow_html=True)
            with col_s: st.markdown(f'<div style="color:#64748b;padding-top:6px">{r["Sport"]}</div>', unsafe_allow_html=True)
            with col_or: st.markdown(f'<div style="color:#f59e0b;text-align:center;padding-top:6px">{int(r["Or"])} Or</div>', unsafe_allow_html=True)
            with col_ag: st.markdown(f'<div style="color:#94a3b8;text-align:center;padding-top:6px">{int(r["Argent"])} Ag</div>', unsafe_allow_html=True)
            with col_br: st.markdown(f'<div style="color:#c2794a;text-align:center;padding-top:6px">{int(r["Bronze"])} Br</div>', unsafe_allow_html=True)
            with col_tot: st.markdown(f'<div style="color:#f1f5f9;font-weight:700;text-align:center;padding-top:6px">{int(r["Total"])}</div>', unsafe_allow_html=True)
            st.markdown('<hr>', unsafe_allow_html=True)
    else:
        st.info("Aucune donnée pour ces filtres.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Nouvelles générations montantes ──
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Nouvelles générations montantes")
    st.caption("Athlètes ayant débuté après 2008 et déjà médaillés")

    # Filtrage sur le précalcul
    df_ng = new_gen_df
    if sp_a != "Tous": df_ng = df_ng[df_ng["Sport"] == sp_a]
    if co_a != "Tous": df_ng = df_ng[df_ng["NOC"]   == co_a]

    ng = (df_ng.groupby(["Name", "NOC", "Sport", "First"])
          .agg(Total=("Has_Medal", "sum"), Score=("Medal_Score", "sum"))
          .reset_index()
          .sort_values("Score", ascending=False)
          .head(20))

    if not ng.empty:
        fig = px.bar(ng, x="Score", y="Name", orientation="h",
                     color="NOC", text="Score",
                     hover_data=["Sport", "Total", "First"],
                     title="Top 20 athlètes montants — Score de médailles (début ≥ 2008)")
        fig.update_traces(textposition="inside", insidetextanchor="middle",
                          textfont=dict(color="white", size=11))
        fig.update_layout(yaxis=dict(autorange="reversed"))
        fig.update_xaxes(range=[0, ng["Score"].max() * 1.1])
        st.plotly_chart(theme(fig, 520), use_container_width=True)
        top_ng = ng.iloc[0]
        insight(f"{top_ng['Name']} ({top_ng['NOC']}, {top_ng['Sport']}) est l'athlète montant "
                f"le plus performant avec un score de {int(top_ng['Score'])} "
                f"depuis ses débuts en {int(top_ng['First'])}. "
                "Ces profils sont à surveiller pour les JO 2028.")
    else:
        st.info("Aucun athlète montant pour ces filtres.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Cotes athlètes 2028 ──
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Cotes des athlètes par discipline — JO 2028")
    st.caption("Score de forme basé sur les performances récentes 2012–2016")

    sp_cote = st.selectbox("Discipline", ["Toutes"] + SPORTS, key="sp_cote")

    # Filtrage sur le précalcul
    ct = cotes_df
    if sp_cote != "Toutes": ct = ct[ct["Sport"] == sp_cote]
    if co_a    != "Tous":   ct = ct[ct["NOC"]   == co_a]

    ct["Cote_2028"] = (ct["Score"] * 0.6 + ct["Parts"] * 2 +
                       (ct["Last"] >= 2016).astype(int) * 5).round(1)
    ct = ct[ct["Cote_2028"] > 0].sort_values("Cote_2028", ascending=False).head(20)

    if not ct.empty:
        fig = px.bar(ct, x="Cote_2028", y="Name", orientation="h",
                     color="Cote_2028", color_continuous_scale="RdYlGn",
                     text="Cote_2028",
                     hover_data=["NOC", "Sport", "Med", "Last"],
                     title="Top 20 athlètes — Score de forme pour JO 2028")
        fig.update_traces(textposition="inside", insidetextanchor="middle",
                          textfont=dict(color="white", size=10))
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        fig.update_xaxes(range=[0, ct["Cote_2028"].max() * 1.08])
        st.plotly_chart(theme(fig, 520), use_container_width=True)
        top_ct = ct.iloc[0]
        insight(f"{top_ct['Name']} ({top_ct['NOC']}, {top_ct['Sport']}) "
                f"a le meilleur score de forme pour 2028 ({top_ct['Cote_2028']}). "
                "Ce score combine médailles récentes, régularité et activité récente jusqu'en 2016.")
    else:
        st.info("Aucune donnée pour cette discipline.")
    st.markdown("</div>", unsafe_allow_html=True)


# ONGLET 4 — TIMELINE DES RECORDS

if page == "Timeline":

    f1, f2 = st.columns(2)
    with f1: sp_tl = st.selectbox("Sport",  ["Tous"] + SPORTS, key="sp_tl")
    with f2: sa_tl = st.selectbox("Saison", ["Summer", "Winter"], key="sa_tl")

    df_tl = df
    if sp_tl != "Tous": df_tl = df_tl[df_tl["Sport"]  == sp_tl]
    df_tl = df_tl[df_tl["Season"] == sa_tl]

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Timeline des records")

    first_sport = (df_tl.groupby("Sport")["Year"].min().reset_index()
                   .rename(columns={"Year": "Année"}))
    tl_disc = (first_sport.groupby("Année").size().reset_index(name="Nouvelles")
               .sort_values("Année"))
    tl_disc["Total cumulé"] = tl_disc["Nouvelles"].cumsum()

    if not tl_disc.empty:
        col_tl1, col_tl2 = st.columns(2)

        with col_tl1:
            fig = px.area(tl_disc, x="Année", y="Total cumulé",
                          title="Disciplines au programme — Total cumulé",
                          color_discrete_sequence=["#3b82f6"])
            fig.update_traces(line=dict(width=2.5), fillcolor="rgba(59,130,246,0.15)")
            fig.add_vline(x=2016, line_dash="dot", line_color="#ef4444",
                          annotation_text=" 2016", annotation_position="top right")
            fig.update_layout(yaxis_title="Nb disciplines", xaxis_title="Année", showlegend=False)
            limit_year_axis(fig)
            st.plotly_chart(theme(fig, 360), use_container_width=True)

        with col_tl2:
            fig2 = px.bar(tl_disc, x="Année", y="Nouvelles",
                          title="Nouvelles disciplines par édition",
                          color="Nouvelles", color_continuous_scale="Blues")
            fig2.add_vline(x=2016, line_dash="dot", line_color="#ef4444",
                           annotation_text=" 2016", annotation_position="top right")
            fig2.update_layout(yaxis_title="Nouvelles disciplines", coloraxis_showscale=False)
            limit_year_axis(fig2)
            st.plotly_chart(theme(fig2, 360), use_container_width=True)

        new_this_year = first_sport[first_sport["Année"] == year_sel]["Sport"].tolist()
        total_disciplines = int(tl_disc[tl_disc["Année"] <= year_sel]["Nouvelles"].sum())
        if new_this_year:
            insight(f"En {year_sel}, {len(new_this_year)} nouvelle(s) discipline(s) : "
                    f"{', '.join(new_this_year)}. "
                    f"Total au programme : {total_disciplines} disciplines.")
        else:
            insight(f"En {year_sel}, aucune nouvelle discipline. "
                    f"{total_disciplines} disciplines au programme olympique à cette date. "
                    "Le programme s'est enrichi progressivement depuis 1896.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Évolution du nombre de médailles au fil des éditions")

    tl_total = (df_tl.groupby("Year")["Has_Medal"].sum().reset_index().sort_values("Year"))
    tl_total.columns = ["Année", "Médailles"]

    if not tl_total.empty:
        saison_lbl_tl = "d'été" if sa_tl == "Summer" else "d'hiver"
        fig = px.line(tl_total, x="Année", y="Médailles", markers=True,
                      title=f"Total médailles par édition — JO {saison_lbl_tl}",
                      color_discrete_sequence=["#3b82f6"])
        fig.update_traces(line=dict(width=3), marker=dict(size=7),
                          fill="tozeroy", fillcolor="rgba(59,130,246,0.1)")
        fig.add_vline(x=2016, line_dash="dot", line_color="#ef4444",
                      annotation_text=" 2016", annotation_position="top right")
        for yr, txt in [(1916,"WWI"),(1940,"WWII"),(1980,"Boycott"),(1984,"Boycott")]:
            if tl_total["Année"].between(yr-2, yr+2).any():
                fig.add_vline(x=yr, line_dash="dash",
                              line_color="rgba(255,100,100,0.25)",
                              annotation_text=txt, annotation_position="top left",
                              annotation_font_size=10)
        fig.update_layout(yaxis_title="Total médailles", xaxis_title="Année")
        limit_year_axis(fig)
        st.plotly_chart(theme(fig, 380), use_container_width=True)
        peak = tl_total.sort_values("Médailles", ascending=False).iloc[0]
        insight(f"Le pic est {int(peak['Médailles'])} médailles en {int(peak['Année'])}. "
                "Les chutes correspondent aux guerres mondiales (1916, 1940) et aux boycotts (1980, 1984).")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Performances des pays — JO d'été vs JO d'hiver")

    n_ev = st.slider("Nombre de pays", 5, 15, 10, key="n_ev")

    col_ev1, col_ev2 = st.columns(2)
    for col_ev, season_ev, scale_ev, label_ev in [
        (col_ev1, "Summer", "Blues",   "JO d'été"),
        (col_ev2, "Winter", "Purples", "JO d'hiver")
    ]:
        with col_ev:
            df_s = df
            if sp_tl != "Tous": df_s = df_s[df_s["Sport"] == sp_tl]
            top_s = (df_s[df_s["Season"] == season_ev]
                     .groupby("NOC")["Has_Medal"].sum()
                     .sort_values(ascending=False).head(n_ev).reset_index())
            top_s.columns = ["Pays", "Médailles"]
            fig = px.bar(top_s, x="Médailles", y="Pays", orientation="h",
                         color="Médailles", color_continuous_scale=scale_ev,
                         text="Médailles", title=f"Top {n_ev} pays — {label_ev}")
            fig.update_traces(textposition="inside", insidetextanchor="middle",
                              textfont=dict(color="white", size=11))
            fig.update_layout(yaxis=dict(autorange="reversed"),
                              coloraxis_showscale=False, showlegend=False)
            fig.update_xaxes(range=[0, top_s["Médailles"].max() * 1.08])
            st.plotly_chart(theme(fig, 420), use_container_width=True)

    top_ete = df[df["Season"]=="Summer"].groupby("NOC")["Has_Medal"].sum().idxmax()
    top_hiv = df[df["Season"]=="Winter"].groupby("NOC")["Has_Medal"].sum().idxmax()
    insight(f"Les JO d'été sont dominés par {top_ete}, les JO d'hiver par {top_hiv}. "
            "Le classement change selon la saison — certains pays excellent uniquement en hiver (Norvège, Autriche) "
            "ou uniquement en été (Kenya, Cuba, Jamaïque).")
    st.markdown("</div>", unsafe_allow_html=True)


# ONGLET 5 — PRÉDICTION 2028 & AVIS
if page == "K-Means":

    with st.spinner("Chargement rapide du clustering K-Means..."):
        df_cluster, inertias, df_cnoc, CLUSTER_FEATURES = get_kmeans_data(df)

    colors_cluster = {"Débutants":"#3b82f6","Intermédiaires":"#f97316","Performants":"#22c55e"}

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Segmentation des athlètes — K-Means Clustering")
    st.caption("Regroupement non supervisé selon profil physique et expérience olympique")
    col_i1,col_i2,col_i3 = st.columns(3)
    with col_i1: kpi("Athlètes analysés", f"{len(df_cluster):,}".replace(","," "), "#3b82f6")
    with col_i2: kpi("Variables utilisées", "5", "#10b981")
    with col_i3: kpi("Clusters retenus", "K = 3", "#f59e0b")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Méthode du coude — Choix du K optimal")
    fig_coude = go.Figure()
    fig_coude.add_trace(go.Scatter(x=list(range(2,11)), y=inertias, mode="lines+markers",
                                   line=dict(color="#3b82f6",width=2.5),
                                   marker=dict(size=8,color="#3b82f6"), name="Inertie"))
    fig_coude.add_vline(x=3, line_dash="dash", line_color="#ef4444",
                        annotation_text="K optimal = 3", annotation_position="top right",
                        annotation_font_color="#ef4444")
    fig_coude.update_layout(xaxis_title="K", yaxis_title="Inertie", title="Méthode du coude")
    st.plotly_chart(theme(fig_coude, 380), use_container_width=True)
    insight("Le coude est visible à K=3 : au-delà, le gain d'inertie devient marginal.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Profils moyens par cluster")
    profile = df_cluster.groupby("Profil")[CLUSTER_FEATURES].mean().round(2).reset_index()
    counts  = df_cluster.groupby("Profil").size().reset_index(name="Nb_Athlètes")
    profile = profile.merge(counts, on="Profil")

    cards_html = '<div style="display:flex;gap:16px;margin-bottom:16px;">'
    for profil in ["Débutants","Intermédiaires","Performants"]:
        row = profile[profile["Profil"]==profil]
        if not row.empty:
            r = row.iloc[0]; color = colors_cluster[profil]
            cards_html += (
                f'<div class="kpi" style="flex:1;border-left:3px solid {color};">'
                f'<div class="label">{profil}</div>'
                f'<div style="color:#94a3b8;font-size:0.85rem;margin-top:6px">'
                f'Age moyen : <b style="color:#f1f5f9">{r["Age"]:.0f} ans</b><br>'
                f'Participations : <b style="color:#f1f5f9">{r["Participation_Count"]:.1f}</b><br>'
                f'Médailles : <b style="color:#f1f5f9">{r["Total_Medals_Athlete"]:.1f}</b><br>'
                f'Nb athlètes : <b style="color:#f1f5f9">{int(r["Nb_Athlètes"]):,}</b>'
                f'</div></div>')
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    fig_bar = go.Figure()
    labels_bar = ["Age moyen","Participations moy.","Médailles moy."]
    for profil, color in colors_cluster.items():
        row = profile[profile["Profil"]==profil]
        if not row.empty:
            r = row.iloc[0]
            fig_bar.add_trace(go.Bar(
                name=profil, x=labels_bar,
                y=[r["Age"],r["Participation_Count"],r["Total_Medals_Athlete"]],
                marker_color=color,
                text=[f"{r['Age']:.0f}",f"{r['Participation_Count']:.1f}",f"{r['Total_Medals_Athlete']:.1f}"],
                textposition="outside", textfont=dict(color="white",size=11)))
    fig_bar.update_layout(barmode="group", title="Comparaison des profils K-Means",
                          legend=dict(orientation="h",y=1.1))
    st.plotly_chart(theme(fig_bar, 400), use_container_width=True)
    insight("Les Performants se distinguent par des participations et médailles nettement supérieures.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Visualisation — Age vs Médailles totales")
    sample = df_cluster.sample(min(5000, len(df_cluster)), random_state=42)
    fig_sc = px.scatter(sample, x="Age", y="Total_Medals_Athlete",
                        color="Profil", color_discrete_map=colors_cluster,
                        opacity=0.5, title="K-Means — Age vs Total Médailles",
                        labels={"Total_Medals_Athlete":"Total Médailles","Age":"Âge"})
    fig_sc.update_traces(marker=dict(size=4))
    st.plotly_chart(theme(fig_sc, 450), use_container_width=True)
    insight("Les Performants (vert) concentrent les athlètes avec le plus de médailles.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Répartition des clusters par pays")
    top_noc = st.selectbox("Pays (NOC)", ["Tous"]+COUNTRIES, key="noc_cluster")
    df_noc_view = df_cnoc.copy()
    if top_noc != "Tous":
        df_noc_view = df_noc_view[df_noc_view["NOC"]==top_noc]
    dist = df_noc_view.groupby("Profil").size().reset_index(name="Athlètes")
    fig_pie = px.pie(dist, names="Profil", values="Athlètes",
                     color="Profil", color_discrete_map=colors_cluster,
                     hole=0.4, title=f"Distribution — {top_noc}")
    st.plotly_chart(theme(fig_pie, 380), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)



if page == "Prédiction 2028":

    # Chargement uniquement quand cette page est ouverte
    with st.spinner("Chargement des données de momentum 2028..."):
        cym_data, r4_data, last_yr_data, w_data = get_momentum(df)

    with st.spinner("Chargement du modèle de prédiction..."):
        rf, lr, scaler, X_test, X_test_s, y_test, y_pred_rf, y_pred_lr, FEATURES = train_models(df)

    # ── Métriques modèles ──
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Modèle de prédiction — Probabilité de médaille")
    st.write("Random Forest entraîné sur 6 variables identiques au notebook : "
             "**Age, Taille, Poids, Participations, Score médailles athlète, Médailles historiques pays**. "
             "Le déséquilibre des classes (85% non-médaillés) est corrigé pour éviter un biais vers la classe majoritaire.")

    m1, m2, m3, m4 = st.columns(4)
    with m1: kpi("Accuracy RF",  f"{accuracy_score(y_test, y_pred_rf):.2%}",  "#3b82f6")
    with m2: kpi("F1 Score RF",  f"{f1_score(y_test, y_pred_rf):.2%}",        "#10b981")
    with m3: kpi("Accuracy LR",  f"{accuracy_score(y_test, y_pred_lr):.2%}",  "#f59e0b")
    with m4: kpi("F1 Score LR",  f"{f1_score(y_test, y_pred_lr):.2%}",        "#ef4444")
    st.markdown("<br>", unsafe_allow_html=True)

    col_roc, col_imp = st.columns(2)

    with col_roc:
        st.subheader("Courbe ROC")
        y_proba_rf = rf.predict_proba(X_test)[:, 1]
        y_proba_lr = lr.predict_proba(X_test_s)[:, 1]
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
        fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
        auc_rf = auc(fpr_rf, tpr_rf)
        auc_lr = auc(fpr_lr, tpr_lr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, name=f"Random Forest (AUC={auc_rf:.3f})",
                                 line=dict(width=2.5, color="#3b82f6")))
        fig.add_trace(go.Scatter(x=fpr_lr, y=tpr_lr, name=f"Logistic Regression (AUC={auc_lr:.3f})",
                                 line=dict(width=2, dash="dash", color="#f59e0b")))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(dash="dot", color="gray"),
                                 showlegend=False))
        fig.update_layout(xaxis_title="Taux faux positifs", yaxis_title="Taux vrais positifs")
        st.plotly_chart(theme(fig, 360), use_container_width=True)
        insight(f"AUC Random Forest ({auc_rf:.3f}) > Régression Logistique ({auc_lr:.3f}). "
                "Le RF capte mieux les relations non-linéaires entre les variables.")

    with col_imp:
        st.subheader("Importance des variables")
        imp = pd.DataFrame({"Variable": FEATURES, "Importance": rf.feature_importances_}
                           ).sort_values("Importance", ascending=False)
        fig = px.bar(imp, x="Importance", y="Variable", orientation="h",
                     text="Importance", color="Importance",
                     color_continuous_scale="Blues",
                     title="Importance — Random Forest")
        fig.update_traces(texttemplate="%{text:.3f}", textposition="inside",
                          insidetextanchor="middle", textfont=dict(color="white", size=11))
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        fig.update_xaxes(range=[0, imp["Importance"].max() * 1.08])
        st.plotly_chart(theme(fig, 360), use_container_width=True)
        insight(f"La variable la plus importante est **{imp.iloc[0]['Variable']}**. "
                "Les variables d'historique dominent les variables physiques.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Simulateur profil athlète ──
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Simuler un profil athlète")

    col_sim, col_res = st.columns(2)
    with col_sim:
        age_i = st.slider("Age",                         10,   60,   25)
        hgt_i = st.slider("Taille (cm)",                130,  230,  175)
        wgt_i = st.slider("Poids (kg)",                  35,  160,   70)
        prt_i = st.slider("Participations olympiques",    1,   10,    2)
        mda_i = st.slider("Score médailles athlète",      0,   30,    0)
        mdc_i = st.slider("Médailles historiques pays",   0, 5000,  500)

    with col_res:
        inp = pd.DataFrame({
            "Age": [age_i], "Height": [hgt_i], "Weight": [wgt_i],
            "Participation_Count": [prt_i],
            "Total_Medals_Athlete": [mda_i],
            "Country_Medals": [mdc_i]
        })
        prob = rf.predict_proba(inp)[0][1]
        pred = rf.predict(inp)[0]
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.progress(float(prob))
        if pred == 1:
            st.success(f" Probabilité de médaille : **{prob:.2%}**")
        else:
            st.error(f" Probabilité de médaille : **{prob:.2%}**")
        insight("Probabilité > 50% : profil similaire aux athlètes médaillés. "
                "L'expérience olympique et l'historique du pays sont les variables "
                "les plus déterminantes.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Prédictions JO 2028 —
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Prédictions JO 2028 — Cotes pays par discipline")
    st.caption("Score de momentum pondéré sur les 4 dernières éditions d'été disponibles jusqu'en 2016 — méthode identique au notebook section 8")

    sp_pred = st.selectbox("Discipline", ["Toutes"] + SPORTS, key="sp_pred")
    nb_pred = st.slider("Nombre de pays", 5, 20, 10, key="nb_pred")

    # Filtrage sur le précalcul — 
    df_cym = cym_data
    if sp_pred != "Toutes":
        df_cym = df_cym[df_cym["Sport"] == sp_pred]

    def mom(grp):
        return sum(w_data[yr] * (grp[grp["Year"] == yr]["Has_Medal"].values[0]
                   if len(grp[grp["Year"] == yr]) > 0 else 0) for yr in w_data)

    pred_pays = df_cym.groupby("NOC").apply(mom).reset_index()
    pred_pays.columns = ["NOC", "Score_2028"]

    last = (df_cym[df_cym["Year"] == last_yr_data]
            .groupby("NOC")["Has_Medal"]
            .sum()
            .reset_index())
    last.columns = ["NOC", f"Med_{last_yr_data}"]

    comp = pred_pays.merge(last, on="NOC", how="left").fillna(0)
    comp = comp.sort_values("Score_2028", ascending=False).head(nb_pred)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            comp,
            x="Score_2028",
            y="NOC",
            orientation="h",
            color="Score_2028",
            color_continuous_scale="Oranges",
            text=comp["Score_2028"].round(0).astype(int),
            title="Score de momentum prédit — JO 2028"
        )
        fig.update_traces(
            textposition="outside",
            textfont=dict(color="white", size=12)
        )
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            xaxis_title="Score 2028",
            yaxis_title="Pays",
            uniformtext_minsize=10,
            uniformtext_mode="show"
        )
        fig.update_xaxes(range=[0, comp["Score_2028"].max() * 1.18])
        st.plotly_chart(theme(fig, 420), use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name=f"Médailles {last_yr_data}",
            y=comp["NOC"],
            x=comp[f"Med_{last_yr_data}"],
            orientation="h",
            marker_color="rgba(96,165,250,0.75)",
            text=comp[f"Med_{last_yr_data}"].astype(int),
            textposition="outside",
            textfont=dict(color="white", size=11)
        ))
        fig.add_trace(go.Bar(
            name="Projection 2028",
            y=comp["NOC"],
            x=comp["Score_2028"],
            orientation="h",
            marker_color="#f59e0b",
            text=comp["Score_2028"].round(0).astype(int),
            textposition="outside",
            textfont=dict(color="white", size=11)
        ))
        fig.update_layout(
            barmode="group",
            yaxis=dict(autorange="reversed"),
            xaxis_title="Médailles / Score",
            yaxis_title="Pays",
            legend=dict(orientation="h", y=1.10, x=0),
            title=f"JO {last_yr_data} vs Projection 2028",
            uniformtext_minsize=9,
            uniformtext_mode="show"
        )
        max_x = max(comp[f"Med_{last_yr_data}"].max(), comp["Score_2028"].max())
        fig.update_xaxes(range=[0, max_x * 1.18])
        st.plotly_chart(theme(fig, 420), use_container_width=True)

    winner = comp.iloc[0]
    insight(f"{winner['NOC']} est le favori pour les JO 2028 "
            f"avec un score de momentum de {winner['Score_2028']:.0f}. "
            f"La comparaison utilise les médailles réelles de {last_yr_data} et une projection basée sur les 4 dernières olympiades disponibles.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Avis utilisateurs ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Avis utilisateurs")
    st.caption("Partagez votre avis sur l'application et les prédictions JO 2028")

    if "reviews" not in st.session_state:
        st.session_state["reviews"] = [
            {"author": "Marie D.", "stars": 5,
             "text": "Dashboard très intuitif, la carte des médailles est impressionnante.", "sport": "Natation"},
            {"author": "Lucas M.", "stars": 4,
             "text": "Les prédictions JO 2028 sont vraiment intéressantes.", "sport": "Athlétisme"},
            {"author": "Sofia R.", "stars": 5,
             "text": "La section nouvelles générations est une très bonne idée.", "sport": "Gymnastique"},
        ]

    with st.expander("Laisser un avis"):
        rv1, rv2, rv3 = st.columns([2, 1, 2])
        with rv1: rev_auth = st.text_input("Prénom / pseudo", placeholder="Jean P.")
        with rv2: rev_note = st.selectbox("Note", [5, 4, 3, 2, 1],
                                           format_func=lambda x: "★" * x + "☆" * (5 - x))
        with rv3: rev_sp = st.selectbox("Sport favori", ["Général"] + SPORTS, key="rev_sp")
        rev_txt = st.text_area("Commentaire", placeholder="Votre avis...", height=80)
        if st.button("Publier", type="primary"):
            if rev_auth.strip() and rev_txt.strip():
                st.session_state["reviews"].insert(0, {
                    "author": rev_auth.strip(), "stars": rev_note,
                    "text": rev_txt.strip(), "sport": rev_sp
                })
                st.success("Avis publié. Merci !")
            else:
                st.warning("Veuillez renseigner votre prénom et un commentaire.")

    avg = round(sum(r["stars"] for r in st.session_state["reviews"])
                / len(st.session_state["reviews"]))
    st.markdown(f"**{len(st.session_state['reviews'])} avis** — Note moyenne : {'★'*avg}{'☆'*(5-avg)}")
    st.markdown("<br>", unsafe_allow_html=True)

    sp_rev  = list(set(r["sport"] for r in st.session_state["reviews"]))
    flt_rev = st.selectbox("Filtrer par sport", ["Tous"] + sp_rev, key="flt_rev")
    shown   = [r for r in st.session_state["reviews"]
               if flt_rev == "Tous" or r["sport"] == flt_rev]

    for r in shown:
        s = "★" * r["stars"] + "☆" * (5 - r["stars"])
        st.markdown(
            f'<div class="review">'
            f'<div class="author">{r["author"]}'
            f'<span style="color:#475569;font-weight:400;font-size:0.82rem"> — {r["sport"]}</span></div>'
            f'<div class="stars">{s}</div>'
            f'<div class="body">{r["text"]}</div>'
            f'</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)





# FOOTER
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Jeux Olympiques 2028")