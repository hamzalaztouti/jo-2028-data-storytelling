import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



# CONFIGURATION PAGE
st.set_page_config(
    page_title="JO 2028 Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)



# CSS
st.markdown("""
<style>
/* Global */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top left, rgba(59,130,246,0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(16,185,129,0.12), transparent 22%),
        linear-gradient(135deg, #020617 0%, #0f172a 45%, #111827 100%);
    color: #f8fafc;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #0b1220 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
    min-width: 320px;
    max-width: 320px;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {
    color: #e5e7eb !important;
}

/* Main container */
.block-container {
    max-width: 1500px;
    padding-top: 1.6rem;
    padding-bottom: 2rem;
}

/* Typography */
html, body, [class*="css"] {
    color: #e5e7eb;
    font-size: 16px;
}

h1 {
    color: #ffffff;
    font-size: 2.3rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
    margin-bottom: 0.1rem;
}

h2, h3 {
    color: #ffffff !important;
    font-weight: 700 !important;
}

small, .caption {
    color: #94a3b8 !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    background: rgba(255,255,255,0.03);
    border-radius: 14px 14px 0 0 !important;
    padding: 14px 26px !important;
    font-size: 17px !important;
    font-weight: 600 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(180deg, rgba(59,130,246,0.18), rgba(59,130,246,0.05));
    color: #ffffff !important;
}

/* Inputs */
div[data-baseweb="select"] > div,
div[data-baseweb="slider"] > div {
    border-radius: 14px !important;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {
    font-weight: 600;
    color: #e5e7eb !important;
}

/* Info/success/error */
div[data-testid="stAlert"] {
    border-radius: 16px;
}

/* Custom cards */
.card {
    background: linear-gradient(145deg, rgba(15,23,42,0.95), rgba(30,41,59,0.88));
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px;
    padding: 18px 20px;
    box-shadow: 0 18px 40px rgba(0,0,0,0.35);
    backdrop-filter: blur(10px);
}

.card h4 {
    margin: 0 0 6px 0;
    color: #cbd5e1;
    font-size: 0.96rem;
    font-weight: 600;
}

.card h2 {
    margin: 0;
    font-size: 2rem;
    font-weight: 800;
    color: #ffffff;
}

.section-card {
    background: linear-gradient(145deg, rgba(2,6,23,0.92), rgba(15,23,42,0.85));
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 22px;
    padding: 16px 18px 10px 18px;
    box-shadow: 0 14px 35px rgba(0,0,0,0.34);
    margin-bottom: 16px;
}

.hero {
    background:
        radial-gradient(circle at top left, rgba(59,130,246,0.24), transparent 30%),
        linear-gradient(135deg, rgba(2,6,23,0.95), rgba(15,23,42,0.92));
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 24px;
    padding: 24px 26px;
    box-shadow: 0 20px 45px rgba(0,0,0,0.38);
    margin-bottom: 18px;
}

.hero p {
    color: #cbd5e1;
    margin-top: 6px;
    margin-bottom: 0;
}

.badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(59,130,246,0.14);
    color: #bfdbfe;
    font-size: 0.85rem;
    font-weight: 700;
    margin-bottom: 8px;
}

hr {
    border: none;
    height: 1px;
    background: rgba(255,255,255,0.07);
    margin: 14px 0 18px 0;
}
</style>
""", unsafe_allow_html=True)



# CHARGEMENT DES DONNÉES
@st.cache_data
def load_data():
    df = pd.read_csv("data/athlete_events.csv")
    return df


df = load_data().copy()



# PRÉPARATION DES DONNÉES
df["Has_Medal"] = df["Medal"].notna().astype(int)

for col in ["Age", "Height", "Weight", "Year"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")



# SIDEBAR
st.sidebar.markdown("## Filtres")
st.sidebar.caption("Affinez l’analyse selon les dimensions olympiques.")

available_years = sorted(df["Year"].dropna().astype(int).unique().tolist())
min_year = int(min(available_years))
max_year = int(max(available_years))

selected_year = st.sidebar.slider(
    "Année",
    min_value=min_year,
    max_value=max_year,
    value=max_year,
    step=4
)

sports = sorted(df["Sport"].dropna().unique().tolist())
selected_sport = st.sidebar.selectbox("Sport", ["Tous"] + sports)

selected_sex = st.sidebar.selectbox("Sexe", ["Tous", "M", "F"])

seasons = sorted(df["Season"].dropna().unique().tolist())
selected_season = st.sidebar.selectbox("Saison", ["Toutes"] + seasons)

countries = sorted(df["NOC"].dropna().unique().tolist())
selected_country = st.sidebar.selectbox("Pays (NOC)", ["Tous"] + countries)



# FILTRAGE
df_filtered = df[df["Year"] == selected_year].copy()

if selected_sport != "Tous":
    df_filtered = df_filtered[df_filtered["Sport"] == selected_sport]

if selected_sex != "Tous":
    df_filtered = df_filtered[df_filtered["Sex"] == selected_sex]

if selected_season != "Toutes":
    df_filtered = df_filtered[df_filtered["Season"] == selected_season]

if selected_country != "Tous":
    df_filtered = df_filtered[df_filtered["NOC"] == selected_country]



# FONCTIONS UTILITAIRES
def kpi_card(title: str, value: str, accent: str):
    st.markdown(
        f"""
        <div class="card" style="border-left: 5px solid {accent};">
            <h4>{title}</h4>
            <h2>{value}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )


def apply_plot_style(fig, height=450):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        margin=dict(l=30, r=20, t=60, b=30),
        title_x=0.02
    )
    return fig



# HEADER HERO
st.markdown(
    f"""
    <div class="hero">
        <div class="badge">Dashboard Analytics</div>
        <h1>JO {selected_year} - Dashboard</h1>
        <p>
            Analyse avancée des performances olympiques, visualisation mondiale
            et estimation intelligente de la probabilité de médaille.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<hr>", unsafe_allow_html=True)



# KPI PRINCIPAUX
col1, col2, col3, col4 = st.columns(4)

with col1:
    kpi_card("Athlètes", f"{len(df_filtered):,}".replace(",", " "), "#3b82f6")

with col2:
    kpi_card("Pays", str(df_filtered["NOC"].nunique()), "#10b981")

with col3:
    kpi_card("Sports", str(df_filtered["Sport"].nunique()), "#f59e0b")

with col4:
    kpi_card("Médailles", str(int(df_filtered["Has_Medal"].sum())), "#ef4444")



# NAVIGATION
tab1, tab2, tab3 = st.tabs(["Analyse", "Carte", "Prédiction"])



# ONGLET 1 : ANALYSE
with tab1:
    left, right = st.columns((1.2, 1))

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Top 10 pays")
        top_countries = (
            df_filtered["NOC"]
            .value_counts()
            .head(10)
            .reset_index()
        )
        top_countries.columns = ["NOC", "Nombre"]

        fig_top = px.bar(
            top_countries,
            x="Nombre",
            y="NOC",
            orientation="h",
            text="Nombre",
            color="Nombre",
            color_continuous_scale="Turbo",
            title="Pays les plus représentés"
        )
        fig_top.update_traces(textposition="outside")
        fig_top = apply_plot_style(fig_top, height=470)
        fig_top.update_layout(
            xaxis_title="Nombre d'athlètes",
            yaxis_title="Pays"
        )
        st.plotly_chart(fig_top, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Répartition par sexe")
        sex_count = (
            df_filtered["Sex"]
            .value_counts()
            .reset_index()
        )
        sex_count.columns = ["Sexe", "Nombre"]

        if not sex_count.empty:
            fig_sex = px.pie(
                sex_count,
                names="Sexe",
                values="Nombre",
                hole=0.55,
                title="Hommes / Femmes",
                color_discrete_sequence=["#38bdf8", "#10b981", "#f59e0b"]
            )
            fig_sex = apply_plot_style(fig_sex, height=470)
            st.plotly_chart(fig_sex, use_container_width=True)
        else:
            st.info("Aucune donnée disponible pour ce filtre.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Évolution des médailles par année")
    medals_year = (
        df.groupby("Year")["Has_Medal"]
        .sum()
        .reset_index()
        .sort_values("Year")
    )

    fig_line = px.line(
        medals_year,
        x="Year",
        y="Has_Medal",
        markers=True,
        title="Nombre total de médailles observées"
    )
    fig_line.update_traces(line=dict(width=4), marker=dict(size=8))
    fig_line = apply_plot_style(fig_line, height=430)
    fig_line.update_layout(
        xaxis_title="Année",
        yaxis_title="Nombre de médailles"
    )
    st.plotly_chart(fig_line, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Corrélations des variables physiques")
    corr_df = df_filtered[["Age", "Height", "Weight"]].dropna()

    if not corr_df.empty and len(corr_df) > 1:
        corr = corr_df.corr(numeric_only=True)
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Heatmap des corrélations"
        )
        fig_corr = apply_plot_style(fig_corr, height=420)
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Pas assez de données pour calculer la heatmap avec ces filtres.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Lecture analytique")
    st.write("""
    - Les filtres permettent d’isoler une année, un sport, un sexe, une saison ou un pays.
    - Les pays les plus représentés varient selon l’édition et la discipline.
    - L’évolution historique des médailles met en évidence des tendances fortes.
    - Les variables physiques apportent des signaux utiles mais ne suffisent pas seules à expliquer la performance.
    """)
    st.markdown("</div>", unsafe_allow_html=True)



# ONGLET 2 : CARTE
with tab2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Carte mondiale des médailles")

    country_medals = (
        df_filtered.groupby("NOC")["Has_Medal"]
        .sum()
        .reset_index()
    )
    country_medals.columns = ["NOC", "Medals"]

    if not country_medals.empty:
        fig_map = px.choropleth(
            country_medals,
            locations="NOC",
            locationmode="ISO-3",
            color="Medals",
            color_continuous_scale="Turbo",
            title=f"Médailles par pays - {selected_year}"
        )
        fig_map = apply_plot_style(fig_map, height=640)
        fig_map.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type="natural earth",
                bgcolor="rgba(0,0,0,0)"
            )
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("Aucune donnée carte disponible pour ce filtre.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Top pays par médailles")
    top_medals = (
        country_medals.sort_values("Medals", ascending=False)
        .head(15)
    )

    if not top_medals.empty:
        fig_medals = px.bar(
            top_medals,
            x="Medals",
            y="NOC",
            orientation="h",
            text="Medals",
            color="Medals",
            color_continuous_scale="Turbo",
            title="Classement des pays par médailles"
        )
        fig_medals.update_traces(textposition="outside")
        fig_medals = apply_plot_style(fig_medals, height=500)
        fig_medals.update_layout(
            xaxis_title="Nombre de médailles",
            yaxis_title="Pays"
        )
        st.plotly_chart(fig_medals, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ONGLET 3 : PRÉDICTION 
with tab3:
    # Entraînement rapide du modèle (inchangé)
    df_model = df[["Age", "Height", "Weight", "Has_Medal"]].dropna().copy()
    X = df_model[["Age", "Height", "Weight"]]
    y = df_model["Has_Medal"]
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)

    st.markdown("### Modèle de Probabilité de Succès")
    
    # Ligne d'inputs
    col_in1, col_in2, col_in3 = st.columns(3)
    age_in = col_in1.number_input("Âge", value=25)
    height_in = col_in2.number_input("Taille (cm)", value=175)
    weight_in = col_in3.number_input("Poids (kg)", value=70)

    # Calcul
    input_data = pd.DataFrame([[age_in, height_in, weight_in]], columns=["Age", "Height", "Weight"])
    prob = model.predict_proba(input_data)[0][1] * 100

    st.markdown("<br>", unsafe_allow_html=True)

    # Affichage des Graphiques
    res_left, res_right = st.columns(2)

    with res_left:
        st.markdown('<p style="text-align:center; font-size:1.5rem; font-weight:600; margin-bottom:-50px;">Potentiel %</p>', unsafe_allow_html=True)
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob,
            number = {'font': {'size': 80, 'color': 'white'}, 'suffix': ""},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                'bar': {'color': "#3b82f6", 'thickness': 0.6},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 100], 'color': 'rgba(255,255,255,0.05)'}
                ],
            }
        ))
        
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(l=50, r=50, t=0, b=0)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with res_right:
        st.markdown('<p style="text-align:center; font-size:1.5rem; font-weight:600; margin-bottom:20px;">Importance</p>', unsafe_allow_html=True)
        
        importances = model.feature_importances_
        # On réordonne pour correspondre aux couleurs de ton screen (Bleu, Orange, Vert)
        labels = ["Age", "Poids", "Taille"]
        values = [importances[0], importances[2], importances[1]] 
        
        fig_donut = px.pie(
            names=labels,
            values=values,
            hole=0.7,
            color_discrete_sequence=['#636EFA', '#EF553B', '#00CC96'] # Couleurs exactes de ton screen
        )
        
        fig_donut.update_traces(textinfo='percent', textfont_size=14, hovertemplate="%{label}: %{percent}")
        fig_donut.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            height=350,
            margin=dict(t=0, b=0, l=0, r=0),
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

# FOOTER
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Projet JO 2028 - Analyse, visualisation interactive et prédiction")