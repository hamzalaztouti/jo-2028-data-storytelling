import streamlit as st
import pandas as pd
import plotly.express as px
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
/* Fond global */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #020617 0%, #071225 40%, #0b1d44 100%);
    color: #f8fafc;
}

/* Header Streamlit */
[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
}

/* Sidebar plus large */
[data-testid="stSidebar"] {
    background: #020817;
    min-width: 320px;
    max-width: 320px;
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Titre sidebar */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #e2e8f0;
}

/* Zone principale */
.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Titres */
h1, h2, h3 {
    color: #ffffff;
}

/* Texte */
html, body, [class*="css"]  {
    color: #e2e8f0;
    font-size: 16px;
}

/* Cartes KPI */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.20);
}

/* Tabs plus grandes */
button[data-baseweb="tab"] {
    font-size: 18px;
    padding: 14px 28px;
    border-radius: 12px 12px 0 0;
}

/* Inputs plus grands */
div[data-baseweb="select"] > div {
    min-height: 50px;
    font-size: 16px;
    border-radius: 12px;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {
    font-size: 17px;
    font-weight: 600;
    color: #e2e8f0;
}

/* Conteneurs */
.custom-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 18px;
}

/* Ligne séparatrice */
hr {
    border: none;
    height: 1px;
    background: rgba(255,255,255,0.08);
}
</style>
""", unsafe_allow_html=True)

# CHARGEMENT DES DONNÉES

@st.cache_data
def load_data():
    df = pd.read_csv("data/athlete_events.csv")
    return df

df = load_data()


# PRÉPARATION DONNÉES

df = df.copy()
df["Has_Medal"] = df["Medal"].notna().astype(int)

# Conversion robuste
for col in ["Age", "Height", "Weight", "Year"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# SIDEBAR
#
st.sidebar.markdown("## Filtres")

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


# HEADER

st.title(f"JO {selected_year} - Dashboard")
st.write("Application de data storytelling pour l’analyse des performances olympiques, la visualisation mondiale et l’estimation de probabilité de médaille.")

st.markdown("<hr>", unsafe_allow_html=True)


# KPI PRINCIPAUX

col1, col2, col3, col4 = st.columns(4)

col1.metric("Athlètes", f"{len(df_filtered):,}".replace(",", " "))
col2.metric("Pays", df_filtered["NOC"].nunique())
col3.metric("Sports", df_filtered["Sport"].nunique())
col4.metric("Médailles", int(df_filtered["Has_Medal"].sum()))


# NAVIGATION

tab1, tab2, tab3 = st.tabs(["Analyse", "Carte", "Prédiction"])


# ONGLET 1 : ANALYSE

with tab1:
    left, right = st.columns((1.2, 1))

    with left:
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
            color_continuous_scale="Blues",
            title="Pays les plus représentés"
        )
        fig_top.update_layout(
            template="plotly_dark",
            height=500,
            yaxis_title="Pays",
            xaxis_title="Nombre d'athlètes",
            title_x=0.02
        )
        fig_top.update_traces(textposition="outside")
        st.plotly_chart(fig_top, use_container_width=True)

    with right:
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
                hole=0.45,
                title="Hommes / Femmes"
            )
            fig_sex.update_layout(
                template="plotly_dark",
                height=500,
                title_x=0.02
            )
            st.plotly_chart(fig_sex, use_container_width=True)
        else:
            st.info("Aucune donnée disponible pour ce filtre.")

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
    fig_line.update_layout(
        template="plotly_dark",
        height=460,
        xaxis_title="Année",
        yaxis_title="Nombre de médailles",
        title_x=0.02
    )
    st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("Corrélations des variables physiques")
    corr_df = df_filtered[["Age", "Height", "Weight"]].dropna()

    if not corr_df.empty and len(corr_df) > 1:
        corr = corr_df.corr(numeric_only=True)
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            title="Heatmap des corrélations"
        )
        fig_corr.update_layout(
            template="plotly_dark",
            height=450,
            title_x=0.02
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Pas assez de données pour calculer la heatmap avec ces filtres.")

    st.subheader("Lecture analytique")
    st.write("""
    - Les filtres permettent d’isoler une année, un sport, un sexe, une saison ou un pays.
    - Les pays les plus représentés varient selon l’édition et la discipline.
    - L’évolution historique des médailles permet de visualiser les tendances globales.
    - Les variables physiques ne suffisent pas seules à expliquer totalement la performance.
    """)


# ONGLET 2 : CARTE

with tab2:
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
            color_continuous_scale="Blues",
            title=f"Médailles par pays - {selected_year}"
        )
        fig_map.update_layout(
            template="plotly_dark",
            height=650,
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type="natural earth",
                bgcolor="rgba(0,0,0,0)"
            ),
            title_x=0.02
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("Aucune donnée carte disponible pour ce filtre.")

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
            color_continuous_scale="Blues",
            title="Classement des pays par médailles"
        )
        fig_medals.update_layout(
            template="plotly_dark",
            height=550,
            yaxis_title="Pays",
            xaxis_title="Nombre de médailles",
            title_x=0.02
        )
        fig_medals.update_traces(textposition="outside")
        st.plotly_chart(fig_medals, use_container_width=True)


# ONGLET 3 : PRÉDICTION

with tab3:
    st.subheader("Prédiction de probabilité de médaille")

    st.write("""
    Le modèle ci-dessous estime la probabilité qu’un athlète obtienne une médaille
    à partir de trois variables physiques : âge, taille et poids.
    """)

    df_model = df[["Age", "Height", "Weight", "Has_Medal"]].dropna().copy()

    if len(df_model) > 100:
        X = df_model[["Age", "Height", "Weight"]]
        y = df_model["Has_Medal"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        model.fit(X_train, y_train)

        c1, c2, c3 = st.columns(3)

        with c1:
            age_input = st.slider("Âge", 10, 60, 25)

        with c2:
            height_input = st.slider("Taille", 130, 230, 175)

        with c3:
            weight_input = st.slider("Poids", 35, 160, 70)

        input_data = pd.DataFrame({
            "Age": [age_input],
            "Height": [height_input],
            "Weight": [weight_input]
        })

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.write("Résultat du modèle")
        st.progress(float(probability))

        if prediction == 1:
            st.success(f"Probabilité estimée de médaille : {probability:.2%}")
        else:
            st.error(f"Probabilité estimée de médaille : {probability:.2%}")

        st.subheader("Interprétation")
        st.write("""
        - Une probabilité élevée signifie que le profil ressemble davantage à des profils d’athlètes médaillés dans l’historique.
        - Cette estimation reste indicative.
        - La performance réelle dépend aussi d’autres facteurs non inclus ici : niveau technique, contexte sportif, entraînement, concurrence et discipline.
        """)

        importance_df = pd.DataFrame({
            "Variable": ["Age", "Height", "Weight"],
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        fig_imp = px.bar(
            importance_df,
            x="Importance",
            y="Variable",
            orientation="h",
            text="Importance",
            color="Importance",
            color_continuous_scale="Blues",
            title="Importance des variables du modèle"
        )
        fig_imp.update_layout(
            template="plotly_dark",
            height=420,
            title_x=0.02
        )
        fig_imp.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        st.plotly_chart(fig_imp, use_container_width=True)

    else:
        st.warning("Pas assez de données pour entraîner correctement le modèle.")

# FOOTER

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Projet JO 2028 - Analyse, visualisation interactive et prédiction")