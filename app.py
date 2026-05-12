import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EDA Portfolio",
    page_icon="📊",
    layout="wide"
)

SALES_C = "#3fb950"
ZOMATO_C = "#f97316"
NETFLIX_C = "#e50914"

# ─────────────────────────────────────────────────────────────
# MATPLOTLIB STYLE
# ─────────────────────────────────────────────────────────────
plt.style.use("dark_background")

# ─────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────
def fig_bar(labels, values, color, title,
            xlabel="", ylabel="", horizontal=False):

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    if horizontal:
        ax.barh(labels, values, color=color)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
    else:
        ax.bar(labels, values, color=color)
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=25)

    ax.set_title(title)
    ax.grid(alpha=0.3)

    return fig


def fig_line(x, y, color, title, xlabel="", ylabel=""):

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    ax.plot(x, y, color=color, linewidth=2.5, marker="o")
    ax.fill_between(x, y, alpha=0.15, color=color)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)

    plt.xticks(rotation=25)

    return fig


def fig_donut(labels, values, colors, title):

    fig, ax = plt.subplots(figsize=(4.5, 3.8))

    ax.pie(
        values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(width=0.5)
    )

    ax.set_title(title)

    return fig


def fig_hist(data, color, title, xlabel="", bins=25):

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    ax.hist(data, bins=bins, color=color, alpha=0.85)

    ax.axvline(data.mean(), linestyle="--", label="Mean")
    ax.axvline(data.median(), linestyle=":", label="Median")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend()

    return fig

# ─────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_sales():

    df = pd.read_csv(os.path.join("data", "sales_data.csv"))

    df["Sale_Date"] = pd.to_datetime(df["Sale_Date"])

    df["month_name"] = df["Sale_Date"].dt.month_name()
    df["quarter"] = df["Sale_Date"].dt.quarter
    df["year"] = df["Sale_Date"].dt.year

    df["Profit"] = (
        (df["Unit_Price"] - df["Unit_Cost"])
        * df["Quantity_Sold"]
    )

    return df


@st.cache_data
def load_zomato():

    df = pd.read_csv(
        os.path.join("data", "Zomato_Dataset.csv"),
        encoding="latin1"
    )

    df = df[df["Cuisines"].notna()].copy()

    df["Price_Label"] = df["Price_range"].map({
        1: "Cheap",
        2: "Affordable",
        3: "Expensive",
        4: "Very Expensive"
    })

    df_c = df.copy()

    df_c["Cuisines_split"] = df_c["Cuisines"].str.split("|")

    df_c = df_c.explode("Cuisines_split")

    df_c["Cuisines_split"] = (
        df_c["Cuisines_split"]
        .str.strip()
    )

    return df, df_c


@st.cache_data
def load_netflix():

    df = pd.read_csv(
        os.path.join("data", "netflix_titles.csv")
    )

    df.fillna({
        "director": "Unknown",
        "country": "Unknown",
        "cast": "Unknown",
        "rating": "Not Rated",
        "duration": "Unknown"
    }, inplace=True)

    df = df[df["date_added"].notna()].copy()

    df["date_added"] = pd.to_datetime(
        df["date_added"].str.strip(),
        format="%B %d, %Y"
    )

    df["month_added"] = df["date_added"].dt.month
    df["year_added"] = df["date_added"].dt.year
    df["day_added"] = df["date_added"].dt.day_name()

    df["movie_duration"] = (
        df["duration"]
        .where(df["type"] == "Movie")
        .str.extract(r"(\d+)")
        .astype(float)
    )

    df_exp = df.copy()

    df_exp["country_split"] = (
        df_exp["country"]
        .str.split(",")
    )

    df_exp["genre_split"] = (
        df_exp["listed_in"]
        .str.split(",")
    )

    df_exp = (
        df_exp
        .explode("country_split")
        .explode("genre_split")
    )

    df_exp["country_split"] = (
        df_exp["country_split"]
        .str.strip()
    )

    df_exp["genre_split"] = (
        df_exp["genre_split"]
        .str.strip()
    )

    return df, df_exp

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:

    st.title("Aryan Pathania")
    st.caption("EDA Portfolio")

    page = st.radio(
        "Navigation",
        ["Home", "Sales", "Zomato", "Netflix"]
    )

# ─────────────────────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────────────────────
if page == "Home":

    st.title("📊 EDA Portfolio")

    st.write("""
    Interactive dashboards for multiple EDA projects
    using Python, Pandas, Matplotlib and Streamlit.
    """)

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("📈 Sales EDA")
        st.write("Revenue trends, regions, products and profit analysis.")

    with c2:
        st.subheader("🍽️ Zomato EDA")
        st.write("Restaurant ratings, cuisines and city analysis.")

    with c3:
        st.subheader("🎬 Netflix EDA")
        st.write("Content trends, genres and country analysis.")

# ─────────────────────────────────────────────────────────────
# SALES DASHBOARD
# ─────────────────────────────────────────────────────────────
elif page == "Sales":

    st.title("📈 Sales Dashboard")

    try:
        df_raw = load_sales()

    except FileNotFoundError:
        st.error("Place sales_data.csv inside data folder")
        st.stop()

    # FILTERS
    with st.sidebar:

        st.subheader("Filters")

        regions = st.multiselect(
            "Region",
            sorted(df_raw["Region"].unique()),
            default=sorted(df_raw["Region"].unique())
        )

        categories = st.multiselect(
            "Product Category",
            sorted(df_raw["Product_Category"].unique()),
            default=sorted(df_raw["Product_Category"].unique())
        )

    df = df_raw[
        df_raw["Region"].isin(regions) &
        df_raw["Product_Category"].isin(categories)
    ]

    # KPIs
    st.subheader("Key Metrics")

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Revenue", f"${df['Sales_Amount'].sum():,.0f}")
    k2.metric("Profit", f"${df['Profit'].sum():,.0f}")
    k3.metric("Orders", f"{len(df):,}")
    k4.metric("Avg Order", f"${df['Sales_Amount'].mean():,.0f}")

    # CHARTS
    st.subheader("Revenue by Region")

    reg = (
        df.groupby("Region")["Sales_Amount"]
        .sum()
        .sort_values(ascending=True)
    )

    st.pyplot(
        fig_bar(
            reg.index,
            reg.values,
            SALES_C,
            "Revenue by Region",
            horizontal=True
        ),
        use_container_width=True
    )

    st.info("""
    North region generated the highest revenue
    compared to other regions.
    """)

    st.subheader("Monthly Revenue Trend")

    month_order = [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ]

    mon = (
        df.groupby("month_name")["Sales_Amount"]
        .sum()
        .reindex(month_order)
        .dropna()
    )

    st.pyplot(
        fig_line(
            mon.index,
            mon.values,
            SALES_C,
            "Monthly Revenue Trend"
        ),
        use_container_width=True
    )

    st.subheader("Sales Channel Split")

    channel = (
        df.groupby("Sales_Channel")["Sales_Amount"]
        .sum()
    )

    st.pyplot(
        fig_donut(
            channel.index,
            channel.values,
            [SALES_C, "#1f1f1f"],
            "Sales Channels"
        ),
        use_container_width=True
    )

    st.subheader("Raw Data")

    st.dataframe(df, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# ZOMATO DASHBOARD
# ─────────────────────────────────────────────────────────────
elif page == "Zomato":

    st.title("🍽️ Zomato Dashboard")

    try:
        df_raw, df_c_raw = load_zomato()

    except FileNotFoundError:
        st.error("Place Zomato_Dataset.csv inside data folder")
        st.stop()

    with st.sidebar:

        st.subheader("Filters")

        cities = st.multiselect(
            "City",
            sorted(df_raw["City"].unique()),
            default=sorted(df_raw["City"].unique())[:10]
        )

    df = df_raw[df_raw["City"].isin(cities)]

    df_c = df_c_raw[df_c_raw["City"].isin(cities)]

    # KPIs
    st.subheader("Key Metrics")

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Restaurants", len(df))
    k2.metric("Cities", df["City"].nunique())
    k3.metric("Avg Rating", round(df["Rating"].mean(), 2))
    k4.metric(
        "Avg Cost for Two",
        f"₹{df['Average_Cost_for_two'].mean():,.0f}"
    )

    # CITY CHART
    st.subheader("Top Cities")

    city = df["City"].value_counts().head(10)

    st.pyplot(
        fig_bar(
            city.index,
            city.values,
            ZOMATO_C,
            "Top Cities",
            horizontal=True
        ),
        use_container_width=True
    )

    # CUISINES
    st.subheader("Popular Cuisines")

    cuis = (
        df_c["Cuisines_split"]
        .value_counts()
        .head(10)
    )

    st.pyplot(
        fig_bar(
            cuis.index,
            cuis.values,
            ZOMATO_C,
            "Popular Cuisines",
            horizontal=True
        ),
        use_container_width=True
    )

    # RATING HISTOGRAM
    st.subheader("Rating Distribution")

    st.pyplot(
        fig_hist(
            df["Rating"],
            ZOMATO_C,
            "Ratings",
            xlabel="Rating"
        ),
        use_container_width=True
    )

    # DELIVERY
    st.subheader("Online Delivery")

    delivery = df["Has_Online_delivery"].value_counts()

    st.pyplot(
        fig_bar(
            delivery.index,
            delivery.values,
            ZOMATO_C,
            "Online Delivery"
        ),
        use_container_width=True
    )

    # BOOKING
    st.subheader("Table Booking")

    booking = df["Has_Table_booking"].value_counts()

    st.pyplot(
        fig_bar(
            booking.index,
            booking.values,
            ZOMATO_C,
            "Table Booking"
        ),
        use_container_width=True
    )

    st.subheader("Raw Data")

    st.dataframe(df, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# NETFLIX DASHBOARD
# ─────────────────────────────────────────────────────────────
elif page == "Netflix":

    st.title("🎬 Netflix Dashboard")

    try:
        df_raw, df_exp_raw = load_netflix()

    except FileNotFoundError:
        st.error("Place netflix_titles.csv inside data folder")
        st.stop()

    with st.sidebar:

        st.subheader("Filters")

        types = st.multiselect(
            "Content Type",
            ["Movie", "TV Show"],
            default=["Movie", "TV Show"]
        )

    df = df_raw[df_raw["type"].isin(types)]

    df_exp = df_exp_raw[df_exp_raw["type"].isin(types)]

    # KPIs
    st.subheader("Key Metrics")

    movies = df[df["type"] == "Movie"]

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Titles", len(df))
    k2.metric("Movies", len(df[df["type"] == "Movie"]))
    k3.metric("TV Shows", len(df[df["type"] == "TV Show"]))

    if not movies.empty:
        k4.metric(
            "Avg Duration",
            f"{movies['movie_duration'].mean():.0f} min"
        )

    # CONTENT TYPES
    st.subheader("Movies vs TV Shows")

    tc = df["type"].value_counts()

    st.pyplot(
        fig_donut(
            tc.index,
            tc.values,
            [NETFLIX_C, "#1f1f1f"],
            "Content Type"
        ),
        use_container_width=True
    )

    # YEARLY GROWTH
    st.subheader("Titles Added Per Year")

    yr = df.groupby("year_added").size()

    st.pyplot(
        fig_line(
            yr.index.astype(str),
            yr.values,
            NETFLIX_C,
            "Titles Added"
        ),
        use_container_width=True
    )

    # COUNTRIES
    st.subheader("Top Countries")

    cc = (
        df_exp[df_exp["country_split"] != "Unknown"]
        ["country_split"]
        .value_counts()
        .head(10)
    )

    st.pyplot(
        fig_bar(
            cc.index,
            cc.values,
            NETFLIX_C,
            "Top Countries",
            horizontal=True
        ),
        use_container_width=True
    )

    # GENRES
    st.subheader("Top Genres")

    gc = (
        df_exp["genre_split"]
        .value_counts()
        .head(10)
    )

    st.pyplot(
        fig_bar(
            gc.index,
            gc.values,
            NETFLIX_C,
            "Top Genres",
            horizontal=True
        ),
        use_container_width=True
    )

    st.subheader("Movie Duration Distribution")

    dur = df["movie_duration"].dropna()

    st.pyplot(
        fig_hist(
            dur,
            NETFLIX_C,
            "Movie Duration",
            xlabel="Minutes"
        ),
        use_container_width=True
    )

    st.subheader("Raw Data")

    st.dataframe(df, use_container_width=True)