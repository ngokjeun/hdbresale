import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="HDB Resale Prices",
                   page_icon=":house:",
                   initial_sidebar_state="collapsed",
                   layout="wide")


@st.cache
def get_data_from_csv():
    df = pd.read_csv("resale_prices.csv")

    # add 'Month' column to df
    df["Month"] = pd.to_datetime(df["month"], format="%Y-%m").dt.month
    return df


df = get_data_from_csv()

# -- sidebar --

st.sidebar.header("Please Filter Here:")

Flat_type = st.sidebar.multiselect(
    "Select the Flat Type:",
    options=df["flat_type"].unique(),
    default=df["flat_type"].unique()
)

Town = st.sidebar.multiselect(
    "Select the Town:",
    options=df["town"].unique(),
    default=df["town"].unique()
)


df_selection = df.query(
    "town == @Town & flat_type == @Flat_type"

)

st.title(":bar_chart: HDB Resale Stats for 2022")
st.markdown("##")

median_sqm = int(df_selection["floor_area_sqm"].median())
median_lease_commence = int(df_selection["lease_commence_date"].median())
median_resale_price = int(df_selection["resale_price"].median())

left_column, middle_column, right_column = st.columns(3)
with left_column:
    st.subheader("Median sqm:")
    st.subheader(f"{median_sqm:,} sqm")
with middle_column:
    st.subheader("Median Year of Lease Commencement:")
    st.subheader(f"{median_lease_commence}")
with right_column:
    st.subheader("Median Resale Price:")
    st.subheader(f"SGD {median_resale_price:,}")

st.markdown("---")

median_price_by_type = (
    df_selection.groupby(["flat_type"]).median()["resale_price"])


fig_type_price = px.bar(
    median_price_by_type,
    y="resale_price",
    x=median_price_by_type.index,
    orientation="v",
    title="<em>Median Resale Price by Type</em>",
    color_discrete_sequence=["#0083B8"]*len(median_price_by_type),
)

fig_type_price.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    yaxis=(dict(showgrid=False))
)

resale_by_month = df_selection.groupby(["Month"]).count()
resale_by_month.head()
fig_resale_month = px.line(
    resale_by_month,
    x=resale_by_month.index,
    y=resale_by_month["month"],
    title="<em>Resale Numbers by Month</em>",
    color_discrete_sequence=["#0083B8"]*len(resale_by_month),
    range_x=[1, 12],

)

fig_resale_month.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    yaxis=(dict(showgrid=False))
)

left_column, right_column = st.columns(2)
left_column.plotly_chart(fig_type_price, use_container_width=True)
right_column.plotly_chart(fig_resale_month, use_container_width=True)


# hide st
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)
