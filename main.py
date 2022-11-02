import pandas as pd
import plotly.express as px
import streamlit as st
import requests
from zipfile import ZipFile
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="HDB Resale Prices",
                   page_icon=":house:",
                   initial_sidebar_state="collapsed",
                   layout="wide")


@st.cache
def get_data_from_csv():
    URL = "https://data.gov.sg/dataset/7a339d20-3c57-4b11-a695-9348adfd7614/download"
    response = requests.get(URL)
    open("data.zip", "wb").write(response.content)
    with ZipFile("data.zip", 'r') as zip:
        zip.extract("resale-flat-prices-based-on-registration-date-from-jan-2017-onwards.csv", "data")
    df = pd.read_csv("data/resale-flat-prices-based-on-registration-date-from-jan-2017-onwards.csv")
    # add 'Month' column to df
    df["Month"] = pd.to_datetime(df["month"], format="%Y-%m").dt.month
    df["year"] = pd.to_datetime(df["month"], format="%Y-%m").dt.year
    df.loc[df['flat_type'] == "1 ROOM", "flat_kind"] = 1
    df.loc[df['flat_type'] == "2 ROOM", "flat_kind"] = 2
    df.loc[df['flat_type'] == "3 ROOM", "flat_kind"] = 3
    df.loc[df['flat_type'] == "4 ROOM", "flat_kind"] = 4
    df.loc[df['flat_type'] == "5 ROOM", "flat_kind"] = 5
    df.loc[df['flat_type'] == "EXECUTIVE", "flat_kind"] = 6
    df.loc[df['flat_type'] == "MULTI_GENERATION", "flat_kind"] = 7
    #clean
    df.dropna(inplace=True) 
    return df


df = get_data_from_csv()

# --LR--
#model
x = df[["floor_area_sqm", "lease_commence_date", "flat_kind"]]
y = df["resale_price"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
coeff_df = pd.DataFrame(lr.coef_,x.columns, columns = ["Coefficient"])


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

Year = st.sidebar.multiselect(
    "Select the Year:",
    options=df["year"].unique(),
    default=2022
)


df_selection = df.query(
    "town == @Town & flat_type == @Flat_type & year == @Year" 
)

st.title(":bar_chart: HDB Resale Stats for 2017-2022")
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
    labels={
            "flat_type": "Flat Type",
            "resale_price": "Resale Price (SGD)",
            },
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
    labels={
            "Month": "Month",
            "month": "Number of Flats Resold",
            },
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

st.markdown("---")

# -- prediction --
st.title("HDB Resale Price Prediction")
st.write("This section is independent of sidebar selections. Price is calculated with a linear regression model.")
st.subheader("Flat Type")
flat_select = st.slider("Number of rooms, 6 for Executive and 7 for Multi-Gen", value=4, min_value=1, max_value=7, step=1)
st.subheader("Lease Commence Year")
year_select = st.slider("Year", value=1998, min_value=int(df.lease_commence_date.min()), max_value=int(df.lease_commence_date.max()), step=1)
st.subheader("Floor Space")
sqm_select = st.slider("In SQM (You may use the tool above to get a representative number)", value=93.0, min_value=float(df.floor_area_sqm.min()), max_value=float(df.floor_area_sqm.max()), step=0.5)

price_predicted = lr.predict([[sqm_select, year_select, flat_select]])[0]

html_str = f"""
<style>
p.a {{
  font: bold 40px Courier;
  color: #0FFF50;
}}
</style>
<p class="a">${price_predicted:.2f}</p>
"""

st.markdown(html_str, unsafe_allow_html=True)


# hide st
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)
