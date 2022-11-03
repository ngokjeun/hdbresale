import pandas as pd
import plotly.express as px
import streamlit as st
import requests
from zipfile import ZipFile
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from streamlit_option_menu import option_menu
import datetime as dt
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor


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
    df.dropna(inplace=True) 
    return df


df = get_data_from_csv()


# --LR--
#clean 
df_town = pd.concat([df,pd.get_dummies(df.town, prefix='town')],axis=1)
df_town['sale_date'] = pd.to_datetime(df_town['month'], format='%Y-%m')
df_town['sale_date'] = df_town['sale_date'].map(dt.datetime.toordinal)

#model
add_prefix = ["town_" + town for town in list(df.town.unique())]
x = ['lease_commence_date', 'flat_kind', 'sale_date', 'town_ANG MO KIO', 'town_BEDOK', 'town_BISHAN', 'town_BUKIT BATOK', 'town_BUKIT MERAH', 'town_BUKIT PANJANG', 'town_BUKIT TIMAH', 'town_CENTRAL AREA', 'town_CHOA CHU KANG', 'town_CLEMENTI', 'town_GEYLANG', 'town_HOUGANG', 'town_JURONG EAST', 'town_JURONG WEST', 'town_KALLANG/WHAMPOA', 'town_MARINE PARADE', 'town_PASIR RIS', 'town_PUNGGOL', 'town_QUEENSTOWN', 'town_SEMBAWANG', 'town_SENGKANG', 'town_SERANGOON', 'town_TAMPINES', 'town_TOA PAYOH', 'town_WOODLANDS', 'town_YISHUN']
x = df_town[x]
y = df_town["resale_price"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# -- sidebar --
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["home", "the model"]
    )

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

# ----
if selected == "home":
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
    st.write("Data is currently incomplete for 2022.")
    st.markdown("---")
# how to write in streamlit 
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
    st.write("This section is independent of sidebar selections. Price is calculated with regression models.")
    st.subheader("Flat Type")
    flat_select = st.slider("Number of rooms, 6 for Executive and 7 for Multi-Gen", value=4, min_value=1, max_value=7, step=1)
    st.subheader("Lease Commence Year")
    year_select = st.slider("Year", value=1998, min_value=int(df.lease_commence_date.min()), max_value=int(df.lease_commence_date.max()), step=1)
    town_select = st.selectbox("Select the Town:",
        options=df["town"].unique(),
        key = "not_sidebar"
        )
    date_select = st.text_input("Key in the date: i.e. YYYY-MM", "2022-09")
    regression_select = st.selectbox("Select the regression method: (Linear is faster, Random Forest is more accurate)",
        options=["Linear Regressor", "Random Forest Regressor"]
        )

    #converting user inputs
    ordinal_date = dt.datetime.toordinal(pd.to_datetime(date_select, format='%Y-%m'))
    town_select_list = [0] * 26
    for i in df_town.columns:
         if town_select in i:
             t_index = df_town.columns.get_loc(i) - 14
             town_select_list = town_select_list[:t_index]+[1]+town_select_list[t_index+1:]
    user_list = [year_select, flat_select, ordinal_date] + town_select_list
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    if regression_select == "Linear Regressor":
        price_predicted = lr.predict([user_list])[0]
    elif regression_select == "Random Forest Regressor":
        rfr = RandomForestRegressor(min_samples_split=3, n_estimators=20, min_samples_leaf = 1)
        rfr.fit(X_train, y_train)
        price_predicted = rfr.predict([user_list])[0]

    html_str = f"""
    <style>
    p.a {{
    font: bold 40px Courier;
    color: #5dc0fc;
    }}
    </style>
    <p class="a">${price_predicted:,.0f}</p>
    """

    st.markdown(html_str, unsafe_allow_html=True)

if selected == "the model":
    st.title("the model")
    st.markdown("**Scikit linear and random forest regression models were used in this demonstration**")
    st.write("To include town as a variable, I represented it as a dummy variable.")
    st.code("df_town = pd.concat([df,pd.get_dummies(df.town, prefix='town')],axis=1)")
    st.write("Converting user inputs was a challenge, as I have yet to understand labeling of nominal categorical variables after encoding(i.e. town). Hence I populated a list of 0s and 1s with a for loop.")
    st.code("""
    town_select_list = [0] * 26
    for i in df_town.columns:
         if town_select in i:
             t_index = df_town.columns.get_loc(i) - 14
             town_select_list = town_select_list[:t_index]+[1]+town_select_list[t_index+1:]
    user_list = [year_select, flat_select, ordinal_date] + town_select_list
    """)
    st.write("As data.gov.sg provided the dataset with the sale timestamp as a string, I converted it into an ordinal date for use with the model. date_select is a text input.")
    st.code("ordinal_date = dt.datetime.toordinal(pd.to_datetime(date_select,format=%Y-%m))")
    st.write("Making suitable DataFrames to train")
    st.code('''
    add_prefix = ["town_" + town for town in list(df.town.unique())]
    x = ['lease_commence_date', 'flat_kind', 'sale_date', 'town_ANG MO KIO', 'town_BEDOK', 'town_BISHAN', 'town_BUKIT BATOK', 'town_BUKIT MERAH', 'town_BUKIT PANJANG', 'town_BUKIT TIMAH', 'town_CENTRAL AREA', 'town_CHOA CHU KANG', 'town_CLEMENTI', 'town_GEYLANG', 'town_HOUGANG', 'town_JURONG EAST', 'town_JURONG WEST', 'town_KALLANG/WHAMPOA', 'town_MARINE PARADE', 'town_PASIR RIS', 'town_PUNGGOL', 'town_QUEENSTOWN', 'town_SEMBAWANG', 'town_SENGKANG', 'town_SERANGOON', 'town_TAMPINES', 'town_TOA PAYOH', 'town_WOODLANDS', 'town_YISHUN']
    x = df_town[x]
    y = df_town["resale_price"]
    ''')
    st.write("Splitting the data into training and testing sets")
    st.code("X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42) #this is used for both linear and random forest regressors")
    st.write("Creating a linear regression object with scikit")
    st.code("lr = LinearRegression()")
    st.code("rfr = RandomForestRegressor() #parameters can be added to increase speed, I used min_samples_split=3, n_estimators=20, min_samples_leaf = 1")
    st.write("Train the model using sets specified")
    st.code("lr.fit(X_train, y_train)")
    st.code("rfr.fit(X_train, y_train)")
    st.write("The coefficients from linear regression")
    st.code('coeff_df = pd.DataFrame(lr.coef_,x.columns, columns = ["Coefficient"])')
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    coeff_df = pd.DataFrame(lr.coef_,x.columns, columns = ["Coefficient"])
    st.dataframe(coeff_df)
    rfr = RandomForestRegressor(min_samples_split=3, n_estimators=20, min_samples_leaf = 1)
    rfr.fit(X_train, y_train)
    st.write("Predict for test set")
    st.code("predictions = lr.predict(X_test)")
    st.code("predictions = rfr.predict(X_test)")
    st.markdown("***Plot predictions against actual***")
    st.code("px.scatter(y_test, predictions)")
    st.code("px.scatter(y_test, predictions_rfr)")
    predictions = lr.predict(X_test)
    predictions_rfr = rfr.predict(X_test)
    fig_scatter = px.scatter(x=y_test, y=predictions)
    fig_scatter_rfr = px.scatter(x=y_test, y=predictions_rfr)
    left_column, right_column = st.columns(2)
    with left_column:
        st.write("Linear Regression")
        st.plotly_chart(fig_scatter, use_container_width=True)
    with right_column:
        st.write("Random Forest Regression")
        st.plotly_chart(fig_scatter_rfr, use_container_width=True)
    st.write("As observed, the random forest regressor produced a more compact, linear 'line'. This is indicative of a more accurate model. In the previous version of this model, the flat area was used as a variable. However, since flat area is dependent on flat type (95% correlated), it was not an independent variable, hence was removed from the model.")
    st.markdown("***Plot prediction distribution***")
    st.code("px.histogram((y_test-predictions))")
    st.code("px.histogram((y_test-predictions_rfr))")
    left_column2, right_column2 = st.columns(2)
    with left_column2:
        st.write("Linear Regression")
        fig_hist = px.histogram((y_test-predictions))
        st.plotly_chart(fig_hist, use_container_width=True)
    with right_column2:    
        st.write("Random Forest Regression")
        fig_hist_rfr = px.histogram((y_test-predictions_rfr))
        st.plotly_chart(fig_hist_rfr, use_container_width=True)
    st.write("This distribution graph is in the bell shape we expect. As the model becomes more accurate, the spread becomes tighter. We can see that the distribution of predictions for the random forest regressor is significantly tighter than the linear regression model.")
    st.write("To find the R^2 value, we use the score function")
    st.code("metrics.r2_score(y_test, predictions)")
    st.code("metrics.r2_score(y_test, predictions_rfr)")
    st.write(metrics.r2_score(y_test, predictions), metrics.r2_score(y_test, predictions_rfr))
    st.write("The R^2 value is around 0.8, which is decent for a linear regressor. The R^2 value of around 0.9 of the random forest regressor is improved")
    st.write("To find the Mean Absolute Error, we use the mean_absolute_error function")
    st.code("metrics.mean_absolute_error(y_test, predictions)")
    st.code("metrics.mean_absolute_error(y_test, predictions_rfr)")
    st.write(metrics.mean_absolute_error(y_test, predictions), metrics.mean_absolute_error(y_test, predictions_rfr))
    st.markdown("The Mean Absolute Error shows that the linear regression model is off by around $57,000 on average.")
    st.markdown("In contrast, the random forest regressor is off by around $34,000 on average. This is a significant improvement.")



# hide st
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)
