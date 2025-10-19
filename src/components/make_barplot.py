import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def create_price_over_symboling(auto_df):
    st.write(f"### Distribution of price vs. make based with symboling")
    return st.bar_chart(
        auto_df,
        x="make",
        y="price",
        color="symboling",
        stack=False
    )

def create_price_over_categoricals(auto_df):
    categoricals_bar = auto_df.select_dtypes(include=["object"]).columns
    categoricals_bar = [feat for feat in categoricals_bar if feat != "make"]
    selected_category_bar = st.selectbox("Column Names: ", categoricals_bar, key="price_over_categorical")

    st.write(f"### Distribution of Price vs. {selected_category_bar} with symboling")
    return st.bar_chart(
        auto_df,
        x=selected_category_bar,
        y="price",
        color="symboling",
        stack=False
    )