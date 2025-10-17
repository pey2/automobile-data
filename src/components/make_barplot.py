import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def create_price_over_symboling(auto_df):
    st.write(f"### Distribution of price over make based on symboling")
    return st.bar_chart(
        auto_df,
        x="make",
        y="price",
        color="symboling"
    )

def create_price_over_categoricals(auto_df):
    categoricals_bar = auto_df.select_dtypes(include=["object"]).columns
    categoricals_bar = [feat for feat in categoricals_bar if feat != "make"]
    selected_category_bar = st.selectbox("Categories: ", categoricals_bar, key="cat_4")

    st.write(f"### Distribution of price over make based on {selected_category_bar}")
    return st.bar_chart(
        auto_df,
        x="make",
        y="price",
        color=selected_category_bar
    )
