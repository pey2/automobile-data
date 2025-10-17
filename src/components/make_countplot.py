import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def create_symboling_count(auto_df):
    st.write("### Distribution of symboling")
    fig, ax = plt.subplots()
    sns.countplot(data=auto_df, x="symboling")
    ax.set_title("Symboling Count")
    
    return st.pyplot(fig)

def create_symboling_categoricals_count(auto_df):
    categoricals_count = auto_df.select_dtypes(include=["object"]).columns
    selected_category_count = st.selectbox("Categories: ", categoricals_count, key="cat_2")

    st.write(f"### Distribution of {selected_category_count} based on symboling")
    fig, ax = plt.subplots(figsize=(30,10))
    sns.countplot(data=auto_df, x=auto_df[selected_category_count], hue="symboling", palette="viridis")
    ax.set_title(f"Distribution of {selected_category_count} based on symboling")

    return st.pyplot(fig)

def create_make_categoricals_count(auto_df):
    categoricals = auto_df.select_dtypes(include=["object"]).columns
    categoricals = [feat for feat in categoricals if feat != "make"]
    selected_category = st.selectbox("Categories: ", categoricals, key="cat_1")

    st.write(f"### Distribution of make based on {selected_category}")
    fig, ax = plt.subplots(figsize=(30,10))
    sns.countplot(data=auto_df, x="make", hue=auto_df[selected_category], ax=ax)
    ax.set_title(f"Distribution of make based on {selected_category}")
    
    return st.pyplot(fig)
