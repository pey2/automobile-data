import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def create_symboling_categoricals_count(auto_df):
    categoricals_count = auto_df.select_dtypes(include=["object"]).columns
    selected_category_count = st.selectbox("Column Names: ", categoricals_count, key="categorical_symboling")

    st.write(f"### Distribution of {selected_category_count} with symboling")
    fig, ax = plt.subplots(figsize=(30,10))
    sns.countplot(data=auto_df, x=auto_df[selected_category_count], hue="symboling", palette="viridis")
    ax.set_title(f"Distribution of {selected_category_count} based on symboling")

    return st.pyplot(fig)