import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def create_heatmap(auto_df):
    st.write("### Heatmap")
    numericals = auto_df.select_dtypes(exclude=["object"])
    corr = numericals.corr()
    fig, ax = plt.subplots(figsize=(13,10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation of Numericals")

    return st.pyplot(fig)