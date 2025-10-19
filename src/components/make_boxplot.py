import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def create_numericals_over_symboling(auto_df):
    numericals_bar = auto_df.drop("symboling", axis=1).select_dtypes(exclude=[object]).columns
    selected_numerical_bar = st.selectbox("Column Names: ", numericals_bar, key="numericals_over_symboling")

    st.write(f"### Distribution of {selected_numerical_bar} vs. symboling")
    fig, ax = plt.subplots(figsize=(15,5))
    sns.boxplot(data=auto_df, x="symboling", y=selected_numerical_bar, ax=ax)

    return st.pyplot(fig)