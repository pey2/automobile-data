import sys

sys.path.insert(1, "src/components")

import streamlit as st
import pandas as pd
import make_countplot, make_barplot, make_heatmap, make_boxplot

def main():
    st.title("Data Visualization")

    auto_df = pd.read_csv("data/automobile_cleaned.csv")
    auto_df = auto_df.drop("Unnamed: 0", axis=1)

    st.header("Automobile Dataset")
    st.dataframe(auto_df)

    # countplot for symboling based on categoricals
    make_countplot.create_symboling_categoricals_count(auto_df)

    # barchart for price vs categoricals
    make_barplot.create_price_over_symboling(auto_df)
    make_barplot.create_price_over_categoricals(auto_df)

    # barchart for numericals vs. symboling
    make_boxplot.create_numericals_over_symboling(auto_df)

    # heatmap
    make_heatmap.create_heatmap(auto_df)

if __name__ == "__main__":
    st.set_page_config(
        page_title = "Automobile Data",
        page_icon = "🚗",
        layout = "wide"
    )
    main()