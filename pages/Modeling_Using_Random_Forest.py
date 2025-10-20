import streamlit as st
import pickle
import pandas as pd

with open("src/models/random_forest.pkl", "rb") as f:
    model = pickle.load(f)
with open("src/models/ohe.pkl", "rb") as f:
    ohe = pickle.load(f)
with open("src/models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("src/models/pca.pkl", "rb") as f:
    pca = pickle.load(f)

auto_df = pd.read_csv("data/automobile_cleaned.csv").drop("Unnamed: 0", axis=1)
numerical_feat1 = auto_df.drop({"symboling"}, axis=1).select_dtypes(exclude=["object"]).columns[:9]
numerical_feat2 = auto_df.drop({"symboling"}, axis=1).select_dtypes(exclude=["object"]).columns[9:]
categorical_feat = auto_df.select_dtypes(include=["object"]).columns

st.title("🚗 Symboling Classifier using Random Forest")
st.write("Enter automobile data to predict its insurance risk rating (symboling).")

col1, col2, col3 = st.columns(3)

inputs = {}

with col1:
    for col in numerical_feat1:
        inputs[col] = st.number_input(col.replace("-", " ").title(), 
                                      min_value=1,  
                                      format="%g",
                                      help="Enter a value greater than 1")

with col2:
    for col in numerical_feat2:
        if (col == "num-of-cylinders") | (col == "num-of-doors"):
            inputs[col] = st.selectbox(col.replace("-", " ").title(), sorted(auto_df[col].unique()))
        else:
            inputs[col] = st.number_input(col.replace("-", " ").title(), 
                                        min_value=1,  
                                        format="%g",
                                        help="Enter a value greater than 1")    
with col3:
    for col in categorical_feat:
        inputs[col] = st.selectbox(col.replace("-", " ").title(), auto_df[col].unique())

if st.button("Predict Symboling"):
    numerical_feat = numerical_feat1.append(numerical_feat2)
    input_df = pd.DataFrame([inputs])

    # One hot encoder
    ohe_encoded = ohe.transform(input_df[categorical_feat])
    ohe_df = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out(categorical_feat))

    input_encoded = pd.concat([input_df.drop(categorical_feat, axis=1), ohe_df], axis=1)

    # Standardization
    scaled_numeric = scaler.transform(input_encoded[numerical_feat])
    scaled_df = pd.DataFrame(
        scaled_numeric,
        columns=scaler.get_feature_names_out(numerical_feat),
        index=input_encoded.index
    )

    X_scaled = pd.concat(
        [scaled_df, input_encoded.drop(columns=scaler.get_feature_names_out(numerical_feat))],
        axis=1
    )

    # PCA
    final_features = pca.transform(X_scaled)

    # Predict
    prediction = model.predict(final_features)
    st.success(f"✅ Predicted Symboling: **{prediction[0]}**")
