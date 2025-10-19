import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Data loading
def load_and_clean_data(path: str):
    df = pd.read_csv(path).drop("Unnamed: 0", axis=1)
    return df


# Data Split
def split_data(df):
    X = df.drop("symboling", axis=1)
    y = df["symboling"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Data preprocessing
def preprocess_data(X_train, X_test):
    numerical_feat = X_train.select_dtypes(exclude=["object"]).columns
    categorical_feat = X_train.select_dtypes(include=["object"]).columns

    # OneHotEncoder
    ohe = OneHotEncoder(drop="first", sparse_output=False)
    ohe_encoded_train = ohe.fit_transform(X_train[categorical_feat])
    ohe_encoded_test = ohe.transform(X_test[categorical_feat])

    ohe_df_train = pd.DataFrame(
        ohe_encoded_train,
        columns=ohe.get_feature_names_out(categorical_feat),
        index=X_train.index
    )
    ohe_df_test = pd.DataFrame(
        ohe_encoded_test,
        columns=ohe.get_feature_names_out(categorical_feat),
        index=X_test.index
    )

    X_train_encoded = pd.concat([X_train.drop(categorical_feat, axis=1), ohe_df_train], axis=1)
    X_test_encoded = pd.concat([X_test.drop(categorical_feat, axis=1), ohe_df_test], axis=1)

    # StandardScaler
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(X_train_encoded[numerical_feat])
    scaled_test = scaler.transform(X_test_encoded[numerical_feat])

    scaled_df_train = pd.DataFrame(
        scaled_train,
        columns=scaler.get_feature_names_out(numerical_feat),
        index=X_train_encoded.index
    )
    scaled_df_test = pd.DataFrame(
        scaled_test,
        columns=scaler.get_feature_names_out(numerical_feat),
        index=X_test_encoded.index
    )

    X_train_scaled = pd.concat([scaled_df_train, X_train_encoded.drop(scaled_df_train.columns, axis=1)], axis=1)
    X_test_scaled = pd.concat([scaled_df_test, X_test_encoded.drop(scaled_df_test.columns, axis=1)], axis=1)

    return X_train_scaled, X_test_scaled, ohe, scaler

# Apply PCA
def apply_pca(X_train, X_test):
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca

# Apply SMOTE
def balance_data(X_train, y_train):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X_train, y_train)

# Train and test
def train_and_evaluate(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=84, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    return clf

# Save models
def save_models(model, ohe, scaler, pca):
    with open("src/models/random_forest.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("src/models/ohe.pkl", "wb") as f:
        pickle.dump(ohe, f)
    with open("src/models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("src/models/pca.pkl", "wb") as f:
        pickle.dump(pca, f)
    print("✅ Model and preprocessors saved successfully!")

def main():
    auto_df = load_and_clean_data("data/automobile_cleaned.csv")
    X_train, X_test, y_train, y_test = split_data(auto_df)
    X_train_scaled, X_test_scaled, ohe, scaler = preprocess_data(X_train, X_test)
    X_train_pca, X_test_pca, pca = apply_pca(X_train_scaled, X_test_scaled)
    X_train_res, y_train_res = balance_data(X_train_pca, y_train)
    model = train_and_evaluate(X_train_res, X_test_pca, y_train_res, y_test)
    save_models(model, ohe, scaler, pca)

if __name__ == "__main__":
    main()
