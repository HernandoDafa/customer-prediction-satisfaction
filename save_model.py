import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle

# Memuat dataset
df = pd.read_csv('onlinefoods.csv')

# Menghapus baris dengan nilai yang hilang
df.dropna(inplace=True)

# Mendefinisikan fitur dan target
X = df.drop('Output', axis=1)
y = df['Output']

# Mengidentifikasi kolom kategorikal dan numerik
categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(include=['number']).columns

# One-hot encoding untuk kolom kategorikal
encoder = OneHotEncoder(drop='first', sparse=False)
encoded_categorical = pd.DataFrame(encoder.fit_transform(X[categorical_cols]))
encoded_categorical.columns = encoder.get_feature_names_out(categorical_cols)
encoded_categorical.index = X.index

# Menggabungkan kembali data setelah encoding
X = X.drop(categorical_cols, axis=1)
X = pd.concat([X, encoded_categorical], axis=1)

# Standard scaling untuk fitur numerik
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Melatih model Decision Tree
dec_tree = DecisionTreeClassifier()
dec_tree.fit(X_train, y_train)

# Melatih model K-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Melatih model Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Menyimpan model ke dalam file 'model.pkl'
models = {
    "log_reg": log_reg,
    "dec_tree": dec_tree,
    "knn": knn,
    "rf_model": rf_model
}

with open('model.pkl', 'wb') as file:
    pickle.dump(models, file)

print("Model telah disimpan ke dalam model.pkl")
