import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier
import joblib

# Load dataset
df = pd.read_csv('dataset_tanaman.csv')

# Pisahkan fitur dan target
X = df.drop('jenis_tanaman', axis=1)
y = df['jenis_tanaman']

# Preprocessing
numeric_features = ['ph', 'nitrogen', 'phosphor', 'potassium', 'suhu',
                    'curah_hujan', 'ketinggian', 'intensitas_cahaya']
categorical_features = ['tekstur_tanah']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(), categorical_features)
])

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline MLP
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('mlp', MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Eval
y_pred = pipeline.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Akurasi:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(pipeline, 'mlp_sklearn_model.pkl')
