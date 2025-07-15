
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load dataset
df = pd.read_csv('dataset_tanaman.csv')

# Pisahkan fitur dan target
X = df.drop('jenis_tanaman', axis=1)
y = df['jenis_tanaman']

# Preprocessing pipeline
numeric_features = ['ph', 'nitrogen', 'phosphor', 'potassium', 'suhu', 'curah_hujan', 'ketinggian', 'intensitas_cahaya']
categorical_features = ['tekstur_tanah']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(), categorical_features)
])

# Preprocessing input
X_processed = preprocessor.fit_transform(X)

# One-hot encoding untuk label
encoder = OneHotEncoder(sparse=False)
y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)

# Build MLP model
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=0)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Akurasi pada data uji: {accuracy * 100:.2f}%")
