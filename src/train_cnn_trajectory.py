import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Input, Flatten
from joblib import dump


def create_sequences(data, sequence_length):
    """
    Converte um array de dados em sequências de entrada (X) e saída (y).
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        # O conjunto de entrada é a sequência de 'sequence_length' passos
        X.append(data[i:(i + sequence_length)])
        # O conjunto de saída é o passo IMEDIATAMENTE SEGUINTE à sequência (CORREÇÃO)
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)


def build_cnn_model(sequence_length, num_features):
    # ... (a função build_cnn_model continua a mesma)
    model = Sequential([
        Input(shape=(sequence_length, num_features)),
        Conv1D(filters=32, kernel_size=2, padding='causal',
               dilation_rate=1, activation='relu'),
        Conv1D(filters=32, kernel_size=2, padding='causal',
               dilation_rate=2, activation='relu'),
        Conv1D(filters=32, kernel_size=2, padding='causal',
               dilation_rate=4, activation='relu'),
        Conv1D(filters=32, kernel_size=2, padding='causal',
               dilation_rate=8, activation='relu'),
        Conv1D(filters=1, kernel_size=1),
        Flatten(),
        Dense(units=num_features)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    return model


# --- 1. Carregamento e Pré-processamento dos Dados ---
df = pd.read_csv('robot_trajectory.csv')
trajectory_data = df[['position', 'velocity']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(trajectory_data)

SEQUENCE_LENGTH = 30
NUM_FEATURES = 2

X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# --- 2. Construção e Treinamento do Modelo CNN ---
model = build_cnn_model(SEQUENCE_LENGTH, NUM_FEATURES)
history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_data=(X_test, y_test), verbose=1)

# --- 3. Avaliação e Visualização ---
# ... (a parte de plotagem continua a mesma)

# --- 4. Salvamento do Modelo e Normalizador ---
print("\nSalvando o modelo CNN e o normalizador...")
model.save('cnn_trajectory_model.h5')
dump(scaler, 'scaler_cnn.gz')
print("Modelo CNN salvo como 'cnn_trajectory_model.h5'")
print("Normalizador salvo como 'scaler_cnn.gz'")
