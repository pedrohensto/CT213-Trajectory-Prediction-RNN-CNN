from joblib import dump
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

def create_sequences(data, sequence_length):
    """
    Converte um array de dados em sequências de entrada (X) e saída (y).
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        # O conjunto de entrada é a sequência de 'sequence_length' passos
        X.append(data[i:(i + sequence_length)])
        # O conjunto de saída é o passo imediatamente seguinte à sequência
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# --- 1. Carregamento e Pré-processamento dos Dados ---

# Carrega o dataset de trajetória gerado anteriormente
try:
    df = pd.read_csv('robot_trajectory.csv')
    print("Dataset 'robot_trajectory.csv' carregado com sucesso!")
except FileNotFoundError:
    print("Erro: Arquivo 'robot_trajectory.csv' não encontrado. Certifique-se de gerá-lo primeiro.")
    exit()

# Extrai os dados de posição e velocidade
trajectory_data = df[['position', 'velocity']].values

# Normaliza os dados para o intervalo [0, 1], que é ideal para LSTMs
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(trajectory_data)
print(f"Dados normalizados. Shape: {scaled_data.shape}")

# Define o tamanho da sequência de entrada para a LSTM
# Usaremos os últimos 20 pontos para prever o próximo
SEQUENCE_LENGTH = 20

# Cria as sequências de dados para treinamento
X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)
print(f"Sequências de entrada (X) criadas com shape: {X.shape}")
print(f"Sequências de saída (y) criadas com shape: {y.shape}")

# Divide os dados em conjuntos de treino e teste (80% para treino, 20% para teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Construção do Modelo LSTM ---

print("\nConstruindo o modelo LSTM...")
model = Sequential([
    Input(shape=(SEQUENCE_LENGTH, 2)), # 2 features: posição e velocidade
    LSTM(units=50, return_sequences=False), # 50 neurônios na camada LSTM
    Dense(units=2) # Camada de saída com 2 neurônios para prever (posição, velocidade)
])

# Compila o modelo
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# --- 3. Treinamento do Modelo ---

print("\nIniciando o treinamento do modelo...")
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)
print("Treinamento concluído!")

# --- 4. Avaliação e Visualização dos Resultados ---

print("\nGerando gráficos de avaliação...")

# Gráfico 1: Curva de Loss (Erro) do Treinamento
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Erro de Treinamento')
plt.plot(history.history['val_loss'], label='Erro de Validação')
plt.title('Curva de Aprendizado do Modelo LSTM')
plt.xlabel('Época')
plt.ylabel('Erro Quadrático Médio (Loss)')
plt.legend()
plt.grid(True)
plt.savefig('results/lstm_learning_curve.png')
plt.show()


# Gráfico 2: Comparação da Previsão vs. Real
# Faz as previsões no conjunto de teste
predicted_scaled = model.predict(X_test)

# Desfaz a normalização para comparar na escala original
predicted_trajectory = scaler.inverse_transform(predicted_scaled)
real_trajectory = scaler.inverse_transform(y_test)

plt.figure(figsize=(14, 7))
# Plota a posição real vs. prevista
plt.subplot(1, 2, 1)
plt.plot(real_trajectory[:, 0], label='Posição Real', color='blue', alpha=0.7)
plt.plot(predicted_trajectory[:, 0], label='Posição Prevista', color='red', linestyle='--')
plt.title('Previsão da Posição do Robô')
plt.xlabel('Passo de Tempo (Amostra de Teste)')
plt.ylabel('Posição')
plt.legend()
plt.grid(True)

# Plota a velocidade real vs. prevista
plt.subplot(1, 2, 2)
plt.plot(real_trajectory[:, 1], label='Velocidade Real', color='blue', alpha=0.7)
plt.plot(predicted_trajectory[:, 1], label='Velocidade Prevista', color='red', linestyle='--')
plt.title('Previsão da Velocidade do Robô')
plt.xlabel('Passo de Tempo (Amostra de Teste)')
plt.ylabel('Velocidade')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('results/lstm_prediction_vs_real.png')
plt.show()

# --- 5. Salvamento do Modelo e Normalizador ---

print("\nSalvando o modelo e o normalizador...")

# Salva o modelo treinado em um arquivo H5
model.save('data/lstm_trajectory_model.h5')

# Salva o objeto 'scaler' que foi usado para normalizar os dados
dump(scaler, 'data/scaler.gz')

print("Modelo salvo como 'data/lstm_trajectory_model.h5'")
print("Normalizador salvo como 'scaler.gz'")
