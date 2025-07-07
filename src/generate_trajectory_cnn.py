import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from joblib import load

# --- 1. Carregar Modelo, Normalizador e Dados ---
print("Carregando modelo CNN, normalizador e dados...")
model = load_model('data/cnn_trajectory_model.h5')
scaler = load('data/scaler_cnn.gz')
df_real = pd.read_csv('data/robot_trajectory.csv')
real_data = df_real[['position', 'velocity']].values

# Usa o mesmo tamanho de sequência do treinamento
SEQUENCE_LENGTH = 30
PREDICTION_LENGTH = 200  # Número de passos a serem "imaginados" pelo modelo

# --- 2. Preparar a Sequência "Semente" ---
seed_data_real = real_data[:SEQUENCE_LENGTH]
seed_data_scaled = scaler.transform(seed_data_real)
current_sequence = seed_data_scaled.reshape(1, SEQUENCE_LENGTH, 2)

# --- 3. Loop de Geração Autônoma ---
print(
    f"Iniciando previsão generativa com CNN para {PREDICTION_LENGTH} passos...")
generated_trajectory_scaled = []

for _ in range(PREDICTION_LENGTH):
    # 1. Prever o próximo passo
    predicted_step = model.predict(current_sequence)

    # 2. Armazenar a previsão
    generated_trajectory_scaled.append(predicted_step[0])

    # 3. Atualizar a sequência para a próxima previsão
    # A previsão é um array (1, 2), precisamos remodelar para (1, 1, 2) para concatenar
    predicted_step_reshaped = predicted_step.reshape(1, 1, 2)
    new_sequence = np.append(
        current_sequence[:, 1:, :], predicted_step_reshaped, axis=1)
    current_sequence = new_sequence

print("Geração concluída.")

# --- 4. Visualização dos Resultados ---
generated_trajectory = scaler.inverse_transform(generated_trajectory_scaled)
real_comparison_data = real_data[SEQUENCE_LENGTH:
                                 SEQUENCE_LENGTH + PREDICTION_LENGTH]

plt.figure(figsize=(12, 8))
plt.plot(real_comparison_data[:, 0], real_comparison_data[:, 1],
         'b-', label='Trajetória Real', linewidth=3, alpha=0.7)
plt.plot(generated_trajectory[:, 0], generated_trajectory[:, 1],
         'g--', label='Trajetória Prevista (CNN)', linewidth=2)
plt.title('Previsão Generativa de Trajetória com CNN (WaveNet-like)')
plt.xlabel('Posição')
plt.ylabel('Velocidade')
plt.legend()
plt.grid(True)
plt.savefig('results/cnn_generative_prediction.png')
plt.show()
