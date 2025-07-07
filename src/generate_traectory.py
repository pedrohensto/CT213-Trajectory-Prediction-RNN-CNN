import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from joblib import load

# --- 1. Carregar Modelo, Normalizador e Dados ---

print("Carregando modelo, normalizador e dados...")
# Carrega o modelo LSTM pré-treinado
model = load_model('data/lstm_trajectory_model.h5')
# Carrega o normalizador usado no treinamento
scaler = load('data/scaler.gz')
# Carrega a trajetória real para usarmos como ponto de partida e para comparação
df_real = pd.read_csv('data/robot_trajectory.csv')
real_data = df_real[['position', 'velocity']].values

# Define o mesmo tamanho de sequência usado no treinamento
SEQUENCE_LENGTH = 20
# Define quantos passos no futuro queremos prever
PREDICTION_LENGTH = 200

# --- 2. Preparar a Sequência "Semente" ---

# Pega os primeiros SEQUENCE_LENGTH pontos da trajetória real como nosso "empurrão inicial"
seed_data_real = real_data[:SEQUENCE_LENGTH]
# Normaliza a semente, assim como foi feito no treino
seed_data_scaled = scaler.transform(seed_data_real)

# A lista 'current_sequence' será nossa memória de curto prazo, que será atualizada a cada passo
current_sequence = seed_data_scaled.reshape(1, SEQUENCE_LENGTH, 2)

# --- 3. Loop de Geração Autônoma ---

print(f"Iniciando previsão generativa para {PREDICTION_LENGTH} passos...")
generated_trajectory_scaled = []  # Lista para armazenar as previsões

for _ in range(PREDICTION_LENGTH):
    # 1. Prever o próximo passo com base na sequência atual
    predicted_step = model.predict(current_sequence)

    # 2. Armazenar a previsão
    generated_trajectory_scaled.append(predicted_step[0])

    # 3. Atualizar a sequência: remove o passo mais antigo e adiciona a nova previsão no final
    # Isso faz com que o modelo "use sua própria imaginação" para o futuro
    new_sequence = np.append(current_sequence[:, 1:, :], [
                             predicted_step], axis=1)
    current_sequence = new_sequence

print("Geração concluída.")

# --- 4. Visualização dos Resultados ---

# Desfaz a normalização da trajetória gerada para vermos na escala real
generated_trajectory = scaler.inverse_transform(generated_trajectory_scaled)

# Pega a parte correspondente da trajetória real para comparação
real_comparison_data = real_data[SEQUENCE_LENGTH:
                                 SEQUENCE_LENGTH + PREDICTION_LENGTH]

# Plota o gráfico final
plt.figure(figsize=(12, 8))
plt.plot(real_comparison_data[:, 0], real_comparison_data[:, 1],
         'b-', label='Trajetória Real', linewidth=3, alpha=0.7)
plt.plot(generated_trajectory[:, 0], generated_trajectory[:, 1],
         'r--', label='Trajetória Prevista (Gerada)', linewidth=2)
plt.title('Previsão Generativa de Trajetória com LSTM')
plt.xlabel('Posição')
plt.ylabel('Velocidade')
plt.legend()
plt.grid(True)
plt.savefig('results/lstm_generative_prediction.png')
plt.show()
