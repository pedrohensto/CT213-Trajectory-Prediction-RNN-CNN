# Previsão de Trajetória de Robô Móvel com RNN e CNN

**Autor:** Pedro Henrique
**Curso:** CT-213 - Inteligência Artificial para Robótica Móvel (ITA)

## Descrição do Projeto

Este projeto final explora e compara duas arquiteturas de Deep Learning — Long Short-Term Memory (LSTM) e uma Rede Neural Convolucional (CNN) inspirada na WaveNet — para a tarefa de previsão de trajetória de um robô móvel.

O objetivo é aprender um modelo da dinâmica de um agente treinado no ambiente "Mountain Car" e avaliar a capacidade de cada arquitetura em realizar previsões de curto prazo e gerar trajetórias autônomas de longo prazo.

## Estrutura dos Arquivos

```
.
├── data/                 # Contém o dataset e os modelos treinados
├── results/              # Contém os gráficos e figuras geradas
├── report/               # Contém o relatório final em PDF
├── src/                  # Contém todo o código-fonte
├── environment.yml       # Arquivo de dependências do Conda
└── README.md             # Este arquivo
```

## Como Replicar os Experimentos

### 1. Configuração do Ambiente

Este projeto utiliza o Conda para gerenciamento de ambientes. Para recriar o ambiente com todas as dependências necessárias, execute o seguinte comando na pasta raiz do projeto:

```bash
conda env create -f environment.yml
```

Em seguida, ative o ambiente recém-criado:

```bash
conda activate ct213-exame-env
```

### 2. Etapas da Execução

Os scripts devem ser executados a partir da pasta `src/`.

**a) Treinamento dos Modelos (Opcional, pois os modelos já estão salvos em `/data`)**

Para retreinar os modelos do zero:
```bash
# Treina o modelo LSTM e salva em data/lstm_trajectory_model.h5
python src/train_lstm_trajectory.py

# Treina o modelo CNN e salva em data/cnn_trajectory_model.h5
python src/train_cnn_trajectory.py
```

**b) Geração das Trajetórias Autônomas (Resultado Principal)**

Para usar os modelos já treinados e gerar os gráficos de trajetória autônoma:
```bash
# Gera a trajetória com o modelo LSTM
python src/generate_trajectory_lstm.py

# Gera a trajetória com o modelo CNN
python src/generate_trajectory_cnn.py
```

## Resultados Principais

A principal conclusão do trabalho é a comparação da robustez generativa entre as duas arquiteturas.

**Trajetória Gerada pela LSTM:**
*(Adicione aqui a imagem `results/lstm_generative_prediction.png`)*

**Trajetória Gerada pela CNN:**
*(Adicione aqui a imagem `results/cnn_generative_prediction.png`)*

Observa-se que a LSTM foi mais robusta em manter a estabilidade da dinâmica do sistema em previsões de longo prazo.