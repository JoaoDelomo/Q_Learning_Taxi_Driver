import random
from IPython.display import clear_output
import gymnasium as gym
import numpy as np
from QLearning import QLearning
import matplotlib.pyplot as plt

# Definindo os valores de alpha a serem testados
alpha_values = [0.01, 0.1, 0.5]

# Criando o ambiente do problema do Taxi Driver
env = gym.make("Taxi-v3", render_mode='ansi').env

# Armazenar os resultados para plotar
actions_per_episode_all = {}

# Treinando o agente com diferentes valores de alpha
for alpha in alpha_values:
    print(f'Treinando com alpha={alpha}')
    qlearn = QLearning(env, alpha=alpha, gamma=0.99, epsilon=0.7, epsilon_min=0.05, epsilon_dec=0.99, episodes=5000)
    
    # Armazenar a Q-table (sem descompactar dois valores)
    q_table = qlearn.train('data/q-table-taxi-driver-alpha-{}.csv'.format(alpha), None)
    
    # Recuperar o número de ações por episódio armazenado na classe QLearning
    actions_per_episode = qlearn.actions_per_episode
    actions_per_episode_all[alpha] = actions_per_episode

# Plotar o número de ações por episódio para cada valor de alpha
plt.figure(figsize=(10, 6))
for alpha, actions in actions_per_episode_all.items():
    plt.plot(range(len(actions)), actions, label=f'alpha={alpha}')

# Configurações do Gráfico
plt.xlabel('Episódios')
plt.ylabel('Número de Ações')
plt.title('Comparação do Número de Ações por Episódio para Diferentes Valores de Alpha')
plt.legend()
plt.grid(True)
plt.show()
