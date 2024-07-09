import pandas as pd
import matplotlib.pyplot as plt

# Carregar os dados do arquivo CSV
file_path = 'tot_energy.csv'
df = pd.read_csv(file_path)

# Definir os dados para o gráfico
clients = df['Client']
low = df['LOW']
moderate = df['MODERATE']
high = df['HIGH']

# Configurar o gráfico
plt.figure(figsize=(10, 6))  # Tamanho da figura

# Plotar as linhas
plt.plot(clients, low, marker='o', linestyle='-', color='b', label='LOW')
plt.plot(clients, moderate, marker='o', linestyle='-', color='g', label='MODERATE')
plt.plot(clients, high, marker='o', linestyle='-', color='r', label='HIGH')

# Adicionar título e rótulos aos eixos
plt.title('Energy Levels by Client')
plt.xlabel('Client')
plt.ylabel('Energy Level')

# Adicionar legenda
plt.legend()

# Exibir o gráfico
plt.grid(True)
plt.tight_layout()
plt.show()
