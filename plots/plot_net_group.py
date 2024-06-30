import pandas as pd
import matplotlib.pyplot as plt

# Ler o arquivo CSV
df = pd.read_csv('./results/csv/ns3/simulator_ns3.csv')

# Filtrar os dados onde a latência é diferente de "inf"
df_filtered = df[df['Latency (s)'] != 'inf']

# Converter colunas necessárias para tipo numérico, se ainda não estiverem
df_filtered['Latency (s)'] = pd.to_numeric(df_filtered['Latency (s)'])
df_filtered['Packet Loss Ratio (%)'] = pd.to_numeric(df_filtered['Packet Loss Ratio (%)'])
df_filtered['Throughput (Mbps)'] = pd.to_numeric(df_filtered['Throughput (Mbps)'])
df_filtered['Energy Consumed (J)'] = pd.to_numeric(df_filtered['Energy Consumed (J)'])

# Configurar o layout dos gráficos em uma grade 2x2
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Gráfico de linha para Latência
axs[0, 0].plot(df_filtered['Flow ID'], df_filtered['Latency (s)'], marker='o', linestyle='-', color='b')
axs[0, 0].set_title('Latency per Client ID')
axs[0, 0].set_xlabel('Client ID')
axs[0, 0].set_ylabel('Latency (s)')
axs[0, 0].legend(['Latency (s)'], loc='upper right')

# Adicionar anotações nos pontos
for i in range(len(df_filtered)):
    axs[0, 0].annotate(f"{df_filtered['Latency (s)'].iloc[i]:.2f}", 
                       (df_filtered['Flow ID'].iloc[i], df_filtered['Latency (s)'].iloc[i]), 
                       textcoords="offset points", xytext=(0,10), ha='center')

# Gráfico de linhas verticais para Taxa de Perda de Pacotes
axs[0, 1].bar(df_filtered['Flow ID'], df_filtered['Packet Loss Ratio (%)'], color='r')
axs[0, 1].set_title('Packet Loss Rate per Client ID')
axs[0, 1].set_xlabel('Client ID')
axs[0, 1].set_ylabel('Packet Loss Ratio (%)')
axs[0, 1].legend(['Packet Loss Ratio (%)'], loc='upper right')

# Adicionar anotações nas barras
for i in range(len(df_filtered)):
    axs[0, 1].annotate(f"{df_filtered['Packet Loss Ratio (%)'].iloc[i]:.2f}", 
                       (df_filtered['Flow ID'].iloc[i], df_filtered['Packet Loss Ratio (%)'].iloc[i]), 
                       textcoords="offset points", xytext=(0,10), ha='center')

# Gráfico de linhas verticais para Vazão
axs[1, 0].bar(df_filtered['Flow ID'], df_filtered['Throughput (Mbps)'], color='g')
axs[1, 0].set_title('Throughput per Client ID')
axs[1, 0].set_xlabel('Client ID')
axs[1, 0].set_ylabel('Throughput (Mbps)')
axs[1, 0].legend(['Throughput (Mbps)'], loc='upper right')

# Adicionar anotações nas barras
for i in range(len(df_filtered)):
    axs[1, 0].annotate(f"{df_filtered['Throughput (Mbps)'].iloc[i]:.2f}", 
                       (df_filtered['Flow ID'].iloc[i], df_filtered['Throughput (Mbps)'].iloc[i]), 
                       textcoords="offset points", xytext=(0,10), ha='center')

# Gráfico de linha para Consumo Energético
axs[1, 1].plot(df_filtered['Flow ID'], df_filtered['Energy Consumed (J)'], marker='o', linestyle='-', color='m')
axs[1, 1].set_title('Energy Consumption per Client ID')
axs[1, 1].set_xlabel('Client ID')
axs[1, 1].set_ylabel('Energy Consumed (J)')
axs[1, 1].legend(['Energy Consumed (J)'], loc='upper right')

# Adicionar anotações nos pontos
for i in range(len(df_filtered)):
    axs[1, 1].annotate(f"{df_filtered['Energy Consumed (J)'].iloc[i]:.2f}", 
                       (df_filtered['Flow ID'].iloc[i], df_filtered['Energy Consumed (J)'].iloc[i]), 
                       textcoords="offset points", xytext=(0,10), ha='center')

# Adicionar título principal ao quadro de gráficos
fig.suptitle('Networks Simulation', fontsize=16)

# Ajustar layout
plt.tight_layout()

# Salvar os gráficos em um arquivo PNG
plt.savefig('./images/figura0.png')

# Exibir os gráficos
plt.show()
