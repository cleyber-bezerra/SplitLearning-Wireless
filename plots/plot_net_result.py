import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ler o arquivo CSV
df = pd.read_csv('./results/csv/ns3/simulator_ns3.csv')

# Filtrar as linhas onde "Latency (s)" não é "inf"
df = df[df['Latency (s)'] != 'inf']

# Converter as colunas para os tipos apropriados, se necessário
df['Latency (s)'] = pd.to_numeric(df['Latency (s)'])
df['Packet Loss Ratio (%)'] = pd.to_numeric(df['Packet Loss Ratio (%)'])
df['Throughput (Mbps)'] = pd.to_numeric(df['Throughput (Mbps)'])
df['Energy Consumed (J)'] = pd.to_numeric(df['Energy Consumed (J)'])

# Gráfico 1: Latência em relação ao Flow ID
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Flow ID', y='Latency (s)', marker='o')
plt.title('Latency in relation to the Client ID')
plt.xlabel('Client ID')
plt.ylabel('Latency (s)')
plt.legend(['Latency (s)'])
plt.grid(True)
plt.savefig('./results/img/net_latencia_vs_clientID.png')  # Salvar o gráfico como .png
plt.show()

# Gráfico 2: Taxa de perda de pacote em relação ao Flow ID
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Flow ID', y='Packet Loss Ratio (%)')
plt.title('Packet loss rate in relation to the Client ID')
plt.xlabel('Client ID')
plt.ylabel('Packet Loss Ratio (%)')
plt.legend(['Packet Loss Ratio (%)'])
plt.grid(True)
plt.savefig('./results/img/net_packet_loss_vs_clientID.png')  # Salvar o gráfico como .png
plt.show()

# Gráfico 3: Intervalo de confiança da taxa de transferência em relação ao Flow ID
plt.figure(figsize=(10, 6))
sns.pointplot(data=df, x='Flow ID', y='Throughput (Mbps)', errorbar='sd', join=False)
plt.title('Throughput in relation to the Client ID')
plt.xlabel('Client ID')
plt.ylabel('Throughput (Mbps)')
plt.legend(['Throughput (Mbps)'])
plt.grid(True)
plt.savefig('./results/img/net_throughput_vs_clientID.png')  # Salvar o gráfico como .png
plt.show()
