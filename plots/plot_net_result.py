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
ax1 = sns.lineplot(data=df, x='Flow ID', y='Latency (s)', marker='o')
plt.title('Latency in relation to the Client ID')
plt.xlabel('Client ID')
plt.ylabel('Latency (s)')
plt.legend(['Latency (s)'])
plt.grid(True)

# Adicionar anotações
for i, row in df.iterrows():
    ax1.annotate(f"{row['Latency (s)']:.2f}", (row['Flow ID'], row['Latency (s)']), textcoords="offset points", xytext=(0,10), ha='center')

plt.savefig('./results/img/net_latencia_vs_clientID.png')  # Salvar o gráfico como .png
plt.show()

# Gráfico 2: Taxa de perda de pacote em relação ao Flow ID
plt.figure(figsize=(10, 6))
ax2 = sns.barplot(data=df, x='Flow ID', y='Packet Loss Ratio (%)')
plt.title('Packet loss rate in relation to the Client ID')
plt.xlabel('Client ID')
plt.ylabel('Packet Loss Ratio (%)')
plt.legend(['Packet Loss Ratio (%)'])
plt.grid(True)

# Adicionar anotações
for p in ax2.patches:
    height = p.get_height()
    ax2.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom', xytext=(0, 10), textcoords='offset points')

plt.savefig('./results/img/net_packet_loss_vs_clientID.png')  # Salvar o gráfico como .png
plt.show()

'''
# Gráfico 3: Vazão em relação ao Flow ID
plt.figure(figsize=(10, 6))
ax3 = sns.lineplot(data=df, x='Flow ID', y='Throughput (Mbps)', errorbar='sd', join=False)
plt.title('Throughput in relation to the Client ID')
plt.xlabel('Client ID')
plt.ylabel('Throughput (Mbps)')
plt.legend(['Throughput (Mbps)'])
plt.grid(True)

# Adicionar anotações
for i, row in df.iterrows():
    ax3.annotate(f"{row['Throughput (Mbps)']:.2f}", (row['Flow ID'], row['Throughput (Mbps)']), textcoords="offset points", xytext=(0,10), ha='center')

plt.savefig('./results/img/net_throughput_vs_clientID.png')  # Salvar o gráfico como .png
plt.show()
'''

# Gráfico 3: Vazão em relação ao Flow ID (modificado para gráfico de barras)
plt.figure(figsize=(10, 6))
ax3 = sns.barplot(data=df, x='Flow ID', y='Throughput (Mbps)')
plt.title('Throughput in relation to the Client ID')
plt.xlabel('Client ID')
plt.ylabel('Throughput (Mbps)')
plt.legend(['Throughput (Mbps)'])
plt.grid(True)

# Adicionar anotações
for p in ax3.patches:
    height = p.get_height()
    ax3.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom', xytext=(0, 10), textcoords='offset points')

plt.savefig('./results/img/net_throughput_vs_clientID.png')  # Salvar o gráfico como .png
plt.show()

