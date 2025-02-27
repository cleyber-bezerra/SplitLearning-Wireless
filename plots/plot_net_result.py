import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ler o arquivo CSV
df = pd.read_csv('./csv/ns3/simulator_ns3.csv')

# Filtrar as linhas onde "Delay (s)" não é "inf"
df = df[df['Delay (s)'] != 'inf']

# Converter as colunas para os tipos apropriados, se necessário
df['Delay (s)'] = pd.to_numeric(df['Delay (s)'])
df['Packet Loss Ratio (%)'] = pd.to_numeric(df['Packet Loss Ratio (%)'])
df['Throughput (Mbps)'] = pd.to_numeric(df['Throughput (Mbps)'])
df['Energy Consumed (J)'] = pd.to_numeric(df['Energy Consumed (J)'])
df['Flow ID'] = df['Flow ID'].astype(int)  # Converter Flow ID para inteiro

# Gráfico 1: Latência em relação ao Flow ID
plt.figure(figsize=(10, 6))
ax1 = sns.lineplot(data=df, x='Flow ID', y='Delay (s)', marker='o')
plt.title('Delay in relation to the Client ID - Network')
plt.xlabel('Client ID')
plt.ylabel('Delay (s)')
plt.legend(['Delay (s)'])
plt.grid(True)

# Adicionar anotações
for i, row in df.iterrows():
    ax1.annotate(f"{row['Delay (s)']:.2f}", (row['Flow ID'], row['Delay (s)']), textcoords="offset points", xytext=(0,10), ha='center')

# Definir rótulos de eixo x como inteiros
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

plt.savefig('./images/figure1.png')  # Salvar o gráfico como .png
#plt.show()

# Gráfico 2: Taxa de perda de pacote em relação ao Flow ID
plt.figure(figsize=(10, 6))
ax2 = sns.barplot(data=df, x='Flow ID', y='Packet Loss Ratio (%)')
plt.title('Packet loss rate in relation to the Client ID - Network')
plt.xlabel('Client ID')
plt.ylabel('Packet Loss Ratio (%)')
plt.legend(['Packet Loss Ratio (%)'])
plt.grid(True)

# Adicionar anotações
for p in ax2.patches:
    height = p.get_height()
    ax2.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom', xytext=(0, 10), textcoords='offset points')

# Definir rótulos de eixo x como inteiros
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

plt.savefig('./images/figure2.png')  # Salvar o gráfico como .png
#plt.show()

# Gráfico 3: Vazão em relação ao Flow ID (modificado para gráfico de barras)
plt.figure(figsize=(10, 6))
ax3 = sns.barplot(data=df, x='Flow ID', y='Throughput (Mbps)')
plt.title('Throughput in relation to the Client ID - Network')
plt.xlabel('Client ID')
plt.ylabel('Throughput (Mbps)')
plt.legend(['Throughput (Mbps)'])
plt.grid(True)

# Adicionar anotações
for p in ax3.patches:
    height = p.get_height()
    ax3.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom', xytext=(0, 10), textcoords='offset points')

# Definir rótulos de eixo x como inteiros
ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

plt.savefig('./images/figure3.png')  # Salvar o gráfico como .png
#plt.show()

# Gráfico 4: Consumo de energia em relação ao Flow ID (modificado para gráfico de linha)
plt.figure(figsize=(10, 6))
ax4 = sns.lineplot(data=df, x='Flow ID', y='Energy Consumed (J)', marker='o')
plt.title('Energy Consumption in relation to the Client ID - Network')
plt.xlabel('Client ID')
plt.ylabel('Energy Consumed (J)')
plt.legend(['Energy Consumed (J)'])
plt.grid(True)

# Adicionar anotações
for i, row in df.iterrows():
    ax4.annotate(f"{row['Energy Consumed (J)']:.2f}", (row['Flow ID'], row['Energy Consumed (J)']), textcoords="offset points", xytext=(0,10), ha='center')

# Definir rótulos de eixo x como inteiros
ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

plt.savefig('./images/figure4.png')  # Salvar o gráfico como .png
#plt.show()
