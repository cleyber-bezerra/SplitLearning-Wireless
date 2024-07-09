import pandas as pd
import matplotlib.pyplot as plt

# Carregar os dados do arquivo CSV
file_path = './csv/ia/result_train_sync.csv'
data = pd.read_csv(file_path)

# Convertendo as colunas que estão como strings de listas para valores únicos
data['Validation Accuracy'] = data['Validation Accuracy'].str.strip('[]').astype(float)
data['Comm Time'] = data['Comm Time'].astype(float)

# Gerar o gráfico de Validation Accuracy por Client
plt.figure(figsize=(10, 6))
plt.plot(data['Client'], data['Validation Accuracy'], marker='o', linestyle='-')
plt.xlabel('Client')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy por Client')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Adicionar rótulos aos pontos (Validation Accuracy)
for x, y in zip(data['Client'], data['Validation Accuracy']):
    plt.text(x, y, f'{y:.2f}', ha='center', va='bottom')

plt.show()

# Gerar o gráfico de Comm Time por Client
plt.figure(figsize=(10, 6))
plt.plot(data['Client'], data['Comm Time'], marker='o', linestyle='-')
plt.xlabel('Client')
plt.ylabel('Comm Time')
plt.title('Comm Time por Client')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Adicionar rótulos aos pontos (Comm Time)
for x, y in zip(data['Client'], data['Comm Time']):
    plt.text(x, y, f'{y:.2f}', ha='center', va='bottom')

plt.show()

