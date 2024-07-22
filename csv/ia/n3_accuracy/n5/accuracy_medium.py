import matplotlib.pyplot as plt
import pandas as pd

# Carregar o arquivo CSV
file_path = 'result_train_sync_medium.csv'
data = pd.read_csv(file_path)

# Converter as colunas de acurácia de validação para listas de floats
data['Validation Accuracy'] = data['Validation Accuracy'].apply(lambda x: float(x.strip('[]')))

# Calcular a média de acurácia de validação por cliente
mean_accuracy_per_client = data.groupby('Client')['Validation Accuracy'].mean()

# Plotar o gráfico de barras com rótulos de valor em cada barra
plt.figure(figsize=(10, 6))
bars = plt.bar(mean_accuracy_per_client.index, mean_accuracy_per_client.values, color='darkblue')

# Adicionar rótulos de valor em cada barra
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 4), va='bottom')  # Usei '%.4f' para mostrar 4 casas decimais

plt.title('Média de Acurácia de Validação por Cliente - Medium')
plt.xlabel('Cliente')
plt.ylabel('Média de Acurácia de Validação')
plt.xticks(rotation=45)
plt.tight_layout()

# Salvar o gráfico como um arquivo PNG
plt.savefig('media_acuracia_por_cliente_medium.png')

# Mostrar o gráfico
plt.show()

