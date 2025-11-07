import csv

# Listas auxiliares:
dados = []
aux1 = ['loam', 'sandy', 'clay']        # loam = 0, sandy = 1, clay = 2  
aux2 = ['daily', 'weekly', 'bi-weekly'] # daily = 0, weekly = 1, bi-weekly = 2
aux3 = ['chemical', 'organic', 'none']  # chemical = 0, organical = 1, none = 2

# Abre o arquivo de entrada e salva o .csv na lista auxiliar:
with open('data/raw/plant_growth_data.csv') as f_in:
    reader = csv.reader(f_in)
    
    for linha in reader:
        dados.append(linha)

# Substitui cada termo em cada coluna pelo seu índice na lista auxiliar:
for linha in dados:
    for item in aux1:
        if linha[0] == item:
            linha[0] = str(aux1.index(item))
        
    for item2 in aux2:
        if linha[2] == item2:
            linha[2] = str(aux2.index(item2))

    for item3 in aux3:
        if linha[3] == item3:
            linha[3] = str(aux3.index(item3))

# Salva as modificações em um novo .csv:
with open('data/processed/plant_data_processed.csv', "w", newline='') as f_out:
    writer = csv.writer(f_out)    
    writer.writerows(dados)