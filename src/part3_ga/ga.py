import random
import matplotlib.pyplot as plt

def gerar_itens(numItens, pesoMax, valorMax):
    """Gera uma lista de itens com base em quantidade, peso máximo e valor máximo."""
    random.seed(42) # Seed de geração aletaória
    listaItens = []
    for _ in range(numItens):
        pesoItem = random.randint(1,pesoMax)
        valorItem = random.randint(1,valorMax)
        item = (pesoItem, valorItem)
        listaItens.append(item)
    return listaItens

def gerar_gene():
    """Gera um gene binário com base no número de itens."""
    lista = []
    for _ in range(numItens):
        num = random.randint(0,1)
        lista.append(num)
    
    return lista

def gerar_lista_genes(tam_populacao):
    """Gera uma lista de genes binários com base no tamanho da população."""
    lista_genes = []
    for _ in range(tam_populacao):
        gene = gerar_gene()
        lista_genes.append(gene)
    return lista_genes

def gerar_fitness(gene, peso_max):
    """Gera o fitness de um gene com base na soma dos valores de seus itens.
    O fitness é 0 caso a soma dos pesos dos itens exceda o peso máximo da mochila."""
    preco = 0
    peso = 0
    for i in range(len(gene)):
        it = gene[i]
        if it == 1:
            preco += itens[i][1]
            peso += itens[i][0]

    if peso > peso_max:
        preco = 0

    return preco

def gerar_lista_fitness(lista_genes, peso_max):
    """Gera uma lista com os fitness de todos os genes dentro de uma lista de genes."""
    lista_fitness = []
    for gene in lista_genes:
        fitness = gerar_fitness(gene, peso_max)
        lista_fitness.append(fitness)
    return lista_fitness

def porcentagens(lista_fitness):
    """Gera as porcentagens dos genes com base na lista de fitness correspondente."""
    lista_porcents = []
    soma_fitness = sum(lista_fitness)
    porcentagem = 0

    for i in lista_fitness:
        porcentagem = i/soma_fitness
        lista_porcents.append(porcentagem)

    return lista_porcents

def gerar_pai(lista_genes, porcentagensRoleta):
    """Gera um gene pai aleatório dentro da lista de genes com base no peso da porcentagem."""
    pai = random.choices(lista_genes, weights=porcentagensRoleta, k=1)
    pai = pai[0]
    return pai

def gerar_filhos(pai1, pai2):
    """Gera dois filhos de dois pais, fazendo o crossover
    de genes com base em um ponto de corte aleatório e válido."""
    ponto_corte = random.randint(1, len(pai1)-1)

    pai1_1 = pai1[:ponto_corte]
    pai1_2 = pai1[ponto_corte:]
    
    pai2_1 = pai2[:ponto_corte]
    pai2_2 = pai2[ponto_corte:]

    filho1 = pai1_1 + pai2_2
    filho2 = pai2_1 + pai1_2

    return filho1, filho2

def mutacao(lista_fihos, taxaMutacao):
    """Aplica uma mutação dentro de uma lista de genes
    com base em uma taxa de mutação."""
    for filho in lista_fihos:
        if random.random() < taxaMutacao:
            i, j = random.sample(range(len(filho)), 2)
            filho[i], filho[j] = filho[j], filho[i]
    return lista_filhos

def plot_grafico(lista_f, y_label):
    """Plota os resultados de uma lista em um gráfico.
    Aplicado para plotar a média de fitness ou o maior
    fitness por geração."""

    x = range(len(lista_f))
    y = lista_f
    plt.plot(x, y)
    plt.xlabel('Gerações')
    plt.ylabel(y_label)
    plt.title('Resultados do AG para o Knapsack 0/1')
    plt.grid(True)
    plt.show()

def print_terminal(i, lista_genes, lista_fitness):
    print("Geração: ", i)
    print(lista_genes, "\n")
    print(lista_fitness, "\n")

def elitismo(lista_genes, lista_fitness, lista_filhos):
    """Salva os dois maiores valores da geração anterior nos filhos da atual,
    se os dois maiores tiverem um fitness maior que 0."""
    lista_fitness_aux = lista_fitness
    
    fitness_max = max(lista_fitness_aux)
    fitness_max_index = lista_fitness_aux.index(fitness_max)
    lista_fitness_aux.remove(fitness_max)

    fitness_max_2 = max(lista_fitness_aux)
    fitness_max2_index = lista_fitness_aux.index(fitness_max_2)

    melhor_gene_1 = lista_genes[fitness_max_index]
    melhor_gene_2 = lista_genes[fitness_max2_index]

    if (fitness_max > 0 and fitness_max_2 > 0):
        lista_filhos.append(melhor_gene_1)
        lista_filhos.append(melhor_gene_2)
        return True
    else:
        return False


if __name__ == '__main__':
    # Variáveis da geração de itens:
    numItens       = 10 
    pesoMax        = 5  
    valorMax       = 8

    # Limite de peso da mochila:
    pesoMochila  = 10  

    # Variáveis gerais do algoritmo genético:
    tamPopulacao = 400
    numGeracoes  = 1000
    taxaMutacao  = 0.01

    # Geração de itens e genes iniciais:
    itens = gerar_itens(numItens, pesoMax, valorMax)
    print(itens, "\n")
    lista_genes = gerar_lista_genes(tamPopulacao)

    lista_media_fitness = []
    lista_maior_fitness = []
    
    for i in range(numGeracoes):
        # Geração dos fitness e suas porcentagens da roleta viciada:
        lista_fitness       = gerar_lista_fitness(lista_genes, pesoMochila)
        porcentagensRoleta  = porcentagens(lista_fitness)

        # Geração de pais e filhos:
        lista_filhos = []
        num_casais = 0

        # Aplica o elitismo antes de gerar pais e filhos:
        if elitismo(lista_genes, lista_fitness, lista_filhos) == True:
            num_casais = int((tamPopulacao/2) - 1)
        else:
            num_casais = int(tamPopulacao/2)

        # Gera pais e filhos:
        for _ in range(num_casais):
            pai1 = gerar_pai(lista_genes, porcentagensRoleta)
            pai2 = gerar_pai(lista_genes, porcentagensRoleta)
            filho1, filho2 = gerar_filhos(pai1, pai2)
            lista_filhos.append(filho1)
            lista_filhos.append(filho2)

        # Obtenção da média de fitness e o maior fitness por geração:
        media_fitness = sum(lista_fitness)/len(lista_fitness)
        maior_fitness = max(lista_fitness)
        lista_media_fitness.append(media_fitness)
        lista_maior_fitness.append(maior_fitness)

        # Aplicação da mutação na lista de filhos gerados:
        lista_filhos_mutados = mutacao(lista_filhos, taxaMutacao)

        # Substituição da geração anterior pela atual na lista de genes:
        lista_genes = lista_filhos_mutados
    
    plot_grafico(lista_media_fitness, 'Média do Fitness da Geração')
    plot_grafico(lista_maior_fitness, 'Maior Fitness da Geração')