<h1 align="center">Trabalho Prático IA [Algoritmos] (2025/2)</h1>

<div align="center">

![Python](https://img.shields.io/badge/python-blue?style=for-the-badge&logo=python&logoColor=white)
![VS Code](https://img.shields.io/badge/visual%20studio%20code-blue?style=for-the-badge)
![Ubuntu](https://img.shields.io/badge/ubuntu-orange?style=for-the-badge&logo=ubuntu&logoColor=white)

📖:
[Visão Geral](#visão-geral) |
[Como reproduzir](#como-reproduzir) |
[Decisões Técnicas](#decisões-técnicas)

</div>


## Visão Geral

<div align="justify">
O objetivo do trabalho, realizado na disciplina de Inteligência Artificial ofertada pelo professor Tiago Aves de Oliveira, foi de compreender, implementar e comparar algoritmos clássicos de IA e Computação Natural, preparando e analisando dados reais.
</div>

### Partes do trabalho:
O trabalho foi dividido em quatro partes:

- Parte 1 - Árvore de decisão manual
- Parte 2 - Supervisionado (Kaggle/UCI): KNN, SVM e Árvore 
- Parte 3 - Algoritmo Genético (AG)
- Parte 4 - Enxame e Imunes

### Estrutura do Repositório:
```
TRABALHO PRÁTICO IA/
│   .gitattributes
│   README.md
│   requirements.txt
│   svm.model
│   
├───data
│   ├───processed
│   │       benchmark_results.csv
│   │       comparison_report.txt
│   │       confusion_matrix_dt_100000.png
│   │       confusion_matrix_knn_100000.png
│   │       confusion_matrix_svm_100000.png
│   │       decision_tree_visualization.png
│   │       decision_tree_visualization_100000.png
│   │       X_test.csv
│   │       X_test_scaled.csv
│   │       X_train.csv
│   │       X_train_scaled.csv
│   │       y_test.csv
│   │       y_train.csv
│   │
│   └───raw
│           Watera.csv
│
└───src
    ├───part1_tree_manual
    │       tree_diagram.md
    │       tree_image.png
    │       tree_manual.py
    │
    ├───part2_ml
    │   │   preprocess.py
    │   │   train_knn.py
    │   │   train_svm.py
    │   │   train_tree.py
    │   │   util_metrics.py
    │   │
    │   └───__pycache__
    │           preprocess.cpython-310.pyc
    │           preprocess.cpython-311.pyc
    │           util_metrics.cpython-310.pyc
    │
    └───part3_ga
            ga.py
```

<!--
```
TREE-DECISION/
│
├── data/
│   ├── raw/                    # Dados brutos
│   │   └── plant_growth_data.csv
│   └── processed/              # Dados processados
│       ├── X_train.csv         # Features de treino (sem escalonamento)
│       ├── X_train_scaled.csv  # Features de treino (escalonadas)
│       ├── y_train.csv         # Target de treino
│       └── ...
│
├── src/
│   ├── part1_tree_manual/      # Implementação manual
│   │   └── tree_manual.py      # Classe Tree customizada [Exemplo: 32 correntes filosóficas]
│   │   
│   │
│   └── part2_ml/               # Machine Learning
│       ├── preprocess.py       # Pré-processamento completo
│       ├── train_tree.py       # Treinar Decision Tree
│       ├── train_knn.py        # Treinar KNN
│       ├── train_svm.py        # Treinar SVM
│       └── util_metrics.py     # Funções de métricas
│
├── requirements.txt            # Dependências
└── README.md                   # Este arquivo
```
-->

<!--
Este projeto explora **Árvores de Decisão** de duas formas:

1. **Parte 1 - Implementação Manual (`src/part1_tree_manual/`)**
   - Estrutura de dados de árvore binária do zero
   - Visualização com NetworkX e Matplotlib
   - Exemplo: Árvore de decisão filosófica com 32 correntes (6 níveis)

2. **Parte 2 - Machine Learning (`src/part2_ml/`)**
   - Pré-processamento robusto de dados
   - Treinamento de modelos: Decision Tree, KNN, SVM
   - Métricas de avaliação e validação cruzada 
-->



## Como reproduzir

### Pré-requisitos

- **Python 3.8+** instalado
- **pip** (gerenciador de pacotes Python)
- **Git** (opcional, para clonar o repositório)

---

### Instalação Rápida

#### Usando run.sh (Linux):
```bash
# Concede permissões de execução:
chmod +x run.sh

# Cria .venv e baixa depenências nele:
./run.sh
```

#### Usando Makefile (Alternativa):

```bash
# Instalar dependências
make install

# Executar Parte 1 (Árvore Manual Filosófica)
make part1

# Executar Parte 2 completa (ML: pré-processamento + treinos)
make part2

# Ou executar partes específicas:
make part2-dt         # Apenas Decision Tree
make part2-knn        # Apenas KNN
make part2-svm        # Apenas SVM

# Executar Parte 3 (Algoritmo Genético)
make part3

# Ver resultados
make results

# Limpar arquivos gerados
make clean
```

#### Instalação Manual

```bash
pip install -r requirements.txt
```

---

### Bibliotecas Utilizadas

| Biblioteca | Versão | Por Que Usamos? | Decisão de Implementação |
|------------|--------|-----------------|--------------------------|
| **pandas** | 2.1.4 | Manipulação de dados tabulares (CSV, DataFrames) | Escolhido por sua eficiência em operações de leitura/escrita de CSV e transformações de dados. A API intuitiva permite operações complexas em uma linha. |
| **numpy** | 1.26.3 | Operações numéricas eficientes (arrays, matrizes) | Base para computação científica em Python. Vetorização de operações acelera cálculos em 10-100x comparado a loops Python puros. |
| **scikit-learn** | 1.3.2 | **Biblioteca principal de ML**: Decision Tree, KNN, SVM, pré-processamento, métricas | API consistente entre algoritmos facilita experimentação. Implementações otimizadas em C/Cython garantem performance. Amplamente testada e documentada. |
| **scipy** | 1.11.4 | Algoritmos científicos e estatísticos (dependência do scikit-learn) | Fornece estruturas de dados especializadas (sparse matrices) e algoritmos de otimização usados internamente pelo scikit-learn. |
| **matplotlib** | 3.8.2 | Visualização de dados (gráficos, plots) | Padrão *de facto* para visualização científica em Python. Flexibilidade para customização detalhada de gráficos. |
| **seaborn** | 0.13.0 | Visualização estatística (matriz de confusão) | Built on top do matplotlib, oferece temas visuais mais modernos e funções especializadas para análise estatística. |

---

<!--
### Sobre a escolha de cada Biblioteca

#### **scikit-learn** 
```python
# Modelos de ML
from sklearn.tree import DecisionTreeClassifier       # Árvore de Decisão
from sklearn.neighbors import KNeighborsClassifier    # KNN
from sklearn.svm import SVC                           # SVM

# Pré-processamento
from sklearn.preprocessing import StandardScaler      # Escalonamento (KNN/SVM)
from sklearn.preprocessing import LabelEncoder        # String → Número

# Métricas e Validação
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
```

#### **pandas** + **numpy**
```python
import pandas as pd    # Ler CSV, manipular tabelas
import numpy as np     # Operações matemáticas rápidas
```
-->

## Decisões Técnicas

### **Parte 1: Árvore Manual (Filosófica)**

#### Execução

```bash
make part1
```

#### O que faz
- Sistema interativo com 6 níveis de perguntas filosóficas
- Identifica 32 correntes filosóficas diferentes
- Recomendações de livros personalizadas (200+ obras catalogadas)

#### Decisões de Implementação

**1. Estrutura de Dados Customizada**
```python
class Tree:
    def __init__(self, question, left=None, right=None, leaf_value=None):
        self.question = question
        self.left = left
        self.right = right
        self.leaf_value = leaf_value
```
- **Por que classe própria?** Controle total sobre a lógica de navegação e visualização. Permite adicionar métodos customizados (ex: `collect_all_nodes()`, `get_depth()`).
- **Alternativa considerada:** Usar `sklearn.tree.DecisionTreeClassifier`, mas não permite construção manual da árvore com perguntas textuais.

**2. Hierarquia de 6 Níveis**
```
Nível 1: Conhecimento (Racionalismo vs Empirismo)
Nível 2: Realidade (Materialismo vs Idealismo)
Nível 3: Ética (Deontologia vs Consequencialismo)
Nível 4: Existência (Determinismo vs Livre-arbítrio)
Nível 5: Política (Individualismo vs Coletivismo)
Nível 6: Estética (Objetividade vs Subjetividade)
```
- **Por que 6 níveis?** 2^6 = 64 folhas possíveis, mas usamos 32 correntes (algumas folhas compartilham ramos). Profundidade equilibra especificidade com usabilidade.
- **Decisão de design:** Ordem das perguntas segue progressão lógica (fundamentos → aplicações práticas).


---

### **Parte 2: Machine Learning (Supervisionado)**

#### Execução Completa

```bash
# Execução de todas os algoritmos:
make part2

# Execução individual dos algoritmos:
make part2-preprocess
make part2-dt
make part2-knn
make part2-svm
```

---

#### **1. Pré-processamento (`preprocess.py`)**

##### Decisões de Implementação

**A. Tratamento de Valores Nulos**
```python
# Numéricos: Mediana (não média!)
imputer_num = SimpleImputer(strategy='median')
X[numerical_cols] = imputer_num.fit_transform(X[numerical_cols])

# Categóricos: Moda
imputer_cat = SimpleImputer(strategy='most_frequent')
X[categorical_cols] = imputer_cat.fit_transform(X[categorical_cols])
```
- **Por que mediana e não média?** 
  - Média é sensível a outliers. Se 99 valores são ~10 e 1 valor é 10.000, a média será distorcida.
  - Mediana é robusta: sempre retorna o valor central.
  - **Exemplo real:** Em dados de qualidade de água, um sensor defeituoso pode gerar pH=999. Mediana ignora esse erro.

**B. Label Encoding vs One-Hot Encoding**
```python
# Label Encoding (usado no projeto)
le = LabelEncoder()
X['Soil_Type'] = le.fit_transform(X['Soil_Type'])
# ['loam', 'sandy', 'clay'] → [0, 1, 2]

# One-Hot Encoding (comentado, mas disponível)
X_encoded = pd.get_dummies(X, columns=['Soil_Type'], drop_first=True)
# Soil_Type_loam | Soil_Type_sandy | Soil_Type_clay
#       1        |        0        |       0
```
- **Por que Label Encoding?**
  - **Decision Trees:** Não precisam de One-Hot. Árvores podem lidar com ordinais (0, 1, 2) diretamente.
  - **KNN/SVM:** Label Encoding funciona quando há ordem natural (ex: Pequeno=0, Médio=1, Grande=2).
  - **Quando usar One-Hot:** Categorias sem ordem (ex: cores: vermelho, azul, verde). Evita o modelo assumir que "azul" (1) está "entre" vermelho (0) e verde (2).
- **Trade-off:** One-Hot aumenta dimensionalidade. 10 categorias → 10 colunas. Afeta performance em datasets grandes.

**C. Escalonamento: StandardScaler vs MinMaxScaler**
```python
# StandardScaler (média=0, desvio=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# MinMaxScaler (valores entre 0 e 1) - usado no projeto
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```
- **Por que MinMaxScaler?**
  - **KNN:** Distâncias euclidianas são afetadas por escala. Feature "Renda" (0-100k) dominaria "Idade" (0-100).
  - **SVM:** Kernel RBF usa distâncias. Features com ranges diferentes quebram a simetria do kernel.
  - **MinMaxScaler vs StandardScaler:** 
    - MinMaxScaler preserva distribuição original (boa para dados sem outliers extremos).
    - StandardScaler melhor quando há outliers (normaliza pelo desvio padrão).
  - **Decision Trees NÃO precisam:** Árvores fazem splits baseados em thresholds relativos (ex: "pH > 7?"). Escala não importa.

**D. Divisão Estratificada**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y,  # CRUCIAL!
    random_state=42
)
```
- **Por que estratificação?**
  - **Problema:** Dataset com 90% classe A, 10% classe B. Split aleatório pode gerar treino com 95% A, teste com 85% A.
  - **Solução:** `stratify=y` garante que treino E teste tenham 90% A, 10% B.
  - **Impacto:** Sem estratificação, métricas podem ser enviesadas. Modelo "aprende" distribuição diferente da realidade.

**E. Amostragem Aleatória (10k, 50k, 100k)**
```python
df_sample = df.sample(n=sample_size, random_state=42)
```
- **Por que amostrar?**
  - **Benchmarking:** Comparar performance dos algoritmos em diferentes escalas de dados.
  - **Trade-off tempo vs acurácia:** SVM com 100k linhas pode levar horas. 10k linhas permite iteração rápida.
  - **`random_state=42`:** Reprodutibilidade. Mesma amostra em execuções diferentes.

---

#### **2. Treinamento dos Modelos**

##### **Decision Tree (`train_tree.py`)**

```python
dt = DecisionTreeClassifier(
    max_depth=10,              # Limita profundidade
    min_samples_split=20,      # Mínimo de amostras para split
    min_samples_leaf=10,       # Mínimo de amostras por folha
    random_state=42
)
```

**Decisões de Hiperparâmetros:**

| Parâmetro | Valor | Por Que? | Risco se Diferente |
|-----------|-------|----------|-------------------|
| `max_depth=10` | 10 níveis | Profundidade média para datasets tabulares. Previne overfitting em dados com ruído. | **Muito alto (ex: 50):** Árvore memoriza treino (overfitting). **Muito baixo (ex: 3):** Underfitting, não captura padrões. |
| `min_samples_split=20` | 20 amostras | Evita splits em subconjuntos pequenos (pouco representativos). | **Muito baixo (ex: 2):** Árvore cria regras específicas para poucos exemplos (overfitting). |
| `min_samples_leaf=10` | 10 amostras | Garante que folhas tenham exemplos suficientes para generalizar. | **Muito baixo (ex: 1):** Folhas com 1 exemplo = decorar dataset. |

**Por que Decision Trees são robustas:**
- **Não precisam de escalonamento:** Splits baseados em thresholds (ex: "Temperatura > 25°C?").
- **Lidam com não-linearidade:** Capturam interações complexas (ex: "SE temperatura > 30 E umidade < 40 ENTÃO...").
- **Interpretabilidade:** Regras legíveis por humanos.
- **Overfitting fácil:** Sem regularização, decoram o treino. Por isso os hiperparâmetros acima.

---

##### **KNN (`train_knn.py`)**

```python
knn = KNeighborsClassifier(
    n_neighbors=5,    # K=5 vizinhos
    n_jobs=-1         # Paralelização (usa todos os cores)
)
```

**Seleção do K=5:**
```python
# Testamos K de 1 a 20 e plotamos acurácia
k_range = list(range(1, 21))
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # ... treinar e avaliar
# Gráfico mostra K=5 como melhor trade-off
```

| K | Comportamento | Por Que Não Escolher? |
|---|---------------|----------------------|
| **K=1** | Classifica baseado no vizinho mais próximo | **Overfitting:** Sensível a ruído. Um outlier mislabeled pode quebrar predição. |
| **K=5** | **Balanceado** | Suaviza ruído sem perder granularidade. |
| **K=20** | Classifica baseado em 20 vizinhos | **Underfitting:** Fronteiras de decisão muito suaves. Perde detalhes. |

---

##### **SVM (`train_svm.py`)**

```python
svm = SVC(
    kernel='rbf',        # Radial Basis Function (não-linear)
    C=1.0,               # Regularização
    probability=True,    # Habilita predict_proba() para ROC-AUC
    random_state=42
)
```

**Decisões de Hiperparâmetros:**

| Parâmetro | Valor | Por Que? |
|-----------|-------|----------|
| `kernel='rbf'` | Radial Basis Function | **Não-linear:** Mapeia dados para espaço de alta dimensão onde são linearmente separáveis. Alternativas: `'linear'` (mais rápido, assume separabilidade linear), `'poly'` (polinomial, caro computacionalmente). |
| `C=1.0` | Penalização padrão | **Trade-off:** C alto → Margem estreita (overfitting). C baixo → Margem larga (underfitting). 1.0 é balanceado. |
| `probability=True` | Habilita probabilidades | **Necessário para ROC-AUC:** `predict()` retorna classes (0, 1). `predict_proba()` retorna probabilidades (0.0-1.0). Aumenta tempo de treino (~2x), mas essencial para métricas. |

**PCA Opcional (Redução de Dimensionalidade):**
```python
if use_pca:
    pca = PCA(n_components=2)  # Reduz para 2 dimensões
    X_train_pca = pca.fit_transform(X_train_scaled)
```
- **Por que PCA?** SVM é O(n² a n³) em número de amostras. Com muitas features, treino fica lento. PCA reduz features mantendo variância.
- **Trade-off:** Perda de informação. 2 componentes podem capturar 80% da variância, mas 20% é perdido.

---

#### **3. Métricas de Avaliação (`util_metrics.py`)**

```python
metrics = {
    'accuracy': accuracy_score(y_true, y_pred),
    'precision': precision_score(y_true, y_pred, average='macro'),
    'recall': recall_score(y_true, y_pred, average='macro'),
    'f1_score': f1_score(y_true, y_pred, average='macro'),
    'roc_auc': roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
}
```

**Por que múltiplas métricas?**

| Métrica | O que mede | Quando usar |
|---------|------------|-------------|
| **Acurácia** | % de predições corretas | Classes balanceadas. **Cuidado:** 90% acurácia em dataset com 90% classe A = modelo inútil (prediz sempre A). |
| **Precisão** | % de positivos previstos que são realmente positivos | Falsos positivos são caros. Ex: spam (marcar email legítimo como spam). |
| **Recall** | % de positivos reais que foram identificados | Falsos negativos são caros. Ex: diagnóstico de câncer (não detectar doença). |
| **F1-Score** | Média harmônica de Precisão e Recall | Trade-off entre FP e FN. Classes desbalanceadas. |
| **ROC-AUC** | Área sob curva ROC (varia de 0 a 1) | Avalia performance em diferentes thresholds. 1.0 = perfeito, 0.5 = aleatório. |


---


### **Parte 3: Algoritmo Genético**

```bash
make part3
```

*(Implementação e decisões técnicas serão adicionadas)*

---

## Resultados e Comparações

### Visualizar Resultados

```bash
make results
```

Este comando exibe:
- **Relatório comparativo** (`comparison_report.txt`) com métricas de todos os algoritmos
- **Benchmark CSV** (`benchmark_results.csv`) com dados tabulares
- **Matrizes de confusão** (`.png`) para análise visual de erros
- **Gráfico de seleção de K** (KNN) mostrando por que K=5 foi escolhido
- **Visualização da Decision Tree** (estrutura completa da árvore)

### Interpretação dos Resultados

**Comparação Típica (Water Quality Dataset):**

| Algoritmo | Acurácia Teste | Tempo Treino | Overfitting | Interpretabilidade |
|-----------|----------------|--------------|-------------|-------------------|
| **Decision Tree** | ~98.7% | ~0.45s | Baixo (0.008%) | ⭐⭐⭐⭐⭐ |
| **KNN** | ~92.6% | ~0.23s | Médio (2.8%) | ⭐⭐ |
| **SVM** | ~95.0% | ~590s | Baixo (-0.07%) | ⭐ |

**Análise:**
- **Decision Tree:** Melhor acurácia e interpretabilidade. Rápida de treinar. **Escolha ideal para este dataset.**
- **KNN:** Treino rápido, mas performance inferior. Overfitting moderado (decora padrões locais do treino).
- **SVM:** Boa acurácia, mas treino MUITO lento (10 minutos para 100k linhas). Modelo "black box" (difícil interpretar).

**Por que Decision Tree venceu aqui?**
1. **Dataset tabular com features numéricas:** Árvores são naturalmente adequadas para dados estruturados.
2. **Interações não-lineares:** Água potável depende de combinações (ex: pH alto + Cloro baixo = não potável).
3. **Dados limpos:** Poucas outliers extremas, então robustez do KNN não foi necessária.
4. **Interpretabilidade requerida:** Podemos extrair regras (ex: "SE Hardness < 200 E Solids > 20000 ENTÃO potável").

---

## Estrutura de Arquivos Gerados

```
data/processed/
├── benchmark_results.csv           # Métricas de todos os treinos
├── comparison_report.txt           # Relatório formatado
├── confusion_matrix_dt_100000.png  # Matriz de confusão (Decision Tree)
├── confusion_matrix_knn_100000.png # Matriz de confusão (KNN)
├── confusion_matrix_svm_100000.png # Matriz de confusão (SVM)
├── decision_tree_visualization_100000.png  # Árvore completa
├── knn_k_selection.png             # Gráfico de seleção de K
├── X_train.csv                     # Features de treino (não escalonadas)
├── X_train_scaled.csv              # Features de treino (escalonadas)
├── X_test.csv                      # Features de teste (não escalonadas)
├── X_test_scaled.csv               # Features de teste (escalonadas)
├── y_train.csv                     # Labels de treino
└── y_test.csv                      # Labels de teste
```


---

## Referências 

- **"Introduction to Machine Learning with Python"** - Andreas Müller & Sarah Guido
  - Capítulos 2-3: Pré-processamento e validação
  - Capítulo 5: Decision Trees, KNN, SVM
- **"Hands-On Machine Learning"** - Aurélien Géron
  - Capítulo 6: Decision Trees e Random Forests
  - Capítulo 5: SVM e Kernel Trick

### Documentação Oficial
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [pandas Documentation](https://pandas.pydata.org/docs/)

---

## Licença e Autoria

**Disciplina:** Inteligência Artificial (2025/2)  
**Professor:** Tiago Alves de Oliveira  
**Instituição:** Cefet-MG Divinópolis
**Alunos:** João Pedro Rodrigues Silva e Samuel Silva Gomes

---

## Contato

Para dúvidas ou sugestões, abra uma issue no repositório ou entre em contato com os autores.

