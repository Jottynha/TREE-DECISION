import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import sklearn.preprocessing as prep
from sklearn.impute import SimpleImputer
import os

def preprocess_data(input_path, output_dir='data/processed/', test_size=0.2, random_state=42):
    """
    <============================================================>
                PRÉ-PROCESSAMENTO COMPLETO DE DADOS
    <============================================================>
    [1] VALORES NULOS (Missing Data):
        Preenche dados faltantes para evitar erros
        Numéricos: usa MEDIANA (robusto a outliers)
        Categóricos: usa MODA (valor mais frequente)
    [2] LABEL ENCODING (String → Número):
        Transforma categorias em números ordinais
        ['low', 'medium', 'high'] → [0, 1, 2]
        ['loam', 'sandy', 'clay'] → [0, 1, 2]
        NECESSÁRIO: ML trabalha apenas com números
    [3] ONE-HOT ENCODING (Categorias → Colunas Binárias):
        Cria colunas separadas para cada categoria
        Exemplo: 'cor'=['red','blue'] → 'cor_red'=[1,0], 'cor_blue'=[0,1]
        Útil quando NÃO há ordem natural entre categorias
    [4] ESCALONAMENTO (StandardScaler):
        Padroniza features: média=0, desvio padrão=1
        Transforma [10, 20, 30] e [1000, 2000, 3000] para mesma escala
        ESSENCIAL para KNN e SVM (sensíveis à magnitude)
        NÃO necessário para árvores de decisão
    [5] DIVISÃO ESTRATIFICADA (Stratified Split):
        Mantém proporção das classes em treino/teste
        Se dataset tem 70% classe A, treino terá 70% classe A
        Evita desbalanceamento entre conjuntos
    [6] VALIDAÇÃO CRUZADA K-FOLD:
        Divide dados em K partes (folds)
        Treina K vezes, usando parte diferente para validar
        Reduz overfitting e dá estimativa mais confiável
    <============================================================>
    """
    # Cria diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    # ETAPA 1: CARREGAMENTO DOS DADOS
    df = pd.read_csv(input_path)
    print("\n" + "="*70)
    print("ETAPA 1: CARREGAMENTO DOS DADOS")
    print("="*70)
    print(f"Dataset original: {df.shape[0]} linhas × {df.shape[1]} colunas")
    
    # AMOSTRAGEM ALEATÓRIA: Pegar apenas 10k linhas (se dataset for maior)
    if df.shape[0] > 10000:
        print(f"\nDataset muito grande! Pegando amostra aleatória de 10.000 linhas...")
        df = df.sample(n=10000, random_state=42)  # random_state garante reprodutibilidade
        print(f"Amostra selecionada: {df.shape[0]} linhas × {df.shape[1]} colunas")
        print("   (Amostragem aleatória garante representatividade das classes)")
    else:
        print(f"\n✓ Dataset possui {df.shape[0]} linhas (menor que 10k, usando todas)")
    
    print(f"\nColunas encontradas: {list(df.columns)}")
    print(f"\nValores nulos por coluna:\n{df.isnull().sum()}")

    # ETAPA 2: SEPARAR FEATURES (X) E TARGET (y)
    print("\n" + "="*70)
    print("ETAPA 2: SEPARAÇÃO DE FEATURES E TARGET")
    print("="*70)
    X = df.iloc[:, :-1]  # Todas as colunas exceto a última
    y = df.iloc[:, -1]   # Última coluna é o target
    print(f"Features (X): {X.shape[1]} colunas")
    print(f"Target (y): '{y.name}' com {y.nunique()} classes distintas")

    # ETAPA 3: TRATAMENTO DE VALORES NULOS
    print("\n" + "="*70)
    print("ETAPA 3: TRATAMENTO DE VALORES NULOS")
    print("="*70)
    # 3.1 - Colunas numéricas: preencher com MEDIANA
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        imputer_num = SimpleImputer(strategy='median')
        X[numerical_cols] = imputer_num.fit_transform(X[numerical_cols])
        print(f"Colunas numéricas preenchidas com MEDIANA: {list(numerical_cols)}")
    # 3.2 - Colunas categóricas: preencher com MODA
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        X[categorical_cols] = imputer_cat.fit_transform(X[categorical_cols])
        print(f"Colunas categóricas preenchidas com MODA: {list(categorical_cols)}")
    print(f"\nValores nulos após tratamento: {X.isnull().sum().sum()} (deve ser 0)")
    
    # ETAPA 4: LABEL ENCODING (String → Número)
    print("\n" + "="*70)
    print("ETAPA 4: LABEL ENCODING (Transformar Strings em Números)")
    print("="*70)
    print("Label Encoding converte categorias em valores numéricos ordinais.\n")
    label_encoders = {}  # Guardar encoders para uso futuro
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        
        # Mostrar mapeamento
        mapping = {original: encoded for original, encoded in zip(le.classes_, range(len(le.classes_)))}
        print(f"'{col}': {mapping}")
    
    # ETAPA 5: ONE-HOT ENCODING (opcional)
    print("\n" + "="*70)
    print("ETAPA 5: ONE-HOT ENCODING")
    print("="*70)
    print("Útil quando NÃO há ordem natural entre categorias.\n")
    
    # Aqui depois nos pode escolher quais colunas aplicar one-hot
    # Por padrão, vou aplicar em todas as categóricas já convertidas
    # Se não quiser one-hot, só descomentar as linhas abaixo
    
    # X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    # print(f"Após one-hot encoding: {X_encoded.shape}")
    
    # Por enquanto, vou usar apenas Label Encoding (mais simples)
    X_encoded = X.copy()
    print(f"Usando apenas Label Encoding (sem one-hot): {X_encoded.shape}")
    print(f"Colunas finais: {list(X_encoded.columns)}")
    
    # ETAPA 6: CODIFICAR TARGET (y)
    print("\n" + "="*70)
    print("ETAPA 6: CODIFICAR TARGET")
    print("="*70)
    if y.dtype == 'object':
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)
        mapping_target = {original: encoded for original, encoded in zip(le_target.classes_, range(len(le_target.classes_)))}
        print(f"Target codificado: {mapping_target}")
    else:
        y_encoded = y.values
        print(f"Target já é numérico")

    # ETAPA 7: DIVISÃO TREINO/TESTE ESTRATIFICADA
    print("\n" + "="*70)
    print("ETAPA 7: DIVISÃO TREINO/TESTE ESTRATIFICADA")
    print("="*70)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y_encoded  # ESTRATIFICAÇÃO: mantém proporção das classes
    )
    print(f"Conjunto de TREINO: {X_train.shape[0]} amostras ({(1-test_size)*100:.0f}%)")
    print(f"Conjunto de TESTE:  {X_test.shape[0]} amostras ({test_size*100:.0f}%)")
    # Mostrar distribuição das classes
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    print(f"\nDistribuição das classes:")
    print(f"   TREINO: {dict(zip(unique_train, counts_train))}")
    print(f"   TESTE:  {dict(zip(unique_test, counts_test))}")
    
    # ETAPA 8: ESCALONAMENTO (StandardScaler)
    print("\n" + "="*70)
    print("ETAPA 8: ESCALONAMENTO (StandardScaler)")
    print("="*70)
<<<<<<< Updated upstream
    #scaler = prep.StandardScaler()
    scaler = prep.MinMaxScaler()
=======
    scaler = prep.StandardScaler()
>>>>>>> Stashed changes
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ETAPA 9: SALVAR DADOS PROCESSADOS
    print("\n" + "="*70)
    print("ETAPA 9: SALVANDO DADOS PROCESSADOS")
    print("="*70)
    # Salvando versão ESCALONADA (para KNN/SVM)
    pd.DataFrame(X_train_scaled, columns=X_encoded.columns).to_csv(
        f'{output_dir}X_train_scaled.csv', index=False
    )
    pd.DataFrame(X_test_scaled, columns=X_encoded.columns).to_csv(
        f'{output_dir}X_test_scaled.csv', index=False
    )
    print(f"Salvos: X_train_scaled.csv e X_test_scaled.csv (para KNN/SVM)")
    # Salvando versão NÃO ESCALONADA (para árvores de decisão)
    X_train.to_csv(f'{output_dir}X_train.csv', index=False)
    X_test.to_csv(f'{output_dir}X_test.csv', index=False)
    print(f"Salvos: X_train.csv e X_test.csv (para Árvores de Decisão)")
    # Salvando targets
    pd.DataFrame(y_train, columns=['target']).to_csv(
        f'{output_dir}y_train.csv', index=False
    )
    pd.DataFrame(y_test, columns=['target']).to_csv(
        f'{output_dir}y_test.csv', index=False
    )
    print(f"Salvos: y_train.csv e y_test.csv")  
    print(f"\nTodos os arquivos salvos em: '{output_dir}'")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoders, X


def get_kfold_splits(n_splits=5, random_state=42):
    """
    VALIDAÇÃO CRUZADA K-FOLD ESTRATIFICADA
        Divide dados em K partes (folds)
        Treina K vezes, cada vez usando 1 fold para validação
        Estratificada: mantém proporção das classes em cada fold
        Reduz overfitting e dá estimativa mais confiável do modelo
    Exemplo com K=5:
    Fold 1: [Treino: 80%] [Val: 20%] ← fold 1
    Fold 2: [Treino: 80%] [Val: 20%] ← fold 2
    Fold 3: [Treino: 80%] [Val: 20%] ← fold 3
    Fold 4: [Treino: 80%] [Val: 20%] ← fold 4
    Fold 5: [Treino: 80%] [Val: 20%] ← fold 5
    Resultado final: média das 5 acurácias
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


if __name__ == "__main__":
    # Trampo atoa de deixar isso aqui bonito
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*15 + "INICIANDO PRÉ-PROCESSAMENTO" + " "*22 + "║")
    print("╚" + "="*68 + "╝")
    input_file = 'data/raw/plant_growth_data.csv'
    X_train, X_test, y_train, y_test, scaler, encoders = preprocess_data(input_file)
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*18 + "PRÉ-PROCESSAMENTO CONCLUÍDO" + " "*16 + "║")
    print("╚" + "="*68 + "╝")

    # DEMONSTRAÇÃO: VALIDAÇÃO CRUZADA K-FOLD
    print("\n" + "="*70)
    print("DEMONSTRAÇÃO: VALIDAÇÃO CRUZADA K-FOLD (5 Folds)")
    print("="*70)
    kfold = get_kfold_splits(n_splits=5)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train), 1):
        print(f"Fold {fold}: Treino={len(train_idx)} amostras, Validação={len(val_idx)} amostras")