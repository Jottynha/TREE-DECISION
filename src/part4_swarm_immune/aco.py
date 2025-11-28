# <= ACO (Ant Colony Optimization) para o Problema da Mochila (Knapsack Problem) =>
# O problema da mochila consiste em selecionar itens com valores e pesos, 
# maximizando o valor total sem exceder a capacidade da mochila.
# Adaptação do ACO:
#   - Cada formiga constrói uma solução selecionando itens
#   - Feromônios representam a "qualidade" de incluir cada item
#   - Heurística: razão valor/peso (eficiência do item)
import numpy as np
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass
from typing import List, Tuple


@dataclass # Simplificação da classe Item
class Item:
    """Representa um item da mochila"""
    id: int
    value: int
    weight: int
    efficiency: float  # valor/peso 
    def __repr__(self):
        return f"Item({self.id}: v={self.value}, w={self.weight}, eff={self.efficiency:.2f})"

class KnapsackACO:
    """ 
    items: Lista de itens disponíveis
    capacity: Capacidade máxima da mochila
    n_ants: Número de formigas por iteração
    n_iterations: Número de iterações
    alpha: Importância do feromônio (padrão: 1.0)
    beta: Importância da heurística (padrão: 2.0)
    rho: Taxa de evaporação (padrão: 0.5)
    Q: Quantidade de feromônio depositado (padrão: 100)
    """
    def __init__(self, items: List[Item], capacity: int, 
                 n_ants: int = 50, n_iterations: int = 100,
                 alpha: float = 1.0, beta: float = 2.0, 
                 rho: float = 0.5, Q: float = 100):
        
        self.items = items
        self.capacity = capacity
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha  
        self.beta = beta   
        self.rho = rho      
        self.Q = Q          
        # Inicializar feromônios (todos os itens começam com o mesmo nível)
        self.pheromone = np.ones(len(items))
        # Heurística: eficiência (valor/peso)
        self.heuristic = np.array([item.efficiency for item in items])
        # Histórico para plotar evolução
        self.best_values_history = []
        self.avg_values_history = []
        
    def construct_solution(self) -> Tuple[List[int], int, int]:
        """
        Uma formiga constrói uma solução selecionando itens probabilisticamente
        Retorna:(solução, valor_total, peso_total)
        """
        solution = []
        total_value = 0
        total_weight = 0
        available_items = list(range(len(self.items)))
        while available_items:
            # Calcular probabilidades de seleção
            probabilities = self._calculate_probabilities(available_items, total_weight)          
            if probabilities.sum() == 0:
                break  # Nenhum item viável
            # Selecionar item baseado nas probabilidades
            probabilities /= probabilities.sum()
            selected_idx = np.random.choice(len(available_items), p=probabilities)
            item_id = available_items[selected_idx]
            item = self.items[item_id]
            # Adicionar item se couber
            if total_weight + item.weight <= self.capacity:
                solution.append(item_id)
                total_value += item.value
                total_weight += item.weight
            available_items.pop(selected_idx) # Remover item da lista de disponíveis
        return solution, total_value, total_weight
    def _calculate_probabilities(self, available_items: List[int], current_weight: int) -> np.ndarray:
        """
        Calcula probabilidades de seleção para itens disponíveis
        P[i] = (pheromone[i]^alpha * heuristic[i]^beta) / sum(...)
        """
        probabilities = np.zeros(len(available_items))  
        for idx, item_id in enumerate(available_items):
            item = self.items[item_id]
            # Item não cabe na mochila
            if current_weight + item.weight > self.capacity:
                probabilities[idx] = 0
                continue
            # Fórmula ACO: tau^alpha * eta^beta
            tau = self.pheromone[item_id] ** self.alpha
            eta = self.heuristic[item_id] ** self.beta
            probabilities[idx] = tau * eta
        return probabilities
    
    def update_pheromones(self, all_solutions: List[Tuple[List[int], int, int]]):
        """
        Atualiza feromônios baseado nas soluções das formigas
        1. Evaporação: tau = tau * (1 - rho)
        2. Depósito: tau[i] += Q / total_weight (para cada item na solução)
        """
        # Evaporação
        self.pheromone *= (1 - self.rho)
        # Depósito de feromônio pelas formigas
        for solution, value, weight in all_solutions:
            # Quanto melhor a solução (maior valor), mais feromônio
            deposit = self.Q * value / self.capacity    
            for item_id in solution:
                self.pheromone[item_id] += deposit
    
    def run(self, verbose: bool = True) -> Tuple[List[int], int, int]:
        """
        Executa o algoritmo ACO
        Retorna:(melhor solução, melhor_valor, peso_total)
        """
        best_solution = None
        best_value = 0
        best_weight = 0
        start_time = time.time()
        for iteration in range(self.n_iterations):
            # Cada formiga constrói uma solução
            all_solutions = []
            iteration_values = []
            for _ in range(self.n_ants):
                solution, value, weight = self.construct_solution()
                all_solutions.append((solution, value, weight))
                iteration_values.append(value)
                # Atualizar melhor solução
                if value > best_value:
                    best_solution = solution
                    best_value = value
                    best_weight = weight
            # Atualizar feromônios
            self.update_pheromones(all_solutions)
            # Guardar estatísticas
            self.best_values_history.append(best_value)
            self.avg_values_history.append(np.mean(iteration_values))
            # Log
            if verbose and (iteration + 1) % 20 == 0:
                elapsed = time.time() - start_time
                print(f"Iteração {iteration+1}/{self.n_iterations} | "
                      f"Melhor Valor: {best_value} | "
                      f"Valor Médio: {np.mean(iteration_values):.1f} | "
                      f"Tempo: {elapsed:.2f}s")
        total_time = time.time() - start_time
        if verbose:
            print("\n" + "="*70)
            print("RESULTADO FINAL ACO")
            print("="*70)
            print(f"Melhor Valor: {best_value}")
            print(f"Peso Total: {best_weight}/{self.capacity}")
            print(f"Utilização: {best_weight/self.capacity*100:.1f}%")
            print(f"Itens Selecionados: {len(best_solution)}")
            print(f"Tempo Total: {total_time:.2f}s")
            print("="*70)
        return best_solution, best_value, best_weight


def generate_knapsack_instance(n_items: int, max_value: int = 100, 
                               max_weight: int = 50, seed: int = 42) -> Tuple[List[Item], int]:
    """
    Gera uma instância aleatória do problema da mochila
    n_items: Número de itens
    max_value: Valor máximo de um item
    max_weight: Peso máximo de um item
    seed: Seed para reprodutibilidade
    Retorna: lista_de_itens, capacidade_da_mochila)
    """
    np.random.seed(seed)
    items = []
    for i in range(n_items):
        value = np.random.randint(10, max_value + 1)
        weight = np.random.randint(5, max_weight + 1)
        efficiency = value / weight
        items.append(Item(id=i, value=value, weight=weight, efficiency=efficiency))
    # Capacidade da mochila: ~50% do peso total dos itens
    total_weight = sum(item.weight for item in items)
    capacity = int(total_weight * 0.5)    
    return items, capacity


def greedy_knapsack(items: List[Item], capacity: int) -> Tuple[List[int], int, int]:
    """
    Algoritmo Guloso (comparação por baseline): seleciona itens por eficiência (valor/peso)
    """
    # Ordenar por eficiência decrescente
    sorted_items = sorted(enumerate(items), key=lambda x: x[1].efficiency, reverse=True)
    solution = []
    total_value = 0
    total_weight = 0
    for item_id, item in sorted_items:
        if total_weight + item.weight <= capacity:
            solution.append(item_id)
            total_value += item.value
            total_weight += item.weight
    return solution, total_value, total_weight


def plot_convergence(aco: KnapsackACO, greedy_value: int, save_path: str = "data/processed/aco_convergence.png"):
    """Plota a evolução do algoritmo ACO"""
    plt.figure(figsize=(12, 6))
    iterations = range(1, len(aco.best_values_history) + 1)
    plt.plot(iterations, aco.best_values_history, 'b-', linewidth=2, label='Melhor Solução')
    plt.plot(iterations, aco.avg_values_history, 'g--', linewidth=1.5, label='Média das Formigas')
    plt.axhline(y=greedy_value, color='r', linestyle=':', linewidth=2, label='Algoritmo Guloso')
    plt.xlabel('Iteração', fontsize=12, fontweight='bold')
    plt.ylabel('Valor Total', fontsize=12, fontweight='bold')
    plt.title('Convergência do ACO - Problema da Mochila', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nGráfico de convergência salvo em: {save_path}")
    plt.close()


def plot_pheromone_distribution(aco: KnapsackACO, solution: List[int], 
                                save_path: str = "data/processed/aco_pheromone.png"):
    """Plota a distribuição de feromônios"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    # Gráfico 1: Feromônios por item
    item_ids = [item.id for item in aco.items]
    colors = ['green' if i in solution else 'gray' for i in item_ids]
    ax1.bar(item_ids, aco.pheromone, color=colors, alpha=0.7)
    ax1.set_xlabel('ID do Item', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Nível de Feromônio', fontsize=11, fontweight='bold')
    ax1.set_title('Distribuição de Feromônios (Verde = Selecionado)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    # Gráfico 2: Eficiência vs Feromônio
    efficiencies = [item.efficiency for item in aco.items]
    colors_scatter = ['green' if i in solution else 'red' for i in item_ids]
    ax2.scatter(efficiencies, aco.pheromone, c=colors_scatter, s=100, alpha=0.6)
    ax2.set_xlabel('Eficiência (Valor/Peso)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Nível de Feromônio', fontsize=11, fontweight='bold')
    ax2.set_title('Feromônio vs Eficiência (Verde = Selecionado)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico de feromônios salvo em: {save_path}")
    plt.close()


def plot_solution_comparison(items: List[Item], capacity: int,
                            aco_solution: List[int], aco_value: int, aco_weight: int,
                            greedy_solution: List[int], greedy_value: int, greedy_weight: int,
                            save_path: str = "data/processed/aco_comparison.png"):
    """Compara visualmente as soluções ACO vs Guloso"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    solutions = [
        (aco_solution, aco_value, aco_weight, "ACO", axes[0]),
        (greedy_solution, greedy_value, greedy_weight, "Guloso", axes[1])
    ]
    for solution, value, weight, name, ax in solutions:
        selected_items = [items[i] for i in solution]
        not_selected = [item for item in items if item.id not in solution]   
        # Plotar itens selecionados vs não selecionados
        if selected_items:
            sel_weights = [item.weight for item in selected_items]
            sel_values = [item.value for item in selected_items]
            ax.scatter(sel_weights, sel_values, c='green', s=100, alpha=0.7, 
                      label=f'Selecionados ({len(selected_items)})', marker='o')
        if not_selected:
            not_weights = [item.weight for item in not_selected]
            not_values = [item.value for item in not_selected]
            ax.scatter(not_weights, not_values, c='red', s=50, alpha=0.4, 
                      label=f'Não Selecionados ({len(not_selected)})', marker='x')
        ax.set_xlabel('Peso', fontsize=11, fontweight='bold')
        ax.set_ylabel('Valor', fontsize=11, fontweight='bold')
        ax.set_title(f'{name}\nValor: {value} | Peso: {weight}/{capacity} ({weight/capacity*100:.1f}%)', 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico de comparação salvo em: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ACO (ANT COLONY OPTIMIZATION) - PROBLEMA DA MOCHILA")
    print("="*70)
    n_items = 50 # Gerar instância do problema
    items, capacity = generate_knapsack_instance(n_items, max_value=100, max_weight=50, seed=42) # Gerar instância do problema
    print(f"\nInstância do Problema:")
    print(f"  Número de Itens: {n_items}")
    print(f"  Capacidade da Mochila: {capacity}")
    print(f"  Valor Total Disponível: {sum(item.value for item in items)}")
    print(f"  Peso Total Disponível: {sum(item.weight for item in items)}")
    print("\n" + "-"*70)
    print("EXECUTANDO ALGORITMO GULOSO (Baseline)")
    print("-"*70)
    greedy_start = time.time()
    greedy_solution, greedy_value, greedy_weight = greedy_knapsack(items, capacity)
    greedy_time = time.time() - greedy_start
    print(f"Valor: {greedy_value}")
    print(f"Peso: {greedy_weight}/{capacity} ({greedy_weight/capacity*100:.1f}%)")
    print(f"Itens: {len(greedy_solution)}")
    print(f"Tempo: {greedy_time:.4f}s")
    print("\n" + "-"*70)
    print("EXECUTANDO ACO")
    print("-"*70)
    aco = KnapsackACO(
        items=items,
        capacity=capacity,
        n_ants=30,
        n_iterations=100,
        alpha=1.0,      
        beta=2.0,       
        rho=0.5,        
        Q=100           
    )
    aco_solution, aco_value, aco_weight = aco.run(verbose=True)
    print("\n" + "="*70)
    print("COMPARAÇÃO: ACO vs GULOSO")
    print("="*70)
    improvement = ((aco_value - greedy_value) / greedy_value * 100) if greedy_value > 0 else 0
    print(f"ACO Valor:    {aco_value:>6} | Peso: {aco_weight:>4}/{capacity}")
    print(f"Guloso Valor: {greedy_value:>6} | Peso: {greedy_weight:>4}/{capacity}")
    print(f"Melhoria ACO: {improvement:>+6.2f}%")
    print("="*70)
    print("\nGerando visualizações...")
    plot_convergence(aco, greedy_value)
    plot_pheromone_distribution(aco, aco_solution)
    plot_solution_comparison(items, capacity, aco_solution, aco_value, aco_weight,
                            greedy_solution, greedy_value, greedy_weight)
    print("\n<= Execução concluída =>")
