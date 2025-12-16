# ACO (Ant Colony Optimization) para o Problema da Mochila
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

sys.path.append(str(Path(__file__).parent.parent / 'part3_ga'))
from ga import GA


@dataclass
class Item:
    id: int
    value: int
    weight: int
    efficiency: float

class KnapsackACO:
    def __init__(self, items: List[Item], capacity: int, n_ants: int = 30, 
                 n_iterations: int = 100, alpha: float = 1.0, beta: float = 2.0, 
                 rho: float = 0.5, Q: float = 100):
        self.items = items
        self.capacity = capacity
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.pheromone = np.ones(len(items))
        self.heuristic = np.array([item.efficiency for item in items])
        self.best_values_history = []
        self.avg_values_history = []
        self.diversity_history = []
        self.diversity_history = []

    def construct_solution(self) -> Tuple[List[int], int, int]:
        solution, total_value, total_weight = [], 0, 0
        available = list(range(len(self.items)))
        
        while available:
            probs = np.zeros(len(available))
            for idx, item_id in enumerate(available):
                item = self.items[item_id]
                if total_weight + item.weight <= self.capacity:
                    probs[idx] = (self.pheromone[item_id] ** self.alpha) * (self.heuristic[item_id] ** self.beta)
            
            if probs.sum() == 0:
                break
            
            probs /= probs.sum()
            selected = np.random.choice(len(available), p=probs)
            item_id = available.pop(selected)
            item = self.items[item_id]
            
            if total_weight + item.weight <= self.capacity:
                solution.append(item_id)
                total_value += item.value
                total_weight += item.weight
        
        return solution, total_value, total_weight

    def run(self, verbose: bool = True) -> Tuple[List[int], int, int]:
        best_solution, best_value, best_weight = None, 0, 0
        start_time = time.time()

        for iteration in range(self.n_iterations):
            solutions = [self.construct_solution() for _ in range(self.n_ants)]
            values = [val for _, val, _ in solutions]
            
            best_iter = max(solutions, key=lambda x: x[1])
            if best_iter[1] > best_value:
                best_solution, best_value, best_weight = best_iter
            
            self.pheromone *= (1 - self.rho)
            # Depositar feromônio apenas nas top 30% soluções (elitismo)
            solutions_sorted = sorted(solutions, key=lambda x: x[1], reverse=True)
            n_elite = max(1, int(0.3 * len(solutions)))
            for sol, val, _ in solutions_sorted[:n_elite]:
                deposit = self.Q * val / (self.capacity * n_elite)  # Normalizar
                for item_id in sol:
                    self.pheromone[item_id] += deposit
            
            diversity = len(set(tuple(sorted(sol)) for sol, _, _ in solutions))
            self.best_values_history.append(best_value)
            self.avg_values_history.append(np.mean(values))
            self.diversity_history.append(diversity)
            
            if verbose and (iteration + 1) % 20 == 0:
                diversity = len(set(tuple(sol) for sol, _, _ in solutions))
                print(f"Iteração {iteration+1}/{self.n_iterations} | Melhor: {best_value} | "
                      f"Médio: {np.mean(values):.1f} | Diversidade: {diversity}/{self.n_ants} | "
                      f"Tempo: {time.time()-start_time:.2f}s")
        
        if verbose:
            print(f"\n{'='*70}\nRESULTADO FINAL ACO\n{'='*70}")
            print(f"Valor: {best_value} | Peso: {best_weight}/{self.capacity} "
                  f"({best_weight/self.capacity*100:.1f}%) | Itens: {len(best_solution)} | "
                  f"Tempo: {time.time()-start_time:.2f}s\n{'='*70}")
        
        return best_solution, best_value, best_weight


class GAWrapper:
    def __init__(self, items: List[Item], capacity: int, population_size: int = 100, 
                 generations: int = 100):
        self.items = items
        self.capacity = capacity
        self.items_tuples = [(item.weight, item.value) for item in items]
        self.ga = GA(len(items), max(item.weight for item in items), 
                    max(item.value for item in items), capacity, 
                    population_size, generations, 0.01)
        self.best_values_history = []
        self.avg_values_history = []

    def run(self, verbose: bool = True) -> Tuple[List[int], int, int]:
        start_time = time.time()
        genes = self.ga.gerar_lista_genes(self.ga.tamPopulacao)
        best_solution, best_value, best_weight = [], 0, 0

        for gen in range(self.ga.numGeracoes):
            fitness = self.ga.gerar_lista_fitness(genes, self.capacity, self.items_tuples)
            probs = self.ga.porcentagens(fitness)
            
            best_idx = fitness.index(max(fitness))
            if fitness[best_idx] > best_value:
                gene = genes[best_idx]
                best_solution = [i for i, bit in enumerate(gene) if bit == 1]
                best_value = sum(self.items[i].value for i in best_solution)
                best_weight = sum(self.items[i].weight for i in best_solution)
            
            self.best_values_history.append(best_value)
            self.avg_values_history.append(sum(fitness) / len(fitness))
            
            if verbose and (gen + 1) % 20 == 0:
                print(f"Geração {gen+1}/{self.ga.numGeracoes} | Melhor: {best_value} | "
                      f"Médio: {sum(fitness)/len(fitness):.1f} | Tempo: {time.time()-start_time:.2f}s")
            
            filhos = []
            for _ in range(self.ga.tamPopulacao // 2):
                p1 = self.ga.gerar_pai(genes, probs)
                p2 = self.ga.gerar_pai(genes, probs)
                f1, f2 = self.ga.gerar_filhos(p1, p2)
                filhos.extend([f1, f2])
            genes = self.ga.mutacao(filhos, self.ga.taxaMutacao)
        
        if verbose:
            print(f"\n{'='*70}\nRESULTADO FINAL GA\n{'='*70}")
            print(f"Valor: {best_value} | Peso: {best_weight}/{self.capacity} "
                  f"({best_weight/self.capacity*100:.1f}%) | Itens: {len(best_solution)} | "
                  f"Tempo: {time.time()-start_time:.2f}s\n{'='*70}")
        
        return best_solution, best_value, best_weight


def plot_comparison(aco: KnapsackACO, ga: GAWrapper, items: List[Item], capacity: int,
                   aco_sol: List[int], ga_sol: List[int]):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(aco.best_values_history, 'b-', label='ACO', linewidth=2)
    axes[0].plot(ga.best_values_history, 'r-', label='GA', linewidth=2)
    axes[0].set_xlabel('Iteração/Geração')
    axes[0].set_ylabel('Melhor Valor')
    axes[0].set_title('Convergência')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    colors = ['green' if i in aco_sol else 'gray' for i in range(len(items))]
    axes[1].bar(range(len(items)), aco.pheromone, color=colors, alpha=0.7)
    axes[1].set_xlabel('Item')
    axes[1].set_ylabel('Feromônio')
    axes[1].set_title('Distribuição de Feromônios ACO')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('data/processed/aco_ga_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    np.random.seed(42)
    n_items = 50
    items = [Item(i, np.random.randint(10, 101), np.random.randint(5, 51), 0) 
             for i in range(n_items)]
    for item in items:
        item.efficiency = item.value / item.weight
    capacity = sum(item.weight for item in items) // 2

    print(f"\n{'='*70}\nCOMPARAÇÃO: ACO vs GA - PROBLEMA DA MOCHILA\n{'='*70}")
    print(f"Instância: {n_items} itens | Capacidade: {capacity}\n{'-'*70}")

    print("ALGORITMO GENÉTICO (GA)\n" + "-"*70)
    ga = GAWrapper(items, capacity, population_size=100, generations=100)
    ga_sol, ga_val, ga_wt = ga.run(verbose=True)

    print(f"\n{'-'*70}\nANT COLONY OPTIMIZATION (ACO)\n{'-'*70}")
    aco = KnapsackACO(items, capacity, n_ants=30, n_iterations=100)
    aco_sol, aco_val, aco_wt = aco.run(verbose=True)

    diff = ((aco_val - ga_val) / ga_val * 100) if ga_val > 0 else 0
    print(f"\n{'='*70}\nCOMPARAÇÃO FINAL\n{'='*70}")
    print(f"GA:  Valor={ga_val:>4} | Peso={ga_wt:>3}/{capacity} | Itens={len(ga_sol)}")
    print(f"ACO: Valor={aco_val:>4} | Peso={aco_wt:>3}/{capacity} | Itens={len(aco_sol)}")
    print(f"Diferença: {diff:>+6.2f}% | Resultado: {'ACO melhor' if aco_val > ga_val else 'GA melhor' if ga_val > aco_val else 'Empate'}")
    
    common = len(set(aco_sol) & set(ga_sol))
    print(f"Itens em comum: {common}/{min(len(aco_sol), len(ga_sol))}\n{'='*70}")
    
    plot_comparison(aco, ga, items, capacity, aco_sol, ga_sol)
