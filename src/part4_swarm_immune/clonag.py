# CLONALG (Clonal Selection Algorithm) para o Problema da Mochila
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

sys.path.append(str(Path(__file__).parent.parent / 'part3_ga'))
sys.path.append(str(Path(__file__).parent))
from ga import GA
from aco import Item, GAWrapper, KnapsackACO


class CLONALG:
    def __init__(self, items: List[Item], capacity: int, pop_size: int = 50, 
                 n_select: int = 10, beta: float = 1.5, n_random: int = 30, 
                 generations: int = 100):
        self.items = items
        self.capacity = capacity
        self.pop_size = pop_size
        self.n_select = n_select
        self.beta = beta
        self.n_random = n_random
        self.generations = generations
        self.n_items = len(items)
        self.best_values_history = []
        self.avg_values_history = []
        self.diversity_history = []

    def _evaluate(self, solution: List[int]) -> Tuple[int, int]:
        value = sum(self.items[i].value for i, bit in enumerate(solution) if bit == 1)
        weight = sum(self.items[i].weight for i, bit in enumerate(solution) if bit == 1)
        return value, weight

    def _repair(self, solution: List[int]) -> List[int]:
        sol = solution[:]
        _, weight = self._evaluate(sol)
        if weight <= self.capacity:
            return sol
        
        included = [(i, self.items[i].efficiency) for i, bit in enumerate(sol) if bit == 1]
        included.sort(key=lambda x: x[1])
        
        for idx, _ in included:
            sol[idx] = 0
            _, weight = self._evaluate(sol)
            if weight <= self.capacity:
                break
        return sol

    def _mutate(self, solution: List[int], rate: float) -> List[int]:
        return [1 - bit if np.random.random() < rate else bit for bit in solution]

    def _hamming_distance(self, sol1: List[int], sol2: List[int]) -> int:
        """Calcula distância de Hamming entre duas soluções."""
        return sum(b1 != b2 for b1, b2 in zip(sol1, sol2))

    def _apply_hypermutation(self, population: List[Tuple], threshold: int = 3) -> List[Tuple]:
        """Aplica hipermutação em indivíduos muito similares ao melhor."""
        if not population:
            return population
        
        best_sol = population[0][0]
        hypermutated = []
        
        for sol, val, wt in population:
            distance = self._hamming_distance(sol, best_sol)
            # Se muito similar ao melhor (< threshold bits de diferença), hipermuta
            if distance < threshold and len(hypermutated) < self.pop_size // 4:
                # Taxa de mutação muito alta (0.4-0.6)
                rate = np.random.uniform(0.4, 0.6)
                new_sol = self._mutate(sol[:], rate)
                new_sol = self._repair(new_sol)
                new_val, new_wt = self._evaluate(new_sol)
                hypermutated.append((new_sol, new_val, new_wt))
            else:
                hypermutated.append((sol, val, wt))
        
        return hypermutated

    def run(self, verbose: bool = True) -> Tuple[List[int], int, int]:
        start_time = time.time()
        
        population = []
        for _ in range(self.pop_size):
            sol = [np.random.randint(0, 2) for _ in range(self.n_items)]
            sol = self._repair(sol)
            val, wt = self._evaluate(sol)
            population.append((sol, val, wt))
        
        best_solution, best_value, best_weight = None, 0, 0

        for gen in range(self.generations):
            population.sort(key=lambda x: x[1], reverse=True)
            selected = population[:self.n_select]
            
            if selected[0][1] > best_value:
                best_solution, best_value, best_weight = selected[0]
            
            clones = []
            for rank, (sol, val, _) in enumerate(selected, start=1):
                n_clones = max(1, int(self.beta * self.pop_size / rank))
                max_val = selected[0][1]
                
                for _ in range(n_clones):
                    # Taxa de mutação adaptativa: aumenta para indivíduos piores
                    val_norm = val / (max_val + 1e-9) if max_val > 0 else 0
                    # Taxa base maior (0.5) e varia entre 0.1 (melhor) e 0.5 (pior)
                    rate = 0.5 * (1.0 - val_norm * 0.8)
                    clone = self._mutate(sol[:], rate)
                    clone = self._repair(clone)
                    cv, cw = self._evaluate(clone)
                    clones.append((clone, cv, cw))
            
            randoms = []
            for _ in range(self.n_random):
                sol = [np.random.randint(0, 2) for _ in range(self.n_items)]
                sol = self._repair(sol)
                val, wt = self._evaluate(sol)
                randoms.append((sol, val, wt))
            
            population = (population + clones + randoms)
            population.sort(key=lambda x: x[1], reverse=True)
            
            # Aplicar hipermutação para aumentar diversidade
            population = self._apply_hypermutation(population, threshold=5)
            population.sort(key=lambda x: x[1], reverse=True)
            
            # Seleção com penalização de clones: manter top 50% por fitness, 
            # e 50% por diversidade (remover duplicatas exatas)
            elite_size = self.pop_size // 2
            diverse_size = self.pop_size - elite_size
            
            # Elite: melhores por fitness
            elite = population[:elite_size]
            
            # Diversidade: selecionar indivíduos únicos do restante
            remaining = population[elite_size:]
            unique_sols = {}
            for sol, val, wt in remaining:
                sol_tuple = tuple(sol)
                if sol_tuple not in unique_sols:
                    unique_sols[sol_tuple] = (sol, val, wt)
            
            diverse = list(unique_sols.values())[:diverse_size]
            
            # Se não houver diversidade suficiente, gerar novos aleatórios
            while len(diverse) < diverse_size:
                sol = [np.random.randint(0, 2) for _ in range(self.n_items)]
                sol = self._repair(sol)
                val, wt = self._evaluate(sol)
                diverse.append((sol, val, wt))
            
            population = elite + diverse
            
            diversity = len(set(tuple(sol) for sol, _, _ in population))
            self.best_values_history.append(best_value)
            self.avg_values_history.append(np.mean([val for _, val, _ in population]))
            self.diversity_history.append(diversity)
            
            if verbose and (gen + 1) % 20 == 0:
                diversity = len(set(tuple(sol) for sol, _, _ in population))
                print(f"Geração {gen+1}/{self.generations} | Melhor: {best_value} | "
                      f"Médio: {np.mean([val for _, val, _ in population]):.1f} | "
                      f"Diversidade: {diversity}/{self.pop_size} | "
                      f"Tempo: {time.time()-start_time:.2f}s")
        
        if verbose:
            print(f"\n{'='*70}\nRESULTADO FINAL CLONALG\n{'='*70}")
            print(f"Valor: {best_value} | Peso: {best_weight}/{self.capacity} "
                  f"({best_weight/self.capacity*100:.1f}%) | Itens: {sum(best_solution)} | "
                  f"Tempo: {time.time()-start_time:.2f}s\n{'='*70}")
        
        solution_indices = [i for i, bit in enumerate(best_solution) if bit == 1]
        return solution_indices, best_value, best_weight


def plot_comparison_all(aco, ga, clonalg, aco_time, ga_time, clonalg_time):
    # Figura 1: Convergência e Evolução do Valor Médio
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1.1 Convergência do melhor valor
    axes1[0].plot(aco.best_values_history, 'b-', label='ACO', linewidth=2.5)
    axes1[0].plot(ga.best_values_history, 'r-', label='GA', linewidth=2.5)
    axes1[0].plot(clonalg.best_values_history, 'g-', label='CLONALG', linewidth=2.5)
    axes1[0].set_xlabel('Iteração/Geração', fontsize=11)
    axes1[0].set_ylabel('Melhor Valor', fontsize=11)
    axes1[0].set_title('Convergência: ACO vs GA vs CLONALG', fontsize=12, fontweight='bold')
    axes1[0].legend(fontsize=10)
    axes1[0].grid(True, alpha=0.3)
    
    # 1.2 Evolução do valor médio
    axes1[1].plot(aco.avg_values_history, 'b--', label='ACO', linewidth=2.5)
    axes1[1].plot(ga.avg_values_history, 'r--', label='GA', linewidth=2.5)
    axes1[1].plot(clonalg.avg_values_history, 'g--', label='CLONALG', linewidth=2.5)
    axes1[1].set_xlabel('Iteração/Geração', fontsize=11)
    axes1[1].set_ylabel('Valor Médio', fontsize=11)
    axes1[1].set_title('Evolução do Valor Médio', fontsize=12, fontweight='bold')
    axes1[1].legend(fontsize=10)
    axes1[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/processed/fig1_convergencia_media.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figura 2: Diversidade e Taxa de Convergência
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    
    # 2.1 Evolução da diversidade
    axes2[0].plot(aco.diversity_history, 'b-', label='ACO', linewidth=2.5)
    axes2[0].plot(ga.diversity_history if hasattr(ga, 'diversity_history') else [100]*len(ga.best_values_history), 
                  'r-', label='GA', linewidth=2.5, alpha=0.5)
    axes2[0].plot(clonalg.diversity_history, 'g-', label='CLONALG', linewidth=2.5)
    axes2[0].set_xlabel('Iteração/Geração', fontsize=11)
    axes2[0].set_ylabel('Soluções Únicas', fontsize=11)
    axes2[0].set_title('Evolução da Diversidade Populacional', fontsize=12, fontweight='bold')
    axes2[0].legend(fontsize=10)
    axes2[0].grid(True, alpha=0.3)
    
    # 2.2 Taxa de convergência (improvement rate)
    def compute_improvement_rate(history, window=10):
        rates = []
        for i in range(len(history)):
            if i < window:
                rates.append(0)
            else:
                improvement = history[i] - history[i-window]
                rates.append(improvement / window)
        return rates
    
    aco_rate = compute_improvement_rate(aco.best_values_history)
    ga_rate = compute_improvement_rate(ga.best_values_history)
    clonalg_rate = compute_improvement_rate(clonalg.best_values_history)
    
    axes2[1].plot(aco_rate, 'b-', label='ACO', linewidth=2.5, alpha=0.7)
    axes2[1].plot(ga_rate, 'r-', label='GA', linewidth=2.5, alpha=0.7)
    axes2[1].plot(clonalg_rate, 'g-', label='CLONALG', linewidth=2.5, alpha=0.7)
    axes2[1].set_xlabel('Iteração/Geração', fontsize=11)
    axes2[1].set_ylabel('Taxa de Melhoria (valor/iteração)', fontsize=11)
    axes2[1].set_title('Taxa de Convergência (janela=10)', fontsize=12, fontweight='bold')
    axes2[1].legend(fontsize=10)
    axes2[1].grid(True, alpha=0.3)
    axes2[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/processed/fig2_diversidade_taxa.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figura 3: Eficiência e Comparação Final
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    
    algorithms = ['ACO', 'GA', 'CLONALG']
    final_values = [aco.best_values_history[-1], ga.best_values_history[-1], clonalg.best_values_history[-1]]
    times = [aco_time, ga_time, clonalg_time]
    efficiency = [v/t for v, t in zip(final_values, times)]
    colors = ['blue', 'red', 'green']
    
    # 3.1 Eficiência (valor final / tempo)
    bars = axes3[0].bar(algorithms, efficiency, color=colors, alpha=0.6, edgecolor='black', linewidth=1.5)
    axes3[0].set_ylabel('Eficiência (Valor/Segundo)', fontsize=11)
    axes3[0].set_title('Eficiência Computacional', fontsize=12, fontweight='bold')
    axes3[0].grid(True, alpha=0.3, axis='y')
    
    # Adicionar valores nas barras
    for bar, eff in zip(bars, efficiency):
        height = bar.get_height()
        axes3[0].text(bar.get_x() + bar.get_width()/2., height,
                     f'{eff:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 3.2 Comparação final (valor e tempo)
    x = np.arange(len(algorithms))
    width = 0.35
    
    ax2 = axes3[1]
    bars1 = ax2.bar(x - width/2, final_values, width, label='Valor Final', 
                    color=colors, alpha=0.6, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Valor da Solução', color='black', fontsize=11)
    ax2.set_xlabel('Algoritmo', fontsize=11)
    ax2.set_title('Comparação: Qualidade vs Tempo', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms)
    ax2.tick_params(axis='y')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Adicionar valores nas barras
    for bar, val in zip(bars1, final_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3 = ax2.twinx()
    bars2 = ax3.bar(x + width/2, times, width, label='Tempo (s)', 
                    color='orange', alpha=0.5, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Tempo de Execução (s)', color='orange', fontsize=11)
    ax3.tick_params(axis='y', labelcolor='orange')
    
    # Adicionar valores nas barras de tempo
    for bar, t in zip(bars2, times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{t:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold', color='orange')
    
    # Combinar legendas
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('data/processed/fig3_eficiencia_comparacao.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nGráficos salvos:")
    print(f"  - data/processed/fig1_convergencia_media.png")
    print(f"  - data/processed/fig2_diversidade_taxa.png")
    print(f"  - data/processed/fig3_eficiencia_comparacao.png")


if __name__ == "__main__":
    # IMPORTANTE: Usar sementes diferentes para cada algoritmo!
    n_items = 50
    
    # Gerar itens com seed fixa para reprodutibilidade
    np.random.seed(42)
    items = [Item(i, np.random.randint(10, 101), np.random.randint(5, 51), 0) 
             for i in range(n_items)]
    for item in items:
        item.efficiency = item.value / item.weight
    capacity = sum(item.weight for item in items) // 2

    print(f"\n{'='*70}\nCOMPARAÇÃO: ACO vs GA vs CLONALG - PROBLEMA DA MOCHILA\n{'='*70}")
    print(f"Instância: {n_items} itens | Capacidade: {capacity}\n{'-'*70}")

    print("ALGORITMO GENÉTICO (GA)\n" + "-"*70)
    np.random.seed(100)  # Seed diferente para GA
    import time as time_module
    ga_start = time_module.time()
    ga = GAWrapper(items, capacity, population_size=100, generations=100)
    ga_sol, ga_val, ga_wt = ga.run(verbose=True)
    ga_time = time_module.time() - ga_start

    print(f"\n{'-'*70}\nANT COLONY OPTIMIZATION (ACO)\n{'-'*70}")
    np.random.seed(200)  # Seed diferente para ACO
    aco_start = time_module.time()
    aco = KnapsackACO(items, capacity, n_ants=30, n_iterations=100)
    aco_sol, aco_val, aco_wt = aco.run(verbose=True)
    aco_time = time_module.time() - aco_start

    print(f"\n{'-'*70}\nCLONAL SELECTION ALGORITHM (CLONALG)\n{'-'*70}")
    np.random.seed(300)  # Seed diferente para CLONALG
    clonalg_start = time_module.time()
    clonalg = CLONALG(items, capacity, pop_size=80, n_select=15, beta=1.2, 
                     n_random=30, generations=100)
    clonalg_sol, clonalg_val, clonalg_wt = clonalg.run(verbose=True)
    clonalg_time = time_module.time() - clonalg_start

    print(f"\n{'='*70}\nCOMPARAÇÃO FINAL\n{'='*70}")
    results = [
        ('GA', ga_val, ga_wt, len(ga_sol)),
        ('ACO', aco_val, aco_wt, len(aco_sol)),
        ('CLONALG', clonalg_val, clonalg_wt, len(clonalg_sol))
    ]
    results.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, val, wt, n) in enumerate(results, 1):
        symbol = '🥇' if i == 1 else '🥈' if i == 2 else '🥉'
        print(f"{symbol} {name}: Valor={val:>4} | Peso={wt:>3}/{capacity} ({wt/capacity*100:.1f}%) | Itens={n}")
    
    best_val = results[0][1]
    print(f"\n{'='*70}")
    for name, val, _, _ in results[1:]:
        diff = ((best_val - val) / val * 100) if val > 0 else 0
        print(f"Melhor que {name}: +{diff:.2f}%")
    
    print(f"{'='*70}")
    plot_comparison_all(aco, ga, clonalg, aco_time, ga_time, clonalg_time)
