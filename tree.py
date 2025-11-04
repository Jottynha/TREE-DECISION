# Código responsável por modularizar a estrutura de uma árvore de decisão;
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque

class Tree:
    """
    Classe que representa um nó de uma árvore de decisão binária;
    Adotando 'Não' como direito e 'Sim' como esquerdo;
    """
    def __init__(self, value=None):
        self.value = value
        self.left = None
        self.right = None
    def add_left(self, value):
        """Adiciona um filho à esquerda (Sim)"""
        if isinstance(value, Tree):
            self.left = value
        else:
            self.left = Tree(value)
        return self.left
    def add_right(self, value):
        """Adiciona um filho à direita (Não)"""
        if isinstance(value, Tree):
            self.right = value
        else:
            self.right = Tree(value)
        return self.right
    def is_leaf(self):
        """Verifica se o nó é uma folha (não tem filhos)"""
        return (self.left is None and self.right is None)
    def sim(self):
        """Retorna o filho esquerdo (Sim)"""
        return self.left
    def nao(self):
        """Retorna o filho direito (Não)"""
        return self.right
    def from_list(nodes):
        """
        Cria uma árvore a partir de uma lista em ordem de largura (BFS).
        None representa ausência de nó.
        
        Exemplo:
            nodes = ["Raiz", "Sim1", "Não1", "Sim2", None, "Não2", None]
            Cria:
                    Raiz
                   /    \
                Sim1    Não1
                /         \
              Sim2        Não2
        """
        if not nodes or nodes[0] is None:
            return None
        root = Tree(nodes[0])
        queue = deque([root])
        i = 1
        while queue and i < len(nodes):
            current = queue.popleft()
            # Adicionando filho esquerdo (Sim)
            if i < len(nodes) and nodes[i] is not None:
                current.left = Tree(nodes[i])
                queue.append(current.left)
            i += 1
            # Adicionando filho direito (Não)
            if i < len(nodes) and nodes[i] is not None:
                current.right = Tree(nodes[i])
                queue.append(current.right)
            i += 1
        return root
    def to_list(self):
        """
        Converte a árvore para uma lista em ordem de largura (BFS);
        Inclui None para posições vazias;
        Precisa salvar/restaurar a estrutura exata da árvore.
        """
        if not self.value:
            return []
        result = []
        queue = deque([self])
        while queue:
            current = queue.popleft()
            if current:
                result.append(current.value)
                queue.append(current.left)
                queue.append(current.right)
            else:
                result.append(None)
        # Remove None's do final
        while result and result[-1] is None:
            result.pop()
        return result
    def traverse_bfs(self):
        """
        Retorna lista de valores em ordem de largura (BFS);
        Precisa apenas listar/processar os nós existentes;
        """
        if not self.value:
            return []
        result = []
        queue = deque([self])
        while queue:
            current = queue.popleft()
            result.append(current.value)
            if current.left:
                queue.append(current.left)
            if current.right:
                queue.append(current.right)
        return result
    def traverse_dfs_preorder(self):
        """
        Retorna lista de valores em pré-ordem (DFS)
        """
        result = []
        if self.value:
            result.append(self.value)
            if self.left:
                result.extend(self.left.traverse_dfs_preorder())
            if self.right:
                result.extend(self.right.traverse_dfs_preorder())
        return result
    def get_height(self):
        """
        Retorna a altura da árvore
        """
        if self.is_leaf():
            return 0
        left_height = self.left.get_height() if self.left else 0
        right_height = self.right.get_height() if self.right else 0
        return 1 + max(left_height, right_height)
    def get_node_count(self):
        """
        Retorna o número total de nós
        """
        count = 1
        if self.left:
            count += self.left.get_node_count()
        if self.right:
            count += self.right.get_node_count()
        return count
    def find_node(self, value):
        """
        Busca um nó pelo valor (BFS)
        """
        queue = deque([self])
        while queue:
            current = queue.popleft()
            if current.value == value:
                return current
            if current.left:
                queue.append(current.left)
            if current.right:
                queue.append(current.right)
        return None
    def visualize(self, title="Árvore de Decisão", figsize=(12, 8), save_path=None):
        """
        Visualiza a árvore de decisão usando matplotlib e networkx.
        Argumentos:
            title: Título do gráfico
            figsize: Tamanho da figura (largura, altura)
            save_path: Caminho para salvar a imagem (opcional)
        """
        def build_graph(tree, graph, pos, x=0, y=0, layer=1, parent=None, direction=""):
            if tree is None:
                return graph, pos
            node_id = id(tree)
            graph.add_node(node_id, label=tree.value)
            pos[node_id] = (x, y)
            if parent is not None:
                graph.add_edge(parent, node_id, label=direction)
            width = 4 / (2 ** layer)
            if tree.left:
                build_graph(tree.left, graph, pos, x - width, y - 1, layer + 1, node_id, "Sim")
            if tree.right:
                build_graph(tree.right, graph, pos, x + width, y - 1, layer + 1, node_id, "Não")
            return graph, pos
        G = nx.DiGraph()
        positions = {}
        build_graph(self, G, positions)
        plt.figure(figsize=figsize)
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_nodes(G, positions, node_color='lightblue', node_size=3000, node_shape='o')
        nx.draw_networkx_edges(G, positions, edge_color='gray', arrows=True, arrowsize=20, width=2)
        nx.draw_networkx_labels(G, positions, labels, font_size=10, font_weight='bold')
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, positions, edge_labels, font_size=9, font_color='red')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    def print(self):
        """
        Alias para visualize()
        """
        self.visualize()
    def __str__(self):
        """
        Representação em string da árvore
        """
        return f"Árvore(valores={self.value}, altura={self.get_height()}, nós={self.get_node_count()})"
    def __repr__(self):
        return self.__str__()
    
# Exemplo de uso:
if __name__ == "__main__":
    print("=" * 60)
    print("EXEMPLO 1: Construção manual (método original)")
    print("=" * 60)
    root = Tree("É um animal?")
    root.add_left("É um mamífero?")
    root.add_right("É uma planta?")
    root.sim().add_left("É um cão?")
    root.sim().add_right("É um pássaro?")
    root.nao().add_left("É uma flor?")
    root.nao().add_right("É uma árvore?")
    print(f"Informações da árvore: {root}")
    print(f"Travessia BFS: {root.traverse_bfs()}")
    print(f"Travessia DFS: {root.traverse_dfs_preorder()}")
    print(f"Lista (formato vetor): {root.to_list()}")
    root.visualize(title="Exemplo 1: Construção Manual")
    print("\n" + "=" * 60)
    print("EXEMPLO 2: Construção a partir de vetor (BFS)")
    print("=" * 60)
    # Criando a mesma árvore usando lista
    nodes = [
        "É um animal?",
        "É um mamífero?", "É uma planta?",
        "É um cão?", "É um pássaro?", "É uma flor?", "É uma árvore?"
    ]
    root2 = Tree.from_list(nodes)
    print(f"Informações da árvore: {root2}")
    print(f"Travessia BFS: {root2.traverse_bfs()}")
    root2.visualize(title="Exemplo 2: Construção por Vetor")
    print("\n" + "=" * 60)
    print("EXEMPLO 3: Árvore com nós ausentes")
    print("=" * 60)
    # Árvore parcial com None indicando ausência de nós
    nodes_partial = [
        "Tem patas?",
        "Quantas patas?", None,  # Sem filho direito no nível 1
        "4 patas", "2 patas"      # Filhos do "Quantas patas?"
    ]
    root3 = Tree.from_list(nodes_partial)
    print(f"Informações da árvore: {root3}")
    print(f"Travessia BFS: {root3.traverse_bfs()}")
    root3.visualize(title="Exemplo 3: Árvore Parcial")
