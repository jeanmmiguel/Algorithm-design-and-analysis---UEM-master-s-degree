import numpy as np
import random
import math
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from itertools import combinations
from typing import List, Tuple

# Função auxiliar para calcular o comprimento total da árvore
def calculate_tree_length(points: np.ndarray, connections: List[Tuple[int, int]]) -> float:
    """
    Calcula o comprimento total da árvore dada seus pontos e conexões.

    :param points: Array Numpy dos pontos.
    :param connections: Lista de conexões (arestas) entre os pontos.
    :return: Comprimento total da árvore.
    """
    length = 0
    for i, j in connections:
        length += np.linalg.norm(points[i] - points[j])
    return length

# Função auxiliar para calcular a matriz de distâncias
def calculate_distance_matrix(points: np.ndarray) -> np.ndarray:
    """
    Calcula a matriz de distâncias para um conjunto de pontos.

    :param points: Array Numpy dos pontos.
    :return: Matriz de distâncias como um array Numpy.
    """
    return distance_matrix(points, points)

# Função auxiliar para encontrar a Árvore Geradora Mínima (MST) usando o algoritmo de Kruskal
def find_mst(points: np.ndarray) -> List[Tuple[int, int]]:
    """
    Encontra a Árvore Geradora Mínima (MST) para um conjunto de pontos usando o algoritmo de Kruskal.

    :param points: Array Numpy dos pontos.
    :return: Lista de conexões (arestas) da MST.
    """
    dist_matrix = calculate_distance_matrix(points)
    edges = [(i, j, dist_matrix[i, j]) for i, j in combinations(range(len(points)), 2)]
    edges.sort(key=lambda x: x[2])

    parent = list(range(len(points)))

    def find(x: int) -> int:
        if parent[x] == x:
            return x
        parent[x] = find(parent[x])
        return parent[x]

    mst = []
    for u, v, weight in edges:
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            parent[root_u] = root_v
            mst.append((u, v))
    
    return mst

# Função principal para gerar uma solução inicial (MST + Pontos de Steiner)
def initial_solution(terminals: np.ndarray, num_steiner_points: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Gera uma solução inicial calculando a MST dos pontos terminais e potencialmente adicionando pontos de Steiner.
    
    :param terminals: Array Numpy dos pontos terminais.
    :param num_steiner_points: Número de pontos de Steiner a serem adicionados.
    :return: Tupla de pontos (terminais + pontos de Steiner) e as conexões da MST.
    """
    # Combinando terminais e potenciais pontos de Steiner (se usados)
    all_points = terminals  # Supondo que os pontos de Steiner não sejam adicionados inicialmente

    # Encontra a MST para os pontos combinados
    mst = find_mst(all_points)
    
    return all_points, mst

# Função para perturbar a solução atual
def perturb_solution(points: np.ndarray, connections: List[Tuple[int, int]], terminals: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Perturba a solução atual adicionando, removendo ou movendo um ponto de Steiner e recalcula a MST.

    :param points: Array Numpy dos pontos atuais (terminais + pontos de Steiner).
    :param connections: Lista das conexões atuais da MST.
    :param terminals: Array Numpy dos pontos terminais.
    :return: Tupla de pontos atualizados e novas conexões da MST.
    """
    num_points = len(points)
    num_terminals = len(terminals)
    num_steiner_points = num_points - num_terminals

    perturbed = False

    while not perturbed:
        choice = random.choice(['add', 'remove', 'move'])
        
        if choice == 'add' and num_steiner_points < num_terminals - 2:
            # Adicionar um novo ponto de Steiner dentro dos limites dos pontos existentes
            new_point = np.random.uniform(low=points.min(axis=0), high=points.max(axis=0))
            points = np.vstack((points, new_point))
            perturbed = True
            
        elif choice == 'remove' and num_steiner_points > 0:
            # Remover um ponto de Steiner aleatório
            remove_index = random.choice(range(num_terminals, num_points))
            points = np.delete(points, remove_index, axis=0)
            perturbed = True
            
        elif choice == 'move' and num_steiner_points > 0:
            # Mover um ponto de Steiner aleatório para uma nova posição
            random_index = random.choice(range(num_terminals, num_points))
            old_point = points[random_index].copy()
            points[random_index] = old_point + np.random.uniform(low=-0.1, high=0.1, size=2)
            perturbed = True

    # Recalcula a MST para os pontos perturbados
    new_connections = find_mst(points)
    
    return points, new_connections

# Função para realizar o algoritmo de Tempera Simulada (Simulated Annealing)
def simulated_annealing(
    terminals: np.ndarray,
    num_steiner_points: int,
    initial_temp: float,
    cooling_rate: float,
    max_iterations: int,
    save_interval: int = 1,
    record_interations: bool = False
) -> Tuple[np.ndarray, List[Tuple[int, int]], float]:
    """
    Realiza o algoritmo de Tempera Simulada para encontrar uma solução otimizada para o problema da MST.

    :param terminals: Array Numpy dos pontos terminais.
    :param num_steiner_points: Número de pontos de Steiner a serem adicionados.
    :param initial_temp: Temperatura inicial para o processo de tempera.
    :param cooling_rate: Taxa de resfriamento para a diminuição da temperatura.
    :param max_iterations: Número máximo de iterações a serem realizadas.
    :param save_interval: Intervalo em que a solução atual é salva e plotada.
    :param record_interations: Booleano para decidir se as iterações devem ser plotadas e salvas.
    :return: Tupla dos melhores pontos, suas conexões e o comprimento da MST.
    """
    points, connections = initial_solution(terminals, num_steiner_points)
    current_length = calculate_tree_length(points, connections)
    current_points = points
    current_connections = connections
    
    best_length = current_length
    best_points = current_points.copy()
    best_connections = current_connections.copy()
    
    temperature = initial_temp
    
    history = {
        'iteration': [],
        'temperature': [],
        'length': [],
        'accepted': [],
        'worse_accepted': []
    }

    for iteration in range(max_iterations):
        new_points, new_connections = perturb_solution(current_points.copy(), current_connections.copy(), terminals)
        new_length = calculate_tree_length(new_points, new_connections)
        print(f'Iteração: {iteration}, Temperatura: {temperature}, Comprimento: {new_length}')

        delta_length = new_length - current_length
        accepted = False
        worse_accepted = False
        
        # Salva e plota em intervalos especificados
        if iteration % save_interval == 0 and record_interations:
            plot_solution(
                new_points, new_connections,  
                current_points, current_connections,         
                best_points, best_connections,      
                terminals,
                title=f'Iteração {iteration} - Busca Atual: {new_length:.2f}, Melhor Atual: {current_length:.2f}, Melhor Geral: {best_length:.2f}',
                save_path=f'plots/step_{iteration}_comparison.png'
            )
        if delta_length < 0 or math.exp(-delta_length / temperature) > random.random():
            current_points = new_points
            current_connections = new_connections
            current_length = new_length
            accepted = True
            worse_accepted = delta_length > 0

            if new_length < best_length:
                best_length = new_length
                best_points = new_points
                best_connections = new_connections

        # Registro do histórico
        history['iteration'].append(iteration)
        history['temperature'].append(temperature)
        history['length'].append(current_length)
        history['accepted'].append(accepted)
        history['worse_accepted'].append(worse_accepted)

        temperature *= cooling_rate
        
        if temperature < 1e-10:
            break
    
    # Plotar o histórico de aceitação
    plot_acceptance_history(history)
    plot_solution(
        current_points, current_connections,  
        best_points, best_connections,      
        best_points, best_connections,      
        terminals,
        title=f'Solução Final - Comprimento: {best_length:.2f}',
        save_path='plots/final_comparison.png'
    )
    return best_points, best_connections, best_length


# Função para plotar a solução atual, melhor atual e melhor geral lado a lado
def plot_solution(
    current_points: np.ndarray,
    current_connections: List[Tuple[int, int]],
    best_current_points: np.ndarray,
    best_current_connections: List[Tuple[int, int]],
    best_overall_points: np.ndarray,
    best_overall_connections: List[Tuple[int, int]],
    terminals: np.ndarray,
    title: str,
    save_path: str = None
) -> None:
    """
    Plota a solução atual, a melhor solução atual e a melhor solução geral lado a lado.

    :param current_points: Array Numpy dos pontos atuais.
    :param current_connections: Lista das conexões atuais da MST.
    :param best_current_points: Array Numpy dos melhores pontos atuais.
    :param best_current_connections: Lista das melhores conexões atuais da MST.
    :param best_overall_points: Array Numpy dos melhores pontos gerais.
    :param best_overall_connections: Lista das melhores conexões gerais da MST.
    :param terminals: Array Numpy dos pontos terminais.
    :param title: Título do plot.
    :param save_path: Caminho para salvar a imagem do plot. Se None, mostra o plot.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plotar a solução da busca atual
    axes[0].set_title('Busca Atual')
    for i, j in current_connections:
        axes[0].plot([current_points[i][0], current_points[j][0]], [current_points[i][1], current_points[j][1]], 'b-')
    axes[0].plot(terminals[:, 0], terminals[:, 1], 'ro', label='Terminais')
    axes[0].plot(current_points[len(terminals):, 0], current_points[len(terminals):, 1], 'go', label='Pontos de Steiner')
    axes[0].legend()

    # Plotar a melhor solução atual
    axes[1].set_title('Melhor Atual')
    for i, j in best_current_connections:
        axes[1].plot([best_current_points[i][0], best_current_points[j][0]], [best_current_points[i][1], best_current_points[j][1]], 'b-')
    axes[1].plot(terminals[:, 0], terminals[:, 1], 'ro', label='Terminais')
    axes[1].plot(best_current_points[len(terminals):, 0], best_current_points[len(terminals):, 1], 'go', label='Pontos de Steiner')
    axes[1].legend()

    # Plotar a melhor solução geral
    axes[2].set_title('Melhor Geral')
    for i, j in best_overall_connections:
        axes[2].plot([best_overall_points[i][0], best_overall_points[j][0]], [best_overall_points[i][1], best_overall_points[j][1]], 'b-')
    axes[2].plot(terminals[:, 0], terminals[:, 1], 'ro', label='Terminais')
    axes[2].plot(best_overall_points[len(terminals):, 0], best_overall_points[len(terminals):, 1], 'go', label='Pontos de Steiner')
    axes[2].legend()

    fig.suptitle(title)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


# Função para plotar o histórico de aceitação
def plot_acceptance_history(history: dict):
    """
    Plota o histórico de aceitação mostrando a temperatura e o comprimento ao longo das iterações.
    Destaca os pontos onde uma solução pior foi aceita.

    :param history: Dicionário contendo os dados do histórico de aceitação.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    iterations = history['iteration']
    temperatures = history['temperature']
    lengths = history['length']
    accepted = history['accepted']
    worse_accepted = history['worse_accepted']

    ax1.set_xlabel('Iteração')
    ax1.set_ylabel('Temperatura', color='tab:blue')
    ax1.plot(iterations, temperatures, 'b-', label='Temperatura')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Comprimento', color='tab:red')
    ax2.plot(iterations, lengths, 'r-', label='Comprimento da MST')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # for i in range(len(iterations)):
    #     if worse_accepted[i]:
    #         ax2.plot(iterations[i], lengths[i], 'go', label='Pior Aceita' if i == 0 else "", markersize=5)

    fig.tight_layout()
    fig.legend(loc='upper left')
    plt.title('Histórico de Aceitação do Tempera Simulada')
    plt.show()

def plot_terminals(terminals: np.ndarray):
    """
    Plota os pontos terminais.

    :param terminals: Array Numpy dos pontos terminais.
    """
    plt.figure(figsize=(6, 6))
    plt.plot(terminals[:, 0], terminals[:, 1], 'ro', label='Terminais')
    plt.title('Pontos Terminais')
    plt.legend()
    plt.show()

# Definir os pontos terminais
terminals = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # 4 pontos terminais
#terminals = np.array([[-4, 3.45], [-3.7, 3.03], [-2.74, 3.47], [-2, 2], [1.22, 0.57], [1, -2], [-1.98, -3.33], [5.38, -0.63]])

plot_terminals(terminals    )
# Parâmetros para o Tempera Simulada
num_steiner_points = 2
initial_temp = 5
cooling_rate = 0.99
max_iterations = 100000

# Gerar e plotar a solução inicial
initial_points, initial_connections = initial_solution(terminals, num_steiner_points)
initial_length = calculate_tree_length(initial_points, initial_connections)
print(f'Comprimento inicial: {initial_length}')
print(f'Pontos iniciais: {initial_points}')
print(f'Conexões iniciais: {initial_connections}')

# Executar o algoritmo de Tempera Simulada
best_points, best_connections, best_length = simulated_annealing(terminals, num_steiner_points, initial_temp, cooling_rate, max_iterations, record_interations=True)
