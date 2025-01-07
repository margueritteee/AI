from collections import deque 
import heapq
from typing import Dict, List, Set, Tuple, Optional
import time


# Définition de la carte de la ville
class City:
    def __init__(self):
        self.graph = {
            'A': {'B': 4, 'C': 3},
            'B': {'A': 4, 'D': 5, 'E': 6},
            'C': {'A': 3, 'F': 5},
            'D': {'B': 5, 'G': 4},
            'E': {'B': 6, 'G': 3},
            'F': {'C': 5, 'G': 6},
            'G': {'D': 4, 'E': 3, 'F': 6}
        }
        
        self.heuristic = {
            'A': 9,
            'B': 6,
            'C': 8,
            'D': 3,
            'E': 2,
            'F': 4,
            'G': 0
        }
def calculate_path_cost(city: City, path: List[str]) -> float:
    """Calculate the total cost of a path"""
    if not path or len(path) < 2:
        return 0
    
    total_cost = 0
    for i in range(len(path) - 1):
        total_cost += city.graph[path[i]][path[i + 1]]
    return total_cost

def measure_performance(func, *args):
    """
    Measure wall-clock time and calculate path cost for a given function.
    """
    start_time = time.perf_counter()
    result = func(*args)
    elapsed_time = time.perf_counter() - start_time
    
    # Calculate cost if a valid path was found
    cost = calculate_path_cost(args[0], result) if result else float('inf')
    
    return result, elapsed_time, cost


def rechercheenlargeur(city: City, start: str, goal: str) -> Optional[List[str]]:
    """elle utilise une structure de données appelée "file" (queue), où les éléments sont traités dans l'ordre d'ajout (FIFO : First In, First Out)"""
    queue = deque([[start]]) #File pour stocker les chemins à explorer
    visited = set() #Ensemble pour suivre les nœuds visités

    while queue: #As long as there are cities left to explore, keep looping
        path = queue.popleft() #Take the first path from the queue 
        node = path[-1] #Get the last city (node) in the path to see where we are now.

        if node == goal: #if the current node is the goal we return the path that weve passed
            return path

        if node not in visited: #Si le nœud n'a pas été visité
            visited.add(node) # Marquer comme visité
            for neighbor in city.graph[node]:# Parcourir les voisins
                if neighbor not in visited: # Si le voisin n'a pas été visité
                    new_path = list(path)
                    new_path.append(neighbor) # Construire un nouveau chemin
                    queue.append(new_path) # L'ajouter à la file
    return None #ida 3gbna elihm gae w malginash goal we return none

def rechercheenprofondeur(city: City, start: str, goal: str) -> Optional[List[str]]:
    """Recherche en profondeur avec exploration des voisins par ordre alphabétique"""
    stack = [[start]]       # Une pile qui contient le chemin initial (juste le point de départ)
    visited = set()         # Un ensemble pour garder trace des nœuds déjà visités

    while stack:
        path = stack.pop()  # On prend le dernier chemin ajouté à la pile
        node = path[-1]     # Le nœud actuel est le dernier élément de ce chemin 

        if node == goal:    # Si on a atteint le but, on retourne le chemin complet
            return path

        if node not in visited:  # Si le nœud n'a pas été visité
            visited.add(node)    # On marque le nœud comme visité
            neighbors = sorted(city.graph[node], reverse=True) # On trie les voisins par ordre alphabétique
            for neighbor in neighbors:            # Pour chaque voisin trié
                if neighbor not in visited:       # Si le voisin n'a pas été visité
                    new_path = list(path)         # On crée une copie du chemin actuel
                    new_path.append(neighbor)     # On ajoute le voisin au nouveau chemin
                    stack.append(new_path)        # On ajoute ce nouveau chemin à la pile
    return None



def rechercheenprofondeurlimitée(city: City, start: str, goal: str, depth_limit: int):
    # Fonction principale qui prend la ville (graphe), point de départ, but et limite de profondeur
    
    def recursive_dls(path: List[str], depth: int) -> Optional[List[str]]:
        # Fonction récursive interne qui prend le chemin actuel et la profondeur restante
        
        if depth == -1:
            # Si on a dépassé la limite de profondeur
            return None
        
        node = path[-1]  
        # Récupère le dernier nœud du chemin actuel
        
        if node == goal:
            # Si on a trouvé le but
            return path
        
        for neighbor in city.graph[node]:
            # Pour chaque voisin du nœud actuel
            if neighbor not in path:
                # Si le voisin n'a pas déjà été visité dans ce chemin
                new_path = recursive_dls(path + [neighbor], depth - 1)
                # Appel récursif avec le nouveau chemin et profondeur-1
                if new_path:
                    # Si un chemin valide est trouvé
                    return new_path
        
        return None  # Aucun chemin trouvé à cette branche
    
    return recursive_dls([start], depth_limit)
    # Démarre la recherche avec le nœud initial et la limite de profondeur

def rechercheenprofondeuritérative(city: City, start: str, goal: str, max_depth: int):
    # Essaie différentes profondeurs de 0 jusqu'à max_depth
    for depth in range(max_depth + 1):
        # Pour chaque profondeur, essaie de trouver un chemin
        result = rechercheenprofondeurlimitée(city, start, goal, depth)
        if result:  # Si un chemin est trouvé
            return result
    return None    # Si aucun chemin n'est trouvé après toutes les profondeurs

def best_first_search(city: City, start: str, goal: str):
    # Initialisation avec une file de priorité
    # Le tuple contient (valeur_heuristique, chemin)
    pq = [(city.heuristic[start], [start])]
    
    # Ensemble des nœuds visités
    visited = set()

    while pq:  # Tant que la file n'est pas vide
        #_ est utilisé pour ignorer la valeur heuristique (car elle n’est plus nécessaire à ce stade)
        #Le chemin correspondant est stocké dans la variable path
        _, path = heapq.heappop(pq)  # heapq.heappop(pq) pour retirer l'élément ayant la plus petite valeur heuristique dans la file de priorité. 
        node = path[-1]  #récupérez le dernier nœud du chemin actuel (path) pour continuer l'exploration.

        if node == goal:  # Si on a trouvé le but
            return path

        if node not in visited:
            visited.add(node)  # Marque le nœud comme visité
            #parcourez tous les voisins du nœud actuel en utilisant le graphe (city.graph).
            for neighbor in city.graph[node]:
                #Si un voisin (neighbor) n’a pas encore été visité, vous le considérez pour exploration.
                if neighbor not in visited:
                    #Créer un nouveau chemin 
                    new_path = list(path) #copiez le chemin actuel (path) dans une nouvelle liste (new_path).
                    new_path.append(neighbor) #ajoutez le voisin (neighbor) au nouveau chemin.
                    # Ajoute le nouveau chemin avec son heuristique
                    heapq.heappush(pq, (city.heuristic[neighbor], new_path)) #ajoutez le nouveau chemin à la file de priorité (pq) avec sa valeur heuristique.
                    #Le tuple ajouté est (city.heuristic[neighbor], new_path) 
                    # new_path: Le chemin mis à jour.
                    #city.heuristic[neighbor]: La valeur heuristique estimée pour atteindre le but (goal) à partir du voisin.

def astar(city: City, start: str, goal: str):
    # Initialisation avec une file de priorité contenant : (f_score, g_score, chemin)
    # f_score = g_score + heuristique (coût total estimé)
    # g_score = coût réel depuis le départ
    pq = [(city.heuristic[start], 0, [start])]
    visited = set()

    while pq:
        _, g_score, path = heapq.heappop(pq)  # On récupère le nœud avec le plus petit f_score
        node = path[-1]  # Nœud actuel

        if node == goal:
            return path

        if node not in visited:
            visited.add(node)
            # Pour chaque voisin et sa distance
            for neighbor, distance in city.graph[node].items():
                if neighbor not in visited:
                    new_g = g_score + distance  # Nouveau coût réel
                    new_f = new_g + city.heuristic[neighbor]  # Nouveau coût total estimé
                    new_path = list(path)
                    new_path.append(neighbor)
                    heapq.heappush(pq, (new_f, new_g, new_path))

# Test des algorithmes
def test_all_algorithms():
    city = City()
    start, goal = 'A', 'G'
    
    print(f"\nRecherche de chemin de {start} à {goal}:")
    print("=" * 80)
    
    # Dictionary to store performance results
    performance_results = {}
    
    # Test BFS
    path, cpu_time, cost = measure_performance(rechercheenlargeur, city, start, goal)
    performance_results['BFS'] = {
        'path': path,
        'cpu_time': cpu_time,
        'cost': cost
    }
    
    # Test DFS
    path, cpu_time, cost = measure_performance(rechercheenprofondeur, city, start, goal)
    performance_results['DFS'] = {
        'path': path,
        'cpu_time': cpu_time,
        'cost': cost
    }
    
    # Test DLS with different limits
    for limit in [2, 3]:
        path, cpu_time, cost = measure_performance(
            rechercheenprofondeurlimitée, city, start, goal, limit
        )
        performance_results[f'DLS (limit={limit})'] = {
            'path': path,
            'cpu_time': cpu_time,
            'cost': cost
        }
    
    # Test IDS with different max depths
    for depth in [2, 3]:
        path, cpu_time, cost = measure_performance(
            rechercheenprofondeuritérative, city, start, goal, depth
        )
        performance_results[f'IDS (max_depth={depth})'] = {
            'path': path,
            'cpu_time': cpu_time,
            'cost': cost
        }
    
    # Test Best-First Search
    path, cpu_time, cost = measure_performance(best_first_search, city, start, goal)
    performance_results['Best-First Search'] = {
        'path': path,
        'cpu_time': cpu_time,
        'cost': cost
    }
    
    # Test A*
    path, cpu_time, cost = measure_performance(astar, city, start, goal)
    performance_results['A*'] = {
        'path': path,
        'cpu_time': cpu_time,
        'cost': cost
    }
    
    # Print results in a formatted table
    print("\nRésultats de performance des algorithmes:")
    print("-" * 90)
    print(f"{'Algorithme':<25} {'Chemin':<25} {'Coût':<15} {'Temps CPU (s)':<15}")
    print("-" * 90)
    
    for algo, data in performance_results.items():
        path_str = str(data['path']) if data['path'] else "Pas de chemin trouvé"
        cost_str = f"{data['cost']:.2f}" if data['cost'] != float('inf') else "∞"
        print(f"{algo:<25} {path_str:<25} {cost_str:>15} {data['cpu_time']:>14.6f}")

if __name__ == "__main__":
    test_all_algorithms()