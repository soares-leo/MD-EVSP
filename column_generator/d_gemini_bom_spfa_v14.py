# Imports originais e novos
from initializer.inputs import *
from initializer.utils import calculate_cost
from collections import deque, namedtuple, defaultdict
from datetime import timedelta, datetime
import math
import time

# Estrutura otimizada para dados das arestas
EdgeData = namedtuple('EdgeData', ['dist', 'time', 'reduced_cost'])

# Funções auxiliares para conversão de tempo
def datetime_to_minutes(dt):
    """Converte datetime para minutos desde meia-noite."""
    if dt is None:
        return None
    return dt.hour * 60 + dt.minute + dt.second / 60.0

def minutes_to_datetime(minutes, reference_date=None):
    """Converte minutos de volta para datetime."""
    if minutes is None:
        return None
    if reference_date is None:
        # Usa uma data de referência padrão se não fornecida
        reference_date = datetime.now().date()
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    secs = int((minutes % 1) * 60)
    return datetime.combine(reference_date, datetime.min.time()) + timedelta(hours=hours, minutes=mins, seconds=secs)

# A classe Label agora usa float para current_time
class Label:
    __slots__ = [
        'node', 'source_trip', 'total_weight', 'total_dist', 'total_time',
        'dist_since_charge', 'time_since_charge', 'current_time', 'path',
        'total_dh_dist', 'total_dh_time', 'total_travel_dist', 
        'total_travel_time', 'total_wait_time', 'total_charge_time'
    ]
    
    def __init__(self, node, source_trip, total_weight, total_dist, total_time,
                 dist_since_charge, time_since_charge, current_time, path,
                 total_dh_dist, total_dh_time, total_travel_dist, 
                 total_travel_time, total_wait_time, total_charge_time):
        self.node = node
        self.source_trip = source_trip
        self.total_weight = total_weight
        self.total_dist = total_dist
        self.total_time = total_time
        self.dist_since_charge = dist_since_charge
        self.time_since_charge = time_since_charge
        self.current_time = current_time  # Agora é float (minutos)
        self.path = path
        self.total_dh_dist = total_dh_dist
        self.total_dh_time = total_dh_time
        self.total_travel_dist = total_travel_dist
        self.total_travel_time = total_travel_time
        self.total_wait_time = total_wait_time
        self.total_charge_time = total_charge_time
    
    def __eq__(self, other):
        """Requerido para a verificação de pertencimento."""
        if not isinstance(other, Label):
            return False
        return (self.node == other.node and 
                self.source_trip == other.source_trip and
                self.path == other.path)
    
    def __hash__(self):
        """Requerido para usar objetos Label em um set para consulta O(1)."""
        return hash((self.node, self.source_trip, tuple(self.path)))

def run_spfa(graph, source_node, D, T_d, dh_df, duals, last_route_number=None, filter_graph=True):
    """
    Algoritmo SPFA otimizado com cache para encontrar caminhos mínimos com restrições.
    Mantém a corretude do algoritmo original visitando todos os caminhos válidos.
    """
    # Pré-filtragem do grafo
    if filter_graph:
        valid_nodes = {node for node in graph.nodes() 
                       if node.startswith(("l", "c")) or node == source_node}
        graph = graph.subgraph(valid_nodes)
    
    # ========== OTIMIZAÇÕES DE CACHE ==========
    #print("Building caches...")
    cache_start = time.time()
    
    # Cache de atributos dos nós com conversão de datetime para minutos
    node_attrs = {}
    reference_date = None  # Será definido com base no primeiro datetime encontrado
    
    for node in graph.nodes():
        attrs = dict(graph.nodes[node])
        
        # Converte start_time para minutos se existir
        if 'start_time' in attrs and attrs['start_time'] is not None:
            if reference_date is None:
                reference_date = attrs['start_time'].date()
            attrs['start_time_minutes'] = datetime_to_minutes(attrs['start_time'])
        
        node_attrs[node] = attrs
    
    # Cache de vizinhos válidos (pré-filtra nós depot)
    neighbors_cache = {}
    for node in graph.nodes():
        valid_neighbors = [v for v in graph.neighbors(node) if not v.startswith("d")]
        neighbors_cache[node] = valid_neighbors
    
    # Cache de dados das arestas usando namedtuple para acesso mais rápido
    edge_cache = {}
    for u in graph.nodes():
        for v in graph.neighbors(u):
            if not v.startswith("d"):
                data = graph[u][v]
                edge_cache[(u, v)] = EdgeData(
                    dist=data.get('dist', 0),
                    time=data.get('time', 0),
                    reduced_cost=data.get('reduced_cost', 0)
                )
    
    # Pré-cálculo dos nós de viagem
    trip_nodes = {n for n, d in node_attrs.items() if d.get('type') == 'T'}
    
    cache_time = time.time() - cache_start
    #print(f"Cache building completed in {cache_time:.2f} seconds")
    
    # ========== INICIALIZAÇÃO ==========
    # Rótulo inicial
    start = Label(
        node=source_node, source_trip=None, total_weight=0.0,
        total_dist=0.0, total_time=0.0, dist_since_charge=0.0,
        time_since_charge=0.0, current_time=None, path=[source_node],
        total_dh_dist=0.0, total_dh_time=0.0, total_travel_dist=0.0,
        total_travel_time=0.0, total_wait_time=0.0, total_charge_time=0.0
    )
    
    best_labels = {}
    queue = deque([start])
    queue_set = {start}  # Set auxiliar para consulta O(1)
    
    #print(f"Finding shortest paths for source {source_node}...")
    spfa_start = time.time()
    
    iterations = 0
    
    # ========== LOOP PRINCIPAL SPFA COM DEQUE (MANTÉM CORRETUDE) ==========
    while queue:
        u_label = queue.popleft()
        queue_set.remove(u_label)
        
        u = u_label.node
        iterations += 1
        need_recharge = False
        
        # Processa todos os vizinhos usando cache
        for v in neighbors_cache.get(u, []):
            
            # ========== PROCESSAMENTO DE ESTAÇÃO DE RECARGA ==========
            if v.startswith("c"):
                if not need_recharge or u.startswith(("d", "c")):
                    continue
                elif u.startswith("l"):
                    edge_data = edge_cache.get((u, v))
                    if edge_data:
                        new_label = _process_charging_station_optimized(
                            u, v, u_label, edge_data, T_d
                        )
                        if new_label:
                            prev = best_labels.get(v)
                            if prev is None or new_label.total_weight < prev.total_weight:
                                best_labels[v] = new_label
                                if new_label not in queue_set:
                                    queue.append(new_label)
                                    queue_set.add(new_label)
                                need_recharge = False
                continue
            
            # ========== PROCESSAMENTO DE NÓ DE VIAGEM ==========
            if v.startswith("l"):
                v_attrs = node_attrs[v]
                edge_data = edge_cache.get((u, v))
                if not edge_data:
                    continue
                
                # Usa tempo em minutos para comparações mais rápidas
                v_start_time_minutes = v_attrs.get("start_time_minutes")
                if v_start_time_minutes is None:
                    continue
                
                # Verificações de tempo para nós de recarga
                if u.startswith("c"):
                    if u_label.current_time is not None:
                        if v_start_time_minutes < u_label.current_time + edge_data.time:
                            continue
                
                if need_recharge:
                    continue
                
                # Determina a primeira viagem
                if len(u_label.path) == 1:
                    first_trip = v
                else:
                    first_trip = u_label.source_trip
                
                # Usa dados cacheados da aresta
                dh_dist = edge_data.dist
                dh_time = edge_data.time
                
                # Cálculo do tempo atual (agora em minutos float)
                if u_label.current_time is None:
                    clock = v_start_time_minutes - dh_time
                else:
                    clock = u_label.current_time
                    if clock + dh_time > v_start_time_minutes:
                        continue
                
                if u.startswith("c"):
                    if v_start_time_minutes < clock + dh_time:
                        continue
                
                # Cálculo do tempo de espera (operação simples com floats)
                wait_time = v_start_time_minutes - (clock + dh_time)
                
                travel_time = v_attrs["planned_travel_time"]
                new_total_time = (
                    u_label.total_time + dh_time + wait_time + travel_time
                )
                
                # Verificação de restrição de tempo
                if new_total_time >= T_d:
                    continue
                
                # Cálculo de distância
                travel_dist = v_attrs.get('travel_dist')
                new_dist_since_charge = u_label.dist_since_charge + dh_dist + travel_dist
                
                # Verificação de necessidade de recarga
                if new_dist_since_charge >= D:
                    need_recharge = True
                    continue
                
                # Usa custo reduzido cacheado
                weight = edge_data.reduced_cost
                
                # Cria novo rótulo (current_time agora é float)
                travel_label = Label(
                    node=v,
                    source_trip=first_trip,
                    total_weight=u_label.total_weight + weight,
                    total_dist=u_label.total_dist + dh_dist + travel_dist,
                    total_time=new_total_time,
                    dist_since_charge=new_dist_since_charge,
                    time_since_charge=u_label.time_since_charge + dh_time + travel_time,
                    current_time=clock + dh_time + wait_time + travel_time,  # Float em minutos
                    path=u_label.path + [v],
                    total_dh_dist=u_label.total_dh_dist + dh_dist,
                    total_dh_time=u_label.total_dh_time + dh_time,
                    total_travel_dist=u_label.total_travel_dist + travel_dist,
                    total_travel_time=u_label.total_travel_time + travel_time,
                    total_wait_time=u_label.total_wait_time + wait_time,
                    total_charge_time=u_label.total_charge_time
                )
                
                # Atualiza se encontrou caminho melhor
                prev = best_labels.get(v)
                if prev is None or travel_label.total_weight < prev.total_weight:
                    best_labels[v] = travel_label
                    if travel_label not in queue_set:
                        queue.append(travel_label)
                        queue_set.add(travel_label)
    
    #print(f"Shortest paths finding process is done. Total iterations: {iterations}")
    spfa_runtime = time.time() - spfa_start
    #print(f"Time spent finding paths: {spfa_runtime:.2f} seconds.")
    
    # ========== CORREÇÃO DE RÓTULOS ==========
    #print("Label correcting process started...")
    label_corr_start = time.time()
    
    trip_routes = _build_trip_routes_optimized(
        trip_nodes, best_labels, graph, edge_cache, duals, reference_date
    )
    
    #print("Label correcting process is done.")
    label_corr_runtime = time.time() - label_corr_start
    #print(f"Time spent correcting labels: {label_corr_runtime:.2f} seconds.")
    
    return trip_routes

def _process_charging_station_optimized(u, v, u_label, edge_data, T_d):
    """Processa a transição para uma estação de recarga usando dados cacheados."""
    weight_cs = edge_data.reduced_cost
    dist_cs = edge_data.dist
    time_cs = edge_data.time
    
    # Cálculo do tempo de recarga
    charge_minutes = (15.6 * ((u_label.dist_since_charge + dist_cs) / 20) / 30) * 60
    
    # Agora simplesmente soma floats em vez de criar timedelta
    new_clock = (u_label.current_time + time_cs + charge_minutes 
                 if u_label.current_time is not None else None)
    
    charge_label = Label(
        node=v,
        source_trip=u_label.source_trip,
        total_weight=u_label.total_weight + weight_cs,
        total_dist=u_label.total_dist + dist_cs,
        total_time=u_label.total_time + time_cs + charge_minutes,
        dist_since_charge=0.0,
        time_since_charge=0.0,
        current_time=new_clock,  # Float em minutos
        path=u_label.path + [v],
        total_dh_dist=u_label.total_dh_dist + dist_cs,
        total_dh_time=u_label.total_dh_time + time_cs,
        total_travel_dist=u_label.total_travel_dist,
        total_travel_time=u_label.total_travel_time,
        total_wait_time=u_label.total_wait_time,
        total_charge_time=u_label.total_charge_time + charge_minutes
    )
    
    if charge_label.total_time >= T_d:
        return None
    
    return charge_label

def _build_trip_routes_optimized(trip_nodes, best_labels, graph, edge_cache, duals, reference_date=None):
    """Constrói as rotas de viagem usando caches otimizados."""
    
    # Agrupa todos os rótulos pela sua viagem de origem em uma única passagem
    labels_by_source_trip = defaultdict(list)
    for label in best_labels.values():
        if len(label.path) > 1:
            source_trip = label.path[1]  # A primeira viagem na rota
            labels_by_source_trip[source_trip].append(label)
    
    trip_routes = {}
    
    # Itera sobre os grupos já formados
    for trip_id, trip_labels in labels_by_source_trip.items():
        best_cols = []
        
        for a in trip_labels:
            a_full_path = a.path + [a.path[0]]
            a_rec_times = sum(1 for x in a.path if x.startswith("c"))
            
            # Pula se o penúltimo nó é uma estação de recarga
            if a_full_path[-2].startswith("c"):
                continue
            
            # Primeiro tenta usar o cache, depois o grafo original se necessário
            return_edge = edge_cache.get((a.path[0], a_full_path[-2]))
            if return_edge:
                return_time = return_edge.time
            elif graph.has_edge(a.path[0], a_full_path[-2]):
                return_time = graph[a.path[0]][a_full_path[-2]]["time"]
            else:
                continue
            
            a_dh_time = a.total_dh_time + return_time
            
            # Obtém custo reduzido da última aresta
            a_last_trip = a.path[-1]
            last_edge = edge_cache.get((a_last_trip, a.path[0]))
            if last_edge:
                a_beta = last_edge.reduced_cost
            elif graph.has_edge(a_last_trip, a.path[0]):
                a_beta = graph[a_last_trip][a.path[0]]["reduced_cost"]
            else:
                continue
            
            a_final_weight = a.total_weight + a_beta
            
            # Calcula custo total
            a_cost = calculate_cost(
                a_rec_times, 
                a.total_travel_time, 
                a_dh_time, 
                a.total_wait_time,
            )
            
            # Calcula custo reduzido
            only_trips = [t for t in a_full_path if t.startswith("l")]
            trip_duals_sum = sum(duals["alpha"][graph.nodes[t]['id']] for t in only_trips)
            a_reduced_cost = a_cost - trip_duals_sum + duals["beta"][a.path[0]]
            
            # Converte current_time de volta para datetime para o output
            final_time_datetime = minutes_to_datetime(a.current_time, reference_date) if a.current_time is not None else None
            
            route = {
                "Path": a_full_path,
                "Cost": a_cost,
                "ReducedCost": a_reduced_cost,
                "Data": {
                    "total_dh_dist": a.total_dh_dist,
                    "total_dh_time": a.total_dh_time,
                    "total_travel_dist": a.total_travel_dist,
                    "total_travel_time": a.total_travel_time,
                    "total_wait_time": a.total_wait_time,
                    "final_time": final_time_datetime,  # Convertido de volta para datetime
                    "total_charge_time": a.total_charge_time
                }
            }
            
            best_cols.append(route)
        
        trip_routes[trip_id] = best_cols
    
    return trip_routes