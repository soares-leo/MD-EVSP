from datetime import datetime, timedelta
import random
import pandas as pd


def generate_timetable(cp_id, num_trips, start_time="05:50", min_interval=5, max_interval=15,
                       interval_bias=8, travel_time_range=(70, 75)):
    """
    Gera um timetable sintético para um CP (ponto de controle), incluindo os intervalos entre viagens.

    Args:
        cp_id (str): identificador do CP.
        num_trips (int): número de partidas a gerar.
        start_time (str): hora da primeira viagem.
        min_interval (int): intervalo mínimo entre viagens.
        max_interval (int): intervalo máximo.
        interval_bias (int): valor central mais provável.
        travel_time_range (tuple): tempo de viagem (min, max)

    Returns:
        pd.DataFrame com columns = ['trip_id', 'cp', 'departure_time', 'departure_interval', 'planned_travel_time']
    """
    current_time = datetime.strptime(start_time, "%H:%M")
    trips = []

    for i in range(1, num_trips + 1):
        if i == 1:
            interval = None
        else:
            interval = int(random.gauss(interval_bias, 2))
            interval = max(min_interval, min(interval, max_interval))
            current_time += timedelta(minutes=interval)

        travel_time = random.randint(*travel_time_range)

        trips.append({
            'trip_id': f"{cp_id}_{i}",
            'cp': cp_id,
            'departure_time': current_time.strftime("%H:%M:%S"),
            'departure_interval': interval,
            'planned_travel_time': travel_time
        })

    return pd.DataFrame(trips)
