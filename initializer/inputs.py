lines_info = {
    "4": {
        "cp1": {
            "name": "Xuejiadao",
            "latlon": (120.187, 35.917),
            "depot_id": "d1_4",
            "depot_capacity": 90
        },
        "cp2": {
            "name": "Lingshanwei",
            "latlon": (120.118, 35.917),
            "depot_id": "d2_4",
            "depot_capacity": 61
        },
        "cp_distance_km": 24,
        "num_trips": 290
    },
    "59": {
        "cp1": {
            "name": "Anzi",
            "latlon": (120.233, 35.977),
            "depot_id": "d1_59",
            "depot_capacity": 120
        },
        "cp2": {
            "name": "Jimiya",
            "latlon": (120.155, 35.923),
            "depot_id": "d2_59",
            "depot_capacity": 11
        },
        "cp_distance_km": 11.4,
        "num_trips": 100
    },
    "60": {
        "cp1": {
            "name": "Anzi",
            "latlon": (120.233, 35.977),
            "depot_id": "d1_59",
            "depot_capacity": 120
        },
        "cp2": {
            "name": "Xingguangdao",
            "latlon": (120.103, 35.896),
            "depot_id": "d2_60",
            "depot_capacity": 10
        },
        "cp_distance_km": 19,
        "num_trips": 120
    }
}

cp_depot_distances = {
    "cp1_l4": {
        "d1_4": 0.0,
        "d2_60": 3.24,
        "d1_59": 4.08,
        "d2_59": 14.76,
        "d2_4": 28.8
    },
    "cp2_l4": {
        "d2_4": 0.0,
        "d2_59": 5.64,
        "d1_59": 16.32,
        "d2_60": 23.04,
        "d1_4": 28.8
    },
    "cp1_l59": {
        "d1_59": 0.0,        
        "d1_4": 4.08,
        "d2_60": 6.60,        
        "d2_59": 13.68,
        "d2_4": 15.84
    },
    "cp2_l59": {
        "d2_59": 0.0,        
        "d2_4": 5.64,
        "d1_59": 13.68,
        "d1_4": 14.76,
        "d2_60": 22.80
    },
    "cp1_l60": {
        "d2_59": 0.0,
        "d2_4": 15.84,        
        "d1_59": 13.68,
        "d1_4": 4.08,
        "d2_60": 6.6
    },
    "cp2_l60": {
        "d2_60": 0.0,        
        "d1_4": 3.24,
        "d1_59": 6.60,
        "d2_59": 22.80,
        "d2_4": 23.04
    }
}


depots = {
    "d1_4": {
        "line": ["4"],
        "location": "Xuejiadao",
        "capacity": 90,
        "departed": 0
    },
    "d2_4": {
        "line": ["4"],
        "location": "Lingshanwei",
        "capacity": 61,
        "departed": 0
    },
    "d1_59": {
        "line": ["59", "60"],  # Compartilhado
        "location": "Anzi",
        "capacity": 120,
        "departed": 0
    },
    "d2_59": {
        "line": ["59"],
        "location": "Jimiya",
        "capacity": 11,
        "departed": 0
    },
    "d2_60": {
        "line": ["60"],
        "location": "Xingguangdao",
        "capacity": 10,
        "departed": 0
    }
}

problem_params = {
    "fixed_cost": 700,                   # RMB por veículo
    "waiting_cost_per_min": 0.7,         # RMB/min
    "deadhead_cost_per_min": 1.6,        # RMB/min
    "travel_cost_per_min": 1.6,          # RMB/min
    "charging_cost_per_event": 35,       # RMB por carga

    # Especificações do veículo elétrico
    "battery_capacity_kWh": 133.79,
    "soc_lower_bound": 0.30,             # 30%
    "charging_rate_kW_per_hour": 30,     # 30kW por hora
    "consumption_rate_kWh_per_hour": 15.6,  # Consumo médio

    # Autonomia máxima calculada no artigo (~360 min)
    "max_drive_time_min": 360,

    # Parâmetros do Column Generation
    "theta_min": 0.7,     # para arredondamento
    "cg_stop_delta": 50,  # Zmin
    "cg_max_no_improve": 10,  # I
    "cg_max_columns_added": 5,  # K
}
