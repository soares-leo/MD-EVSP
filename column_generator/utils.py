def add_reduced_cost_info(graph, duals, dh_times_df):
    for u, v, data in graph.edges(data=True):
        if graph.nodes[v]["type"] == "T":
            dh_cost = data["time"] * 1.6
            reduced_cost = dh_cost - duals["alpha"][graph.nodes[v]['id']]
        elif graph.nodes[v]["type"] == "K":
            dh_cost = data["time"] * 1.6
            reduced_cost = dh_cost + duals["beta"][v]
        else:
            j = v.replace("c", "d")
            end_cp = graph.nodes[u]["end_cp"]
            dh_time = float(dh_times_df.loc[end_cp, j])
            reduced_cost = dh_time * 1.6
        graph.edges[u, v]["reduced_cost"] = reduced_cost
    return graph