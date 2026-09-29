import streamlit as st
import math
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import io

st.set_page_config(page_title="AI Lab 6 - Informed Search", page_icon="🤖", layout="wide")

st.title("🤖 AI Lab 6: Informed Search Techniques")
st.markdown("**CLO-2:** Apply classical AI techniques including uninformed and informed search algorithms.")
st.markdown("---")

# Sidebar navigation
task = st.sidebar.radio("Select Task", [
    "Task 1: Warehouse Heuristic",
    "Task 2: Airport GBFS",
    "Task 3: Hospital A* Search",
    "Task 4: Drone Weighted A*",
    "Task 6: 8-Puzzle Heuristics",
    "Task 7: Grid GBFS vs BFS"
])

# ============================================================
# TASK 1
# ============================================================
if task == "Task 1: Warehouse Heuristic":
    st.header("Task 1: Designing a Heuristic for a Warehouse Robot")

    locations = {
        "Receiving_Area": (0, 0),
        "Storage_A": (2, 1),
        "Storage_B": (1, 4),
        "Sorting_Area": (4, 2),
        "Inspection_Area": (5, 5),
        "Packing_Station": (7, 4)
    }

    goal = "Packing_Station"

    edges = [
        ("Receiving_Area", "Storage_A", 2.2),
        ("Receiving_Area", "Storage_B", 4.1),
        ("Storage_A", "Sorting_Area", 2.2),
        ("Storage_B", "Sorting_Area", 6.0),
        ("Storage_B", "Inspection_Area", 5.0),
        ("Sorting_Area", "Inspection_Area", 3.2),
        ("Sorting_Area", "Packing_Station", 5.0),
        ("Inspection_Area", "Packing_Station", 2.2)
    ]

    def heuristic(current, goal_node=goal):
        x1, y1 = locations[current]
        x2, y2 = locations[goal_node]
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Heuristic Table
    st.subheader("Heuristic Values (Euclidean Distance to Goal)")
    table_data = {"Location": [], "h(n)": []}
    for loc in locations:
        table_data["Location"].append(loc)
        table_data["h(n)"].append(f"{heuristic(loc):.4f}")
    st.table(table_data)

    # Consistency Check
    st.subheader("Consistency Verification: h(n) ≤ cost(n,m) + h(m)")
    consistency_data = {"Edge": [], "h(n)": [], "cost + h(m)": [], "Valid": []}
    for u, v, cost in edges:
        h_u = heuristic(u)
        h_v = heuristic(v)
        rhs = cost + h_v
        is_valid = round(h_u, 2) <= round(rhs, 2)
        consistency_data["Edge"].append(f"{u} → {v}")
        consistency_data["h(n)"].append(f"{h_u:.2f}")
        consistency_data["cost + h(m)"].append(f"{rhs:.2f}")
        consistency_data["Valid"].append("✅" if is_valid else "❌")
    st.table(consistency_data)

    # Graph Visualization
    st.subheader("Graph Visualization")
    G = nx.DiGraph()
    for loc, pos in locations.items():
        G.add_node(loc, pos=pos)
    for u, v, weight in edges:
        G.add_edge(u, v, weight=weight)

    pos = nx.get_node_attributes(G, 'pos')
    edge_labels = nx.get_edge_attributes(G, 'weight')

    fig, ax = plt.subplots(figsize=(9, 6))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=2500,
            font_size=8, font_weight="bold", arrows=True, arrowsize=20, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red", ax=ax)
    ax.set_title("Warehouse Graph with Edge Costs")
    st.pyplot(fig)

# ============================================================
# TASK 2
# ============================================================
elif task == "Task 2: Airport GBFS":
    st.header("Task 2: Greedy Best-First Search for Airport Baggage Cart")

    locations = {
        "Baggage_Area": (0, 0),
        "Security": (2, 1),
        "Checkpoint": (1, 4),
        "Food_Court": (4, 2),
        "Terminal_Hall": (5, 5),
        "Departure_Gate": (8, 6)
    }

    graph = {
        "Baggage_Area": [("Security", 2.2), ("Checkpoint", 4.1)],
        "Security": [("Food_Court", 2.2)],
        "Checkpoint": [("Terminal_Hall", 5.0)],
        "Food_Court": [("Terminal_Hall", 3.2), ("Departure_Gate", 6.0)],
        "Terminal_Hall": [("Departure_Gate", 3.2)],
        "Departure_Gate": []
    }

    start_node = "Baggage_Area"
    goal_node = "Departure_Gate"

    def heuristic(current, goal=goal_node):
        x1, y1 = locations[current]
        x2, y2 = locations[goal]
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Heuristic Table
    st.subheader("Heuristic Values")
    table_data = {"Location": [], "h(n)": []}
    for loc in locations:
        table_data["Location"].append(loc)
        table_data["h(n)"].append(f"{heuristic(loc):.4f}")
    st.table(table_data)

    # GBFS
    def greedy_best_first_search(start, goal):
        priority_queue = [(heuristic(start), start, [start], 0.0)]
        visited = set()
        expansion_order = []
        while priority_queue:
            priority_queue.sort()
            h, current, path, cost = priority_queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            expansion_order.append(current)
            if current == goal:
                return path, expansion_order, cost
            for neighbor, edge_cost in graph[current]:
                if neighbor not in visited:
                    priority_queue.append((heuristic(neighbor), neighbor, path + [neighbor], cost + edge_cost))
        return None, expansion_order, 0.0

    solution_path, expanded_nodes, total_cost = greedy_best_first_search(start_node, goal_node)

    st.subheader("GBFS Results")
    st.write(f"**Node Expansion Order:** {' → '.join(expanded_nodes)}")
    st.write(f"**Solution Path:** {' → '.join(solution_path)}")
    st.write(f"**Total Path Cost:** {total_cost:.2f}")

    # Graph
    st.subheader("Graph Visualization")
    G = nx.DiGraph()
    for loc, pos in locations.items():
        G.add_node(loc, pos=pos)
    for u, neighbors in graph.items():
        for v, weight in neighbors:
            G.add_edge(u, v, weight=weight)

    pos = nx.get_node_attributes(G, 'pos')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    node_colors = ["lightgreen" if node in solution_path else "lightblue" for node in G.nodes()]

    fig, ax = plt.subplots(figsize=(9, 6))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=2600,
            font_size=8, font_weight="bold", arrows=True, arrowsize=20, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red", ax=ax)
    ax.set_title("Airport Graph - GBFS Solution Path (Green Nodes)")
    st.pyplot(fig)

# ============================================================
# TASK 3
# ============================================================
elif task == "Task 3: Hospital A* Search":
    st.header("Task 3: A* Search for Hospital Delivery Robot")

    locations = {
        "Pharmacy": (0, 0),
        "Main_Corridor": (2, 1),
        "Patient_Wing": (1, 4),
        "Nursing_Station": (4, 2),
        "Laboratory": (5, 5),
        "Emergency_Ward": (8, 6)
    }

    start_node = "Pharmacy"
    goal_node = "Emergency_Ward"

    graph = {
        "Pharmacy": [("Main_Corridor", 2.2), ("Patient_Wing", 4.1)],
        "Main_Corridor": [("Nursing_Station", 2.2)],
        "Patient_Wing": [("Laboratory", 5.0)],
        "Nursing_Station": [("Laboratory", 3.2), ("Emergency_Ward", 6.0)],
        "Laboratory": [("Emergency_Ward", 3.2)],
        "Emergency_Ward": []
    }

    def heuristic(current, goal=goal_node):
        x1, y1 = locations[current]
        x2, y2 = locations[goal]
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Heuristic Table
    st.subheader("Heuristic Values")
    table_data = {"Location": [], "h(n)": []}
    for loc in locations:
        table_data["Location"].append(loc)
        table_data["h(n)"].append(f"{heuristic(loc):.4f}")
    st.table(table_data)

    # A* Search
    def a_star_search(start, goal):
        priority_queue = [(0.0 + heuristic(start), start, [start], 0.0)]
        visited = set()
        expansion_order = []
        while priority_queue:
            priority_queue.sort()
            h, current, path, cost = priority_queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            expansion_order.append(current)
            if current == goal:
                return path, expansion_order, cost
            for neighbor, edge_cost in graph[current]:
                if neighbor not in visited:
                    new_g = cost + edge_cost
                    f = new_g + heuristic(neighbor)
                    priority_queue.append((f, neighbor, path + [neighbor], new_g))
        return None, expansion_order, 0.0

    solution_path, expanded_nodes, total_cost = a_star_search(start_node, goal_node)

    st.subheader("A* Search Results")
    st.write(f"**Node Expansion Order:** {' → '.join(expanded_nodes)}")
    st.write(f"**Solution Path:** {' → '.join(solution_path)}")
    st.write(f"**Total Path Cost:** {total_cost:.2f}")

    # Graph
    st.subheader("Graph Visualization")
    G = nx.DiGraph()
    for loc, pos in locations.items():
        G.add_node(loc, pos=pos)
    for u, neighbors in graph.items():
        for v, weight in neighbors:
            G.add_edge(u, v, weight=weight)

    pos = nx.get_node_attributes(G, 'pos')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    node_colors = ["lightgreen" if node in solution_path else "lightblue" for node in G.nodes()]

    fig, ax = plt.subplots(figsize=(9, 6))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=2600,
            font_size=8, font_weight="bold", arrows=True, arrowsize=20, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red", ax=ax)
    ax.set_title("Hospital Graph - A* Optimal Path (Green Nodes)")
    st.pyplot(fig)

# ============================================================
# TASK 4
# ============================================================
elif task == "Task 4: Drone Weighted A*":
    st.header("Task 4: Weighted A* for Autonomous Delivery Drone")

    locations = {
        "Distribution_Center": (0, 0),
        "Zone_A": (2, 1),
        "Zone_B": (1, 4),
        "Zone_C": (4, 2),
        "Zone_D": (5, 5),
        "Customer_Building": (8, 6)
    }

    start_node = "Distribution_Center"
    goal_node = "Customer_Building"

    graph = {
        "Distribution_Center": [("Zone_A", 2.2), ("Zone_B", 3.0)],
        "Zone_A": [("Zone_C", 2.2)],
        "Zone_B": [("Zone_D", 5.0)],
        "Zone_C": [("Zone_D", 3.2), ("Customer_Building", 6.0)],
        "Zone_D": [("Customer_Building", 3.2)],
        "Customer_Building": []
    }

    def heuristic(current, goal=goal_node):
        x1, y1 = locations[current]
        x2, y2 = locations[goal]
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Heuristic Table
    st.subheader("Heuristic Values")
    table_data = {"Location": [], "h(n)": []}
    for loc in locations:
        table_data["Location"].append(loc)
        table_data["h(n)"].append(f"{heuristic(loc):.4f}")
    st.table(table_data)

    # Weighted A*
    def weighted_a_star_search(start, goal, weight=1.0):
        initial_f = 0.0 + weight * heuristic(start)
        priority_queue = [(initial_f, start, [start], 0.0)]
        visited = set()
        expansion_order = []
        while priority_queue:
            priority_queue.sort()
            f, current, path, g_cost = priority_queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            expansion_order.append(current)
            if current == goal:
                return path, expansion_order, g_cost
            for neighbor, edge_cost in graph[current]:
                if neighbor not in visited:
                    new_g = g_cost + edge_cost
                    f = new_g + weight * heuristic(neighbor)
                    priority_queue.append((f, neighbor, path + [neighbor], new_g))
        return None, expansion_order, 0.0

    # Run for all weights
    weights = [1.0, 1.5, 2.0, 3.0]
    results = []
    for w in weights:
        path, expanded, cost = weighted_a_star_search(start_node, goal_node, weight=w)
        path_str = " → ".join(path) if path else "None"
        results.append({
            "Weight w": w,
            "Solution Path": path_str,
            "Total Cost": f"{cost:.2f}",
            "Nodes Expanded": len(expanded)
        })

    st.subheader("Weight Comparison Table")
    st.table(results)

    # Graph
    st.subheader("Graph Visualization (Optimal A* Path Highlighted)")
    optimal_path, _, _ = weighted_a_star_search(start_node, goal_node, weight=1.0)
    path_edges = list(zip(optimal_path[:-1], optimal_path[1:]))

    G = nx.DiGraph()
    for loc, pos in locations.items():
        G.add_node(loc, pos=pos)
    for u, neighbors in graph.items():
        for v, weight in neighbors:
            G.add_edge(u, v, weight=weight)

    pos = nx.get_node_attributes(G, 'pos')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    node_colors = ["lightgreen" if node in optimal_path else "lightblue" for node in G.nodes()]
    edge_colors = ["green" if (u, v) in path_edges else "gray" for u, v in G.edges()]
    edge_widths = [3.0 if (u, v) in path_edges else 1.2 for u, v in G.edges()]

    fig, ax = plt.subplots(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=2800,
            font_size=8, font_weight="bold", arrows=True, arrowsize=20,
            edge_color=edge_colors, width=edge_widths, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red", ax=ax)
    ax.set_title("Drone Delivery Graph - Optimal A* Path Highlighted (Green)")
    st.pyplot(fig)

# ============================================================
# TASK 6
# ============================================================
elif task == "Task 6: 8-Puzzle Heuristics":
    st.header("Task 6: Heuristic Function Evaluation on 8-Puzzle")

    goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    initial_state = [[1, 2, 3], [4, 0, 6], [7, 5, 8]]
    intermediate_state_1 = [[1, 2, 3], [4, 5, 6], [7, 0, 8]]
    intermediate_state_2 = [[1, 2, 3], [0, 4, 6], [7, 5, 8]]

    def h1_misplaced_tiles(state, goal=goal_state):
        misplaced = 0
        for r in range(3):
            for c in range(3):
                tile = state[r][c]
                if tile != 0 and tile != goal[r][c]:
                    misplaced += 1
        return misplaced

    def find_pos(state, value):
        for r in range(3):
            for c in range(3):
                if state[r][c] == value:
                    return r, c
        return None

    def h2_manhattan_distance(state, goal=goal_state):
        total_distance = 0
        for tile in range(1, 9):
            r1, c1 = find_pos(state, tile)
            r2, c2 = find_pos(goal, tile)
            total_distance += abs(r1 - r2) + abs(c1 - c2)
        return total_distance

    # Display States
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write("**Goal State**")
        for row in goal_state:
            st.write(row)
    with col2:
        st.write("**Initial State**")
        for row in initial_state:
            st.write(row)
    with col3:
        st.write("**Intermediate 1**")
        for row in intermediate_state_1:
            st.write(row)
    with col4:
        st.write("**Intermediate 2**")
        for row in intermediate_state_2:
            st.write(row)

    # Heuristic Table
    st.subheader("Heuristic Values")
    states = [
        ("Initial State", initial_state),
        ("Intermediate State 1 (Move 5 Up)", intermediate_state_1),
        ("Intermediate State 2 (Move 4 Right)", intermediate_state_2)
    ]

    table_data = {"State": [], "h1 (Misplaced)": [], "h2 (Manhattan)": []}
    for name, s in states:
        table_data["State"].append(name)
        table_data["h1 (Misplaced)"].append(h1_misplaced_tiles(s))
        table_data["h2 (Manhattan)"].append(h2_manhattan_distance(s))
    st.table(table_data)

    st.subheader("Admissibility & Consistency")
    st.write("**Both h1 and h2 are admissible** — they never overestimate the true cost to reach the goal.")
    st.write("**h2 (Manhattan) is also consistent** — it satisfies h(n) ≤ cost(n,m) + h(m) for all edges.")
    st.write("**h2 dominates h1** — h2(n) ≥ h1(n) for all states, so A* with h2 expands fewer nodes.")

# ============================================================
# TASK 7
# ============================================================
elif task == "Task 7: Grid GBFS vs BFS":
    st.header("Task 7: Greedy Best-First Search on a 4×4 Weighted Grid")

    grid = [
        [0, 1, 4, 7],
        [2, 'X', 1, 3],
        [3, 2, 2, 1],
        [4, 3, 1, 0]
    ]

    start = (0, 0)
    goal = (3, 3)
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def heuristic(pos, goal_pos=goal):
        r, c = pos
        gr, gc = goal_pos
        return abs(gr - r) + abs(gc - c)

    # Display Grid
    st.subheader("Grid Layout")
    grid_display = [["S", "1", "4", "7"], ["2", "X", "1", "3"], ["3", "2", "2", "1"], ["4", "3", "1", "G"]]
    st.table(grid_display)

    # GBFS
    def gbfs(start, goal):
        priority_queue = [(heuristic(start), start, [start], 0)]
        visited = set()
        expansion_order = []
        while priority_queue:
            priority_queue.sort()
            h, current, path, cost = priority_queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            expansion_order.append(current)
            if current == goal:
                return path, expansion_order, cost
            r, c = current
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 4 and 0 <= nc < 4:
                    if grid[nr][nc] != 'X' and (nr, nc) not in visited:
                        move_cost = grid[nr][nc]
                        priority_queue.append((heuristic((nr, nc)), (nr, nc), path + [(nr, nc)], cost + move_cost))
        return None, expansion_order, 0

    # BFS
    def bfs(start, goal):
        queue = deque([(start, [start], 0)])
        visited = {start}
        expansion_order = []
        while queue:
            current, path, cost = queue.popleft()
            expansion_order.append(current)
            if current == goal:
                return path, expansion_order, cost
            r, c = current
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 4 and 0 <= nc < 4:
                    if grid[nr][nc] != 'X' and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        move_cost = grid[nr][nc]
                        queue.append(((nr, nc), path + [(nr, nc)], cost + move_cost))
        return None, expansion_order, 0

    gbfs_path, gbfs_expanded, gbfs_cost = gbfs(start, goal)
    bfs_path, bfs_expanded, bfs_cost = bfs(start, goal)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("GBFS Results")
        st.write(f"**Nodes Expanded:** {len(gbfs_expanded)}")
        st.write(f"**Expansion Order:** {' → '.join(str(p) for p in gbfs_expanded)}")
        st.write(f"**Solution Path:** {' → '.join(str(p) for p in gbfs_path)}")
        st.write(f"**Total Cost:** {gbfs_cost}")
    with col2:
        st.subheader("BFS Results")
        st.write(f"**Nodes Expanded:** {len(bfs_expanded)}")
        st.write(f"**Expansion Order:** {' → '.join(str(p) for p in bfs_expanded)}")
        st.write(f"**Solution Path:** {' → '.join(str(p) for p in bfs_path)}")
        st.write(f"**Total Cost:** {bfs_cost}")

    st.subheader("Comparison")
    st.write("**GBFS** uses heuristic (Manhattan distance) to guide search — expands fewer nodes but may not find optimal path.")
    st.write("**BFS** explores level by level — guarantees shortest path in terms of hops but ignores edge costs.")
