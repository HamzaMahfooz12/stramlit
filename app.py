import streamlit as st
import math
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

st.set_page_config(page_title="AI Lab 6 - Informed Search", page_icon="🤖", layout="wide")

st.title("🤖 AI Lab 6: Informed Search Techniques")
st.markdown("**CLO-2:** Apply classical AI techniques including uninformed and informed search algorithms.")
st.markdown("---")

# Graph Data - All locations and edges
locations = {
    "Receiving_Area": (0, 0),
    "Storage_A": (2, 1),
    "Storage_B": (1, 4),
    "Sorting_Area": (4, 2),
    "Inspection_Area": (5, 5),
    "Packing_Station": (7, 4),
    "Baggage_Area": (10, 0),
    "Security": (12, 1),
    "Checkpoint": (11, 4),
    "Food_Court": (14, 2),
    "Terminal_Hall": (15, 5),
    "Departure_Gate": (18, 6),
    "Pharmacy": (20, 0),
    "Main_Corridor": (22, 1),
    "Patient_Wing": (21, 4),
    "Nursing_Station": (24, 2),
    "Laboratory": (25, 5),
    "Emergency_Ward": (28, 6),
    "Distribution_Center": (30, 0),
    "Zone_A": (32, 1),
    "Zone_B": (31, 4),
    "Zone_C": (34, 2),
    "Zone_D": (35, 5),
    "Customer_Building": (38, 6)
}

edges = [
    ("Receiving_Area", "Storage_A", 2.2),
    ("Receiving_Area", "Storage_B", 4.1),
    ("Storage_A", "Sorting_Area", 2.2),
    ("Storage_B", "Sorting_Area", 6.0),
    ("Storage_B", "Inspection_Area", 5.0),
    ("Sorting_Area", "Inspection_Area", 3.2),
    ("Sorting_Area", "Packing_Station", 5.0),
    ("Inspection_Area", "Packing_Station", 2.2),
    ("Baggage_Area", "Security", 2.2),
    ("Baggage_Area", "Checkpoint", 4.1),
    ("Security", "Food_Court", 2.2),
    ("Checkpoint", "Terminal_Hall", 5.0),
    ("Food_Court", "Terminal_Hall", 3.2),
    ("Food_Court", "Departure_Gate", 6.0),
    ("Terminal_Hall", "Departure_Gate", 3.2),
    ("Pharmacy", "Main_Corridor", 2.2),
    ("Pharmacy", "Patient_Wing", 4.1),
    ("Main_Corridor", "Nursing_Station", 2.2),
    ("Patient_Wing", "Laboratory", 5.0),
    ("Nursing_Station", "Laboratory", 3.2),
    ("Nursing_Station", "Emergency_Ward", 6.0),
    ("Laboratory", "Emergency_Ward", 3.2),
    ("Distribution_Center", "Zone_A", 2.2),
    ("Distribution_Center", "Zone_B", 3.0),
    ("Zone_A", "Zone_C", 2.2),
    ("Zone_B", "Zone_D", 5.0),
    ("Zone_C", "Zone_D", 3.2),
    ("Zone_C", "Customer_Building", 6.0),
    ("Zone_D", "Customer_Building", 3.2),
]

# Build adjacency list
graph = {loc: [] for loc in locations}
for u, v, w in edges:
    graph[u].append((v, w))

# Scenarios
scenarios = {
    "Warehouse Robot": {
        "start": "Receiving_Area",
        "goal": "Packing_Station",
        "nodes": ["Receiving_Area", "Storage_A", "Storage_B", "Sorting_Area", "Inspection_Area", "Packing_Station"]
    },
    "Airport Baggage": {
        "start": "Baggage_Area",
        "goal": "Departure_Gate",
        "nodes": ["Baggage_Area", "Security", "Checkpoint", "Food_Court", "Terminal_Hall", "Departure_Gate"]
    },
    "Hospital Robot": {
        "start": "Pharmacy",
        "goal": "Emergency_Ward",
        "nodes": ["Pharmacy", "Main_Corridor", "Patient_Wing", "Nursing_Station", "Laboratory", "Emergency_Ward"]
    },
    "Delivery Drone": {
        "start": "Distribution_Center",
        "goal": "Customer_Building",
        "nodes": ["Distribution_Center", "Zone_A", "Zone_B", "Zone_C", "Zone_D", "Customer_Building"]
    }
}

# Heuristic function
def heuristic(current, goal_node):
    x1, y1 = locations[current]
    x2, y2 = locations[goal_node]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# GBFS
def greedy_best_first_search(start, goal, adj):
    pq = [(heuristic(start, goal), start, [start], 0.0)]
    visited = set()
    expansion_order = []
    while pq:
        pq.sort()
        h, current, path, cost = pq.pop(0)
        if current in visited:
            continue
        visited.add(current)
        expansion_order.append(current)
        if current == goal:
            return path, expansion_order, cost
        for neighbor, edge_cost in adj[current]:
            if neighbor not in visited:
                pq.append((heuristic(neighbor, goal), neighbor, path + [neighbor], cost + edge_cost))
    return None, expansion_order, 0.0

# A* Search
def a_star_search(start, goal, adj):
    pq = [(0.0 + heuristic(start, goal), start, [start], 0.0)]
    visited = set()
    expansion_order = []
    while pq:
        pq.sort()
        f, current, path, cost = pq.pop(0)
        if current in visited:
            continue
        visited.add(current)
        expansion_order.append(current)
        if current == goal:
            return path, expansion_order, cost
        for neighbor, edge_cost in adj[current]:
            if neighbor not in visited:
                new_g = cost + edge_cost
                f = new_g + heuristic(neighbor, goal)
                pq.append((f, neighbor, path + [neighbor], new_g))
    return None, expansion_order, 0.0

# Weighted A*
def weighted_a_star_search(start, goal, adj, weight=1.0):
    pq = [(0.0 + weight * heuristic(start, goal), start, [start], 0.0)]
    visited = set()
    expansion_order = []
    while pq:
        pq.sort()
        f, current, path, g_cost = pq.pop(0)
        if current in visited:
            continue
        visited.add(current)
        expansion_order.append(current)
        if current == goal:
            return path, expansion_order, g_cost
        for neighbor, edge_cost in adj[current]:
            if neighbor not in visited:
                new_g = g_cost + edge_cost
                f = new_g + weight * heuristic(neighbor, goal)
                pq.append((f, neighbor, path + [neighbor], new_g))
    return None, expansion_order, 0.0

# BFS
def bfs_search(start, goal, adj):
    queue = deque([(start, [start], 0.0)])
    visited = {start}
    expansion_order = []
    while queue:
        current, path, cost = queue.popleft()
        expansion_order.append(current)
        if current == goal:
            return path, expansion_order, cost
        for neighbor, edge_cost in adj[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor], cost + edge_cost))
    return None, expansion_order, 0.0

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("⚙️ Search Configuration")

scenario = st.sidebar.selectbox("📍 Select Scenario", list(scenarios.keys()))
sc = scenarios[scenario]

start_node = st.sidebar.selectbox("🟢 Start Node", sc["nodes"], index=sc["nodes"].index(sc["start"]))
goal_node = st.sidebar.selectbox("🔴 Goal Node", sc["nodes"], index=sc["nodes"].index(sc["goal"]))

algorithm = st.sidebar.selectbox("🧠 Select Algorithm", [
    "Greedy Best-First Search (GBFS)",
    "A* Search",
    "Weighted A* (w=1.5)",
    "Weighted A* (w=2.0)",
    "Weighted A* (w=3.0)",
    "BFS (Uninformed)",
    "Compare All"
])

run_btn = st.sidebar.button("🚀 Run Search", use_container_width=True)

# ============================================================
# BUILD SCENARIO GRAPH
# ============================================================
sc_nodes = sc["nodes"]
sc_edges = [(u, v, w) for u, v, w in edges if u in sc_nodes and v in sc_nodes]
sc_graph = {n: [] for n in sc_nodes}
for u, v, w in sc_edges:
    sc_graph[u].append((v, w))

# ============================================================
# DISPLAY GRAPH + HEURISTIC TABLE
# ============================================================
col_graph, col_info = st.columns([2, 1])

with col_info:
    st.subheader("📊 Heuristic Table")
    st.markdown(f"**h(n)** = Euclidean distance to `{goal_node}`")
    h_data = {"Node": [], "h(n)": []}
    for n in sc_nodes:
        h_data["Node"].append(n)
        h_data["h(n)"].append(f"{heuristic(n, goal_node):.4f}")
    st.table(h_data)

with col_graph:
    st.subheader(f"🗺️ {scenario} Graph")
    G = nx.DiGraph()
    sc_locs = {n: locations[n] for n in sc_nodes}
    for n, pos in sc_locs.items():
        G.add_node(n, pos=pos)
    for u, v, w in sc_edges:
        G.add_edge(u, v, weight=w)

    pos = nx.get_node_attributes(G, 'pos')
    edge_labels = nx.get_edge_attributes(G, 'weight')

    fig, ax = plt.subplots(figsize=(9, 5))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=2500,
            font_size=7, font_weight="bold", arrows=True, arrowsize=20, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red", font_size=8, ax=ax)
    ax.set_title(f"{scenario} - All Nodes & Edges")
    st.pyplot(fig)

# ============================================================
# RUN SEARCH & SHOW RESULTS
# ============================================================
if run_btn:
    st.markdown("---")

    if algorithm == "Compare All":
        st.header("📈 Comparison of All Algorithms")

        algos = {
            "GBFS": lambda: greedy_best_first_search(start_node, goal_node, sc_graph),
            "A*": lambda: a_star_search(start_node, goal_node, sc_graph),
            "Weighted A* (w=1.5)": lambda: weighted_a_star_search(start_node, goal_node, sc_graph, 1.5),
            "Weighted A* (w=2.0)": lambda: weighted_a_star_search(start_node, goal_node, sc_graph, 2.0),
            "Weighted A* (w=3.0)": lambda: weighted_a_star_search(start_node, goal_node, sc_graph, 3.0),
            "BFS": lambda: bfs_search(start_node, goal_node, sc_graph),
        }

        comparison = {"Algorithm": [], "Solution Path": [], "Total Cost": [], "Nodes Expanded": []}

        for name, func in algos.items():
            path, expanded, cost = func()
            comparison["Algorithm"].append(name)
            comparison["Solution Path"].append(" → ".join(path) if path else "No path")
            comparison["Total Cost"].append(f"{cost:.2f}")
            comparison["Nodes Expanded"].append(len(expanded))

        st.table(comparison)

        # Show all paths on graph
        st.subheader("🗺️ All Solution Paths Visualized")
        colors_map = ["green", "blue", "orange", "purple", "red", "cyan"]
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        nx.draw(G, pos, with_labels=True, node_color="lightyellow", node_size=2500,
                font_size=7, font_weight="bold", arrows=True, arrowsize=15, ax=ax2, edge_color="lightgray")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="gray", font_size=7, ax=ax2)

        for i, (name, func) in enumerate(algos.items()):
            path, _, _ = func()
            if path:
                path_edges = list(zip(path[:-1], path[1:]))
                nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color=colors_map[i % len(colors_map)],
                                       width=2.5, style="solid", arrows=True, arrowsize=18, ax=ax2,
                                       label=name)
        ax2.legend(loc="upper left", fontsize=7)
        ax2.set_title("All Algorithm Paths Overlaid")
        st.pyplot(fig2)

    else:
        # Single algorithm run
        if algorithm == "Greedy Best-First Search (GBFS)":
            path, expanded, cost = greedy_best_first_search(start_node, goal_node, sc_graph)
            algo_name = "GBFS"
        elif algorithm == "A* Search":
            path, expanded, cost = a_star_search(start_node, goal_node, sc_graph)
            algo_name = "A*"
        elif algorithm == "Weighted A* (w=1.5)":
            path, expanded, cost = weighted_a_star_search(start_node, goal_node, sc_graph, 1.5)
            algo_name = "Weighted A* (w=1.5)"
        elif algorithm == "Weighted A* (w=2.0)":
            path, expanded, cost = weighted_a_star_search(start_node, goal_node, sc_graph, 2.0)
            algo_name = "Weighted A* (w=2.0)"
        elif algorithm == "Weighted A* (w=3.0)":
            path, expanded, cost = weighted_a_star_search(start_node, goal_node, sc_graph, 3.0)
            algo_name = "Weighted A* (w=3.0)"
        else:
            path, expanded, cost = bfs_search(start_node, goal_node, sc_graph)
            algo_name = "BFS"

        st.header(f"🔍 {algo_name} Results")

        if path:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Path Cost", f"{cost:.2f}")
            col2.metric("Nodes Expanded", len(expanded))
            col3.metric("Path Length", len(path))

            st.subheader("📋 Details")
            st.write(f"**Expansion Order:** {' → '.join(expanded)}")
            st.write(f"**Solution Path:** {' → '.join(path)}")

            # Highlighted graph
            st.subheader("🗺️ Solution Path Highlighted")
            path_edges = list(zip(path[:-1], path[1:]))
            node_colors = ["#90EE90" if n in path else "lightblue" for n in G.nodes()]
            edge_colors = ["green" if (u, v) in path_edges else "lightgray" for u, v in G.edges()]
            edge_widths = [3.5 if (u, v) in path_edges else 1.0 for u, v in G.edges()]

            fig3, ax3 = plt.subplots(figsize=(10, 6))
            nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=2800,
                    font_size=7, font_weight="bold", arrows=True, arrowsize=20,
                    edge_color=edge_colors, width=edge_widths, ax=ax3)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red", font_size=8, ax=ax3)
            ax3.set_title(f"{algo_name}: {start_node} → {goal_node} (Cost: {cost:.2f})")
            st.pyplot(fig3)
        else:
            st.error("❌ No path found between the selected nodes!")

# ============================================================
# 8-PUZZLE SECTION (Separate)
# ============================================================
st.markdown("---")
with st.expander("🧩 Task 6: 8-Puzzle Heuristic Evaluation", expanded=False):
    goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    initial_state = [[1, 2, 3], [4, 0, 6], [7, 5, 8]]
    intermediate_state_1 = [[1, 2, 3], [4, 5, 6], [7, 0, 8]]
    intermediate_state_2 = [[1, 2, 3], [0, 4, 6], [7, 5, 8]]

    def h1_misplaced(state, goal=goal_state):
        count = 0
        for r in range(3):
            for c in range(3):
                if state[r][c] != 0 and state[r][c] != goal[r][c]:
                    count += 1
        return count

    def find_pos(state, val):
        for r in range(3):
            for c in range(3):
                if state[r][c] == val:
                    return r, c

    def h2_manhattan(state, goal=goal_state):
        dist = 0
        for tile in range(1, 9):
            r1, c1 = find_pos(state, tile)
            r2, c2 = find_pos(goal, tile)
            dist += abs(r1 - r2) + abs(c1 - c2)
        return dist

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.write("**Goal State**")
        for row in goal_state: st.write(row)
    with c2:
        st.write("**Initial State**")
        for row in initial_state: st.write(row)
    with c3:
        st.write("**Intermediate 1**")
        for row in intermediate_state_1: st.write(row)
    with c4:
        st.write("**Intermediate 2**")
        for row in intermediate_state_2: st.write(row)

    states = [
        ("Initial State", initial_state),
        ("Intermediate 1 (Move 5 Up)", intermediate_state_1),
        ("Intermediate 2 (Move 4 Right)", intermediate_state_2)
    ]
    tbl = {"State": [], "h1 (Misplaced)": [], "h2 (Manhattan)": []}
    for name, s in states:
        tbl["State"].append(name)
        tbl["h1 (Misplaced)"].append(h1_misplaced(s))
        tbl["h2 (Manhattan)"].append(h2_manhattan(s))
    st.table(tbl)
    st.info("Both h1 and h2 are **admissible** (never overestimate). h2 **dominates** h1, so A* with h2 expands fewer nodes.")

# ============================================================
# GRID SEARCH SECTION
# ============================================================
with st.expander("🔲 Task 7: 4×4 Grid - GBFS vs BFS", expanded=False):
    grid = [[0, 1, 4, 7], [2, 'X', 1, 3], [3, 2, 2, 1], [4, 3, 1, 0]]
    start_g = (0, 0)
    goal_g = (3, 3)
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def h_grid(pos):
        return abs(goal_g[0] - pos[0]) + abs(goal_g[1] - pos[1])

    def gbfs_grid():
        pq = [(h_grid(start_g), start_g, [start_g], 0)]
        visited = set()
        exp = []
        while pq:
            pq.sort()
            h, cur, path, cost = pq.pop(0)
            if cur in visited: continue
            visited.add(cur)
            exp.append(cur)
            if cur == goal_g: return path, exp, cost
            for dr, dc in dirs:
                nr, nc = cur[0]+dr, cur[1]+dc
                if 0<=nr<4 and 0<=nc<4 and grid[nr][nc]!='X' and (nr,nc) not in visited:
                    pq.append((h_grid((nr,nc)), (nr,nc), path+[(nr,nc)], cost+grid[nr][nc]))
        return None, exp, 0

    def bfs_grid():
        q = deque([(start_g, [start_g], 0)])
        visited = {start_g}
        exp = []
        while q:
            cur, path, cost = q.popleft()
            exp.append(cur)
            if cur == goal_g: return path, exp, cost
            for dr, dc in dirs:
                nr, nc = cur[0]+dr, cur[1]+dc
                if 0<=nr<4 and 0<=nc<4 and grid[nr][nc]!='X' and (nr,nc) not in visited:
                    visited.add((nr,nc))
                    q.append(((nr,nc), path+[(nr,nc)], cost+grid[nr][nc]))
        return None, exp, 0

    st.write("**Grid Layout:** (S=Start, G=Goal, X=Obstacle)")
    st.table([["S","1","4","7"],["2","X","1","3"],["3","2","2","1"],["4","3","1","G"]])

    g1, g2 = st.columns(2)
    gp, ge, gc = gbfs_grid()
    bp, be, bc = bfs_grid()
    with g1:
        st.write("**🟢 GBFS Results**")
        st.write(f"Nodes Expanded: **{len(ge)}**")
        st.write(f"Path: {' → '.join(str(p) for p in gp)}")
        st.write(f"Cost: **{gc}**")
    with g2:
        st.write("**🔵 BFS Results**")
        st.write(f"Nodes Expanded: **{len(be)}**")
        st.write(f"Path: {' → '.join(str(p) for p in bp)}")
        st.write(f"Cost: **{bc}**")
    st.info("**GBFS** uses heuristic to guide search (fewer expansions). **BFS** explores level-by-level (guarantees shortest hop path).")
