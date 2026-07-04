import math
import random
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def sample_subgraph(adj, partition, pr=None, n_comms=6, per_comm=50, seed=0):
    members = defaultdict(list)
    for n, c in partition.items():
        members[c].append(n)
    top = sorted(members, key=lambda c: -len(members[c]))[:n_comms]
    sampled, node_comm = set(), {}
    for c in top:
        ms = set(members[c])
        if pr:
            seed_node = max(ms, key=lambda x: pr.get(x, 0))
        else:
            seed_node = max(ms, key=lambda x: sum(1 for v in adj[x] if v in ms))
        seen, dq, order = {seed_node}, deque([seed_node]), []
        while dq and len(order) < per_comm:
            u = dq.popleft()
            order.append(u)
            for v in adj[u]:
                if v in ms and v not in seen:
                    seen.add(v)
                    dq.append(v)
        for u in order:
            sampled.add(u)
            node_comm[u] = c
    sub = {u: [v for v in adj[u] if v in sampled] for u in sampled}
    return sub, node_comm


def spring_layout(sub, iterations=100, seed=0):
    rng = random.Random(seed)
    nodes = list(sub)
    n = len(nodes)
    if n == 0:
        return {}
    pos = {u: [rng.uniform(0, 1), rng.uniform(0, 1)] for u in nodes}
    edges = set()
    for u in sub:
        for v in sub[u]:
            edges.add((u, v) if str(u) < str(v) else (v, u))
    k = math.sqrt(1.0 / n)
    t = 0.1
    dt = t / (iterations + 1)
    for _ in range(iterations):
        disp = {u: [0.0, 0.0] for u in nodes}
        for i in range(n):
            ui = nodes[i]; xi, yi = pos[ui]
            for j in range(i + 1, n):
                uj = nodes[j]
                dx = xi - pos[uj][0]; dy = yi - pos[uj][1]
                d = math.hypot(dx, dy) or 1e-6
                rep = k * k / d
                ux, uy = dx / d * rep, dy / d * rep
                disp[ui][0] += ux; disp[ui][1] += uy
                disp[uj][0] -= ux; disp[uj][1] -= uy
        for u, v in edges:
            dx = pos[u][0] - pos[v][0]; dy = pos[u][1] - pos[v][1]
            d = math.hypot(dx, dy) or 1e-6
            att = d * d / k
            ax_, ay_ = dx / d * att, dy / d * att
            disp[u][0] -= ax_; disp[u][1] -= ay_
            disp[v][0] += ax_; disp[v][1] += ay_
        for u in nodes:
            dx, dy = disp[u]
            d = math.hypot(dx, dy) or 1e-6
            lim = min(d, t)
            pos[u][0] += dx / d * lim
            pos[u][1] += dy / d * lim
        t -= dt
    return {u: tuple(p) for u, p in pos.items()}


def plot_communities(sub, node_comm, pos, pr=None, title=None):
    fig, ax = plt.subplots(figsize=(9, 9))
    for u in sub:
        for v in sub[u]:
            if str(u) < str(v):
                ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                        color='#cccccc', lw=0.3, alpha=0.5, zorder=1)
    comms = sorted(set(node_comm.values()))
    cmap = plt.cm.tab10
    color = {c: cmap(i % 10) for i, c in enumerate(comms)}
    nodes = list(sub)
    if pr:
        mx = max((pr.get(u, 0) for u in nodes), default=0) or 1.0
        sizes = [15 + 220 * (pr.get(u, 0) / mx) for u in nodes]
    else:
        sizes = [25] * len(nodes)
    ax.scatter([pos[u][0] for u in nodes], [pos[u][1] for u in nodes],
               c=[color[node_comm[u]] for u in nodes], s=sizes,
               edgecolors='white', linewidths=0.3, zorder=2)
    handles = [mpatches.Patch(color=color[c], label=f'Comunidad {c}') for c in comms]
    ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.9)
    ax.set_title(title or 'Comunidades de amistad (muestra del LCC)')
    ax.axis('off')
    plt.tight_layout(); plt.show()
