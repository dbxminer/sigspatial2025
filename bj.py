import pandas as pd
import numpy as np
import ast
import networkx as nx
import time
from collections import defaultdict
from typing import Dict, List, Any
from tqdm import tqdm

def generate_fixed_length_paths(
    G: nx.Graph,
    start: Any,
    path_length: int,
    num_paths: int
) -> List[List[Any]]:
    """
    Generate up to `num_paths` simple paths of exactly `path_length` nodes,
    starting from node `start`, using DFS exploration.
    """
    paths = []
    stack = [(start, [start])]
    while stack and len(paths) < num_paths:
        node, path = stack.pop()
        if len(path) == path_length:
            paths.append(path)
            continue
        for nbr in G.neighbors(node):
            if nbr in path:
                continue
            stack.append((nbr, path + [nbr]))
    return paths

# ── Data loading & utils ─────────────────────────────────────────────────────

def load_data(pois_fp: str, roads_fp: str, traj_fp: str):
    pois  = pd.read_csv(pois_fp)
    roads = pd.read_csv(roads_fp)
    traj  = pd.read_csv(traj_fp, sep=';')
    return pois, roads, traj

def parse_list_column(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

def detect_column(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in df.columns:
        if c.lower() in candidates:
            return c
    raise KeyError(f"None of {candidates} in {df.columns.tolist()}")

def cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    dot = sum(a[k] * b.get(k, 0.0) for k in a)
    na  = np.sqrt(sum(v*v for v in a.values()))
    nb  = np.sqrt(sum(v*v for v in b.values()))
    return dot/(na*nb) if na>0 and nb>0 else 0.0

# ── Visit & context statistics ────────────────────────────────────────────────

def get_visited_trajectories_and_stats(
    pois_fp: str, roads_fp: str, traj_fp: str,
    threshold_seconds: float, zipf_param: float = 2.0, min_edge_visits: int = 1
) -> Dict[str, Any]:
    pois, roads, traj = load_data(pois_fp, roads_fp, traj_fp)

    geo_col     = detect_column(roads,    ["geoid","geo_id","geo id"])
    lat_col     = detect_column(roads,    ["lat","latitude"])
    lon_col     = detect_column(roads,    ["lon","lng","longitude"])
    poi_lat_col = detect_column(pois,     ["lat","latitude"])
    poi_lon_col = detect_column(pois,     ["lon","lng","longitude"])
    poi_cat_col = detect_column(pois,     ["cat","category"])

    pois = pois.reset_index().rename(columns={"index":"poi_index"})
    roads = roads.merge(
        pois[["poi_index", poi_lat_col, poi_lon_col]],
        left_on=[lat_col, lon_col],
        right_on=[poi_lat_col, poi_lon_col],
        how="left"
    )

    traj["path"]  = parse_list_column(traj, "path")
    traj["tlist"] = parse_list_column(traj, "tlist")

    poi_visit_counts     = {int(idx):0 for idx in pois["poi_index"]}
    edge_visit_counts    = {}
    edge_nonvisit_counts = {}

    def visited_cats_for_one(path, tlist):
        visited = set()
        if len(path)<2 or len(tlist)<2:
            return visited
        prev_ts = tlist[0]
        for edge_id, curr_ts in zip(path[1:], tlist[1:]):
            duration = curr_ts - prev_ts
            prev_ts = curr_ts
            edge_visit_counts.setdefault(edge_id, 0)
            edge_nonvisit_counts.setdefault(edge_id, 0)
            if duration >= threshold_seconds:
                edge_visit_counts[edge_id] += 1
                poi_idxs = (
                    roads.loc[roads[geo_col]==edge_id, "poi_index"]
                         .dropna().astype(int).unique()
                )
                for poi_idx in poi_idxs:
                    poi_visit_counts[poi_idx] += 1
                cats = roads.loc[roads[geo_col]==edge_id, "cat"].unique()
                visited.update(cats)
            else:
                edge_nonvisit_counts[edge_id] += 1
        return visited

    traj["visited_cats"] = traj.apply(
        lambda r: visited_cats_for_one(r["path"], r["tlist"]), axis=1
    )

    keywords = [f"kw{i}" for i in range(1,101)]
    poi_context: Dict[int, Dict[str,float]] = {}
    for poi_idx in pois["poi_index"].astype(int):
        chosen = np.random.choice(keywords, size=20, replace=False)
        w      = np.random.zipf(zipf_param, size=20).astype(float)
        w     /= w.sum()
        poi_context[poi_idx] = dict(zip(chosen, w))

    category_context: Dict[str, Dict[str,float]] = {}
    for cat in roads["cat"].unique():
        poi_idxs = pois.loc[pois[poi_cat_col]==cat, "poi_index"].astype(int).tolist()
        freqs    = np.array([poi_visit_counts[i] for i in poi_idxs], float)
        if freqs.sum() == 0:
            category_context[cat] = {k:0.0 for k in keywords}
        else:
            weights = freqs / freqs.sum()
            rep = {k:0.0 for k in keywords}
            for poi_idx, wt in zip(poi_idxs, weights):
                for kw, val in poi_context[poi_idx].items():
                    rep[kw] += wt * val
            category_context[cat] = rep

    edge_utility: Dict[int,float] = {}
    for edge_id, cnt in edge_visit_counts.items():
        if cnt >= min_edge_visits:
            cat         = roads.loc[roads[geo_col]==edge_id, "cat"].iloc[0]
            cat_context = category_context[cat]
            poi_idxs    = (
                roads.loc[roads[geo_col]==edge_id, "poi_index"]
                     .dropna().astype(int).unique()
            )
            if len(poi_idxs) == 0:
                continue
            edge_ctx = {k: 0.0 for k in keywords}
            for poi_idx in poi_idxs:
                for kw, val in poi_context.get(int(poi_idx), {}).items():
                    edge_ctx[kw] += val
            for kw in edge_ctx:
                edge_ctx[kw] /= len(poi_idxs)
            cos_sim_val = cosine_similarity(cat_context, edge_ctx)
            edge_utility[edge_id] = 1 - cos_sim_val

    cleaned = traj[
        traj["path"].apply(lambda p: any(edge_utility.get(e, 0.0) > 0 for e in p))
    ].reset_index(drop=True)

    return {
        "cleaned_trajectories": cleaned,
        "edge_utility":        edge_utility,
        "edge_visit_counts":   edge_visit_counts,
        "edge_nonvisit_counts": edge_nonvisit_counts,
        "category_list":       list(category_context.keys()),
        "poi_context":         poi_context,
        "category_context":    category_context,
        "keywords":            keywords,
        "roads_df":            roads
    }

def generate_next_level(prev_itemsets, transactions, level1_set):
    new_counts = defaultdict(int)
    prev_set   = set(prev_itemsets)
    for I in tqdm(prev_itemsets, desc="Generating next level", unit="iset"):
        hi = max(I)
        for trans in transactions:
            if not trans or hi not in trans:
                continue
            TI = tuple(e for e in trans if e > hi)
            for e2 in TI:
                if e2 in I:
                    break
            else:
                if TI and TI not in prev_set:
                    C = tuple(sorted(TI + (hi,)))
                    new_counts[C] += 1
    return dict(new_counts)

def high_utility_pattern_mining(
    transactions: List[List[int]],
    edge_visit_counts: Dict[int,int],
    k: int,
    max_levels: int,
    max_itemsets_per_level: int,
    poi_context: Dict[int,Dict[str,float]],
    category_context: Dict[str, float],
    query_context: Dict[str, float],
    roads_df: pd.DataFrame,
    geo_col: str,
    keywords: List[str],
    chi: int,
    edge_utility: Dict[int,float]
) -> (List[tuple], Dict[tuple, float]):
    topk = sorted(
        edge_visit_counts,
        key=lambda e: edge_visit_counts[e] * edge_utility.get(e, 0.0),
        reverse=True
    )[:k]
    level1_set = set(topk)
    pruned = [set(e for e in path if e in level1_set) for path in transactions]
    pruned = [t for t in pruned if t]
    itemsets = {1: { (e,): edge_visit_counts[e] for e in topk }}
    lvl = 1
    while lvl < max_levels:
        prev = list(itemsets[lvl].keys())
        nxt  = generate_next_level(prev, pruned, level1_set)
        if not nxt:
            break
        scored = []
        for I, freq in nxt.items():
            poi_weights = defaultdict(int)
            for e in I:
                poi_idxs = roads_df.loc[roads_df[geo_col]==e, "poi_index"]\
                                 .dropna().astype(int).unique()
                for poi_idx in poi_idxs:
                    poi_weights[poi_idx] += 1
            total_w = sum(poi_weights.values()) or 1
            ctx_I = {k:0.0 for k in keywords}
            for poi_idx, w in poi_weights.items():
                for kw, val in poi_context[poi_idx].items():
                    ctx_I[kw] += (w / total_w) * val
            cos_sim = cosine_similarity(category_context, ctx_I)
            utility_val = 1 - cos_sim
            scored.append((I, freq * utility_val))
        scored.sort(key=lambda x: x[1], reverse=True)
        keep = [I for I,_ in scored[:max_itemsets_per_level]]
        itemsets[lvl+1] = {I: nxt[I] for I in keep}
        lvl += 1
    all_items = {}
    for lvl_sets in itemsets.values():
        all_items.update(lvl_sets)
    sims = {}
    for I in all_items:
        ctx_I = {k:0.0 for k in keywords}
        total_pois = 0
        for e in I:
            poi_idxs = roads_df.loc[roads_df[geo_col]==e, "poi_index"]\
                             .dropna().astype(int).unique()
            for poi_idx in poi_idxs:
                for kw, val in poi_context.get(int(poi_idx), {}).items():
                    ctx_I[kw] += val
            total_pois += len(poi_idxs)
        if total_pois > 0:
            for kw in ctx_I:
                ctx_I[kw] /= total_pois
        sims[I] = cosine_similarity(query_context, ctx_I)
    top_chi = [I for I,_ in sorted(sims.items(), key=lambda x: x[1], reverse=True)[:chi]]
    return top_chi, sims

def frequent_pattern_mining(
    transactions: List[List[int]],
    edge_visit_counts: Dict[int,int],
    k: int,
    max_levels: int,
    max_itemsets_per_level: int,
    query_context: Dict[str, float],
    poi_context: Dict[int,Dict[str,float]],
    roads_df: pd.DataFrame,
    geo_col: str,
    keywords: List[str],
    chi: int
) -> (List[tuple], Dict[tuple, float]):
    topk = sorted(edge_visit_counts, key=lambda e: edge_visit_counts[e], reverse=True)[:k]
    level1_set = set(topk)
    pruned = [set(e for e in path if e in level1_set) for path in transactions]
    pruned = [t for t in pruned if t]
    itemsets = {1: { (e,): edge_visit_counts[e] for e in topk }}
    lvl = 1
    while lvl < max_levels:
        prev = list(itemsets[lvl].keys())
        nxt  = generate_next_level(prev, pruned, level1_set)
        if not nxt:
            break
        sorted_items = sorted(nxt.items(), key=lambda x: x[1], reverse=True)
        keep = [I for I,_ in sorted_items[:max_itemsets_per_level]]
        itemsets[lvl+1] = {I: nxt[I] for I in keep}
        lvl += 1
    all_items = {}
    for lvl_sets in itemsets.values():
        all_items.update(lvl_sets)
    sims = {}
    for I in all_items:
        ctx_I = {k:0.0 for k in keywords}
        total_pois = 0
        for e in I:
            poi_idxs = roads_df.loc[roads_df[geo_col]==e, "poi_index"]\
                             .dropna().astype(int).unique()
            for poi_idx in poi_idxs:
                for kw, val in poi_context.get(int(poi_idx), {}).items():
                    ctx_I[kw] += val
            total_pois += len(poi_idxs)
        if total_pois > 0:
            for kw in ctx_I:
                ctx_I[kw] /= total_pois
        sims[I] = cosine_similarity(query_context, ctx_I)
    top_chi = [I for I,_ in sorted(all_items.items(), key=lambda x: x[1], reverse=True)[:chi]]
    return top_chi, sims

def build_edge_graph(rel_fp: str, origin_col="origin_id", dest_col="destination_id") -> nx.Graph:
    df = pd.read_csv(rel_fp)
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(int(row[origin_col]), int(row[dest_col]))
    return G

def compute_top5_edges(edge_visit_counts: Dict[int,int]) -> List[int]:
    return sorted(edge_visit_counts, key=lambda e: edge_visit_counts[e], reverse=True)[:10]

def compute_path_utility(
    G: nx.Graph,
    edge_pairs: List[int],
    chi_itemsets: List[tuple],
    sims: Dict[tuple,float],
    num_paths: int,
    top_k_paths: int,
    selection_mode: str,
    path_length: int
) -> float:
    total = 0.0
    for e1 in edge_pairs:
        paths = generate_fixed_length_paths(G, e1, path_length, num_paths)
        if selection_mode == "freq":
            scores = [
                (p, sum(1 for I in chi_itemsets if set(I).issubset(p)))
                for p in paths
            ]
        else:
            scores = []
            for p in paths:
                score = 0.0
                for I in chi_itemsets:
                    overlap = len(set(I).intersection(p)) / len(I) if I else 0
                    score += overlap * sims.get(I, 0.0)
                scores.append((p, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        chosen = [p for p, _ in scores[:top_k_paths]]
        for path in chosen:
            for I in chi_itemsets:
                overlap = len(set(I).intersection(path)) / len(I) if I else 0
                total += overlap * sims.get(I, 0.0)
    return total

if __name__ == "__main__":
    POIS_CSV   = "BJ_POIs.csv"
    ROADS_CSV  = "BJ_roads_with_poi.csv"
    TRAJ_CSV   = "bj_trajectories.csv"
    REL_CSV    = "bj_roadmap_edge.rel"

    threshold    = 300.0
    zipf_param   = 1.1
    min_visits   = 5
    topk_items   = 300
    max_lvls     = 5
    max_per_lvl  = 300
    chi          = 100

    stats = get_visited_trajectories_and_stats(
        POIS_CSV, ROADS_CSV, TRAJ_CSV,
        threshold, zipf_param, min_visits
    )
    cleaned          = stats["cleaned_trajectories"]
    freq_counts      = stats["edge_visit_counts"]
    nonvisit_counts  = stats["edge_nonvisit_counts"]
    total_counts     = {e: freq_counts.get(e,0) + nonvisit_counts.get(e,0)
                        for e in set(freq_counts) | set(nonvisit_counts)}
    categories       = stats["category_list"]
    poi_context      = stats["poi_context"]
    category_context = stats["category_context"]
    keywords         = stats["keywords"]
    roads            = stats["roads_df"]
    geo_col          = detect_column(roads, ["geoid","geo_id","geo id"])
    top5_edges_global= compute_top5_edges(freq_counts)
    G                = build_edge_graph(REL_CSV)

    query_context: Dict[str, Dict[str,float]] = {}
    for cat in categories:
        chosen = np.random.choice(keywords, 20, replace=False)
        w = np.random.rand(20); w /= w.sum()
        qc = {k:0.0 for k in keywords}
        for kw, wt in zip(chosen, w):
            qc[kw] = wt
        query_context[cat] = qc

    high_itemsets = {}
    freq_itemsets = {}
    high_sims_all = {}
    freq_sims_all = {}

    for cat in categories:
        paths = cleaned.loc[
            cleaned["visited_cats"].apply(lambda s, c=cat: c in s), "path"
        ].tolist()
        print(f"Category {cat}: {len(paths)} transactions")

        util_build_start = time.time()
        high_chi, sims_high = high_utility_pattern_mining(
            paths, freq_counts,
            k=topk_items,
            max_levels=max_lvls,
            max_itemsets_per_level=max_per_lvl,
            poi_context=poi_context,
            category_context=category_context[cat],
            query_context=query_context[cat],
            roads_df=roads,
            geo_col=geo_col,
            keywords=keywords,
            chi=chi,
            edge_utility=stats["edge_utility"]
        )
        util_build_time = time.time() - util_build_start
        print(str(cat))
        print(str(util_build_time))
        high_itemsets[cat] = high_chi
        high_sims_all[cat]  = sims_high

        freq_build_start = time.time()
        freq_chi, sims_freq = frequent_pattern_mining(
            paths, total_counts,
            k=topk_items,
            max_levels=max_lvls,
            max_itemsets_per_level=max_per_lvl,
            query_context=query_context[cat],
            poi_context=poi_context,
            roads_df=roads,
            geo_col=geo_col,
            keywords=keywords,
            chi=chi
        )
        freq_build_time = time.time() - freq_build_start
        print(str(cat))
        print(str(freq_build_time))
        freq_itemsets[cat] = freq_chi
        freq_sims_all[cat] = sims_freq

    # experiment: vary one parameter at a time
    k_paths_list     = list(range(5, 15))
    num_paths_list   = list(range(50, 151, 10))
    path_lengths_list= list(range(20, 101, 20))

    DEFAULT_K_PATHS     = 10
    DEFAULT_NUM_PATHS   = 100
    DEFAULT_PATH_LENGTH = 60

    results = []

    # vary top_k_paths
    for top_k_paths in k_paths_list:
        total_high = total_freq = util_time = freq_time = 0.0
        for cat in categories:
            t0 = time.time()
            hi = compute_path_utility(
                G, top5_edges_global,
                high_itemsets[cat], high_sims_all[cat],
                DEFAULT_NUM_PATHS, top_k_paths,
                selection_mode="utility",
                path_length=DEFAULT_PATH_LENGTH
            )
            util_time += time.time() - t0
            total_high += hi

            t1 = time.time()
            fe = compute_path_utility(
                G, top5_edges_global,
                freq_itemsets[cat], freq_sims_all[cat],
                DEFAULT_NUM_PATHS, top_k_paths,
                selection_mode="freq",
                path_length=DEFAULT_PATH_LENGTH
            )
            freq_time += time.time() - t1
            total_freq += fe

        results.append({
            'top_k_paths': top_k_paths,
            'num_paths': DEFAULT_NUM_PATHS,
            'path_length': DEFAULT_PATH_LENGTH,
            'util_time': util_time,
            'freq_time': freq_time,
            'total_high': total_high,
            'total_freq': total_freq
        })

    # vary num_paths
    for num_paths in num_paths_list:
        total_high = total_freq = util_time = freq_time = 0.0
        for cat in categories:
            t0 = time.time()
            hi = compute_path_utility(
                G, top5_edges_global,
                high_itemsets[cat], high_sims_all[cat],
                num_paths, DEFAULT_K_PATHS,
                selection_mode="utility",
                path_length=DEFAULT_PATH_LENGTH
            )
            util_time += time.time() - t0
            total_high += hi

            t1 = time.time()
            fe = compute_path_utility(
                G, top5_edges_global,
                freq_itemsets[cat], freq_sims_all[cat],
                num_paths, DEFAULT_K_PATHS,
                selection_mode="freq",
                path_length=DEFAULT_PATH_LENGTH
            )
            freq_time += time.time() - t1
            total_freq += fe

        results.append({
            'top_k_paths': DEFAULT_K_PATHS,
            'num_paths': num_paths,
            'path_length': DEFAULT_PATH_LENGTH,
            'util_time': util_time,
            'freq_time': freq_time,
            'total_high': total_high,
            'total_freq': total_freq
        })

    # vary path_length
    for path_length in path_lengths_list:
        total_high = total_freq = util_time = freq_time = 0.0
        for cat in categories:
            t0 = time.time()
            hi = compute_path_utility(
                G, top5_edges_global,
                high_itemsets[cat], high_sims_all[cat],
                DEFAULT_NUM_PATHS, DEFAULT_K_PATHS,
                selection_mode="utility",
                path_length=path_length
            )
            util_time += time.time() - t0
            total_high += hi

            t1 = time.time()
            fe = compute_path_utility(
                G, top5_edges_global,
                freq_itemsets[cat], freq_sims_all[cat],
                DEFAULT_NUM_PATHS, DEFAULT_K_PATHS,
                selection_mode="freq",
                path_length=path_length
            )
            freq_time += time.time() - t1
            total_freq += fe

        results.append({
            'top_k_paths': DEFAULT_K_PATHS,
            'num_paths': DEFAULT_NUM_PATHS,
            'path_length': path_length,
            'util_time': util_time,
            'freq_time': freq_time,
            'total_high': total_high,
            'total_freq': total_freq
        })

    # print results as CSV
    print("top_k_paths,num_paths,path_length,util_time,freq_time,total_high,total_freq")
    for r in results:
        print(f"{r['top_k_paths']},{r['num_paths']},"
              f"{r['path_length']},{r['util_time']:.2f},"
              f"{r['freq_time']:.2f},"
              f"{r['total_high']:.2f},{r['total_freq']:.2f}")