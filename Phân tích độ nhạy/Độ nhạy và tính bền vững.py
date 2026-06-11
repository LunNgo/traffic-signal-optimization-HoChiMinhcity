"""
=============================================================================
PHÂN TÍCH ĐỘ NHẠY LƯU LƯỢNG – MẠNG LƯỚI 26 NÚT (ĐBP / VTS / LCT)
Thuật toán : NSGA-II + Local Search  |  Trọng số : Dynamic AHP
Tính toán  : Kế thừa 100% công thức chuẩn xác từ hệ thống 26 nút gốc
Cập nhật   : Fix lỗi hiển thị giá trị ảo do cộng dồn Hàm Phạt (Penalty)
=============================================================================
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import time
import random
import os

warnings.filterwarnings('ignore')
os.makedirs('outputs', exist_ok=True)

plt.rcParams['font.family']       = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

COLORS = ['#1565C0', '#C62828', '#2E7D32', '#E65100', '#6A1B9A', '#00838F', '#F9A825']

# ===================== CẤU HÌNH & BIẾN TOÀN CỤC =====================
SCENARIOS = {
    "-30%": "Data_Giam_30.xlsx",
    "-10%": "Data_Giam_10.xlsx",
    "Base": "Data_Cao_Diem.xlsx", 
    "+10%": "Data_Tang_10.xlsx",
}

G_MIN, G_MAX = 15, 80
OFF_MIN, OFF_MAX = 0, 149

df_nodes = pd.DataFrame()
DISTANCES = []
L_MAX_LIST = [] 
BASELINE_C = []
BASELINE_G1 = []
BASELINE_G2 = []
BASELINE_OFF = []
BASELINE_F1 = []
BASELINE_F2 = []
BASELINE_F3 = []

C_MIN, C_MAX = 85, 100 

# ===================== DYNAMIC AHP =====================
def get_dynamic_ahp_matrix(label: str) -> np.ndarray:
    if label == "-30%":
        return np.array([[1.0, 3.0, 1/2], [1/3, 1.0, 1/4], [2.0, 4.0, 1.0]])
    elif label == "-10%":
        return np.array([[1.0, 3.0, 2.0], [1/3, 1.0, 1/2], [1/2, 2.0, 1.0]])
    elif label == "+10%":
        return np.array([[1.0, 2.0, 4.0], [1/2, 1.0, 3.0], [1/4, 1/3, 1.0]])
    else:  # Base
        return np.array([[1.0, 3.0, 4.0], [1/3, 1.0, 2.0], [1/4, 1/2, 1.0]])

def compute_ahp_weights(A: np.ndarray):
    n = A.shape[0]
    geo_means = np.array([np.prod(A[i, :])**(1/n) for i in range(n)])
    weights = geo_means / geo_means.sum()
    lambda_max = np.mean((A @ weights) / weights)
    CI = (lambda_max - n) / (n - 1)
    CR = CI / 0.58
    return weights, CR

def weighted_sum_score(fit, weights, f_min, f_max):
    score = 0
    for j in range(3):
        norm = (fit[j] - f_min[j]) / (f_max[j] - f_min[j]) if f_max[j] > f_min[j] else 0
        score += weights[j] * norm
    return score

# ===================== ĐỌC DỮ LIỆU & RÀNG BUỘC CHU KỲ =====================
def get_actual_filename(base_name):
    csv_name = base_name + " - Sheet1.csv"
    if os.path.exists(csv_name): return csv_name
    if os.path.exists(base_name): return base_name
    return base_name

def load_data_for_scenario(scenario_label, base_filename):
    global df_nodes, DISTANCES, BASELINE_C, BASELINE_G1, BASELINE_G2, BASELINE_OFF
    global BASELINE_F1, BASELINE_F2, BASELINE_F3, L_MAX_LIST
    global C_MIN, C_MAX
    
    file_path = get_actual_filename(base_filename)
    
    try:
        if file_path.endswith('.csv'):
            df_nut = pd.read_csv(file_path)
        else:
            try:
                df_nut = pd.read_excel(file_path, header=3)
                if not any('q_total' in str(c) for c in df_nut.columns):
                    df_nut = pd.read_excel(file_path)
            except:
                df_nut = pd.read_excel(file_path)
    except Exception as e:
        raise Exception(f"Không thể đọc file {file_path}: {e}")

    col_q = next((c for c in df_nut.columns if 'q_total' in str(c) or c == 'q'), df_nut.columns[2])
    col_s = next((c for c in df_nut.columns if 'S' in str(c)), df_nut.columns[3])
    col_name = next((c for c in df_nut.columns if 'Tên' in str(c) or 'name' in str(c)), df_nut.columns[1])
    col_L = next((c for c in df_nut.columns if 'L(' in str(c) or c == 'L'), df_nut.columns[4])
    col_v = next((c for c in df_nut.columns if 'v' in str(c) and 'km' in str(c).lower()), df_nut.columns[5])
    col_qb = next((c for c in df_nut.columns if 'qb' in str(c)), df_nut.columns[6])
    col_g1 = next((c for c in df_nut.columns if 'Pha1' in str(c) or 'g1' in str(c)), df_nut.columns[7])
    col_g2 = next((c for c in df_nut.columns if 'Pha2' in str(c) or 'g2' in str(c)), df_nut.columns[8])
    col_off = next((c for c in df_nut.columns if 'Offset' in str(c) or 'o_bl' in str(c)), df_nut.columns[10])
    
    n_nodes = len(df_nut)
    
    nodes_dict = {
        'k': list(range(1, n_nodes + 1)),
        'name': df_nut[col_name].astype(str).tolist(),
        'q': df_nut[col_q].fillna(1500).astype(float).tolist(),
        'S': df_nut[col_s].fillna(3600).astype(float).tolist(),
        'L': df_nut[col_L].fillna(3).astype(float).tolist(),
        'v': df_nut[col_v].fillna(15.0).astype(float).tolist(),
        'qb': df_nut[col_qb].fillna(15).astype(float).tolist()
    }
    df_nodes = pd.DataFrame(nodes_dict)
    
    DISTANCES = [200.0] * n_nodes
    try:
        dist_file = get_actual_filename('Khoang_Cach_mang_luoi.xlsx')
        if dist_file.endswith('.csv'): df_lk = pd.read_csv(dist_file)
        else: df_lk = pd.read_excel(dist_file)
        col_tu_nut = next(c for c in df_lk.columns if 'Tu nut' in str(c) or 'Tu_Nut' in str(c))
        col_kc = next(c for c in df_lk.columns if 'Khoang_cach' in str(c) or 'Khoang_Cach' in str(c))
        for idx, row in df_lk.iterrows():
            tu = int(row.get(col_tu_nut, idx + 1))
            if tu - 1 < n_nodes: DISTANCES[tu - 1] = float(row.get(col_kc, 200.0))
    except:
        pass

    L_XE = 6.0
    N_LAN = 3
    ALPHA = 0.85
    
    L_MAX_LIST = []
    for d in DISTANCES:
        if d > 0:
            max_veh = round((d / L_XE) * N_LAN * ALPHA)
            L_MAX_LIST.append(max_veh)
        else:
            L_MAX_LIST.append(9999)
            
    BASELINE_C = [90] * n_nodes 
    BASELINE_G1 = df_nut[col_g1].fillna(50).astype(int).tolist()
    BASELINE_G2 = df_nut[col_g2].fillna(34).astype(int).tolist()
    BASELINE_OFF = df_nut[col_off].fillna(0).astype(int).tolist()

    max_L = max([float(l) for l in df_nodes['L']])
    absolute_c_min = max(60, int(2 * G_MIN + 2 * max_L))
    
    if scenario_label == "-30%": c_low, c_high = 65, 80
    elif scenario_label == "-10%": c_low, c_high = 80, 90
    elif scenario_label == "Base": c_low, c_high = 85, 95
    elif scenario_label == "+10%": c_low, c_high = 95, 110
    else: c_low, c_high = 85, 100
        
    C_MIN = max(absolute_c_min, c_low)
    C_MAX = max(C_MIN + 5, c_high)

    BASELINE_F1, BASELINE_F2, BASELINE_F3 = [], [], []
    for i in range(n_nodes):
        q, S, L, v, qb = df_nodes['q'][i], df_nodes['S'][i], df_nodes['L'][i], df_nodes['v'][i], df_nodes['qb'][i]
        c, g1, g2, off = BASELINE_C[i], BASELINE_G1[i], BASELINE_G2[i], BASELINE_OFF[i]
        q1, q2 = q * 0.6, q * 0.4
        
        d1 = calc_uniform_delay(q1, S*0.6, g1, c) + calc_incremental_delay(q1, S*0.6, g1, c) + calc_residual_delay(q1, qb*0.6)
        d2 = calc_uniform_delay(q2, S*0.4, g2, c) + calc_incremental_delay(q2, S*0.4, g2, c) + calc_residual_delay(q2, qb*0.4)
        BASELINE_F1.append((d1*q1 + d2*q2)/3600)
        
        lq1 = calc_lq1_uniform(q1, S*0.6, g1, c) + calc_lq2_random(q1, S*0.6, g1, c)
        lq2 = calc_lq1_uniform(q2, S*0.4, g2, c) + calc_lq2_random(q2, S*0.4, g2, c)
        BASELINE_F2.append(lq1 + lq2) 
        
        t_travel = DISTANCES[i] / (v * 1000/3600 + 1e-6)
        gamma = calc_gamma_wave(off, t_travel, c)
        ns1 = calc_node_stops(q1, S*0.6, g1, c, gamma)
        ns2 = calc_node_stops(q2, S*0.4, g2, c, gamma)
        BASELINE_F3.append(ns1 + ns2)

# ===================== HỆ THỐNG CÔNG THỨC TOÁN HỌC GỐC =====================
def calc_uniform_delay(q, S, g, C):
    if g <= 0 or S <= 0 or C <= 0: return 9999.0
    lam = g / C
    x = (q * C) / (S * g)
    num = C * ((1 - lam) ** 2)
    den = 2 * (1 - min(1.0, x) * lam)
    return max(0.0, num / den)

def calc_incremental_delay(q, S, g, C, T_period=1):
    if g <= 0 or S <= 0: return 0.0
    lam = g / C
    x = (q * C) / (S * g)
    c_cap = S * lam
    term = (8 * 0.5 * 1.0 * x) / (c_cap * T_period + 1e-6)
    d2 = 900 * T_period * ((x - 1) + np.sqrt(max(0.0, (x - 1) ** 2 + term)))
    return max(0.0, d2)

def calc_residual_delay(q, qb, t_accum=30, T_period=1.0):
    if q <= 0: return 0.0
    return (3600.0 * qb * t_accum) / (q * T_period + 1e-6)

def calc_lq1_uniform(q, S, g, C):
    if C <= 0 or g <= 0: return 0.0
    lam = g / C
    x = (q * C) / (S * g)
    x = min(0.95, x)
    r = max(0.0, C - g)
    num = (q / 3600.0) * (r ** 2)
    den = 2 * C * (1 - x * lam + 1e-6)
    return max(0.0, num / den)

def calc_lq2_random(q, S, g, C, T_period=1):
    if g <= 0 or S <= 0: return 0.0
    lam = g / C
    x = (q * C) / (S * g)
    if x <= 0.5: return 0.0
    c_cap = S * lam
    
    # Sử dụng công thức chuẩn với T_period
    term = ((x - 1) ** 2) + (8 * 0.5 * 1.0 * x) / (c_cap * T_period + 1e-6)
    lq2 = 0.25 * c_cap * T_period * ((x - 1) + np.sqrt(max(0.0, term)))
    
    return max(0.0, lq2)
def calc_gamma_wave(offset_k, t_travel, C, beta=0.3):
    diff = abs(offset_k - t_travel) % C
    if diff > C / 2: diff = C - diff
    return max(0.05, 1.0 - beta * np.exp(-diff / (C / 4 + 1e-6)))

def calc_node_stops(q, S, g, C, gamma):
    if C <= 0 or q <= 0: return 0.0
    x = (q * C) / (S * g + 1e-6)
    term_red = (C - g) / C
    term_sat = 1.0 / (1.0 - min(0.95, x) + 1e-6)
    return q * term_red * term_sat * gamma

# BỔ SUNG: Tính toán giá trị thực (không cộng Penalty) dùng để in ra báo cáo
def calculate_real_objectives(individual):
    n = len(df_nodes)
    G1 = individual[:n]
    OFF = individual[n:2*n]
    c = individual[-1]

    f1_real, f2_real, f3_real = 0.0, 0.0, 0.0

    for i in range(n):
        q, S, L, v, qb = df_nodes['q'][i], df_nodes['S'][i], df_nodes['L'][i], df_nodes['v'][i], df_nodes['qb'][i]
        g1, off_k = G1[i], OFF[i]
        g2 = max(G_MIN, c - g1 - 2*int(L))
        q1, q2 = q * 0.6, q * 0.4
        S1, S2 = S * 0.6, S * 0.4

        d1 = calc_uniform_delay(q1, S1, g1, c) + calc_incremental_delay(q1, S1, g1, c) + calc_residual_delay(q1, qb*0.6)
        d2 = calc_uniform_delay(q2, S2, g2, c) + calc_incremental_delay(q2, S2, g2, c) + calc_residual_delay(q2, qb*0.4)
        f1_real += (d1 * q1 + d2 * q2) / 3600.0

        lq1 = calc_lq1_uniform(q1, S1, g1, c) + calc_lq2_random(q1, S1, g1, c)
        lq2 = calc_lq1_uniform(q2, S2, g2, c) + calc_lq2_random(q2, S2, g2, c)
        f2_real += (lq1 + lq2) 

        t_travel = DISTANCES[i] / (v * 1000/3600 + 1e-6)
        gamma = calc_gamma_wave(off_k, t_travel, c)
        ns1 = calc_node_stops(q1, S1, g1, c, gamma)
        ns2 = calc_node_stops(q2, S2, g2, c, gamma)
        f3_real += (ns1 + ns2)

    return f1_real, f2_real, f3_real

def evaluate_individual(individual):
    n = len(df_nodes)
    G1 = individual[:n]
    OFF = individual[n:2*n]
    c = individual[-1]

    f1_total, f2_total, f3_total = 0.0, 0.0, 0.0
    X_MAX = 0.95 

    for i in range(n):
        q, S, L, v, qb = df_nodes['q'][i], df_nodes['S'][i], df_nodes['L'][i], df_nodes['v'][i], df_nodes['qb'][i]
        g1, off_k = G1[i], OFF[i]
        g2 = max(G_MIN, c - g1 - 2*int(L))
        q1, q2 = q * 0.6, q * 0.4
        S1, S2 = S * 0.6, S * 0.4

        d1 = calc_uniform_delay(q1, S1, g1, c) + calc_incremental_delay(q1, S1, g1, c) + calc_residual_delay(q1, qb*0.6)
        d2_delay = calc_uniform_delay(q2, S2, g2, c) + calc_incremental_delay(q2, S2, g2, c) + calc_residual_delay(q2, qb*0.4)
        f1_total += (d1 * q1 + d2_delay * q2) / 3600.0

        x1 = (q1 * c) / (S1 * g1 + 1e-6)
        x2 = (q2 * c) / (S2 * g2 + 1e-6)
        
        penalty_x = 0
        if x1 > X_MAX: penalty_x += (x1 - X_MAX) * 10000 
        if x2 > X_MAX: penalty_x += (x2 - X_MAX) * 10000
            
        f1_total += penalty_x
        f2_total += penalty_x
        f3_total += penalty_x

        lq1 = calc_lq1_uniform(q1, S1, g1, c) + calc_lq2_random(q1, S1, g1, c)
        lq2_queue = calc_lq1_uniform(q2, S2, g2, c) + calc_lq2_random(q2, S2, g2, c)
        
        penalty_lq = 0
        if lq1 > L_MAX_LIST[i]: penalty_lq += (lq1 - L_MAX_LIST[i]) * 1000
        if lq2_queue > L_MAX_LIST[i]: penalty_lq += (lq2_queue - L_MAX_LIST[i]) * 1000
            
        f1_total += penalty_lq
        f2_total += (lq1 + lq2_queue) + penalty_lq
        f3_total += penalty_lq

        t_travel = DISTANCES[i] / (v * 1000/3600 + 1e-6)
        gamma = calc_gamma_wave(off_k, t_travel, c)
        ns1 = calc_node_stops(q1, S1, g1, c, gamma)
        ns2 = calc_node_stops(q2, S2, g2, c, gamma)
        f3_total += (ns1 + ns2)

    return f1_total, f2_total, f3_total

# ===================== CẤU TRÚC INDIVIDUAL & NSGA-II =====================
def repair_individual(ind):
    n = len(df_nodes)
    ind = list(ind)
    c = int(np.clip(ind[-1], C_MIN, C_MAX))
    ind[-1] = c
    for i in range(n):
        L = int(df_nodes['L'][i])
        g1 = int(np.clip(ind[i], G_MIN, G_MAX))
        if c - g1 - 2*L < G_MIN:
            g1 = int(np.clip(c - G_MIN - 2*L, G_MIN, G_MAX))
        ind[i] = g1
        ind[n+i] = int(ind[n+i] % c) if c > 0 else 0
    return ind

def create_individual(force_c=None):
    n = len(df_nodes)
    c = force_c if force_c else random.randint(C_MIN, C_MAX)
    G1, OFF = [], []
    for i in range(n):
        L = int(df_nodes['L'][i])
        max_g = max(G_MIN, c - G_MIN - 2*L)
        G1.append(random.randint(G_MIN, max(G_MIN, max_g)))
        OFF.append(random.randint(OFF_MIN, c-1))
    return repair_individual(G1 + OFF + [c])

def crossover(p1, p2):
    n = len(p1)
    c1, c2 = list(p1), list(p2)
    for i in range(n):
        if random.random() < 0.5:
            c1[i], c2[i] = c2[i], c1[i]
    return repair_individual(c1), repair_individual(c2)

def mutate(ind, pm=0.2):
    n = len(df_nodes)
    ind = list(ind)
    if random.random() < 0.5: 
        for _ in range(random.randint(2, 6)):
            k = random.randint(0, 2*n) 
            if k < n: 
                ind[k] += random.choice([-5, -3, 3, 5])
            elif k < 2*n: 
                ind[k] += random.choice([-10, 10])
            else: 
                ind[k] += random.choice([-8, 0, 8])
    return repair_individual(ind)

def dominates(a, b):
    return all(ai <= bi for ai, bi in zip(a, b)) and any(ai < bi for ai, bi in zip(a, b))

def fast_non_dominated_sort(pop_fit):
    n = len(pop_fit)
    S_dom, n_dom, rank = [[] for _ in range(n)], [0]*n, [0]*n
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q: continue
            if dominates(pop_fit[p], pop_fit[q]): S_dom[p].append(q)
            elif dominates(pop_fit[q], pop_fit[p]): n_dom[p] += 1
        if n_dom[p] == 0:
            rank[p] = 1
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S_dom[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    rank[q] = i + 2
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return fronts[:-1], rank

def crowding_distance(pop_fit, front):
    n = len(front)
    if n == 0: return []
    dist = [0.0]*n
    for obj in range(len(pop_fit[0])):
        sorted_idx = sorted(range(n), key=lambda i: pop_fit[front[i]][obj])
        dist[sorted_idx[0]] = dist[sorted_idx[-1]] = float('inf')
        f_min, f_max = pop_fit[front[sorted_idx[0]]][obj], pop_fit[front[sorted_idx[-1]]][obj]
        if f_max == f_min: continue
        for k in range(1, n-1):
            dist[sorted_idx[k]] += (pop_fit[front[sorted_idx[k+1]]][obj] - pop_fit[front[sorted_idx[k-1]]][obj]) / (f_max - f_min)
    return dist

def select_tournament(pop, pop_fit, ranks, crowding, k=3):
    candidates = random.sample(range(len(pop)), k)
    best = candidates[0]
    for c in candidates[1:]:
        if ranks[c] < ranks[best] or (ranks[c] == ranks[best] and crowding[c] > crowding[best]):
            best = c
    return pop[best]

def nsga2(pop_size=120, n_gen=80, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    pop = [repair_individual(BASELINE_G1 + BASELINE_OFF + [90])]
    pop += [create_individual(force_c=90) for _ in range(pop_size // 3)]
    pop += [create_individual() for _ in range(pop_size - 1 - pop_size // 3)]

    pop_fit = [evaluate_individual(ind) for ind in pop]
    fronts, ranks = fast_non_dominated_sort(pop_fit)
    crowding = [0.0]*pop_size
    for front in fronts:
        cd = crowding_distance(pop_fit, front)
        for j, idx in enumerate(front): crowding[idx] = cd[j]

    for gen in range(n_gen):
        offspring = []
        while len(offspring) < pop_size:
            p1 = select_tournament(pop, pop_fit, ranks, crowding)
            p2 = select_tournament(pop, pop_fit, ranks, crowding)
            c1, c2 = crossover(p1, p2) 
            offspring.extend([mutate(c1), mutate(c2)])
        offspring = offspring[:pop_size]

        combined = pop + offspring
        combined_fit = pop_fit + [evaluate_individual(ind) for ind in offspring]
        fronts_c, ranks_c = fast_non_dominated_sort(combined_fit)
        
        crowding_c = [0.0]*len(combined)
        for front in fronts_c:
            cd = crowding_distance(combined_fit, front)
            for j, idx in enumerate(front): crowding_c[idx] = cd[j]

        pop, pop_fit, ranks, crowding = [], [], [], []
        for front in fronts_c:
            if len(pop) + len(front) <= pop_size:
                for idx in front:
                    pop.append(combined[idx]); pop_fit.append(combined_fit[idx])
                    ranks.append(ranks_c[idx]); crowding.append(crowding_c[idx])
            else:
                rem = pop_size - len(pop)
                sorted_front = sorted(front, key=lambda i: -crowding_c[i])
                for idx in sorted_front[:rem]:
                    pop.append(combined[idx]); pop_fit.append(combined_fit[idx])
                    ranks.append(ranks_c[idx]); crowding.append(crowding_c[idx])
                break

    pareto_idx = [i for i in range(len(pop)) if ranks[i] == 1]
    return [pop[i] for i in pareto_idx], [pop_fit[i] for i in pareto_idx]

# ===================== LOCAL SEARCH (AHP & f_min, f_max) =====================
def local_search(pareto_pop, pareto_fits, weights, n_iter=60):
    all_fits = np.array(pareto_fits)
    f_min, f_max = all_fits.min(axis=0), all_fits.max(axis=0)

    scores = [weighted_sum_score(f, weights, f_min, f_max) for f in pareto_fits]
    top_k = max(1, int(len(pareto_pop)*0.3)) 
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i])[:top_k]

    best_ind, best_fit, best_score = pareto_pop[top_idx[0]], pareto_fits[top_idx[0]], scores[top_idx[0]]

    for idx in top_idx:
        ind, fit, score = list(pareto_pop[idx]), list(pareto_fits[idx]), scores[idx]
        for _ in range(n_iter):
            neighbor = list(ind)
            n_var = len(df_nodes)
            for k in random.sample(range(2*n_var + 1), random.randint(2, 5)):
                if k < n_var: 
                    neighbor[k] += random.choice([-2, 0, 2])
                elif k < 2*n_var: 
                    neighbor[k] += random.choice([-8, 8])
                else: 
                    neighbor[k] += random.choice([-5, 0, 5])
            
            neighbor = repair_individual(neighbor)
            new_fit = evaluate_individual(neighbor)
            new_score = weighted_sum_score(new_fit, weights, f_min, f_max)

            if new_score < score:
                ind, fit, score = neighbor, list(new_fit), new_score

        if score < best_score:
            best_ind, best_fit, best_score = ind, fit, score

    return best_ind, best_fit, best_score

# ===================== TRỰC QUAN HÓA =====================
def plot_sensitivity(labels, f1_list, f2_list, f3_list, c_list, w_list, bl_f1, bl_f2, bl_f3):
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        'PHÂN TÍCH ĐỘ NHẠY LƯU LƯỢNG – MẠNG LƯỚI 26 NÚT\n'
        'Thuật toán: NSGA-II + Local Search  |  Trọng số: Dynamic AHP',
        fontsize=14, fontweight='bold', y=0.98
    )

    x      = np.arange(len(labels))
    bar_w  = 0.32
    bl_idx = labels.index("Base") if "Base" in labels else 2

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.bar(x - bar_w/2, bl_f1,  bar_w, label='Baseline', color='#90CAF9', edgecolor='white')
    ax1.bar(x + bar_w/2, f1_list, bar_w, label='Tối ưu',   color=COLORS[0], edgecolor='white')
    for i, (bv, ov) in enumerate(zip(bl_f1, f1_list)):
        imp = (bv - ov) / bv * 100 if bv > 0 else 0
        ax1.text(i, max(bv, ov) * 1.02, f'{imp:+.1f}%',
                 ha='center', fontsize=7.5, color='#1B5E20' if imp > 0 else '#B71C1C', fontweight='bold')
    ax1.set_title('f1 – Tổng thời gian trễ (xe.h)', fontweight='bold')
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontweight='bold')
    ax1.set_ylabel('xe.h'); ax1.legend(fontsize=9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:,.0f}'))

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.bar(x - bar_w/2, bl_f2,  bar_w, label='Baseline', color='#FFCCBC', edgecolor='white')
    ax2.bar(x + bar_w/2, f2_list, bar_w, label='Tối ưu',   color=COLORS[1], edgecolor='white')
    for i, (bv, ov) in enumerate(zip(bl_f2, f2_list)):
        imp = (bv - ov) / bv * 100 if bv > 0 else 0
        ax2.text(i, max(bv, ov) * 1.02, f'{imp:+.1f}%',
                 ha='center', fontsize=7.5, color='#1B5E20' if imp > 0 else '#B71C1C', fontweight='bold')
    ax2.set_title('f2 – Chiều dài hàng chờ (xe)', fontweight='bold')
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontweight='bold')
    ax2.set_ylabel('xe'); ax2.legend(fontsize=9)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.bar(x - bar_w/2, np.array(bl_f3)/1e3,  bar_w,
            label='Baseline', color='#C8E6C9', edgecolor='white')
    ax3.bar(x + bar_w/2, np.array(f3_list)/1e3, bar_w,
            label='Tối ưu',   color=COLORS[2],  edgecolor='white')
    for i, (bv, ov) in enumerate(zip(bl_f3, f3_list)):
        imp = (bv - ov) / bv * 100 if bv > 0 else 0
        ax3.text(i, max(bv, ov)/1e3 * 1.02, f'{imp:+.1f}%',
                 ha='center', fontsize=7.5, color='#1B5E20' if imp > 0 else '#B71C1C', fontweight='bold')
    ax3.set_title('f3 – Số lần dừng xe (×1000 lượt/h)', fontweight='bold')
    ax3.set_xticks(x); ax3.set_xticklabels(labels, fontweight='bold')
    ax3.set_ylabel('×1000 lượt/h'); ax3.legend(fontsize=9)

    ax4 = fig.add_subplot(2, 3, 4)
    bars = ax4.bar(x, c_list, color=COLORS[3], edgecolor='white', width=0.5, alpha=0.88)
    ax4.axhline(90, ls='--', color='gray', lw=1.5, label='Baseline C=90s')
    for bar, cv in zip(bars, c_list):
        ax4.text(bar.get_x() + bar.get_width()/2, cv + 0.5,
                 f'{int(cv)}s', ha='center', fontweight='bold', fontsize=10)
    ax4.set_title('Chu kỳ chung C tối ưu theo kịch bản\n(C tăng khi lưu lượng tăng)', fontweight='bold')
    ax4.set_xticks(x); ax4.set_xticklabels(labels, fontweight='bold')
    ax4.set_ylabel('Chu kỳ C (giây)')
    ax4.set_ylim(60, 125)
    ax4.legend(fontsize=9)

    ax5 = fig.add_subplot(2, 3, 5)
    w_arr = np.array(w_list)
    for j, (col, lbl) in enumerate(zip([COLORS[0], COLORS[1], COLORS[2]],
                                        ['w1 (Độ trễ f1)', 'w2 (Hàng chờ f2)', 'w3 (Dừng xe f3)'])):
        ax5.plot(x, w_arr[:, j], 'o-', color=col, lw=2.5, markersize=8, label=lbl)
    ax5.set_title('Trọng số AHP động theo kịch bản lưu lượng', fontweight='bold')
    ax5.set_xticks(x); ax5.set_xticklabels(labels, fontweight='bold')
    ax5.set_ylabel('Trọng số w')
    ax5.set_ylim(0, 0.8)
    ax5.legend(fontsize=9)

    ax6 = fig.add_subplot(2, 3, 6)
    cats     = ['f1 (Trễ)', 'f2 (Hàng chờ)', 'f3 (Dừng)', 'Chu kỳ C']
    baseline = [bl_f1[bl_idx], bl_f2[bl_idx], bl_f3[bl_idx]/1e3, 90.0]

    for i, lbl in enumerate(labels):
        opt  = [f1_list[i], f2_list[i], f3_list[i]/1e3, c_list[i]]
        imps = [(b - o) / b * 100 for b, o in zip(baseline, opt)]
        ax6.plot(cats, imps, 'o-', color=COLORS[i],
                 lw=2, markersize=7, label=lbl, alpha=0.85)

    ax6.axhline(0, ls='--', color='gray', lw=1)
    ax6.set_title('% Cải thiện so với kịch bản Base\n(dương = giảm = tốt hơn)', fontweight='bold')
    ax6.set_ylabel('% Thay đổi')
    ax6.legend(fontsize=8, ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = 'outputs/00_Sensitivity_Analysis_Full.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out

def plot_pareto_distribution(all_pareto, labels):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle('PHÂN TÍCH ĐỘ NHẠY BẰNG PHÂN BỐ TẬP NGHIỆM PARETO\n'
                 '(Sự dịch chuyển của không gian nghiệm tối ưu khi lưu lượng thay đổi)', 
                 fontweight='bold', fontsize=14, y=1.02)

    f1_data = [[p[0] for p in front] if front else [0] for front in all_pareto]
    f2_data = [[p[1] for p in front] if front else [0] for front in all_pareto]
    f3_data = [[p[2]/1e3 for p in front] if front else [0] for front in all_pareto]

    metrics = [
        (f1_data, 'Biến động Độ trễ f1 (xe.h)', 'Tổng thời gian trễ (xe.h)'),
        (f2_data, 'Biến động Hàng chờ f2 (xe)', 'Chiều dài hàng chờ (xe)'),
        (f3_data, 'Biến động Số lần dừng f3 (×1000 lượt)', 'Số lần dừng (×1000 lượt/h)')
    ]

    for idx, (data, title, ylabel) in enumerate(metrics):
        bplot = axes[idx].boxplot(data, patch_artist=True, labels=labels, 
                                  medianprops=dict(color="black", linewidth=1.5))
        axes[idx].set_title(title, fontweight='bold')
        axes[idx].set_ylabel(ylabel)
        axes[idx].yaxis.grid(True, linestyle='--', alpha=0.7)
        
        for patch, color in zip(bplot['boxes'], COLORS[:len(labels)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    plt.tight_layout()
    out = 'outputs/01_Pareto_Distribution.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out

# ===================== MAIN =====================
def main():
    print("=" * 80)
    print("  TỐI ƯU HOÁ & PHÂN TÍCH ĐỘ NHẠY – MẠNG LƯỚI 26 NÚT  |  DYNAMIC AHP")
    print("=" * 80)

    labels = list(SCENARIOS.keys())
    res_f1, res_f2, res_f3, res_C, res_Fit, res_W = [], [], [], [], [], []
    bl_f1_sc, bl_f2_sc, bl_f3_sc, all_pareto = [], [], [], []

    for label, path in SCENARIOS.items():
        t0 = time.time()
        print(f"\n{'─'*70}")
        print(f"  [KỊCH BẢN] {label}  ←  {path}")

        try:
            load_data_for_scenario(label, path)
        except Exception as e:
            print(f"  [CẢNH BÁO] {e} - Đang bỏ qua...")
            continue

        n = len(df_nodes)
        print(f"  [DATA]  {n} nút | C_Tìm_Kiếm = [{C_MIN}, {C_MAX}]s ")

        weights, CR = compute_ahp_weights(get_dynamic_ahp_matrix(label))
        print(f"  [AHP]   w₁={weights[0]:.3f} w₂={weights[1]:.3f} w₃={weights[2]:.3f}  CR={CR:.4f}")
        
        bl_fit = [sum(BASELINE_F1), sum(BASELINE_F2), sum(BASELINE_F3)]

        print("  [NSGA-II] chạy tiến hoá …", end='', flush=True)
        pareto_pop, pareto_fits = nsga2(pop_size=120, n_gen=80, seed=42)
        print(f" xong ({len(pareto_pop)} nghiệm Pareto)")

        print("  [LS]    tinh chỉnh nghiệm AHP …", end='', flush=True)
        best_ind, best_fit, best_sc = local_search(pareto_pop, pareto_fits, weights, n_iter=60)
        print(f" xong  fitness={best_sc:.5f}")

        # BỔ SUNG: Tính toán giá trị THỰC TẾ (bỏ penalty) để in báo cáo chính xác
        real_f1, real_f2, real_f3 = calculate_real_objectives(best_ind)

        res_f1.append(real_f1); res_f2.append(real_f2); res_f3.append(real_f3)
        res_C.append(int(best_ind[-1])); res_Fit.append(best_sc); res_W.append(weights.tolist())
        bl_f1_sc.append(bl_fit[0]); bl_f2_sc.append(bl_fit[1]); bl_f3_sc.append(bl_fit[2])
        all_pareto.append(pareto_fits)
        
        dt = time.time() - t0
        print(f"  ✔  C={int(best_ind[-1])}s | f₁={real_f1:,.1f} | f₂={real_f2:,.1f} | f₃={real_f3:,.0f}  [{dt:.1f}s]")

    if not all_pareto:
        print("Không có dữ liệu hợp lệ để hiển thị.")
        return

    print("\n" + "=" * 95)
    print("  BẢNG TỔNG HỢP KẾT QUẢ PHÂN TÍCH ĐỘ NHẠY  (dùng cho báo cáo)")
    print("=" * 95)

    ahp_labels = [
        "Ưu tiên f3 (Sóng xanh)",        # -30%
        "Ưu tiên nhẹ f1 (Độ trễ)",       # -10%
        "Ưu tiên tối đa f1 (Độ trễ)",    # Base
        "Ưu tiên f2 (Chống Gridlock)",   # +10%
    ][:len(res_C)]
    
    df_rep = pd.DataFrame({
        "Kịch bản"     : labels[:len(res_C)],
        "AHP ưu tiên"  : ahp_labels,
        "w₁ Trễ"       : [f'{w[0]:.3f}' for w in res_W],
        "w₂ Hàng chờ"  : [f'{w[1]:.3f}' for w in res_W],
        "w₃ Dừng"      : [f'{w[2]:.3f}' for w in res_W],
        "C tối ưu (s)" : res_C,
        "f₁ (xe.h)"    : np.round(res_f1, 1),
        "f₂ (xe)"      : np.round(res_f2, 1),
        "f₃ (lượt/h)"  : np.round(res_f3, 0),
    })
    print(df_rep.to_string(index=False))
    
    print("\n  % Cải thiện so với Baseline tương ứng:")
    print(f"  {'Kịch bản':>8}  {'Δf₁%':>8}  {'Δf₂%':>8}  {'Δf₃%':>8}")
    for i, lbl in enumerate(labels[:len(res_C)]):
        d1 = (bl_f1_sc[i] - res_f1[i]) / bl_f1_sc[i] * 100 if bl_f1_sc[i] > 0 else 0
        d2 = (bl_f2_sc[i] - res_f2[i]) / bl_f2_sc[i] * 100 if bl_f2_sc[i] > 0 else 0
        d3 = (bl_f3_sc[i] - res_f3[i]) / bl_f3_sc[i] * 100 if bl_f3_sc[i] > 0 else 0
        print(f"  {lbl:>8}   {d1:>+7.1f}%   {d2:>+7.1f}%   {d3:>+7.1f}%")

    print("=" * 95)
    print("\n[CHART] Đang vẽ biểu đồ …")
    plot_sensitivity(labels[:len(res_C)], res_f1, res_f2, res_f3, res_C, res_W, bl_f1_sc, bl_f2_sc, bl_f3_sc)
    plot_pareto_distribution(all_pareto, labels[:len(res_C)])
    print("\n[DONE] Kết quả đã lưu trong thư mục outputs/")

if __name__ == '__main__':
    main()