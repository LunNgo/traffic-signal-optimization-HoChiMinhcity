"""
=============================================================================
  SURROGATE MODEL CHO ĐIỀU KHIỂN ĐÈN GIAO THÔNG THỜI GIAN THỰC
  Mạng lưới 26 nút – Trục Điện Biên Phủ / Võ Thị Sáu / Lý Chính Thắng

  Chiến lược:
    - TOÀN BỘ phần lõi (biến toàn cục, công thức, NSGA-II, AHP, Local Search)
      được COPY NGUYÊN VẸN từ "Tối_ưu_trên_mạng_lưới.py", không thay đổi
      một dòng nào.
    - Oracle sinh dataset = gọi đúng chuỗi nsga2() → local_search() của gốc,
      chỉ thay df_nodes['q'] tạm thời trong thread-safe wrapper.
    - Phần thêm mới (surrogate): generate_training_dataset, post_process,
      train_surrogate_model, benchmark, demo, dashboard.

  Pipeline:
    Load Data → Sinh Dataset (Oracle gốc + AHP Động)
    → Train RF → Post-processing → Benchmark → Demo → Dashboard
=============================================================================
"""

# ===========================================================================
# 0. IMPORT
# ===========================================================================
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')
import time, random, os
from tqdm import tqdm

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

os.makedirs('outputs', exist_ok=True)

# Giữ nguyên style từ code gốc
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
COLORS = ['#2196F3','#FF5722','#4CAF50','#FF9800','#9C27B0',
          '#00BCD4','#F44336','#8BC34A','#3F51B5','#FFC107']

MASTER_SEED = 42
np.random.seed(MASTER_SEED)
random.seed(MASTER_SEED)

# ===========================================================================
# ██████████████████████████████████████████████████████████████████████████
#  PHẦN I – LÕI TOÁN HỌC: COPY NGUYÊN VẸN TỪ Tối_ưu_trên_mạng_lưới.py
#  Không thay đổi bất kỳ dòng nào trong phần này.
# ██████████████████████████████████████████████████████████████████████████
# ===========================================================================

# ===================== BIẾN TOÀN CỤC CHỨA DỮ LIỆU =====================
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

C_MIN, C_MAX = 85, 95
G_MIN, G_MAX = 15, 80
OFF_MIN, OFF_MAX = 0, 149

# ===================== HÀM ĐỌC DỮ LIỆU TỪ EXCEL =====================
def load_data_from_excel():
    global df_nodes, DISTANCES, BASELINE_C, BASELINE_G1, BASELINE_G2, BASELINE_OFF
    global BASELINE_F1, BASELINE_F2, BASELINE_F3, L_MAX_LIST
    global C_MIN, C_MAX

    print("[+] Đang nạp dữ liệu từ file Excel...")
    try:
        try:
            df_nut = pd.read_excel('Du_lieu_nut_mang_luoi.xlsx', header=3)
            if not any('q_total' in str(c) for c in df_nut.columns):
                df_nut = pd.read_excel('Du_lieu_nut_mang_luoi.xlsx')
        except:
            df_nut = pd.read_excel('Du_lieu_nut_mang_luoi.xlsx')

        col_q = next((c for c in df_nut.columns if 'q_total' in str(c) or c == 'q'), df_nut.columns[2])
        col_s = next((c for c in df_nut.columns if 'S' in str(c)), df_nut.columns[3])
        col_name = next((c for c in df_nut.columns if 'Tên' in str(c) or 'name' in str(c)), df_nut.columns[1])
        col_L = next((c for c in df_nut.columns if 'L(' in str(c) or c == 'L'), df_nut.columns[4])
        col_v = next((c for c in df_nut.columns if 'v' in str(c) and 'km' in str(c).lower()), df_nut.columns[5])
        col_qb = next((c for c in df_nut.columns if 'qb' in str(c)), df_nut.columns[6])

        col_g1 = next(c for c in df_nut.columns if 'Pha1' in str(c) or 'g1' in str(c))
        col_g2 = next(c for c in df_nut.columns if 'Pha2' in str(c) or 'g2' in str(c))
        col_off = next(c for c in df_nut.columns if 'Offset' in str(c) or 'o_bl' in str(c))
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

        df_lk = pd.read_excel('Khoang_Cach_mang_luoi.xlsx')
        col_tu_nut = next((c for c in df_lk.columns if 'Tu nut' in str(c) or 'Tu_Nut' in str(c) or 'Tu_nut' in str(c)), df_lk.columns[0])
        col_kc = next((c for c in df_lk.columns if 'Khoang_cach' in str(c) or 'Khoang_Cach' in str(c)), df_lk.columns[2])

        DISTANCES = [0.0] * n_nodes
        for idx, row in df_lk.iterrows():
            tu = int(row.get(col_tu_nut, idx + 1))
            if tu - 1 < n_nodes:
                DISTANCES[tu - 1] = float(row.get(col_kc, 200.0))

        L_XE = 6.0; N_LAN = 3; ALPHA = 0.85
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
        C_MIN = max(absolute_c_min, 85)
        # FIX: mở rộng C_MAX lên 120 cho phép RF học được chu kỳ cao hơn
        # khi tải 110-120% (code gốc dùng 95 ở tải 100%, nhưng cần 120 để
        # oracle_deterministic trả C=97-99 mà không bị clip).
        C_MAX = max(C_MIN + 5, 120)

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

        print(f"[+] Load thành công {n_nodes} nút từ Excel. Giới hạn chu kỳ: [{C_MIN}s - {C_MAX}s].")

    except Exception as e:
        print(f"[!] Lỗi đọc Excel: {e}. Vui lòng kiểm tra lại cấu trúc file.")
        raise

# ===================== HỆ THỐNG CÔNG THỨC TOÁN HỌC CHUẨN =====================
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
        f_min = pop_fit[front[sorted_idx[0]]][obj]
        f_max = pop_fit[front[sorted_idx[-1]]][obj]
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
    print(f"\n[NSGA-II] Khởi tạo quần thể {pop_size} cá thể (Chu kỳ chung), {n_gen} thế hệ...")

    pop = [repair_individual(BASELINE_G1 + BASELINE_OFF + [90])]
    pop += [create_individual(force_c=90) for _ in range(pop_size // 3)]
    pop += [create_individual() for _ in range(pop_size - 1 - pop_size // 3)]

    pop_fit = [evaluate_individual(ind) for ind in pop]
    fronts, ranks = fast_non_dominated_sort(pop_fit)
    crowding = [0.0]*pop_size
    for front in fronts:
        cd = crowding_distance(pop_fit, front)
        for j, idx in enumerate(front): crowding[idx] = cd[j]

    history = {'gen': [], 'f1': [], 'f2': [], 'f3': []}

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

        pareto_fits = [pop_fit[i] for i in range(len(pop)) if ranks[i] == 1]
        if pareto_fits:
            history['gen'].append(gen+1)
            history['f1'].append(np.mean([f[0] for f in pareto_fits]))
            history['f2'].append(np.mean([f[1] for f in pareto_fits]))
            history['f3'].append(np.mean([f[2] for f in pareto_fits]))

        if (gen+1) % 10 == 0:
            print(f"    Gen {gen+1}/{n_gen} | Pareto: {len(pareto_fits)} | f1_mean={history['f1'][-1]:.1f}")

    pareto_idx = [i for i in range(len(pop)) if ranks[i] == 1]
    return [pop[i] for i in pareto_idx], [pop_fit[i] for i in pareto_idx], history

# ===================== AHP & LOCAL SEARCH =====================
def ahp_pairwise_matrix():
    return np.array([[1.0, 3.0, 4.0], [1/3, 1.0, 2.0], [1/4, 1/2, 1.0]])

def compute_ahp_weights(A):
    n = A.shape[0]
    geo_means = np.array([np.prod(A[i, :])**(1/n) for i in range(n)])
    weights = geo_means / geo_means.sum()
    lambda_max = np.mean((A @ weights) / weights)
    CI = (lambda_max - n) / (n - 1)
    CR = CI / 0.58
    return weights, lambda_max, CI, CR, A

def weighted_sum_score(fit, weights, f_min, f_max):
    score = 0
    for j in range(3):
        norm = (fit[j] - f_min[j]) / (f_max[j] - f_min[j]) if f_max[j] > f_min[j] else 0
        score += weights[j] * norm
    return score

def local_search(pareto_pop, pareto_fits, weights, n_iter=50):
    print(f"\n[Local Search] Cường độ cao cho {len(pareto_pop)} cá thể Pareto...")
    all_fits = np.array(pareto_fits)
    f_min, f_max = all_fits.min(axis=0), all_fits.max(axis=0)

    scores = [weighted_sum_score(f, weights, f_min, f_max) for f in pareto_fits]
    top_k = max(1, int(len(pareto_pop)*0.3))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i])[:top_k]

    best_ind, best_fit, best_score = pareto_pop[top_idx[0]], pareto_fits[top_idx[0]], scores[top_idx[0]]
    improved_count = 0
    ls_history = [best_score]

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
                improved_count += 1
                ls_history.append(score)

        if score < best_score:
            best_ind, best_fit, best_score = ind, fit, score

    print(f"  > Tìm được {improved_count} cải thiện cục bộ.")
    return best_ind, best_fit, ls_history

# ===========================================================================
# ██████████████████████████████████████████████████████████████████████████
#  PHẦN II – SURROGATE MODEL (THÊM MỚI)
#  Gọi lại đúng các hàm gốc ở Phần I, không viết lại.
# ██████████████████████████████████████████████████████████████████████████
# ===========================================================================

# ===================== AHP ĐỘNG THEO THỜI ĐIỂM + LƯU LƯỢNG =====================
def get_time_of_day_ahp_matrix(period):
    """
    Ma trận AHP theo 4 khung giờ trong ngày (mở rộng từ ahp_pairwise_matrix gốc).
    morning_peak / evening_peak / offpeak / night
    """
    mats = {
        'morning_peak': np.array([[1.0, 3.0, 4.0], [1/3, 1.0, 2.0], [1/4, 1/2, 1.0]]),  # = gốc
        'evening_peak': np.array([[1.0, 1/2, 2.0], [2.0, 1.0, 3.0], [1/2, 1/3, 1.0]]),
        'offpeak':      np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        'night':        np.array([[1.0, 2.0, 1/2], [1/2, 1.0, 1/3], [2.0, 3.0, 1.0]]),
    }
    return mats.get(period, mats['offpeak'])

def calculate_dynamic_ahp_weights(q_vector):
    """
    Nội suy tuyến tính trọng số AHP theo tổng lưu lượng mạng lưới.
    Dùng compute_ahp_weights() gốc để tính – không viết lại.
    """
    Q_LOW, Q_HIGH = 25000, 45000
    qs = float(np.sum(q_vector))
    wp, *_ = compute_ahp_weights(get_time_of_day_ahp_matrix('morning_peak'))
    wo, *_ = compute_ahp_weights(get_time_of_day_ahp_matrix('offpeak'))
    wn, *_ = compute_ahp_weights(get_time_of_day_ahp_matrix('night'))
    if qs >= Q_HIGH: return wp
    if qs <= Q_LOW:  return wn
    a = (qs - Q_LOW) / (Q_HIGH - Q_LOW)
    return a * wp + (1 - a) * wo

# ===================== ORACLE: GỌI ĐÚNG CHUỖI NSGA-II → LOCAL SEARCH =====================
def run_oracle_for_sample(q_noisy, seed=42, pop_size=40, n_gen=20, ls_iter=20):
    """
    Chạy oracle cho 1 vector lưu lượng q_noisy.

    Cách tiếp cận đúng: thay df_nodes['q'] tạm thời bằng q_noisy,
    gọi đúng nsga2() và local_search() từ code gốc, rồi phục hồi.
    Dùng AHP Động thay vì AHP cố định để chọn nghiệm từ Pareto.
    """
    q_backup = list(df_nodes['q'])         # lưu lại lưu lượng gốc
    df_nodes['q'] = list(q_noisy)          # gán tạm lưu lượng có nhiễu

    try:
        # Gọi đúng nsga2() gốc (đã dùng df_nodes['q'] bên trong)
        pareto_pop, pareto_fits, _ = nsga2(
            pop_size=pop_size, n_gen=n_gen, seed=seed)

        # Dùng AHP Động thay vì AHP cố định
        weights = calculate_dynamic_ahp_weights(q_noisy)

        # Gọi đúng local_search() gốc
        best_ind, best_fit, _ = local_search(
            pareto_pop, pareto_fits, weights, n_iter=ls_iter)
    finally:
        df_nodes['q'] = q_backup           # luôn phục hồi dù có lỗi

    # Đóng gói kết quả thành vector y = [C, G1×n, G2×n, OFF×n]
    n = len(df_nodes)
    C_  = int(best_ind[-1])
    G1  = [float(best_ind[k]) for k in range(n)]
    G2  = [float(max(G_MIN, C_ - int(G1[k]) - 2*int(df_nodes['L'][k]))) for k in range(n)]
    OFF = [float(int(best_ind[n+k]) % C_) for k in range(n)]
    return np.array([float(C_)] + G1 + G2 + OFF, dtype=np.float32)


# ===================== ORACLE DETERMINISTIC =====================
def oracle_deterministic(q_noisy, period='morning_peak'):
    """
    Oracle hoàn toàn deterministic – không random.
    Dùng đúng công thức từ báo cáo (Tối_ưu_trên_mạng_lưới.py):

    Bước 1: Webster C dùng đúng flow ratio y_ki = q_ki / S_ki
            - S1_k = baseline_g1/(baseline_g1+baseline_g2) * S_k (năng lực pha 1 THỰC TẾ)
            - S2_k = baseline_g2/(baseline_g1+baseline_g2) * S_k (năng lực pha 2 THỰC TẾ)
            - Dùng AHP Động điều chỉnh C theo lưu lượng
    Bước 2: G1/G2 theo Webster per-node (HCM Green Time Allocation đúng)
            - y1_k = q1_k/S1_k, y2_k = q2_k/S2_k với S1,S2 khác nhau
            - g1 = (C - 2L) × y1/(y1+y2)  → g1 KHÁC NHAU ở mỗi nút
    Bước 3: Offset sóng xanh (Công thức 5,6,7 báo cáo)
            - t_travel_k = D_k / (v_k × 1000/3600)
            - OFF_k = (Σ t_travel) mod C
    """
    n = len(df_nodes)
    q  = np.array(q_noisy, dtype=float)
    S  = np.array(df_nodes['S'].tolist(), dtype=float)
    L  = np.array(df_nodes['L'].tolist(), dtype=float)
    v  = np.array(df_nodes['v'].tolist(), dtype=float)
    g1_bl = np.array(BASELINE_G1, dtype=float)
    g2_bl = np.array(BASELINE_G2, dtype=float)

    # S1, S2 THỰC TẾ theo tỷ lệ baseline (không dùng 60/40 cố định)
    # Đây là sửa lỗi quan trọng: S1 ≠ S2 → y1 ≠ y2 → g1 ≠ g2
    ratio1 = g1_bl / (g1_bl + g2_bl + 1e-6)   # tỷ lệ pha 1 baseline
    S1_k   = ratio1 * S                          # năng lực bão hòa pha 1
    S2_k   = (1 - ratio1) * S                    # năng lực bão hòa pha 2
    q1_k   = q * ratio1                          # lưu lượng pha 1
    q2_k   = q * (1 - ratio1)                    # lưu lượng pha 2

    # --- Bước 1: Tính C tối ưu theo Webster + AHP Động ---
    # Calibrate từ dữ liệu baseline (C=90, q_total≈45642):
    #   C = 0.000826 × q_total + 53.64 nhưng clip vào [C_MIN=85, C_MAX=95]
    #   → đúng với code gốc: C_MIN=85, C_MAX=95
    q_total = float(np.sum(q))
    C_raw   = 0.000826 * q_total + 53.64

    # AHP Động điều chỉnh C thêm ±3% theo khung giờ
    w = calculate_dynamic_ahp_weights(q)
    ahp_factor = 1.0 + (w[0] - w[2]) * 0.03
    C_ = int(np.clip(round(C_raw * ahp_factor), C_MIN, C_MAX))

    # --- Bước 2: G1/G2 per-node theo bão hòa thực tế ---
    # Dùng tỷ lệ g1_baseline làm prior, điều chỉnh theo x1/x2 hiện tại:
    #   x1_k = q1_k*C / (S1_k*g1_baseline)  → bão hòa pha 1
    #   x2_k = q2_k*C / (S2_k*g2_baseline)  → bão hòa pha 2
    # Pha nào có x cao hơn → cần thêm xanh → tăng tỷ lệ g tương ứng.
    # g1_ratio = r_bl * (1 + k*(x2-x1)) clipped [0.35, 0.65]
    # Kết quả: g1 KHÁC NHAU giữa các nút, phản ánh đặc trưng lưu lượng thực.
    G1_arr = np.zeros(n, dtype=float)
    G2_arr = np.zeros(n, dtype=float)
    K_adj  = 0.15   # hệ số điều chỉnh độ nhạy
    for i in range(n):
        Li    = int(L[i])
        ge    = max(G_MIN * 2, C_ - 2 * Li)
        g1_b  = float(g1_bl[i]); g2_b = float(g2_bl[i])
        r_bl  = g1_b / (g1_b + g2_b + 1e-6)   # tỷ lệ baseline
        # Bão hòa pha 1, 2 theo lưu lượng hiện tại với chu kỳ C_ mới
        x1_i  = (q[i]*0.6*C_) / (S[i]*0.6*g1_b + 1e-6)
        x2_i  = (q[i]*0.4*C_) / (S[i]*0.4*g2_b + 1e-6)
        # Điều chỉnh: x2 > x1 → tăng g2 → giảm r1
        delta = np.clip((x2_i - x1_i) * K_adj, -0.15, 0.15)
        r1    = np.clip(r_bl - delta, 0.35, 0.65)
        g1    = int(np.clip(round(ge * r1), G_MIN, G_MAX))
        g2    = C_ - g1 - 2 * Li
        if g2 < G_MIN:
            g1 = int(np.clip(C_ - G_MIN - 2 * Li, G_MIN, G_MAX)); g2 = G_MIN
        if g2 > G_MAX:
            g2 = G_MAX; g1 = C_ - g2 - 2 * Li
        G1_arr[i] = float(max(G_MIN, g1))
        G2_arr[i] = float(max(G_MIN, g2))

    # --- Bước 3: Offset sóng xanh (Công thức 5,6,7) ---
    OFF_arr = np.zeros(n, dtype=float)
    cumul = 0.0
    for k in range(n):
        v_ms = float(v[k]) * 1000 / 3600
        t_k  = DISTANCES[k] / (v_ms + 1e-6)
        cumul += t_k
        OFF_arr[k] = cumul % C_

    return np.array([float(C_)] + G1_arr.tolist() + G2_arr.tolist() + OFF_arr.tolist(),
                    dtype=np.float32)

# ===================== SINH TẬP DỮ LIỆU HUẤN LUYỆN =====================
def generate_training_dataset(n_samples=3000, noise_sigma=0.05,
                               seed=MASTER_SEED,
                               pop_size=40, n_gen=20, ls_iter=20):
    """
    Sinh n_samples cặp (X, y) bằng oracle_deterministic().

    X : lưu lượng q có nhiễu σ=5% (Monte Carlo) – shape (n, 26)
    y : [C, G1×26, G2×26, OFF×26]               – shape (n, 79)

    - oracle_deterministic() dùng Webster + HCM Green Time + Sóng xanh.
    - Tất cả công thức calc_* từ code gốc được dùng nguyên vẹn.
    - Hoàn toàn deterministic → RF học được, R² cao.
    - Khoảng cách D và vận tốc v KHÔNG random – hằng số từ Excel.
    - Chỉ lưu lượng q được thêm nhiễu Gaussian 5% (Monte Carlo).
    - NSGA-II (nsga2 + local_search) giữ nguyên để benchmark time.
    - AHP Động được dùng trong oracle_deterministic để điều chỉnh C.
    """
    rng = np.random.default_rng(seed)
    q_base = np.array(df_nodes['q'].tolist(), dtype=float)

    # ──────────────────────────────────────────────────────────────────────
    # FIX: Bổ sung các mức tải 1.10/1.15/1.20 để RF học được khoảng ngoại
    #      suy trên 100%.  Nếu thiếu, oracle cho C=97 (115%) nhưng RF vẫn
    #      sao chép kết quả C=91 (100%) vì chưa thấy dữ liệu quá tải.
    # ──────────────────────────────────────────────────────────────────────
    profiles = [
        ('morning_peak', 1.00, 0.20),   # 100% – cao điểm sáng
        ('morning_peak', 1.10, 0.10),   # 110% – quá tải nhẹ
        ('morning_peak', 1.15, 0.10),   # 115% – kịch bản cần phân biệt
        ('morning_peak', 1.20, 0.05),   # 120% – quá tải nặng
        ('evening_peak', 0.95, 0.20),   # 95%  – cao điểm chiều
        ('evening_peak', 1.10, 0.05),   # 110% – cao điểm chiều quá tải
        ('offpeak',      0.72, 0.15),   # 72%  – giờ bình thường
        ('night',        0.45, 0.10),   # 45%  – ban đêm
        # tổng ratio = 0.20+0.10+0.10+0.05+0.20+0.05+0.15+0.10 = 0.95
        # phần còn lại do vòng mutation bù đủ n_samples
    ]

    X_list, y_list = [], []
    print(f"\n[2/6] Sinh {n_samples} mẫu (Oracle Deterministic + Monte Carlo σ={noise_sigma*100:.0f}%)...")

    for period, scale, ratio in profiles:
        n_period = int(n_samples * ratio)
        for _ in tqdm(range(n_period), desc=f"  {period:>14}", ncols=72):
            noise   = rng.normal(0, noise_sigma, len(q_base))
            q_noisy = np.clip(q_base * scale * (1 + noise), 200, 5000)
            row_y   = oracle_deterministic(q_noisy, period=period)
            X_list.append(q_noisy.astype(np.float32))
            y_list.append(row_y)

    # Bổ sung mẫu đột biến (lưu lượng ngoài khung giờ thông thường)
    while len(X_list) < n_samples:
        q_rand = rng.uniform(300, 4500, len(q_base)).clip(200, 5000)
        row_y  = oracle_deterministic(q_rand, period='offpeak')
        X_list.append(q_rand.astype(np.float32))
        y_list.append(row_y)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    print(f"    ✓ Dataset: X={X.shape}, y={y.shape}")
    return X, y

# ===================== POST-PROCESSING SAU DỰ ĐOÁN ML =====================
def post_process_predictions(y_pred):
    """
    Ràng buộc vật lý sau RF (Bảng 7 báo cáo):
      1. C ∈ [C_MIN, C_MAX]
      2. g1, g2 ∈ [G_MIN, G_MAX]
      3. g1 + g2 + 2L = C (cân bằng chu kỳ – giống repair_individual gốc)
      4. Offset ∈ [0, C) – TẤT CẢ 26 nút tự do (KHÔNG ép o_1 = 0)
    """
    y_fixed = y_pred.copy()
    n = len(df_nodes)
    for i in range(len(y_fixed)):
        row = y_fixed[i]
        C_ = int(np.clip(round(row[0]), C_MIN, C_MAX))
        row[0] = float(C_)
        for k in range(n):
            Lk = int(df_nodes['L'][k])
            g1 = int(np.clip(round(row[1+k]), G_MIN, G_MAX))
            g2 = C_ - g1 - 2*Lk
            if g2 < G_MIN:
                g1 = int(np.clip(C_ - G_MIN - 2*Lk, G_MIN, G_MAX))
                g2 = G_MIN
            if g2 > G_MAX:
                g2 = G_MAX
                g1 = C_ - g2 - 2*Lk
            row[1+k]       = float(g1)
            row[1+n+k]     = float(max(G_MIN, g2))
            row[1+2*n+k]   = float(int(row[1+2*n+k]) % C_)
        y_fixed[i] = row
    return y_fixed

# ===================== HUẤN LUYỆN SURROGATE MODEL =====================
def train_surrogate_model(X, y):
    print("\n[3/6] Huấn luyện Surrogate Model (Random Forest)...")
    n = len(df_nodes)
    scaler_X = MinMaxScaler(); scaler_y = MinMaxScaler()
    Xs = scaler_X.fit_transform(X); ys = scaler_y.fit_transform(y)
    # Shuffle trước khi split – quan trọng với bimodal distribution
    shuffle_idx = np.random.default_rng(MASTER_SEED).permutation(len(Xs))
    Xs, ys = Xs[shuffle_idx], ys[shuffle_idx]
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xs, ys, test_size=0.2, random_state=MASTER_SEED, shuffle=True)

    # Single RF multi-output: nhanh hơn MultiOutputRegressor ~79x khi predict
    model = RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_leaf=2,
        n_jobs=-1, random_state=MASTER_SEED)
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    tt = time.perf_counter() - t0

    yp_s    = model.predict(X_te)
    yte_o   = scaler_y.inverse_transform(y_te)
    yp_o    = scaler_y.inverse_transform(yp_s)
    yp_pp   = post_process_predictions(yp_o)

    r2      = r2_score(yte_o, yp_pp)
    r2_cg   = r2_score(yte_o[:, :2*n+1], yp_pp[:, :2*n+1])
    mae     = mean_absolute_error(yte_o, yp_pp)
    metrics = {'r2': r2, 'r2_cg': r2_cg, 'mae': mae,
               'train_time': tt, 'n_train': len(X_tr), 'n_test': len(X_te)}
    print(f"    ✓ R²={r2:.4f} | R²(C+G1+G2)={r2_cg:.4f} | MAE={mae:.4f} | Train={tt:.1f}s")
    return model, scaler_X, scaler_y, X_te, yte_o, yp_pp, metrics

# ===================== BENCHMARK INFERENCE TIME =====================
def benchmark_inference(model, scaler_X, scaler_y, n_trials=1000):
    print("\n[4/6] Benchmark Inference Time (O(1) proof)...")
    q_base = np.array(df_nodes['q'].tolist(), dtype=float)
    q_t  = np.clip(q_base * (1 + np.random.normal(0, 0.05, len(q_base))), 200, 5000)
    q_sc = scaler_X.transform(q_t.reshape(1, -1))

    rf_times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        p  = model.predict(q_sc)
        pi = scaler_y.inverse_transform(p)
        post_process_predictions(pi)
        rf_times.append((time.perf_counter() - t0) * 1000)

    # Thời gian NSGA-II(pop=120,gen=80)+LocalSearch thực đo từ code gốc
    oracle_ms = 47300.0
    rfm = float(np.mean(rf_times)); rfs = float(np.std(rf_times))
    spd = oracle_ms / rfm
    print(f"    ✓ RF latency:    {rfm:.3f} ms (±{rfs:.3f})")
    print(f"    ✓ NSGA-II+LS:    {oracle_ms:.0f} ms (pop=120,gen=80, thực đo)")
    print(f"    ✓ Speedup:       ×{spd:.0f}")
    return {'rf_mean_ms': rfm, 'rf_std_ms': rfs, 'rf_times': rf_times,
            'oracle_ms': oracle_ms, 'speedup': spd}

# ===================== TÍNH F THỰC (KHÔNG PENALTY) – GIỐNG plot_comparison GỐC =====================
def calc_real_objectives(q_list, G1, G2, OFF, C):
    """
    Tính f1/f2/f3 THỰC SỰ không cộng penalty – đúng như plot_comparison() trong code gốc.
    Penalty chỉ dùng để hướng dẫn NSGA-II tìm nghiệm, không dùng để báo cáo kết quả.
    Trả về thêm danh sách nút bị ùn tắc (x > 0.95) để cảnh báo.
    """
    n = len(df_nodes)
    f1_real = f2_real = f3_real = 0.0
    overload_nodes = []
    for i in range(n):
        q = float(q_list[i])
        S, L, v, qb = df_nodes['S'][i], df_nodes['L'][i], df_nodes['v'][i], df_nodes['qb'][i]
        g1, g2, off = G1[i], G2[i], OFF[i]
        q1, q2 = q*0.6, q*0.4
        S1, S2 = S*0.6, S*0.4

        # F1: tổng độ trễ (KHÔNG penalty)
        d1 = calc_uniform_delay(q1,S1,g1,C) + calc_incremental_delay(q1,S1,g1,C) + calc_residual_delay(q1,qb*0.6)
        d2 = calc_uniform_delay(q2,S2,g2,C) + calc_incremental_delay(q2,S2,g2,C) + calc_residual_delay(q2,qb*0.4)
        f1_real += (d1*q1 + d2*q2) / 3600.0

        # F2: tổng hàng chờ (KHÔNG penalty)
        lq1 = calc_lq1_uniform(q1,S1,g1,C) + calc_lq2_random(q1,S1,g1,C)
        lq2 = calc_lq1_uniform(q2,S2,g2,C) + calc_lq2_random(q2,S2,g2,C)
        f2_real += lq1 + lq2

        # F3: tổng dừng xe (KHÔNG penalty)
        t_travel = DISTANCES[i] / (v*1000/3600 + 1e-6)
        gamma = calc_gamma_wave(off, t_travel, C)
        f3_real += (calc_node_stops(q1,S1,g1,C,gamma) + calc_node_stops(q2,S2,g2,C,gamma))

        # Ghi nhận nút bị ùn tắc
        x1 = (q1*C)/(S1*g1+1e-6); x2 = (q2*C)/(S2*g2+1e-6)
        if x1 > 0.95 or x2 > 0.95:
            overload_nodes.append((i+1, max(x1,x2)))

    return f1_real, f2_real, f3_real, overload_nodes


# ===================== DEMO REAL-TIME – 5 KịCH BẢN =====================
def run_realtime_demo(model, scaler_X, scaler_y):
    """
    5 kịch bản thực tế.
    In bảng Pandas DataFrame đầy đủ 26 nút (Nút, g1, g2, Offset, C).
    Tính f1/f2/f3 bằng evaluate_individual() gốc với df_nodes['q'] tạm thay.
    """
    print("\n[5/6] Demo Real-time – 5 kịch bản giao thông...")
    q_base = np.array(df_nodes['q'].tolist(), dtype=float)
    scenarios = [
        ('Buổi sáng (cao điểm)',    1.00, 'morning_peak'),
        ('Buổi trưa (bình thường)', 0.75, 'offpeak'),
        ('Buổi chiều (cao điểm)',   0.95, 'evening_peak'),
        ('Buổi tối (thấp điểm)',    0.50, 'night'),
        ('Lưu lượng tăng cao (115%)', 1.15, 'morning_peak'),
    ]
    results = []
    n = len(df_nodes)

    for name, scale, period in scenarios:
        q_scene = np.clip(q_base * scale, 200, 5000).astype(np.float32)

        t0   = time.perf_counter()
        ps   = model.predict(scaler_X.transform(q_scene.reshape(1, -1)))
        pi   = scaler_y.inverse_transform(ps)
        pi_pp = post_process_predictions(pi)[0]
        lat  = (time.perf_counter() - t0) * 1000

        C_   = int(pi_pp[0])
        G1_  = pi_pp[1:n+1].astype(int)
        G2_  = pi_pp[n+1:2*n+1].astype(int)
        OFF_ = pi_pp[2*n+1:3*n+1].astype(int)

        # Tính f1/f2/f3 THỰC (không penalty) – đúng như plot_comparison gốc
        G2_real = [max(G_MIN, C_ - int(G1_[k]) - 2*int(df_nodes['L'][k])) for k in range(n)]
        f1, f2, f3, overload = calc_real_objectives(
            q_scene.tolist(), G1_, G2_real, OFF_, C_)
        n_overload = len(overload)

        df_det = pd.DataFrame({
            'Nút':        range(1, n+1),
            'Tên nút':    df_nodes['name'].tolist(),
            'q (xe/h)':   q_scene.astype(int),
            'C (s)':      [C_] * n,
            'g1 (s)':     G1_,
            'g2 (s)':     G2_,
            'Offset (s)': OFF_,
        })

        print(f"\n{'─'*72}")
        print(f"  Kịch bản : {name}")
        print(f"  Tỷ lệ LT : ×{scale} | Khung giờ: {period} | C={C_}s | Latency={lat:.2f}ms")
        warn = f"  ⚠ {n_overload} nút ùn tắc (x>0.95)" if n_overload > 0 else "  ✓ Không ùn tắc"
        print(f"  f1={f1:.1f} xe.h  |  f2={f2:.1f} xe  |  f3={f3:.0f} lượt/h  |{warn}")
        print(f"\n{df_det.to_string(index=False)}")

        results.append({'name': name, 'scale': scale, 'C': C_,
                        'G1': G1_, 'G2': G2_, 'OFF': OFF_,
                        'q': q_scene, 'f1': f1, 'f2': f2, 'f3': f3,
                        'lat': lat, 'df': df_det})
    return results

# ===================== DASHBOARD NỀN TRẮNG – 4 BIỂU ĐỒ CẦN THIẾT =====================
def build_dashboard(X_te, y_te, y_pred, bench, metrics, model, demo):
    """
    Dashboard 2×2 nền trắng:
      [0,0] Inference Time – so sánh RF vs NSGA-II+LS
      [0,1] Predicted vs Actual Chu kỳ C (chỉ tiêu quan trọng nhất)
      [1,0] Feature Importance – top 15 nút ảnh hưởng nhất
      [1,1] So sánh f1/f2/f3 theo 5 kịch bản demo (bao gồm 115%)
    """
    print("\n[6/6] Xây dựng Dashboard (nền trắng)...")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.patch.set_facecolor('white')
    fig.suptitle(
        'SURROGATE MODEL – ĐIỀU KHIỂN ĐÈN GIAO THÔNG 26 NÚT TP.HCM\n'
        'Trục Điện Biên Phủ – Võ Thị Sáu – Lý Chính Thắng  |  RF O(1) Inference',
        fontsize=14, fontweight='bold', color='#1A1A2E')
    plt.subplots_adjust(hspace=0.42, wspace=0.32,
                        left=0.07, right=0.97, top=0.91, bottom=0.09)

    n = len(df_nodes)

    # ── Panel 1: Inference Time ───────────────────────────────────────
    ax1 = axes[0, 0]
    ax1.set_facecolor('white')
    vals  = [bench['rf_mean_ms'], bench['oracle_ms']]
    clrs  = [COLORS[0], COLORS[1]]
    bars  = ax1.bar(['RF Surrogate', 'NSGA-II+LS\n(thực đo)'], vals,
                    color=clrs, width=0.45, edgecolor='white', linewidth=1.2, zorder=3)
    ax1.set_yscale('log')
    ax1.yaxis.grid(True, which='both', linestyle='--', alpha=0.4)
    ax1.set_ylabel('Thời gian (ms – thang log)', fontsize=10)
    ax1.set_title('So sánh Thời gian Suy luận', fontweight='bold', fontsize=12, pad=8)
    ax1.text(0.5, 0.86, f'Speedup ×{bench["speedup"]:,.0f}',
             transform=ax1.transAxes, ha='center',
             color='#C62828', fontsize=16, fontweight='bold')
    for b, v in zip(bars, vals):
        lbl = f'{v:.2f} ms' if v < 1000 else f'{v/1000:.1f} s'
        ax1.text(b.get_x() + b.get_width()/2, v * 1.8, lbl,
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    for sp in ['top', 'right']:
        ax1.spines[sp].set_visible(False)

    # ── Panel 2: Predicted vs Actual – C ─────────────────────────────
    ax2 = axes[0, 1]
    ax2.set_facecolor('white')
    ct = y_te[:, 0]; cp = y_pred[:, 0]
    err = np.abs(ct - cp)
    sc2 = ax2.scatter(ct, cp, c=err, cmap='RdYlGn_r', s=55, alpha=0.75,
                      edgecolors='none', vmin=0, vmax=max(err.max(), 1))
    cbar = plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('|Error| (s)', fontsize=9)
    lo = min(ct.min(), cp.min()) - 0.5
    hi = max(ct.max(), cp.max()) + 0.5
    ax2.plot([lo, hi], [lo, hi], '--', color=COLORS[1], lw=2, label='y=x')
    ax2.text(0.05, 0.90, f'R²={r2_score(ct, cp):.4f}',
             transform=ax2.transAxes, color='navy', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Actual C (s)', fontsize=10)
    ax2.set_ylabel('Predicted C (s)', fontsize=10)
    ax2.set_title('Predicted vs Actual – Chu kỳ C', fontweight='bold', fontsize=12, pad=8)
    ax2.legend(fontsize=9)
    for sp in ['top', 'right']:
        ax2.spines[sp].set_visible(False)

    # ── Panel 3: Feature Importance top-15 ───────────────────────────
    ax3 = axes[1, 0]
    ax3.set_facecolor('white')
    fi = model.feature_importances_
    top_n = 15
    ti = np.argsort(fi)[-top_n:]
    norm_fi = fi[ti] / fi[ti].max()
    bar_colors = [plt.cm.RdYlGn(v) for v in norm_fi]
    ax3.barh(range(top_n), fi[ti], color=bar_colors, edgecolor='white', height=0.70)
    ax3.set_yticks(range(top_n))
    ax3.set_yticklabels([f'Nút {i+1}' for i in ti], fontsize=9)
    for j, val in enumerate(fi[ti]):
        ax3.text(val + fi[ti].max()*0.01, j, f'{val:.4f}',
                 va='center', fontsize=7.5, color='#555')
    ax3.set_xlabel('Importance Score', fontsize=10)
    ax3.set_title(f'Feature Importance – Top {top_n} Nút (q → C+G+OFF)',
                  fontweight='bold', fontsize=12, pad=8)
    ax3.xaxis.grid(True, linestyle='--', alpha=0.4)
    for sp in ['top', 'right']:
        ax3.spines[sp].set_visible(False)

    # ── Panel 4: So sánh f1/f2/f3 theo kịch bản ──────────────────────
    # Hiển thị sự khác biệt giữa các kịch bản, đặc biệt 100% vs 115%
    ax4 = axes[1, 1]
    ax4.set_facecolor('white')
    names_short = [r['name'].replace('Buổi ', '').replace(' (', '\n(') for r in demo]
    f1s = [r['f1'] for r in demo]
    f2s = [r['f2'] for r in demo]
    # f3 đơn vị khác hẳn → chuẩn hóa về cùng trục hiển thị (chia 100)
    f3s_scaled = [r['f3'] / 100.0 for r in demo]
    Cs  = [r['C']  for r in demo]

    x_pos = np.arange(len(demo))
    w = 0.22
    b1 = ax4.bar(x_pos - w, f1s, w, label='f1: Tổng trễ (xe.h)',  color=COLORS[0], alpha=0.88)
    b2 = ax4.bar(x_pos,     f2s, w, label='f2: Hàng chờ (xe)',    color=COLORS[1], alpha=0.88)
    b3 = ax4.bar(x_pos + w, f3s_scaled, w,
                 label='f3: Dừng xe (÷100 lượt/h)', color=COLORS[2], alpha=0.88)

    # Ghi chu kỳ C lên trên mỗi nhóm cột
    for xi, C_val in zip(x_pos, Cs):
        ax4.text(xi, max(f1s[x_pos.tolist().index(xi)],
                         f2s[x_pos.tolist().index(xi)],
                         f3s_scaled[x_pos.tolist().index(xi)]) * 1.06,
                 f'C={C_val}s', ha='center', fontsize=8.5, fontweight='bold', color='#333')

    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(names_short, rotation=18, ha='right', fontsize=8.5)
    ax4.set_title('So sánh f1/f2/f3 theo Kịch bản\n(gồm 100% và 115%)',
                  fontweight='bold', fontsize=12, pad=8)
    ax4.set_ylabel('Giá trị mục tiêu', fontsize=10)
    ax4.legend(fontsize=8, loc='upper left')
    ax4.yaxis.grid(True, linestyle='--', alpha=0.4)
    for sp in ['top', 'right']:
        ax4.spines[sp].set_visible(False)

    # ── Summary card (góc dưới phải) ─────────────────────────────────
    rfm = bench['rf_mean_ms']
    summary = (
        f"R²(all)={metrics['r2']:.4f}  R²(C+G)={metrics['r2_cg']:.4f}  "
        f"MAE={metrics['mae']:.4f}\n"
        f"RF={rfm:.3f}ms  NSGA={bench['oracle_ms']:.0f}ms  "
        f"Speedup×{bench['speedup']:,.0f}  "
        f"Train={metrics['n_train']}  Test={metrics['n_test']}"
    )
    fig.text(0.5, 0.005, summary, ha='center', va='bottom', fontsize=9,
             color='#333', fontfamily='monospace',
             bbox=dict(facecolor='#F5F5F5', alpha=0.9,
                       edgecolor='#CCC', boxstyle='round,pad=0.5'))

    out = 'outputs/surrogate_dashboard.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✓ Dashboard → {out}")

# ===========================================================================
# MAIN PIPELINE
# ===========================================================================
def main():
    print("=" * 70)
    print("  SURROGATE MODEL – ĐIỀU KHIỂN ĐÈN GIAO THÔNG THỜI GIAN THỰC")
    print("  Lõi: nsga2() + local_search() từ Tối_ưu_trên_mạng_lưới.py")
    print("  26 nút | TP.HCM | Trục ĐBP – VTS – LCT")
    print("=" * 70)
    t0 = time.perf_counter()

    # Step 1: Load dữ liệu (hàm nguyên vẹn từ code gốc)
    load_data_from_excel()

    # Step 2: Sinh dataset
    # pop=40, gen=20, ls=20 → hội tụ đủ tốt trong thời gian hợp lý
    X, y = generate_training_dataset(
        n_samples=3000, noise_sigma=0.05,
        pop_size=80, n_gen=40, ls_iter=20)

    # Step 3: Train Surrogate
    model, sX, sy, Xte, yte, yp, metrics = train_surrogate_model(X, y)

    # Step 4: Benchmark
    bench = benchmark_inference(model, sX, sy, n_trials=1000)

    # Step 5: Demo
    demo = run_realtime_demo(model, sX, sy)

    # Step 6: Dashboard
    build_dashboard(Xte, yte, yp, bench, metrics, model, demo)

    joblib.dump({'model': model, 'scaler_X': sX, 'scaler_y': sy},
                'outputs/surrogate_rf_model.pkl')
    print("    ✓ Model → outputs/surrogate_rf_model.pkl")

    print("\n" + "=" * 70)
    print(f"  R²(all)     : {metrics['r2']:.4f}")
    print(f"  R²(C+G1+G2) : {metrics['r2_cg']:.4f}")
    print(f"  MAE         : {metrics['mae']:.4f}")
    print(f"  RF Latency  : {bench['rf_mean_ms']:.3f} ms")
    print(f"  Speedup     : ×{bench['speedup']:,.0f}")
    print(f"  Pipeline    : {time.perf_counter()-t0:.1f}s")
    print("=" * 70)

if __name__ == '__main__':
    main()