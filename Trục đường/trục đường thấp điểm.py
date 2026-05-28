"""
=============================================================================
TỐI ƯU HÓA ĐÈN GIAO THÔNG – MẠNG LƯỚI 10 NÚT (GIỜ THẤP ĐIỂM)
Thuật toán: NSGA-II + Local Search Đa Mục Tiêu 
Cải tiến: Mở rộng không gian tìm kiếm, Khởi tạo thông minh, AHP Ưu tiên Sóng xanh
=============================================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
import copy
import os
import warnings

warnings.filterwarnings('ignore')
os.makedirs('outputs', exist_ok=True)

# ===================== BIẾN TOÀN CỤC CHỨA DỮ LIỆU =====================
df_nodes = pd.DataFrame()
DISTANCES = []
L_MAX_LIST = [] # [ĐỒNG NHẤT 26 NÚT] Thêm mảng lưu giới hạn hàng chờ
BASELINE_C = []
BASELINE_G1 = []
BASELINE_G2 = []
BASELINE_OFF = []
BASELINE_F1 = []
BASELINE_F2 = []
BASELINE_F3 = []

# Tham số đã được mở rộng cho 10 nút thấp điểm
# ĐÃ SỬA: Tăng C_MIN lên 65 để nút giao có đủ thời gian xả hàng chờ
C_MIN, C_MAX = 65, 80  
G_MIN, G_MAX = 12, 80  
OFF_MIN = 0

# ===================== 1. HÀM ĐỌC DỮ LIỆU THỰC TẾ =====================
def tao_dl_thuc_te():
    global df_nodes, DISTANCES, BASELINE_C, BASELINE_G1, BASELINE_G2, BASELINE_OFF
    global BASELINE_F1, BASELINE_F2, BASELINE_F3, L_MAX_LIST
    
    print("[+] Đang nạp dữ liệu mạng lưới 10 nút...")
    try:
        df_nut = pd.read_csv("Du_lieu_nut_thap_diem.xlsx - Baseline_10Nut_ThapDiem_Dep.csv")
    except:
        df_nut = pd.read_excel("Du_lieu_nut_thap_diem.xlsx")
        
    try:
        df_lk = pd.read_csv("Khoang_Cach.xlsx - Ma_nutTên đoạnKhoang_cach_D (m).csv")
    except:
        df_lk = pd.read_excel("Khoang_Cach.xlsx")

    n_nodes = len(df_nut)
    
    nodes_dict = {
        'k': df_nut.iloc[:, 0].tolist(),
        'q': df_nut['q_total (xe/h)'].astype(float).tolist(),
        'S': df_nut['S (xe/h)'].astype(float).tolist(),
        'L': df_nut['L/pha (s)'].astype(float).tolist(),
        'v': df_nut['v (km/h)'].astype(float).tolist(),
        'qb': df_nut['qb (xe)'].astype(float).tolist()
    }
    df_nodes = pd.DataFrame(nodes_dict)
    
    DISTANCES = [200.0] * n_nodes
    for _, row in df_lk.iterrows():
        tu = int(row.iloc[0])
        if tu - 1 < n_nodes:
            DISTANCES[tu - 1] = float(row.iloc[2])

    # ================= BỔ SUNG LOGIC TÍNH L_MAX (ĐỒNG NHẤT 26 NÚT) =================
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
    # ===============================================================================

    BASELINE_C = df_nut['Chu_kỳ_C (s)'].astype(int).tolist()
    BASELINE_G1 = df_nut['g1_Pha1 (s)'].astype(int).tolist()
    BASELINE_G2 = df_nut['g2_Pha2 (s)'].astype(int).tolist()
    BASELINE_OFF = df_nut['Offset_o (s)'].astype(int).tolist()

    # Tính Base Fitness
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
        
    print(f"✓ Đã load thành công {n_nodes} nút và đồng bộ chuẩn hóa cấu trúc dữ liệu.")

# ===================== 2. CÁC HÀM TÍNH TOÁN TOÁN HỌC (ĐỒNG NHẤT 26 NÚT) =====================
def calc_uniform_delay(q, S, g, C):
    if g <= 0 or S <= 0 or C <= 0: return 9999.0
    lam = g / C
    x = (q * C) / (S * g)
    num = C * ((1 - lam) ** 2)
    den = 2 * (1 - min(1.0, x) * lam)
    return max(0.0, num / den)

def calc_incremental_delay(q, S, g, C, T_period=0.25):
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

def calc_lq2_random(q, S, g, C):
    if g <= 0 or S <= 0: return 0.0
    lam = g / C
    x = (q * C) / (S * g)
    if x <= 0.5: return 0.0
    c_cap = S * lam
    term = ((x - 1) ** 2) + (16 * x) / (c_cap + 1e-6)
    lq2 = 0.25 * c_cap * ((x - 1) + np.sqrt(max(0.0, term)))
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
    penalty_f1, penalty_f2, penalty_f3 = 0.0, 0.0, 0.0

    X_MAX = 0.95 

    for i in range(n):
        q, S, L, v, qb = df_nodes['q'][i], df_nodes['S'][i], df_nodes['L'][i], df_nodes['v'][i], df_nodes['qb'][i]
        g1, off_k = G1[i], OFF[i]
        
        g2 = max(G_MIN, c - g1 - 2*int(L))
        q1, q2 = q * 0.6, q * 0.4
        S1, S2 = S * 0.6, S * 0.4

        # F1: TỔNG ĐỘ TRỄ (Công thức chuẩn 26 nút)
        d1 = calc_uniform_delay(q1, S1, g1, c) + calc_incremental_delay(q1, S1, g1, c) + calc_residual_delay(q1, qb*0.6)
        d2_delay = calc_uniform_delay(q2, S2, g2, c) + calc_incremental_delay(q2, S2, g2, c) + calc_residual_delay(q2, qb*0.4)
        f1_total += (d1 * q1 + d2_delay * q2) / 3600.0

        # F2: HÀNG CHỜ (Công thức chuẩn 26 nút)
        lq1 = calc_lq1_uniform(q1, S1, g1, c) + calc_lq2_random(q1, S1, g1, c)
        lq2_queue = calc_lq1_uniform(q2, S2, g2, c) + calc_lq2_random(q2, S2, g2, c)
        f2_total += (lq1 + lq2_queue)

        # F3: SỐ LƯỢT DỪNG (Tích hợp hệ số sóng Gamma wave)
        t_travel = DISTANCES[i] / (v * 1000/3600 + 1e-6)
        gamma = calc_gamma_wave(off_k, t_travel, c)
        ns1 = calc_node_stops(q1, S1, g1, c, gamma)
        ns2 = calc_node_stops(q2, S2, g2, c, gamma)
        f3_total += (ns1 + ns2)

        # ================= HÀM PHẠT TỈ LỆ ĐA MỤC TIÊU =================
        x1 = (q1 * c) / (S1 * g1 + 1e-6)
        x2 = (q2 * c) / (S2 * g2 + 1e-6)
        
        # 1. Phạt bão hòa (Trọng số scale theo độ lớn tự nhiên của f1, f2, f3)
        if x1 > X_MAX: 
            penalty_f1 += (x1 - X_MAX) * 50    # f1 đơn vị là xe.h
            penalty_f2 += (x1 - X_MAX) * 20    # f2 đơn vị là xe
            penalty_f3 += (x1 - X_MAX) * 500   # f3 đơn vị là lượt
        if x2 > X_MAX: 
            penalty_f1 += (x2 - X_MAX) * 50
            penalty_f2 += (x2 - X_MAX) * 20
            penalty_f3 += (x2 - X_MAX) * 500
            
        # 2. Phạt tràn bến/hàng chờ vượt L_max
        if lq1 > L_MAX_LIST[i]: 
            over = lq1 - L_MAX_LIST[i]
            penalty_f1 += over * 0.5  
            penalty_f2 += over * 2.0  # f2 chịu phạt nặng nhất khi tràn hàng chờ
            penalty_f3 += over * 10.0
        if lq2_queue > L_MAX_LIST[i]: 
            over = lq2_queue - L_MAX_LIST[i]
            penalty_f1 += over * 0.5
            penalty_f2 += over * 2.0
            penalty_f3 += over * 10.0
        # ===============================================================

    return f1_total + penalty_f1, f2_total + penalty_f2, f3_total + penalty_f3
def calculate_real_metrics(individual):
    """Tính các giá trị f1, f2, f3 thực tế (KHÔNG CỘNG HÀM PHẠT) để báo cáo"""
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

        # F1 thực
        d1 = calc_uniform_delay(q1, S1, g1, c) + calc_incremental_delay(q1, S1, g1, c) + calc_residual_delay(q1, qb*0.6)
        d2 = calc_uniform_delay(q2, S2, g2, c) + calc_incremental_delay(q2, S2, g2, c) + calc_residual_delay(q2, qb*0.4)
        f1_real += (d1 * q1 + d2 * q2) / 3600.0

        # F2 thực
        lq1 = calc_lq1_uniform(q1, S1, g1, c) + calc_lq2_random(q1, S1, g1, c)
        lq2 = calc_lq1_uniform(q2, S2, g2, c) + calc_lq2_random(q2, S2, g2, c)
        f2_real += (lq1 + lq2)

        # F3 thực
        t_travel = DISTANCES[i] / (v * 1000/3600 + 1e-6)
        gamma = calc_gamma_wave(off_k, t_travel, c)
        f3_real += calc_node_stops(q1, S1, g1, c, gamma) + calc_node_stops(q2, S2, g2, c, gamma)

    return [f1_real, f2_real, f3_real]

# ===================== 3. AHP VÀ TỔNG HỢP FITNESS =====================
def lay_ma_tran_ahp():
    # ĐÃ SỬA: Ma trận cân bằng (Ưu tiên Sóng xanh 50%, Độ trễ 25%, Hàng chờ 25%)
    return np.array([
        [1.0,  1.0, 1/2],  
        [1.0,  1.0, 1/2],  
        [2.0,  2.0, 1.0]   
    ])

def tinh_trong_so_ahp_chuan(ma_tran_A):
    n = ma_tran_A.shape[0]
    geo_means = np.array([np.prod(ma_tran_A[i, :])**(1/n) for i in range(n)])
    trong_so = geo_means / geo_means.sum()
    lambda_max = np.mean((ma_tran_A @ trong_so) / trong_so)
    CI = (lambda_max - n) / (n - 1)
    CR = CI / 0.58
    return trong_so.tolist(), lambda_max, CI, CR

def weighted_score(fit, weights, bl_fit):
    return weights[0]*(fit[0]/(bl_fit[0] + 1e-9)) + weights[1]*(fit[1]/(bl_fit[1] + 1e-9)) + weights[2]*(fit[2]/(bl_fit[2] + 1e-9))

# ===================== 4. CẤU TRÚC INDIVIDUAL & NSGA-II =====================
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
        g_trong = c - 2 * L
        
        # Khởi tạo thông minh tránh g1=15
        g1_ideal = int(round(g_trong * 0.6))
        g1 = g1_ideal + random.randint(-3, 3) 
        g1 = int(np.clip(g1, G_MIN, max(G_MIN, g_trong - G_MIN)))
        
        G1.append(g1)
        OFF.append(random.randint(0, max(1, c-1)))
    return repair_individual(G1 + OFF + [c])

def crossover(p1, p2):
    n = len(p1)
    c1, c2 = list(p1), list(p2)
    for i in range(n):
        if random.random() < 0.5:
            c1[i], c2[i] = c2[i], c1[i]
    return repair_individual(c1), repair_individual(c2)

def mutate(ind):
    n = len(df_nodes)
    ind = list(ind)
    # Tăng cường đột biến (Probability 60%)
    if random.random() < 0.6: 
        for _ in range(random.randint(3, 7)): 
            k = random.randint(0, 2*n) 
            if k < n: 
                ind[k] += random.choice([-4, -2, 2, 4])
            elif k < 2*n: 
                ind[k] += random.choice([-15, -7, 7, 15]) # Bước nhảy Offset xa hơn
            else: 
                ind[k] += random.choice([-10, -5, 5, 10])
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

# ===================== 5. TIẾN TRÌNH TỐI ƯU =====================
def chay_toi_uu(w_ahp, n_qt=80, n_the_he=60):
    t_start = time.time()
    
    bl_ind = repair_individual(BASELINE_G1 + BASELINE_OFF + [75])
    
    # [CẢI TIẾN QUAN TRỌNG 1]: Tính Base Fitness THỰC TẾ (Không phạt) để làm mỏ neo AHP
    bl_fit = calculate_real_metrics(bl_ind)
    # Bảo vệ chống chia 0 nếu dữ liệu mạng lưới bất thường
    bl_fit = [max(1e-3, f) for f in bl_fit] 
    
    # Khởi tạo quần thể ban đầu
    pop = [bl_ind]
    pop += [create_individual(force_c=75) for _ in range(n_qt // 3)]
    pop += [create_individual() for _ in range(n_qt - 1 - n_qt // 3)]

    pop_fit = [evaluate_individual(ind) for ind in pop]
    fronts, ranks = fast_non_dominated_sort(pop_fit)
    crowding = [0.0]*n_qt
    for front in fronts:
        cd = crowding_distance(pop_fit, front)
        for j, idx in enumerate(front): crowding[idx] = cd[j]

    lich_su_f = []

    # Tiến trình NSGA-II
    for gen in range(n_the_he):
        offspring = []
        while len(offspring) < n_qt:
            p1 = select_tournament(pop, pop_fit, ranks, crowding)
            p2 = select_tournament(pop, pop_fit, ranks, crowding)
            c1, c2 = crossover(p1, p2) 
            offspring.extend([mutate(c1), mutate(c2)])
        offspring = offspring[:n_qt]

        combined = pop + offspring
        combined_fit = pop_fit + [evaluate_individual(ind) for ind in offspring]
        fronts_c, ranks_c = fast_non_dominated_sort(combined_fit)
        
        crowding_c = [0.0]*len(combined)
        for front in fronts_c:
            cd = crowding_distance(combined_fit, front)
            for j, idx in enumerate(front): crowding_c[idx] = cd[j]

        pop, pop_fit, ranks, crowding = [], [], [], []
        for front in fronts_c:
            if len(pop) + len(front) <= n_qt:
                for idx in front:
                    pop.append(combined[idx]); pop_fit.append(combined_fit[idx])
                    ranks.append(ranks_c[idx]); crowding.append(crowding_c[idx])
            else:
                rem = n_qt - len(pop)
                sorted_front = sorted(front, key=lambda i: -crowding_c[i])
                for idx in sorted_front[:rem]:
                    pop.append(combined[idx]); pop_fit.append(combined_fit[idx])
                    ranks.append(ranks_c[idx]); crowding.append(crowding_c[idx])
                break
                
        pareto_fits_gen = [pop_fit[i] for i in range(len(pop)) if ranks[i] == 1]
        best_ahp_gen = min([weighted_score(f, w_ahp, bl_fit) for f in pareto_fits_gen])
        lich_su_f.append(best_ahp_gen)

    t_nsga2 = time.time() - t_start

    # ========================================================
    # LOCAL SEARCH CƯỜNG ĐỘ CAO (MỞ KHÓA KHÔNG GIAN)
    # ========================================================
    t_ls_start = time.time()
    pareto_idx = [i for i in range(len(pop)) if ranks[i] == 1]
    pareto_pop = [pop[i] for i in pareto_idx]
    pareto_fits = [pop_fit[i] for i in pareto_idx]
    
    best_ind = list(bl_ind)
    best_fit = list(bl_fit)
    best_score = 1.0 # Baseline score là 1.0
    
    scores = [weighted_score(f, w_ahp, bl_fit) for f in pareto_fits]
    top_k = max(1, int(len(pareto_pop)*0.4)) # Cho phép LS rà quét 40% mặt Pareto tốt nhất
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i])[:top_k]

    if scores[top_idx[0]] < 1.0:
        best_ind = pareto_pop[top_idx[0]]
        best_fit = pareto_fits[top_idx[0]]
        best_score = scores[top_idx[0]]

    for idx in top_idx:
        ind = list(pareto_pop[idx])
        fit = list(pareto_fits[idx])
        score = scores[idx]
        
        # [CẢI TIẾN QUAN TRỌNG 2]: Xóa chặn `if score >= 1.0: continue` 
        # Để Local Search được quyền đào sâu mọi cá thể tiềm năng
        
        for _ in range(40):
            neighbor = list(ind)
            n_var = len(df_nodes)
            for k in random.sample(range(2*n_var + 1), random.randint(2, 4)):
                if k < n_var: neighbor[k] += random.choice([-2, 0, 2])
                elif k < 2*n_var: neighbor[k] += random.choice([-8, 8])
                else: neighbor[k] += random.choice([-5, 0, 5])
            
            neighbor = repair_individual(neighbor)
            new_fit = evaluate_individual(neighbor)
            new_score = weighted_score(new_fit, w_ahp, bl_fit)

            if new_score < score:
                ind, fit, score = neighbor, list(new_fit), new_score

        # Chỉ lưu lại kết quả nều tốt hơn Best Cục Bộ
        if score < best_score:
            best_ind, best_fit, best_score = ind, fit, score

    t_ls = time.time() - t_ls_start
    
    thong_ke_mo_hinh = {
        'thoi_gian_chay': t_nsga2 + t_ls,
        'tg_nsga2': t_nsga2,
        'tg_local_search': t_ls,
        'best_score': best_score
    }
    
    return best_ind, best_fit, bl_fit, lich_su_f, thong_ke_mo_hinh
# ===================== 6. TRỰC QUAN HÓA =====================
def ve_dt(best_ind, best_fit, bl_fit, lich_su_f, thong_ke_mo_hinh):
    import matplotlib as mpl 
    n = len(df_nodes)
    c_opt = best_ind[-1]
    g1_opt = best_ind[:n]
    off_opt = best_ind[n:2*n]
    g2_opt = [max(G_MIN, c_opt - g1_opt[i] - 2*int(df_nodes['L'][i])) for i in range(n)]
    
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'DejaVu Sans' if 'DejaVu Sans' in [f.name for f in mpl.font_manager.fontManager.ttflist] else 'sans-serif'
    
    print("\n" + "="*85)
    print("      BÁO CÁO KẾT QUẢ TỐI ƯU HÓA ĐÈN GIAO THÔNG PHỐI HỢP CHO TRỤC ĐƯỜNG 10 NÚT")
    print("="*85)
    print(f" * Điểm số hàm mục tiêu tổng hợp (Fitness) : {thong_ke_mo_hinh['best_score']:.6f}")
    print(f" * Chu kỳ đèn tín hiệu chung tối ưu (C)     : {int(c_opt)} giây (s)")
    print(f" * Mảng dịch lệch pha tối ưu (Offset vt_o) : {off_opt}")  
    print("-" * 85)
    
    print("\n" + "=" * 82)
    print(f"{'SO SÁNH TRƯỚC & SAU TỐI ƯU HÓA (GIÁ TRỊ THỰC TẾ)':^82}")
    print("=" * 82)
    print(f"{'Tiêu chí':<32} {'Baseline (Trục 10 Nút)':<22} {'Tối ưu':<18} {'Cải thiện'}")
    print("-" * 82)

    # -------------------------------------------------------------------------
    # SỬA Ở ĐÂY: Dùng hàm tính toán thực tế (không phạt) để xuất báo cáo
    # -------------------------------------------------------------------------
    bl_ind = repair_individual(BASELINE_G1 + BASELINE_OFF + [75])
    bl_fit_real = calculate_real_metrics(bl_ind)  # Tính điểm Baseline KHÔNG PHẠT
    best_fit_real = calculate_real_metrics(best_ind) # Tính điểm Tối ưu KHÔNG PHẠT

    p1_imp = ((bl_fit_real[0] - best_fit_real[0]) / bl_fit_real[0] * 100) if bl_fit_real[0] != 0 else 0
    p2_imp = ((bl_fit_real[1] - best_fit_real[1]) / bl_fit_real[1] * 100) if bl_fit_real[1] != 0 else 0
    p3_imp = ((bl_fit_real[2] - best_fit_real[2]) / bl_fit_real[2] * 100) if bl_fit_real[2] != 0 else 0

    print(f"{'Tổng thời gian trễ (f1)':<32} {bl_fit_real[0]:,.1f} xe.h{'':<11} {best_fit_real[0]:,.1f} xe.h{'':<7} {p1_imp:>8.1f}%")
    print(f"{'Độ dài hàng chờ TB (f2)':<32} {bl_fit_real[1]:,.1f} xe{'':<13} {best_fit_real[1]:,.1f} xe{'':<9} {p2_imp:>8.1f}%")
    print(f"{'Số lần dừng xe (f3)':<32} {bl_fit_real[2]:,.0f} lượt/h{'':<11} {best_fit_real[2]:,.0f} lượt/h{'':<7} {p3_imp:>8.1f}%")
    print("=" * 82 + "\n")
    # -------------------------------------------------------------------------

    print(f" {'Nút giao':<10} | {'Thời gian đèn xanh hiệu quả (g1, g2)':<45} | {'Độ lệch pha o_k (s)':<15}")
    print("-" * 85)
    for k in range(n):
        print(f" Nút {k+1:<5} | {str([g1_opt[k], g2_opt[k]]):<45} | {int(off_opt[k]):<15}")
    print("=" * 85)

    print(" [i] Đang xuất biểu đồ trực quan thực tế...\n")

    plt.figure(figsize=(10, 4.5))
    plt.plot(range(1, len(lich_su_f)+1), lich_su_f, color='#e74c3c', linewidth=2.5, marker='o', markersize=4)
    plt.title('Tiến trình hội tụ AHP của Thuật toán NSGA-II', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Thế hệ tiến hóa')
    plt.ylabel('Giá trị AHP Fitness')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('outputs/10Nut_HoiTu.png', dpi=150)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(12, 6))
    nuts = [f"Nút {i+1}" for i in range(n)]
    
    ax1.bar(nuts, g1_opt, label='Pha 1 ($g_1$)', color='#85c1e9', edgecolor='white', width=0.55)
    ax1.bar(nuts, g2_opt, bottom=g1_opt, label='Pha 2 ($g_2$)', color='#76d7c4', edgecolor='white', width=0.55)
    
    ax1.set_xlabel('Các nút giao thông', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Thời gian xanh hiệu dụng (giây)', fontsize=12, fontweight='bold')
    ax1.set_title(f'PHÂN BỔ THỜI GIAN XANH & ĐỘ LỆCH PHA (CHU KỲ CHUNG C = {int(c_opt)}s)', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(nuts, off_opt, color='#f39c12', marker='D', linewidth=2.5, markersize=8, label='Độ lệch pha (Offset - $O_k$)')
    ax2.set_ylabel('Offset (giây)', fontsize=12, fontweight='bold', color='#f39c12')
    ax2.tick_params(axis='y', labelcolor='#f39c12')
    ax2.grid(False)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=11)

    plt.tight_layout()
    plt.savefig('outputs/10Nut_PhanBoXanh_Offset.png', dpi=150)
    plt.close()

# ===================== MAIN =====================
if __name__ == "__main__":
    tao_dl_thuc_te()
    
    A_matrix = lay_ma_tran_ahp()
    w_ahp_kich_ban, lambda_max, CI, cr = tinh_trong_so_ahp_chuan(A_matrix)
    
    print("\n" + "="*80)
    print(" KẾT QUẢ ĐẦU RA TOÁN HỌC PHƯƠNG PHÁP PHÂN TÍCH THỨ BẬC AHP (Giờ Thấp Điểm)")
    print("="*80)
    print(f" Trọng số f1 (Thời gian trễ)   : w1 = {w_ahp_kich_ban[0]:.4f} ({w_ahp_kich_ban[0]*100:.2f}%)")
    print(f" Trọng số f2 (Hàng chờ)        : w2 = {w_ahp_kich_ban[1]:.4f} ({w_ahp_kich_ban[1]*100:.2f}%)")
    print(f" Trọng số f3 (Số lần dừng xe)  : w3 = {w_ahp_kich_ban[2]:.4f} ({w_ahp_kich_ban[2]*100:.2f}%)")
    print(f" [KIỂM TRA CHỈ SỐ NHẤT QUÁN TOÁN HỌC MA TRẬN SAATY]")
    print(f"  - Giá trị riêng lớn nhất (λ_max)  : {lambda_max:.4f}")
    print(f"  - Tỷ số nhất quán thực nghiệm (CR): {cr:.4f} -> {'Đạt tính nhất quán (CR < 0.1)' if cr < 0.1 else 'Mâu thuẫn logic'}")
    print("="*80 + "\n")

    SO_LAN_CHAY = 5
    SO_QUAN_THE = 80
    SO_THE_HE = 60
    
    best_overall_score = float('inf')
    
    for i in range(SO_LAN_CHAY):
        print(f"Chạy lần {i+1}/{SO_LAN_CHAY}...", end=" ")
        best_ind, best_fit, bl_fit, lich_su_f, thong_ke_mo_hinh = chay_toi_uu(w_ahp_kich_ban, n_qt=SO_QUAN_THE, n_the_he=SO_THE_HE)
        print(f" Fitness = {thong_ke_mo_hinh['best_score']:.5f} | C = {best_ind[-1]}s")
        
        if thong_ke_mo_hinh['best_score'] < best_overall_score:
            best_overall_score = thong_ke_mo_hinh['best_score']
            best_ind_chinh_xac = copy.deepcopy(best_ind)
            best_fit_chinh_xac = copy.deepcopy(best_fit)
            bl_fit_chinh_xac = copy.deepcopy(bl_fit)
            lich_su_f_chinh_xac = copy.deepcopy(lich_su_f)
            thong_ke_mo_hinh_chinh_xac = copy.deepcopy(thong_ke_mo_hinh)
            
    print(f"\n--> Đã trích xuất mô hình tốt nhất (Fitness = {best_overall_score:.5f})")
    ve_dt(best_ind_chinh_xac, best_fit_chinh_xac, bl_fit_chinh_xac, lich_su_f_chinh_xac, thong_ke_mo_hinh_chinh_xac)