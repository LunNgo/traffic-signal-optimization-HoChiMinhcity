"""
=============================================================================
TỐI ƯU HÓA ĐÈN GIAO THÔNG – MẠNG LƯỚI 10 NÚT (GIỜ CAO ĐIỂM)
Thuật toán: NSGA-II + Local Search Đa Mục Tiêu
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
L_MAX_LIST = [] 
BASELINE_C = []
BASELINE_G1 = []
BASELINE_G2 = []
BASELINE_OFF = []

# Chu kỳ dài hơn để xả dòng xe lớn (Cao điểm)
# Mở rộng giới hạn để xả dòng xe lớn (Cao điểm)
C_MIN, C_MAX = 80, 100
G_MIN, G_MAX = 15, 120  
OFF_MIN = 0
# ===================== 1. HÀM ĐỌC DỮ LIỆU THỰC TẾ =====================
def tao_dl_thuc_te():
    global df_nodes, DISTANCES, BASELINE_C, BASELINE_G1, BASELINE_G2, BASELINE_OFF, L_MAX_LIST
    
    print("[+] Đang nạp dữ liệu mạng lưới 10 nút (Cao Điểm)...")
    try:
        df_nut = pd.read_csv("Du_lieu_nut_cao_diem.xlsx - Baseline_10Nut_CaoDiem_Dep.csv")
    except:
        df_nut = pd.read_excel("Du_lieu_nut_cao_diem.xlsx")
        
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

    BASELINE_C = df_nut['Chu_kỳ_C (s)'].astype(int).tolist()
    BASELINE_G1 = df_nut['g1_Pha1 (s)'].astype(int).tolist()
    BASELINE_G2 = df_nut['g2_Pha2 (s)'].astype(int).tolist()
    BASELINE_OFF = df_nut['Offset_o (s)'].astype(int).tolist()

    print(f"✓ Đã load thành công {n_nodes} nút và đồng bộ chuẩn hóa cấu trúc dữ liệu.")

# ===================== 2. CÁC HÀM TÍNH TOÁN TOÁN HỌC =====================
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
def evaluate_individual(individual):
    """HÀM EVALUATE BÊ Y HỆT TỪ BẢN 26 NÚT - CHUẨN XÁC 100% VẬT LÝ"""
    n = len(df_nodes)
    G1 = individual[:n]
    OFF = individual[n:2*n]
    c = individual[-1]

    f1_total, f2_total, f3_total = 0.0, 0.0, 0.0
    X_MAX = 1 

    for i in range(n):
        q, S, L, v, qb = df_nodes['q'][i], df_nodes['S'][i], df_nodes['L'][i], df_nodes['v'][i], df_nodes['qb'][i]
        g1, off_k = G1[i], OFF[i]
        g2 = max(G_MIN, c - g1 - 2*int(L))
        
        # CHÍNH XÁC TỶ LỆ Q VÀ S TỪ 26 NÚT
        q1, q2 = q * 0.6, q * 0.4
        S1, S2 = S * 0.6, S * 0.4 

        # F1: TỔNG ĐỘ TRỄ
        d1 = calc_uniform_delay(q1, S1, g1, c) + calc_incremental_delay(q1, S1, g1, c) + calc_residual_delay(q1, qb*0.6)
        d2_delay = calc_uniform_delay(q2, S2, g2, c) + calc_incremental_delay(q2, S2, g2, c) + calc_residual_delay(q2, qb*0.4)
        f1_total += (d1 * q1 + d2_delay * q2) / 3600.0

        # RÀNG BUỘC HỆ SỐ BÃO HÒA x_max (Y hệt 26 nút)
        x1 = (q1 * c) / (S1 * g1 + 1e-6)
        x2 = (q2 * c) / (S2 * g2 + 1e-6)
        
        penalty_x = 0
        if x1 > X_MAX: penalty_x += (x1 - X_MAX) * 10000
        if x2 > X_MAX: penalty_x += (x2 - X_MAX) * 10000
        f1_total += penalty_x
        f2_total += penalty_x
        f3_total += penalty_x

        # RÀNG BUỘC HÀNG CHỜ L_MAX (Y hệt 26 nút)
        lq1 = calc_lq1_uniform(q1, S1, g1, c) + calc_lq2_random(q1, S1, g1, c)
        lq2_queue = calc_lq1_uniform(q2, S2, g2, c) + calc_lq2_random(q2, S2, g2, c)
        
        penalty_lq = 0
        if lq1 > L_MAX_LIST[i]: penalty_lq += (lq1 - L_MAX_LIST[i]) * 1000
        if lq2_queue > L_MAX_LIST[i]: penalty_lq += (lq2_queue - L_MAX_LIST[i]) * 1000
        f1_total += penalty_lq
        f2_total += (lq1 + lq2_queue) + penalty_lq
        f3_total += penalty_lq

        # F3: SỐ LƯỢT DỪNG
        t_travel = DISTANCES[i] / (v * 1000/3600 + 1e-6)
        gamma = calc_gamma_wave(off_k, t_travel, c)
        ns1 = calc_node_stops(q1, S1, g1, c, gamma)
        ns2 = calc_node_stops(q2, S2, g2, c, gamma)
        f3_total += (ns1 + ns2)
        
    # Thêm hàm phạt nếu chu kỳ C đi quá xa so với baseline 90s
    # Phạt nặng hơn ở giờ cao điểm để kìm hãm xu hướng tăng C
    penalty_c = abs(c - 90) * 10.0
    f1_total += penalty_c
    f2_total += penalty_c
    f3_total += penalty_c

    return f1_total, f2_total, f3_total
def calculate_real_metrics(individual):
    """Tính giá trị f1, f2, f3 thực tế để báo cáo (KHÔNG CỘNG HÀM PHẠT)"""
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
        f3_real += calc_node_stops(q1, S1, g1, c, gamma) + calc_node_stops(q2, S2, g2, c, gamma)

    return [f1_real, f2_real, f3_real]

# ===================== 3. AHP VÀ TỔNG HỢP FITNESS =====================
def lay_ma_tran_ahp():
    # Ma trận đã được "hạ nhiệt", f1 vẫn ưu tiên nhưng không quá gắt
    return np.array([
        [1.0, 2.0, 4.0],
        [1/2, 1.0, 2.0],
        [1/4, 1/2, 1.0]
    ])
def tinh_trong_so_ahp_chuan(ma_tran_A):
    n = ma_tran_A.shape[0]
    geo_means = np.array([np.prod(ma_tran_A[i, :])**(1/n) for i in range(n)])
    trong_so = geo_means / geo_means.sum()
    lambda_max = np.mean((ma_tran_A @ trong_so) / trong_so)
    CI = (lambda_max - n) / (n - 1)
    CR = CI / 0.58
    return trong_so.tolist(), lambda_max, CI, CR

def weighted_sum_score(fit, weights, f_min, f_max):
    score = 0
    for j in range(3):
        # Dùng f_min_safe = 0.0 để chặn normalization bị âm
        f_min_safe = 0.0 
        norm = (fit[j] - f_min_safe) / (f_max[j] - f_min_safe) if f_max[j] > f_min_safe else 0
        score += weights[j] * norm
    return score
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

def mutate(ind, pm=0.2): # Thêm tham số pm (dù trong hàm đang fix cứng 0.5)
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

# ===================== 5. TIẾN TRÌNH TỐI ƯU =====================
def chay_toi_uu(w_ahp, n_qt=80, n_the_he=60):
    t_start = time.time()
    
    c_base = int(np.mean(BASELINE_C)) if len(BASELINE_C) > 0 else 90
    bl_ind = repair_individual(BASELINE_G1 + BASELINE_OFF + [c_base])
    bl_fit_real = calculate_real_metrics(bl_ind)
    
    # Lấy điểm neo để tính chuẩn hóa tỷ lệ
    bl_fit_anchor = evaluate_individual(bl_ind)
    bl_fit_anchor = [max(1e-3, f) for f in bl_fit_anchor]
    
    pop = [bl_ind]
    pop += [create_individual(force_c=c_base) for _ in range(n_qt // 3)]
    pop += [create_individual() for _ in range(n_qt - 1 - n_qt // 3)]

    pop_fit = [evaluate_individual(ind) for ind in pop]
    fronts, ranks = fast_non_dominated_sort(pop_fit)
    crowding = [0.0]*n_qt
    for front in fronts:
        cd = crowding_distance(pop_fit, front)
        for j, enumerate_idx in enumerate(front): crowding[enumerate_idx] = cd[j]

    lich_su_f = []

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
            for j, enumerate_idx in enumerate(front): crowding_c[enumerate_idx] = cd[j]

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
                
        # [ĐOẠN CẦN THAY THẾ BẮT ĐẦU TỪ ĐÂY TRONG HÀM chay_toi_uu]
        pareto_fits_gen = [pop_fit[i] for i in range(len(pop)) if ranks[i] == 1]
        if pareto_fits_gen:
            pf_arr = np.array(pareto_fits_gen)
            f_min_gen, f_max_gen = pf_arr.min(axis=0), pf_arr.max(axis=0)
            best_ahp_gen = min([weighted_sum_score(f, w_ahp, f_min_gen, f_max_gen) for f in pareto_fits_gen])
            lich_su_f.append(best_ahp_gen)

    t_nsga2 = time.time() - t_start

    # ========================================================
    # LOCAL SEARCH: Sử dụng chuẩn hóa Min-Max Scaling
    # ========================================================
    t_ls_start = time.time()
    pareto_idx = [i for i in range(len(pop)) if ranks[i] == 1]
    pareto_pop = [pop[i] for i in pareto_idx]
    pareto_fits = [pop_fit[i] for i in pareto_idx]
    
    # Tính f_min, f_max trên toàn bộ tập Pareto hiện tại
    all_fits = np.array(pareto_fits)
    f_min, f_max = all_fits.min(axis=0), all_fits.max(axis=0)
    
    scores = [weighted_sum_score(f, w_ahp, f_min, f_max) for f in pareto_fits]
    top_k = max(1, int(len(pareto_pop)*0.3)) 
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i])[:top_k]

    best_ind = pareto_pop[top_idx[0]]
    best_score = scores[top_idx[0]]

    for idx in top_idx:
        ind = list(pareto_pop[idx])
        score = scores[idx]
        
        for _ in range(50):
            neighbor = list(ind)
            n_var = len(df_nodes)
            for k in random.sample(range(2*n_var + 1), random.randint(2, 5)):
                if k < n_var: neighbor[k] += random.choice([-2, 0, 2])
                elif k < 2*n_var: neighbor[k] += random.choice([-8, 8])
                else: neighbor[k] += random.choice([-5, 0, 5])
            
            neighbor = repair_individual(neighbor)
            new_fit_penalized = evaluate_individual(neighbor)
            
            # Truyền f_min, f_max vào hàm tính điểm mới
            new_score = weighted_sum_score(new_fit_penalized, w_ahp, f_min, f_max)

            if new_score < score:
                ind, score = neighbor, new_score

        if score < best_score:
            best_ind = ind
            best_score = score
            
    t_ls = time.time() - t_ls_start
    best_fit_real = calculate_real_metrics(best_ind)
    
    thong_ke_mo_hinh = {
        'thoi_gian_chay': t_nsga2 + t_ls,
        'tg_nsga2': t_nsga2,
        'tg_local_search': t_ls,
        'best_score': best_score
    }
    
    return best_ind, best_fit_real, bl_fit_real, lich_su_f, thong_ke_mo_hinh

# ===================== 6. TRỰC QUAN HÓA =====================
def ve_dt(best_ind, best_fit_real, bl_fit_real, lich_su_f, thong_ke_mo_hinh):
    import matplotlib as mpl 
    n = len(df_nodes)
    c_opt = best_ind[-1]
    g1_opt = best_ind[:n]
    off_opt = best_ind[n:2*n]
    g2_opt = [max(G_MIN, c_opt - g1_opt[i] - 2*int(df_nodes['L'][i])) for i in range(n)]
    
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'DejaVu Sans' if 'DejaVu Sans' in [f.name for f in mpl.font_manager.fontManager.ttflist] else 'sans-serif'
    
    print("\n" + "="*85)
    print("      BÁO CÁO KẾT QUẢ TỐI ƯU HÓA ĐÈN GIAO THÔNG PHỐI HỢP CHO TRỤC 10 NÚT (CAO ĐIỂM)")
    print("="*85)
    print(f" * Điểm số hàm mục tiêu tổng hợp (Fitness) : {thong_ke_mo_hinh['best_score']:.6f}")
    print(f" * Chu kỳ đèn tín hiệu chung tối ưu (C)     : {int(c_opt)} giây (s)")
    print(f" * Mảng dịch lệch pha tối ưu (Offset vt_o) : {off_opt}")  
    print("-" * 85)
    
    print("\n" + "=" * 82)
    print(f"{'SO SÁNH TRƯỚC & SAU TỐI ƯU HÓA':^82}")
    print("=" * 82)
    print(f"{'Tiêu chí':<32} {'Baseline (Trục 10 Nút)':<22} {'Tối ưu':<18} {'Cải thiện'}")
    print("-" * 82)

    p1_imp = ((bl_fit_real[0] - best_fit_real[0]) / bl_fit_real[0] * 100) if bl_fit_real[0] != 0 else 0
    p2_imp = ((bl_fit_real[1] - best_fit_real[1]) / bl_fit_real[1] * 100) if bl_fit_real[1] != 0 else 0
    p3_imp = ((bl_fit_real[2] - best_fit_real[2]) / bl_fit_real[2] * 100) if bl_fit_real[2] != 0 else 0

    print(f"{'Tổng thời gian trễ (f1)':<32} {bl_fit_real[0]:,.1f} xe.h{'':<11} {best_fit_real[0]:,.1f} xe.h{'':<7} {p1_imp:>8.1f}%")
    print(f"{'Độ dài hàng chờ TB (f2)':<32} {bl_fit_real[1]:,.1f} xe{'':<13} {best_fit_real[1]:,.1f} xe{'':<9} {p2_imp:>8.1f}%")
    print(f"{'Số lần dừng xe (f3)':<32} {bl_fit_real[2]:,.0f} lượt/h{'':<11} {best_fit_real[2]:,.0f} lượt/h{'':<7} {p3_imp:>8.1f}%")
    print("=" * 82 + "\n")

    print(f" {'Nút giao':<10} | {'Thời gian đèn xanh hiệu quả (g1, g2)':<45} | {'Độ lệch pha o_k (s)':<15}")
    print("-" * 85)
    for k in range(n):
        print(f" Nút {k+1:<5} | {str([g1_opt[k], g2_opt[k]]):<45} | {int(off_opt[k]):<15}")
    print("=" * 85)

    print(" [i] Đang xuất biểu đồ trực quan thực tế...\n")

    plt.figure(figsize=(10, 4.5))
    plt.plot(range(1, len(lich_su_f)+1), lich_su_f, color='#e74c3c', linewidth=2.5, marker='o', markersize=4)
    plt.title('Tiến trình hội tụ AHP của Thuật toán NSGA-II (Cao Điểm)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Thế hệ tiến hóa')
    plt.ylabel('Giá trị AHP Fitness')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('outputs/10Nut_CaoDiem_HoiTu.png', dpi=150)
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
    plt.savefig('outputs/10Nut_CaoDiem_PhanBoXanh_Offset.png', dpi=150)
    plt.close()

# ===================== MAIN =====================
# ===================== MAIN =====================
if __name__ == "__main__":
    # 1. Bắt đầu bấm giờ toàn bộ chương trình
    total_start_time = time.time()
    
    tao_dl_thuc_te()
    
    A_matrix = lay_ma_tran_ahp()
    w_ahp_kich_ban, lambda_max, CI, cr = tinh_trong_so_ahp_chuan(A_matrix)
    
    print("\n" + "="*80)
    print(" KẾT QUẢ ĐẦU RA TOÁN HỌC PHƯƠNG PHÁP PHÂN TÍCH THỨ BẬC AHP (Giờ Cao Điểm)")
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
    
    # Biến để lưu trữ mô hình có Fitness AHP tốt nhất
    best_overall_score = float('inf')
    
    for i in range(SO_LAN_CHAY):
        print(f"Chạy lần {i+1}/{SO_LAN_CHAY}...", end=" ")
        best_ind, best_fit_real, bl_fit_real, lich_su_f, thong_ke_mo_hinh = chay_toi_uu(w_ahp_kich_ban, n_qt=SO_QUAN_THE, n_the_he=SO_THE_HE)
        
        # IN LẠI ĐÚNG FITNESS THEO YÊU CẦU CỦA BẠN
        print(f" Fitness = {thong_ke_mo_hinh['best_score']:.5f} | C = {best_ind[-1]}s")
        
        # Chọn nghiệm dựa trên tổng điểm Fitness AHP nhỏ nhất
        if thong_ke_mo_hinh['best_score'] < best_overall_score:
            best_overall_score = thong_ke_mo_hinh['best_score']
            best_ind_chinh_xac = copy.deepcopy(best_ind)
            best_fit_real_chinh_xac = copy.deepcopy(best_fit_real)
            bl_fit_real_chinh_xac = copy.deepcopy(bl_fit_real)
            lich_su_f_chinh_xac = copy.deepcopy(lich_su_f)
            thong_ke_mo_hinh_chinh_xac = copy.deepcopy(thong_ke_mo_hinh)
            
    print(f"\n--> Đã trích xuất mô hình tốt nhất (Fitness = {best_overall_score:.5f})")
    ve_dt(best_ind_chinh_xac, best_fit_real_chinh_xac, bl_fit_real_chinh_xac, lich_su_f_chinh_xac, thong_ke_mo_hinh_chinh_xac) 
    
    # 2. Kết thúc bấm giờ và in ra màn hình
    total_end_time = time.time()
    print(f"\n[!] Tổng thời gian thực thi toàn bộ chương trình: {total_end_time - total_start_time:.2f} giây.")