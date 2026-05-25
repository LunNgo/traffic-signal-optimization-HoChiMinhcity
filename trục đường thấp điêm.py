import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import copy
import time
import pandas as pd
import os

# ==============================================================================
# 1. HÀM ĐỌC VÀ KHỞI TẠO DỮ LIỆU THỰC TẾ (10 NÚT TRỤC THẤP ĐIỂM)
# ==============================================================================
def tao_dl_thuc_te():
    print("[+] Đang nạp dữ liệu mạng lưới 10 nút...")
    
    # Hỗ trợ đọc linh hoạt định dạng file (CSV hoặc Excel)
    try:
        df_nut = pd.read_csv("Du_lieu_nut_thap_diem.xlsx - Trang tính1.csv")
    except:
        df_nut = pd.read_excel("Du_lieu_nut_thap_diem.xlsx")
        
    try:
        df_lk = pd.read_csv("Khoang_Cach.xlsx - Ma_nutTên đoạnKhoang_cach_D (m).csv")
    except:
        df_lk = pd.read_excel("Khoang_Cach.xlsx")

    data_nut = df_nut.to_numpy()
    so_nut = len(np.unique(data_nut[:, 0])) 
    ds_nut = {}
    
    # Đọc dữ liệu nút
    for row in data_nut:
        k = int(row[0])
        if k not in ds_nut:
            ds_nut[k] = {'so_pha': 0, 'q': [], 'S': [], 'w': [], 'v': [], 'qb': [],
                         'P': 0.55, 'fp': 1.0, 'L': float(row[3]),
                         'g_min': [], 'g_max': []}
        
        ds_nut[k]['so_pha'] += 1
        ds_nut[k]['q'].append(float(row[1]))
        ds_nut[k]['S'].append(float(row[2]))
        # Xử lý an toàn cột w nếu bị khuyết
        w_val = float(row[4]) if len(row) > 4 else 20.0
        v_val = float(row[5]) if len(row) > 5 else float(row[4]) 
        qb_val = int(row[6]) if len(row) > 6 else int(row[5]) if len(row) > 5 else 5
        
        ds_nut[k]['w'].append(w_val)
        ds_nut[k]['v'].append(v_val)
        ds_nut[k]['qb'].append(qb_val)
        ds_nut[k]['g_min'].append(18.0)
        ds_nut[k]['g_max'].append(75.0)
    
    # Phân bổ tỷ lệ pha 
    for k in ds_nut:
        nut = ds_nut[k]
        q_total = nut['q'][0]
        S_total = nut['S'][0]
        ratio = 0.51 + 0.09 * np.sin(k)   
        nut['so_pha'] = 2
        nut['q'] = np.array([q_total * ratio, q_total * (1 - ratio)])
        nut['S'] = np.array([S_total * 0.94, S_total * 0.97])
        nut['g_min'] = np.array([18.0, 18.0])
        nut['g_max'] = np.array([90.0, 90.0])
        for key in ['w', 'v', 'qb']:
            nut[key] = np.array([nut[key][0]] * 2)
    
    # Đọc dữ liệu liên kết (khoảng cách)
    data_lien_ket = df_lk.to_numpy()
    D_list = [0.0] * so_nut
    tl_list = [0.0] * so_nut
    L_max_list = [35.0] * so_nut  
    
    for row in data_lien_ket:
        tu = int(row[0])
        if tu - 1 < so_nut:
            dist = float(row[2])
            D_list[tu-1] = dist
            tl_list[tu-1] = float(row[3]) if len(row) > 3 else dist / (15.0 * 1000/3600)
            L_max_list[tu-1] = float(round((dist / 6.5) * 0.6))
    
    dl = {
        'D': np.array(D_list), 
        'tl': np.array(tl_list),
        'L_max_mang': np.array(L_max_list), 
        'T': 1.0, 't_ton': 5.0, 'k_hc': 0.5, 'I_hc': 1.0,
        'beta': 0.28, 'Tp': 1.5, 
        'C_min': 80, 'C_max': 150, 'x_max': 0.90,
        'nut': ds_nut
    }
    print(f"✓ Đã load thành công {so_nut} nút | Đã đồng bộ L_max theo mạng lưới thực tế")
    return dl

# ==============================================================================
# 2. CÁC HÀM TÍNH TOÁN TOÁN HỌC CỐT LÕI 
# ==============================================================================
def tinh_lambda(g, C):
    return g / C if C > 0 else 0

def tinh_x(q, S, lam):
    if S * lam <= 0: return 99.0
    return q / (S * lam)

def tinh_c(S, lam):
    return S * lam

def tinh_pf(P, lam, fp):
    mau = 1 - lam
    return ((1 - P) / np.maximum(0.001, mau)) * fp

def tinh_gamma(o_k, o_truoc, tl, beta, Tp):
    if o_truoc is None: return 1.0
    d_o = np.abs(o_k - o_truoc - tl)
    return 1.0 - beta * np.exp(-d_o / Tp)

def tinh_d1(C, lam, x):
    tu = C * ((1 - lam) ** 2)
    mau = 2 * (1 - np.minimum(1.0, x) * lam)
    return tu / np.maximum(0.001, mau)

def tinh_d2(x, c, T, k_hc, I_hc):
    t1 = x - 1
    t2 = np.sqrt(np.maximum(0.0, (x - 1)**2 + (8 * k_hc * I_hc * x) / (c * T + 1e-5)))
    return 900 * T * (t1 + t2)

def tinh_d3(qb, t_ton, q, T):
    mau = q * T
    if mau <= 0: return 0
    return (3600 * qb * t_ton) / mau

def tinh_f1(C, mt_g, vt_o, dl):
    f1 = 0.0
    for k, n in dl['nut'].items():
        for i in range(n['so_pha']):
            lam = tinh_lambda(mt_g[k-1][i], C)
            x = tinh_x(n['q'][i], n['S'][i], lam)
            c = tinh_c(n['S'][i], lam)
            pf = tinh_pf(n['P'], lam, n['fp'])
            d1 = tinh_d1(C, lam, x)
            d2 = tinh_d2(x, c, dl['T'], dl['k_hc'], dl['I_hc'])
            d3 = tinh_d3(n['qb'][i], dl['t_ton'], n['q'][i], dl['T'])
            f1 += n['q'][i] * ((d1 * pf + d2 + d3) / 3600.0)
    return f1

def tinh_f2(C, mt_g, vt_o, dl):
    f2 = 0.0
    for k, n in dl['nut'].items():
        for i in range(n['so_pha']):
            g_ki = float(mt_g[k-1][i])
            q_gio = float(n['q'][i])
            lam = tinh_lambda(g_ki, C)
            x = tinh_x(q_gio, n['S'][i], lam)
            r = max(0.0, C - g_ki)
            L_q1 = (q_gio / 3600.0) * (r ** 2) / (2 * C * (1 - np.minimum(0.95, x) * lam + 1e-6))
            L_q2 = 0.0
            if x > 0.5:
                L_q2 = 0.25 * x**2 * ( (x - 1) + np.sqrt((x - 1)**2 + (16 * x) / (n['S'][i] * lam + 1e-5)) )
            f2 += max(0.0, L_q1 + L_q2)
    return f2

def tinh_f3(C, mt_g, vt_o, dl):
    f3 = 0.0
    for k, n in dl['nut'].items():
        o_k = vt_o[k-1]
        o_truoc = vt_o[k-2] if k > 1 else None
        ga = tinh_gamma(o_k, o_truoc, dl['tl'][k-1], dl['beta'], dl['Tp'])
        for i in range(n['so_pha']):
            g_ki = mt_g[k-1][i]
            q_ki = n['q'][i]
            S_ki = n['S'][i]
            lambda_ki = g_ki / C
            x_ki = q_ki / (S_ki * lambda_ki + 1e-5)
            term_time = (C - g_ki) / C
            term_sat = 1.0 / (1.0 - np.minimum(0.95, x_ki))
            f3 += q_ki * term_time * term_sat * ga
    return f3

def tinh_F_MAX_tu_dong(dl):
    f1_max = f2_max = f3_max = 0.0
    C_max = dl['C_max']
    for k, n in dl['nut'].items():
        for i in range(n['so_pha']):
            q, S, g_min = n['q'][i], n['S'][i], n['g_min'][i]
            lam_min = g_min / C_max
            x_max_phat = q / (S * lam_min + 1e-5)
            c_min = S * lam_min
            d1_max = 0.5 * (C_max * (1 - lam_min)**2) / (1 - min(1.0, x_max_phat) * lam_min + 1e-6)
            t1 = x_max_phat - 1
            t2 = np.sqrt(max(0.0, t1**2 + (8 * dl['k_hc'] * dl['I_hc'] * x_max_phat) / (c_min * dl['T'] + 1e-5)))
            d2_max = 900 * dl['T'] * (t1 + t2)
            d3_max = (3600 * n['qb'][i] * dl['t_ton']) / (q * dl['T'] + 1e-5)
            f1_max += q * ((d1_max * 1.5 + d2_max + d3_max) / 3600.0) 
            
            r_max = C_max - g_min
            L_q1_max = (q / 3600.0) * (r_max**2) / (2 * C_max * (1 - min(0.95, x_max_phat) * lam_min + 1e-6))
            L_q2_max = 0.25 * x_max_phat**2 * (t1 + np.sqrt(t1**2 + (16 * x_max_phat) / (c_min + 1e-5))) if x_max_phat > 0.5 else 0.0
            
            f2_max += (L_q1_max + L_q2_max)
            f3_max += q * ((C_max - g_min) / C_max) * (1.0 / (1.0 - min(0.95, x_max_phat))) * 1.0

    return [max(20.0, f1_max), max(10.0, f2_max), max(500.0, f3_max)]

# ==============================================================================
# 3. MÔ HÌNH TOÁN HỌC AHP - (CẬP NHẬT THEO BẢNG 4)
# ==============================================================================
def lay_ma_tran_ahp(kich_ban=None):
    """
    Sử dụng cứng ma trận so sánh cặp từ Bảng 4.
    Ưu tiên mạnh f1 (thời gian trễ) so với f3 (số lần dừng).
    """
    return np.array([
        [1.0, 1/2, 1/2], # f1 bằng 1/2 so với f2 và f3
        [2.0, 1.0, 1.0], # f2 gấp 2 lần f1, bằng f3
        [2.0, 1.0, 1.0]  # f3 gấp 2 lần f1, bằng f2
    ])
    

def tinh_trong_so_ahp_chuan(ma_tran_A):
    """
    Tính trọng số chuẩn xác sử dụng trung bình nhân (Geometric Mean) từ ma trận Saaty
    """
    n = ma_tran_A.shape[0]
    geo_means = np.array([np.prod(ma_tran_A[i, :])**(1/n) for i in range(n)])
    trong_so = geo_means / geo_means.sum()
    
    Aw = ma_tran_A @ trong_so
    lambda_max = np.mean(Aw / trong_so)
    CI = (lambda_max - n) / (n - 1)
    RI_dict = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12}
    CR = CI / RI_dict.get(n, 0.58)
    
    return trong_so.tolist(), lambda_max, CI, CR

def tinh_f_tong(f1, f2, f3, F_MAX, w):
    return w[0]*(f1/F_MAX[0]) + w[1]*(f2/F_MAX[1]) + w[2]*(f3/F_MAX[2])

# ==============================================================================
# 4. KIỂM TRA RÀNG BUỘC & CHUẨN HÓA GREEN TIME
# ==============================================================================
def kt_rb(C, mt_g, vt_o, dl, tra_ve_phat=False):
    if not (dl['C_min'] <= C <= dl['C_max']): 
        return False if not tra_ve_phat else 99999
    if vt_o[0] != 0: 
        return False if not tra_ve_phat else 99999
    
    tong_phat = 0.0
    for k, n in dl['nut'].items():
        tong_g = 0.0
        if not (0 <= vt_o[k-1] < C): 
            return False if not tra_ve_phat else 99999
        L_max_nut = dl['L_max_mang'][k-1]
        
        for i in range(n['so_pha']):
            g_ki = mt_g[k-1][i]
            if not (n['g_min'][i] <= g_ki <= n['g_max'][i]): 
                return False if not tra_ve_phat else 99999
            tong_g += g_ki
            lam = tinh_lambda(g_ki, C)
            x = tinh_x(n['q'][i], n['S'][i], lam)
            
            if x > dl['x_max']:
                tong_phat += (x - dl['x_max']) * 300.0
            
            r = max(0.0, C - g_ki)
            L_q1 = (n['q'][i] / 3600.0) * (r ** 2) / (2 * C * (1 - np.minimum(0.95, x) * lam + 1e-6))
            L_q2 = 0.25 * x**2 * ((x - 1) + np.sqrt((x - 1)**2 + (16 * x) / (n['S'][i] * lam + 1e-5))) if x > 0.5 else 0.0
            
            L_q_hien_tai = max(0.0, L_q1 + L_q2)
            if L_q_hien_tai > L_max_nut:
                if not tra_ve_phat: return False 
                tong_phat += (L_q_hien_tai - L_max_nut) * 150.0 
                
        if abs(C - (tong_g + n['L'])) > 1.0: 
            return False if not tra_ve_phat else 99999
        
    if tra_ve_phat: return tong_phat
    return True

def chuan_hoa_green_time(C, mt_g, dl):
    mt_g_moi = []
    for k, n in dl['nut'].items():
        g_trong = C - n['L']
        g_hiendai = np.array(mt_g[k-1]).copy()
        g_hiendai = np.clip(g_hiendai, n['g_min'], n['g_max'])
        diff = int(round(g_trong - g_hiendai.sum()))
        
        if diff != 0:
            idx_max = np.argmax(n['q'])
            g_hiendai[idx_max] = np.clip(g_hiendai[idx_max] + diff, n['g_min'][idx_max], n['g_max'][idx_max])
            diff_con_lai = int(round(g_trong - g_hiendai.sum()))
            if diff_con_lai != 0:
                for idx in range(len(g_hiendai)):
                    g_hiendai[idx] = np.clip(g_hiendai[idx] + diff_con_lai, n['g_min'][idx], n['g_max'][idx])
                    diff_con_lai = int(round(g_trong - g_hiendai.sum()))
                    if diff_con_lai == 0: break
                    
        mt_g_moi.append(g_hiendai.astype(int).tolist())
    return mt_g_moi

# ==============================================================================
# 5. LỚP CÁ THỂ VÀ HỆ TOÁN TỬ TIẾN HÓA 
# ==============================================================================
class CaThe:
    def __init__(self, dl, C_co_dinh=None, w_he_thong=None):
        self.dl = dl
        self.F_MAX = tinh_F_MAX_tu_dong(dl) 
        self.w_ahp = w_he_thong if w_he_thong is not None else [0.637, 0.258, 0.105]
        
        self.C = C_co_dinh if C_co_dinh is not None else random.randint(dl['C_min'], dl['C_max'])
        self.vt_o = [0] + [random.randint(0, self.C-1) for _ in range(len(dl['nut'])-1)]
        
        tam_mt_g = []
        for k, n in dl['nut'].items():
            g_trong = self.C - n['L']
            if n['so_pha'] == 1:
                tam_mt_g.append([int(round(g_trong))])
                continue
                
            ty = n['q'] / n['q'].sum()
            g = np.round(ty * g_trong * (0.85 + 0.3 * np.random.rand())).astype(int)
            g = np.clip(g, n['g_min'], n['g_max'])
            tam_mt_g.append(g.tolist())
        
        self.mt_g = chuan_hoa_green_time(self.C, tam_mt_g, dl)
        self.f1 = self.f2 = self.f3 = self.fit = float('inf')

    def danh_gia(self):
        if not kt_rb(self.C, self.mt_g, self.vt_o, self.dl, tra_ve_phat=False):
            self.f1 = self.f2 = self.f3 = self.fit = 1e6
            return
            
        self.f1 = tinh_f1(self.C, self.mt_g, self.vt_o, self.dl)
        self.f2 = tinh_f2(self.C, self.mt_g, self.vt_o, self.dl)
        self.f3 = tinh_f3(self.C, self.mt_g, self.vt_o, self.dl)
        
        phat = kt_rb(self.C, self.mt_g, self.vt_o, self.dl, tra_ve_phat=True)
        self.fit = tinh_f_tong(self.f1, self.f2, self.f3, self.F_MAX, w=self.w_ahp) + phat

def sx_pareto(qt):
    fronts = [[]]
    for p in qt:
        p.domination_count = 0
        p.dominated_solutions = []
        for q in qt:
            if (p.f1 <= q.f1 and p.f2 <= q.f2 and p.f3 <= q.f3) and (p.f1 < q.f1 or p.f2 < q.f2 or p.f3 < q.f3):
                p.dominated_solutions.append(q)
            elif (q.f1 <= p.f1 and q.f2 <= p.f2 and q.f3 <= p.f3) and (q.f1 < p.f1 or q.f2 < p.f2 or q.f3 < p.f3):
                p.domination_count += 1
        if p.domination_count == 0:
            p.pareto_rank = 0 
            fronts[0].append(p)
    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in p.dominated_solutions:
                q.domination_count -= 1
                if q.domination_count == 0:
                    q.pareto_rank = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return [f for f in fronts if len(f) > 0]

def tinh_kc(front):
    l = len(front)
    if l == 0: return
    for p in front: p.kc_dam_dong = 0.0
    if l <= 2:
        for p in front: p.kc_dam_dong = float('inf')
        return
    for obj in ['f1', 'f2', 'f3']:
        front.sort(key=lambda x: getattr(x, obj))
        front[0].kc_dam_dong = front[-1].kc_dam_dong = float('inf')
        f_max, f_min = getattr(front[-1], obj), getattr(front[0], obj)
        if f_max > f_min:
            for i in range(1, l - 1):
                front[i].kc_dam_dong += (getattr(front[i+1], obj) - getattr(front[i-1], obj)) / (f_max - f_min)

def lai_db(p1, p2, dl):
    con = CaThe(dl)
    con.C = random.choice([p1.C, p2.C])
    con.vt_o = [random.choice([o1, o2]) for o1, o2 in zip(p1.vt_o, p2.vt_o)]
    con.mt_g = copy.deepcopy(random.choice([p1.mt_g, p2.mt_g]))
        
    if random.random() < 0.25:  
        con.C = int(np.clip(con.C + random.choice([-8, -4, 4, 8]), dl['C_min'], dl['C_max']))
        k_rand = random.randint(1, len(dl['nut']) - 1)
        con.vt_o[k_rand] = random.randint(0, con.C - 1)
        
    if random.random() < 0.40: 
        k_rand = random.randint(0, len(dl['nut']) - 1) 
        con.mt_g[k_rand][0] = int(np.clip(con.mt_g[k_rand][0] + random.choice([-5, -3, 3, 5]), 
                                         dl['nut'][k_rand+1]['g_min'][0], dl['nut'][k_rand+1]['g_max'][0]))
        
    con.vt_o = [o % con.C for o in con.vt_o]
    con.vt_o[0] = 0
    con.mt_g = chuan_hoa_green_time(con.C, con.mt_g, dl) 
    return con

def tim_cb(ct, max_buoc=12):
    Best = copy.deepcopy(ct)
    dl = ct.dl
    for _ in range(max_buoc):
        ung_vien = copy.deepcopy(Best)
        n_var = len(dl['nut'])
        
        for k in random.sample(range(2*n_var + 1), random.randint(2, 5)):
            if k < n_var: 
                ung_vien.mt_g[k][0] += random.choice([-2, 0, 2])
            elif k < 2*n_var:
                idx_o = k - n_var
                ung_vien.vt_o[idx_o] += random.choice([-8, 8])
            else: 
                ung_vien.C += random.choice([-5, 0, 5])
                
        ung_vien.C = int(np.clip(ung_vien.C, dl['C_min'], dl['C_max']))
        ung_vien.mt_g = chuan_hoa_green_time(ung_vien.C, ung_vien.mt_g, dl)
        ung_vien.vt_o = [o % ung_vien.C for o in ung_vien.vt_o]
        ung_vien.vt_o[0] = 0
        
        ung_vien.danh_gia()
        if ung_vien.fit < Best.fit: Best = ung_vien
    return Best

# ==============================================================================
# 6. TIẾN TRÌNH CHẠY TỐI ƯU HÓA LAI
# ==============================================================================
def chay_toi_uu(dl, w_kich_ban, n_qt=100, n_the_he=80):
    t_bat_dau = time.time()  
    tg_nsga2 = 0.0
    tg_local_search = 0.0
    
    t_s1 = time.time()
    qt = []
    
    C_goc_ra_deu = np.linspace(dl['C_min'] + 5, dl['C_max'] - 5, n_qt).astype(int)
    for idx in range(n_qt):
        ct = CaThe(dl, C_co_dinh=int(C_goc_ra_deu[idx]), w_he_thong=w_kich_ban)
        ct.danh_gia()
        qt.append(ct)
        
    # Baseline tham chiếu
    C_thuc_te = 75 
    mt_g_thuc_te = [[44, 25], [40, 29], [41, 28], [38, 31], [43, 26], [41, 28], [45, 24], [42, 27], [39, 30], [40, 29]]
    vt_o_thuc_te = [0, 13, 26, 39, 52, 65, 75, 10, 25, 40] 
        
    ca_the_truoc_tu = CaThe(dl, C_co_dinh=C_thuc_te, w_he_thong=w_kich_ban)
    ca_the_truoc_tu.C = C_thuc_te
    ca_the_truoc_tu.mt_g = mt_g_thuc_te  
    ca_the_truoc_tu.vt_o = [o % C_thuc_te for o in vt_o_thuc_te]
    ca_the_truoc_tu.vt_o[0] = 0
    ca_the_truoc_tu.f1 = tinh_f1(C_thuc_te, ca_the_truoc_tu.mt_g, ca_the_truoc_tu.vt_o, dl)
    ca_the_truoc_tu.f2 = tinh_f2(C_thuc_te, ca_the_truoc_tu.mt_g, ca_the_truoc_tu.vt_o, dl)
    ca_the_truoc_tu.f3 = tinh_f3(C_thuc_te, ca_the_truoc_tu.mt_g, ca_the_truoc_tu.vt_o, dl)
    ca_the_truoc_tu.fit = tinh_f_tong(ca_the_truoc_tu.f1, ca_the_truoc_tu.f2, ca_the_truoc_tu.f3, ca_the_truoc_tu.F_MAX, w=w_kich_ban)
        
    tg_nsga2 += (time.time() - t_s1)
    lich_su_f, so_lan_ls, so_lan_ls_thanh_cong, tong_fit_giam_nho_ls = [], 0, 0, 0.0
    
    for th in range(n_the_he):
        t_s2 = time.time()
        fronts = sx_pareto(qt)
        for f in fronts: tinh_kc(f)
        
        ds_con = []
        for _ in range(n_qt):
            p1, p2 = random.sample(qt, 2)
            con = lai_db(p1, p2, dl)
            con.w_ahp = w_kich_ban
            con.danh_gia()
            ds_con.append(con)
                
        qt.sort(key=lambda x: x.fit)
        lich_su_f.append(qt[0].fit)
        tg_nsga2 += (time.time() - t_s2)
        
        t_s3 = time.time()
        n_ls = int(n_qt * 0.25)  
        for i in range(n_ls):
            fit_truoc_ls = qt[i].fit
            qt[i] = tim_cb(qt[i], max_buoc=15)
            qt[i].w_ahp = w_kich_ban
            qt[i].danh_gia()
            so_lan_ls += 1
            if qt[i].fit < fit_truoc_ls:
                so_lan_ls_thanh_cong += 1
                tong_fit_giam_nho_ls += (fit_truoc_ls - qt[i].fit)
        tg_local_search += (time.time() - t_s3)
            
        t_s4 = time.time()
        qt.extend(ds_con)
        fronts = sx_pareto(qt)
        qt_moi = []
        for f in fronts:
            tinh_kc(f)
            if len(qt_moi) + len(f) <= n_qt: qt_moi.extend(f)
            else:
                f.sort(key=lambda x: x.kc_dam_dong, reverse=True)
                qt_moi.extend(f[:n_qt - len(qt_moi)])
                break
        qt = qt_moi[:n_qt]
        tg_nsga2 += (time.time() - t_s4)
        
    qt.sort(key=lambda x: x.fit)
    thong_ke_mo_hinh = {
        'thoi_gian_chay': time.time() - t_bat_dau, 'tg_nsga2': tg_nsga2,
        'tg_local_search': tg_local_search, 'so_lan_ls': so_lan_ls,
        'so_lan_ls_thanh_cong': so_lan_ls_thanh_cong, 'tong_fit_giam_nho_ls': tong_fit_giam_nho_ls
    }
    return qt[0], ca_the_truoc_tu, lich_su_f, thong_ke_mo_hinh

# ==============================================================================
# 7. TRỰC QUAN HÓA KẾT QUẢ ĐẦU RA MÔ HÌNH (ĐÃ CẢI TIẾN TOÀN DIỆN)
# ==============================================================================
def ve_dt(ct_tot, ca_the_truoc_tu, lich_su_f, thong_ke_mo_hinh):
    import matplotlib as mpl 
    ma_tran_g = ct_tot.mt_g
    so_nut = len(ma_tran_g)
    vector_o = [int(x) for x in ct_tot.vt_o]

    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'DejaVu Sans' if 'DejaVu Sans' in [f.name for f in mpl.font_manager.fontManager.ttflist] else 'sans-serif'
    
    tg_chay = thong_ke_mo_hinh['thoi_gian_chay']
    tg_nsga2 = thong_ke_mo_hinh['tg_nsga2']
    tg_ls = thong_ke_mo_hinh['tg_local_search']
    
    print("\n" + "="*85)
    print("      BÁO CÁO KẾT QUẢ TỐI ƯU HÓA ĐÈN GIAO THÔNG PHỐI HỢP CHO TRỤC ĐƯỜNG 10 NÚT")
    print("="*85)
    print(f" * Điểm số hàm mục tiêu tổng hợp (Fitness) : {ct_tot.fit:.6f}")
    print(f" * Chu kỳ đèn tín hiệu chung tối ưu (C)     : {int(ct_tot.C)} giây (s)")
    print(f" * Mảng dịch lệch pha tối ưu (Offset vt_o) : {vector_o}")  
    print("-"*85)
    
    print("\n" + "=" * 82)
    print(f"{'SO SÁNH TRƯỚC & SAU TỐI ƯU HÓA':^82}")
    print("=" * 82)
    print(f"{'Tiêu chí':<32} {'Baseline (Trục 10 Nút)':<22} {'Tối ưu':<18} {'Cải thiện'}")
    print("-" * 82)

    p1_imp = ((ca_the_truoc_tu.f1 - ct_tot.f1) / ca_the_truoc_tu.f1 * 100) if ca_the_truoc_tu.f1 != 0 else 0
    p2_imp = ((ca_the_truoc_tu.f2 - ct_tot.f2) / ca_the_truoc_tu.f2 * 100) if ca_the_truoc_tu.f2 != 0 else 0
    p3_imp = ((ca_the_truoc_tu.f3 - ct_tot.f3) / ca_the_truoc_tu.f3 * 100) if ca_the_truoc_tu.f3 != 0 else 0

    print(f"{'Tổng thời gian trễ (f1)':<32} {ca_the_truoc_tu.f1:,.1f} xe.h{'':<11} {ct_tot.f1:,.1f} xe.h{'':<7} {p1_imp:>8.1f}%")
    print(f"{'Độ dài hàng chờ TB (f2)':<32} {ca_the_truoc_tu.f2:,.1f} xe{'':<13} {ct_tot.f2:,.1f} xe{'':<9} {p2_imp:>8.1f}%")
    print(f"{'Số lần dừng xe (f3)':<32} {ca_the_truoc_tu.f3:,.0f} lượt/h{'':<11} {ct_tot.f3:,.0f} lượt/h{'':<7} {p3_imp:>8.1f}%")
    print("=" * 82 + "\n")

    print(f" {'Nút giao':<10} | {'Thời gian đèn xanh hiệu quả từng pha g_k,i (giây)':<48} | {'Độ lệch pha o_k (s)':<15}")
    print("-"*85)
    for k in range(so_nut):
        print(f" Nút {k+1:<5} | {str([int(x) for x in ma_tran_g[k]]):<48} | {int(vector_o[k]):<15}")
    print("="*85)

    print(" [i] Đang khởi tạo hiển thị đồ thị trực quan thực tế...\n")

    # ĐỒ THỊ 1: Tiến trình hội tụ
    plt.figure(figsize=(10, 4.5))
    plt.plot(range(1, len(lich_su_f)+1), lich_su_f, color='#e74c3c', linewidth=2.5, marker='o', markersize=4)
    plt.title('Tiến trình hội tụ của Thuật toán Lai (Hybrid GA + Local Search)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Thế hệ tiến hóa')
    plt.ylabel('Giá trị hàm mục tiêu tổng hợp (Fitness)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # ĐỒ THỊ 2: Cột chồng g1, g2 + Đường Offset Twin-Axis
    fig, ax1 = plt.subplots(figsize=(12, 6))
    nuts = [f"Nút {i+1}" for i in range(so_nut)]
    
    g1_vals = [ma_tran_g[k][0] for k in range(so_nut)]
    g2_vals = [ma_tran_g[k][1] for k in range(so_nut)]
    
    # Biểu đồ cột chồng - SỬ DỤNG MÀU CŨ THEO YÊU CẦU
    ax1.bar(nuts, g1_vals, label='Pha 1 ($g_1$)', color='#85c1e9', edgecolor='white', width=0.55)
    ax1.bar(nuts, g2_vals, bottom=g1_vals, label='Pha 2 ($g_2$)', color='#76d7c4', edgecolor='white', width=0.55)
    
    ax1.set_xlabel('Các nút giao thông', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Thời gian xanh hiệu dụng (giây)', fontsize=12, fontweight='bold')
    ax1.set_title(f'PHÂN BỔ THỜI GIAN XANH & ĐỘ LỆCH PHA (CHU KỲ CHUNG C = {int(ct_tot.C)}s)', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    # Trục Y thứ 2 cho đường Offset
    ax2 = ax1.twinx()
    ax2.plot(nuts, vector_o, color='#f39c12', marker='D', linewidth=2.5, markersize=8, label='Độ lệch pha (Offset - $O_k$)')
    ax2.set_ylabel('Offset (giây)', fontsize=12, fontweight='bold', color='#f39c12')
    ax2.tick_params(axis='y', labelcolor='#f39c12')
    ax2.grid(False)

    # Hợp nhất Legend cho gọn và đẹp mắt
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=11)

    plt.tight_layout()
    plt.show()
# ==============================================================================
# 8. KHỐI ĐIỀU KHIỂN CHẠY
# ==============================================================================
if __name__ == "__main__":
    dl_dothi = tao_dl_thuc_te()
    
    # Truy xuất ma trận AHP theo Bảng 4 của bạn
    A_matrix = lay_ma_tran_ahp()
    w_ahp_kich_ban, lambda_max, CI, cr = tinh_trong_so_ahp_chuan(A_matrix)
    
    print("\n" + "="*80)
    print(" KẾT QUẢ ĐẦU RA TOÁN HỌC PHƯƠNG PHÁP PHÂN TÍCH THỨ BẬC AHP (Theo Bảng 4)")
    print("="*80)
    print(f" Trọng số f1 (Thời gian trễ)   : w1 = {w_ahp_kich_ban[0]:.4f} ({w_ahp_kich_ban[0]*100:.2f}%)")
    print(f" Trọng số f2 (Hàng chờ)        : w2 = {w_ahp_kich_ban[1]:.4f} ({w_ahp_kich_ban[1]*100:.2f}%)")
    print(f" Trọng số f3 (Số lần dừng xe)  : w3 = {w_ahp_kich_ban[2]:.4f} ({w_ahp_kich_ban[2]*100:.2f}%)")
    print(f" [KIỂM TRA CHỈ SỐ NHẤT QUÁN TOÁN HỌC MA TRẬN SAATY]")
    print(f"  - Giá trị riêng lớn nhất (λ_max)  : {lambda_max:.4f}")
    print(f"  - Tỷ số nhất quán thực nghiệm (CR): {cr:.4f} -> {'Đạt tính nhất quán (CR < 0.1)' if cr < 0.1 else 'Mâu thuẫn logic'}")
    print("="*80 + "\n")

    SO_LAN_CHAY = 5
    SO_QUAN_THE = 45
    SO_THE_HE = 40
    
    best_fitness_all_runs = float('inf')
    
    for i in range(SO_LAN_CHAY):
        print(f"Chạy lần {i+1}/{SO_LAN_CHAY}...", end=" ")
        ct_tot, ca_the_truoc_tu, lich_su_f, thong_ke_mo_hinh = chay_toi_uu(dl_dothi, w_ahp_kich_ban, n_qt=SO_QUAN_THE, n_the_he=SO_THE_HE)
        print(f" Fitness = {ct_tot.fit:.4f}")
        
        if ct_tot.fit < best_fitness_all_runs:
            best_fitness_all_runs = ct_tot.fit
            ct_tot_chinh_xac, ca_the_truoc_tu_chinh_xac = copy.deepcopy(ct_tot), copy.deepcopy(ca_the_truoc_tu)
            lich_su_f_chinh_xac, thong_ke_mo_hinh_chinh_xac = copy.deepcopy(lich_su_f), copy.deepcopy(thong_ke_mo_hinh)
            
    print(f"\n--> Đã trích xuất mô hình tốt nhất (Fitness = {best_fitness_all_runs:.6f})")
    ve_dt(ct_tot_chinh_xac, ca_the_truoc_tu_chinh_xac, lich_su_f_chinh_xac, thong_ke_mo_hinh_chinh_xac)