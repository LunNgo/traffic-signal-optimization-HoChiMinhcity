"""
=============================================================================
PHÂN TÍCH ĐỘ BẤT ĐỊNH (UNCERTAINTY ANALYSIS) – MẠNG LƯỚI 26 NÚT
Phương pháp: Mô phỏng Monte Carlo (N=50 lần, sigma=5%)
Kế thừa: Bộ nghiệm tối ưu (C, g, o) từ NSGA-II + Local Search (AHP)
Hàm mục tiêu: f1 (Độ trễ – xe.h), f2 (Hàng chờ – xe), f3 (Số lần dừng – lượt/h)
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 1 – BIẾN TOÀN CỤC & THAM SỐ HỆ THỐNG
# ─────────────────────────────────────────────────────────────────────────────
df_nodes    = pd.DataFrame()
DISTANCES   = []
L_MAX_LIST  = []
BASELINE_G1 = []
BASELINE_G2 = []
BASELINE_OFF= []

C_MIN, C_MAX = 85,  95
G_MIN, G_MAX = 15,  80
OFF_MIN, OFF_MAX = 0, 149

# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 2 – NẠP DỮ LIỆU TỪ EXCEL
# ─────────────────────────────────────────────────────────────────────────────
def load_data_from_excel(path_nut='Du_lieu_nut_mang_luoi.xlsx',
                         path_kc ='Khoang_Cach_mang_luoi.xlsx'):
    global df_nodes, DISTANCES, L_MAX_LIST
    global BASELINE_G1, BASELINE_G2, BASELINE_OFF
    global C_MIN, C_MAX

    print("[+] Đang nạp dữ liệu từ file Excel...")

    # --- Nút mạng lưới ---
    try:
        df_nut = pd.read_excel(path_nut, header=3)
        if not any('q_total' in str(c) for c in df_nut.columns):
            df_nut = pd.read_excel(path_nut)
    except Exception:
        df_nut = pd.read_excel(path_nut)

    col_q   = next(c for c in df_nut.columns if 'q_total' in str(c) or c == 'q')
    col_s   = next(c for c in df_nut.columns if str(c).startswith('S'))
    col_name= next(c for c in df_nut.columns if 'Tên' in str(c) or 'name' in str(c))
    col_L   = next(c for c in df_nut.columns if 'L/' in str(c) or c == 'L')
    col_v   = next(c for c in df_nut.columns if 'v' in str(c) and 'km' in str(c).lower())
    col_qb  = next(c for c in df_nut.columns if 'qb' in str(c))
    col_g1  = next(c for c in df_nut.columns if 'Pha1' in str(c) or 'g₁' in str(c) or 'g1' in str(c).lower())
    col_g2  = next(c for c in df_nut.columns if 'Pha2' in str(c) or 'g₂' in str(c) or 'g2' in str(c).lower())
    col_off = next(c for c in df_nut.columns if 'Offset' in str(c) or 'o_bl' in str(c))

    n = len(df_nut)
    df_nodes = pd.DataFrame({
        'k'   : list(range(1, n+1)),
        'name': df_nut[col_name].astype(str).tolist(),
        'q'   : df_nut[col_q].fillna(1500).astype(float).tolist(),
        'S'   : df_nut[col_s].fillna(3600).astype(float).tolist(),
        'L'   : df_nut[col_L].fillna(3).astype(float).tolist(),
        'v'   : df_nut[col_v].fillna(15.0).astype(float).tolist(),
        'qb'  : df_nut[col_qb].fillna(15).astype(float).tolist(),
    })

    BASELINE_G1  = df_nut[col_g1].fillna(50).astype(int).tolist()
    BASELINE_G2  = df_nut[col_g2].fillna(34).astype(int).tolist()
    BASELINE_OFF = df_nut[col_off].fillna(0).astype(int).tolist()

    # --- Khoảng cách ---
    df_lk     = pd.read_excel(path_kc)
    col_tu    = next(c for c in df_lk.columns if 'Tu' in str(c))
    col_kc_col= next(c for c in df_lk.columns if 'Khoang' in str(c))
    DISTANCES  = [0.0]*n
    for _, row in df_lk.iterrows():
        tu = int(row[col_tu])
        if tu-1 < n:
            DISTANCES[tu-1] = float(row[col_kc_col])

    # --- Giới hạn hàng chờ L_max ---
    L_XE, N_LAN, ALPHA = 6.0, 3, 0.85
    L_MAX_LIST.clear()
    for d in DISTANCES:
        if d > 0:
            L_MAX_LIST.append(round((d/L_XE)*N_LAN*ALPHA))
        else:
            L_MAX_LIST.append(9999)

    max_L   = max(float(l) for l in df_nodes['L'])
    C_MIN   = max(max(60, int(2*G_MIN + 2*max_L)), 85)
    C_MAX   = max(C_MIN+5, 95)

    print(f"[+] Load thành công {n} nút. C ∈ [{C_MIN}s – {C_MAX}s].")


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 3 – CÔNG THỨC TOÁN HỌC (GIỮ NGUYÊN TỪ MÔ HÌNH GỐC)
# ─────────────────────────────────────────────────────────────────────────────
def calc_uniform_delay(q, S, g, C):
    if g <= 0 or S <= 0 or C <= 0: return 9999.0
    lam = g/C
    x   = (q*C)/(S*g)
    num = C*((1-lam)**2)
    den = 2*(1-min(1.0, x)*lam)
    return max(0.0, num/den)

def calc_incremental_delay(q, S, g, C, T_period=1):
    if g <= 0 or S <= 0: return 0.0
    lam  = g/C
    x    = (q*C)/(S*g)
    c_cap= S*lam
    term = (8*0.5*1.0*x)/(c_cap*T_period+1e-6)
    d2   = 900*T_period*((x-1)+np.sqrt(max(0.0,(x-1)**2+term)))
    return max(0.0, d2)

def calc_residual_delay(q, qb, t_accum=30, T_period=1.0):
    if q <= 0: return 0.0
    return (3600.0*qb*t_accum)/(q*T_period+1e-6)

def calc_lq1_uniform(q, S, g, C):
    if C <= 0 or g <= 0: return 0.0
    lam = g/C
    x   = min(0.95, (q*C)/(S*g))
    r   = max(0.0, C-g)
    num = (q/3600.0)*(r**2)
    den = 2*C*(1-x*lam+1e-6)
    return max(0.0, num/den)

def calc_lq2_random(q, S, g, C, T_period=1):
    if g <= 0 or S <= 0: return 0.0
    lam  = g/C
    x    = (q*C)/(S*g)
    if x <= 0.5: return 0.0
    c_cap= S*lam
    term = ((x-1)**2)+(8*0.5*1.0*x)/(c_cap*T_period+1e-6)
    lq2  = 0.25*c_cap*T_period*((x-1)+np.sqrt(max(0.0, term)))
    return max(0.0, lq2)

def calc_gamma_wave(offset_k, t_travel, C, beta=0.3):
    diff = abs(offset_k-t_travel) % C
    if diff > C/2: diff = C-diff
    return max(0.05, 1.0-beta*np.exp(-diff/(C/4+1e-6)))

def calc_node_stops(q, S, g, C, gamma):
    if C <= 0 or q <= 0: return 0.0
    x       = (q*C)/(S*g+1e-6)
    term_red= (C-g)/C
    term_sat= 1.0/(1.0-min(0.95, x)+1e-6)
    return q*term_red*term_sat*gamma


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 4 – HÀM TÍNH FITNESS (evaluate_individual) – GIỮ NGUYÊN 100%
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_individual(individual, q_arr=None, qb_arr=None):
    """
    Hàm tính toán giá trị thực tế vật lý (không chứa hàm phạt)
    Dùng riêng cho việc vẽ biểu đồ Boxplot và thống kê Monte Carlo.
    """
    n  = len(df_nodes)
    G1 = individual[:n]
    OFF= individual[n:2*n]
    c  = individual[-1]

    f1_real = f2_real = f3_real = 0.0

    for i in range(n):
        q  = float(q_arr[i])  if q_arr  is not None else float(df_nodes['q'][i])
        qb = float(qb_arr[i]) if qb_arr is not None else float(df_nodes['qb'][i])
        S, L, v = float(df_nodes['S'][i]), float(df_nodes['L'][i]), float(df_nodes['v'][i])

        g1    = G1[i]
        off_k = OFF[i]
        g2    = max(G_MIN, c - g1 - 2 * int(L))
        q1, q2 = q * 0.6, q * 0.4
        S1, S2 = S * 0.6, S * 0.4

        # 1. Tính độ trễ thực tế (f1)
        d1 = calc_uniform_delay(q1, S1, g1, c) + calc_incremental_delay(q1, S1, g1, c) + calc_residual_delay(q1, qb * 0.6)
        d2 = calc_uniform_delay(q2, S2, g2, c) + calc_incremental_delay(q2, S2, g2, c) + calc_residual_delay(q2, qb * 0.4)
        f1_real += (d1 * q1 + d2 * q2) / 3600.0

        # 2. Tính chiều dài hàng chờ thực tế (f2)
        lq1 = calc_lq1_uniform(q1, S1, g1, c) + calc_lq2_random(q1, S1, g1, c)
        lq2 = calc_lq1_uniform(q2, S2, g2, c) + calc_lq2_random(q2, S2, g2, c)
        f2_real += (lq1 + lq2)

        # 3. Tính số lần dừng xe thực tế (f3)
        t_travel = DISTANCES[i] / (v * 1000 / 3600 + 1e-6)
        gamma    = calc_gamma_wave(off_k, t_travel, c)
        f3_real += calc_node_stops(q1, S1, g1, c, gamma) + calc_node_stops(q2, S2, g2, c, gamma)

    return f1_real, f2_real, f3_real
# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 5 – BỘ NGHIỆM TỐI ƯU TỐT NHẤT (C, G, O) TỪ NSGA-II + LOCAL SEARCH
#           → Được xác định bằng cách chạy mô hình gốc một lần;
#             ở đây nhúng trực tiếp vào script để tái sử dụng độc lập.
# ─────────────────────────────────────────────────────────────────────────────
def repair_individual(ind):
    n   = len(df_nodes)
    ind = list(ind)
    c   = int(np.clip(ind[-1], C_MIN, C_MAX))
    ind[-1] = c
    for i in range(n):
        L  = int(df_nodes['L'][i])
        g1 = int(np.clip(ind[i], G_MIN, G_MAX))
        if c-g1-2*L < G_MIN:
            g1 = int(np.clip(c-G_MIN-2*L, G_MIN, G_MAX))
        ind[i]   = g1
        ind[n+i] = int(ind[n+i] % c) if c > 0 else 0
    return ind


def get_best_solution():
    """
    Trả về bộ nghiệm tối ưu (individual) được tìm từ mô hình NSGA-II.
    
    GIÁ TRỊ NÀY LẤY TRỰC TIẾP từ kết quả chạy mô hình gốc
    (Tối_ưu_trên_mạng_lưới.py) với tham số:
        pop_size=120, n_gen=80, seed=42, local_search n_iter=60
    Chu kỳ chung tối ưu C* = 90s
    """
    # G1 tối ưu cho 26 nút (từ Local Search AHP-weighted)
    G1_opt = [
        50, 49, 51, 47, 52, 50, 56, 51, 54, 49,
        47, 52, 60, 60, 51, 49, 56, 52, 52, 55,
        58, 51, 55, 47, 47, 50
    ]
    # Offset tối ưu
    OFF_opt = [
        0, 25, 37, 50, 63, 75, 85,  8, 18, 60,
       29, 52, 61,  0, 16, 30, 44, 56, 69, 82,
        1, 10, 43, 70,  8, 43
    ]
    C_opt = 90   # Chu kỳ chung tối ưu (s)

    individual = repair_individual(G1_opt + OFF_opt + [C_opt])
    return individual


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 6 – MÔ PHỎNG MONTE CARLO  (N=50, sigma=5%)
# ─────────────────────────────────────────────────────────────────────────────
def run_monte_carlo(best_individual, N=50, sigma=0.05, seed=0):
    """
    Sinh N bộ dữ liệu nhiễu → đánh giá fitness mỗi lần → lưu kết quả.
    Vectorization: sinh toàn bộ noise matrix (N × n_nodes) một lần duy nhất.
    """
    np.random.seed(seed)
    n         = len(df_nodes)
    q_base    = np.array(df_nodes['q'].tolist(),  dtype=float)
    qb_base   = np.array(df_nodes['qb'].tolist(), dtype=float)

    # ── Sinh noise matrix (N×n) – vectorized, không dùng vòng lặp lồng ──
    noise_q  = np.random.normal(0, sigma, size=(N, n))   # shape (50, 26)
    noise_qb = np.random.normal(0, sigma, size=(N, n))   # shape (50, 26)

    q_noisy_mat  = q_base  * (1 + noise_q)               # broadcast (50, 26)
    qb_noisy_mat = qb_base * (1 + noise_qb)              # broadcast (50, 26)

    # Đảm bảo không âm
    q_noisy_mat  = np.clip(q_noisy_mat,  1.0, None)
    qb_noisy_mat = np.clip(qb_noisy_mat, 0.1, None)

    f1_results, f2_results, f3_results = [], [], []

    print(f"\n{'='*65}")
    print(f"  MÔ PHỎNG MONTE CARLO – N={N} lần, sigma={sigma*100:.0f}% nhiễu")
    print(f"  Chu kỳ tối ưu C* = {best_individual[-1]}s")
    print(f"{'='*65}")

    for run in range(N):
        q_i  = q_noisy_mat[run]    # vector 26 phần tử
        qb_i = qb_noisy_mat[run]

        f1, f2, f3 = evaluate_individual(best_individual, q_arr=q_i, qb_arr=qb_i)
        f1_results.append(f1)
        f2_results.append(f2)
        f3_results.append(f3)

        if (run+1) % 10 == 0 or run == 0:
            print(f"  Run {run+1:3d}/{N}  | f1={f1:8.2f} xe.h  "
                  f"| f2={f2:7.2f} xe  | f3={f3:8.0f} lượt/h")

    return (np.array(f1_results),
            np.array(f2_results),
            np.array(f3_results))


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 7 – IN KẾT QUẢ THỐNG KÊ RA TERMINAL
# ─────────────────────────────────────────────────────────────────────────────
def print_statistics(f1_results, f2_results, f3_results):
    print(f"\n{'='*65}")
    print("  KẾT QUẢ THỐNG KÊ MONTE CARLO (50 lần lặp, nhiễu 5%)")
    print(f"{'='*65}")

    stats = [
        ("f1 – Tổng độ trễ    (xe.h)",   f1_results, "xe.h"),
        ("f2 – Tổng hàng chờ  (xe)",      f2_results, "xe"),
        ("f3 – Tổng lần dừng  (lượt/h)",  f3_results, "lượt/h"),
    ]
    for name, arr, unit in stats:
        mu  = np.mean(arr)
        std = np.std(arr, ddof=1)
        mn  = np.min(arr)
        mx  = np.max(arr)
        cv  = std/mu*100 if mu != 0 else 0
        print(f"  {name}: {mu:.2f} ± {std:.2f} {unit}")
        print(f"       [min={mn:.2f}, max={mx:.2f}, CV={cv:.1f}%]")
    print(f"{'='*65}")


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 8 – VẼ BIỂU ĐỒ BOXPLOT CHUẨN BÀI BÁO KHOA HỌC
# ─────────────────────────────────────────────────────────────────────────────
def plot_uncertainty_boxplots(f1_results, f2_results, f3_results,
                               outfile='outputs/Uncertainty_26nodes.png'):
    # ── Thiết lập style bài báo khoa học ──
    sns.set_theme(style='whitegrid', font_scale=1.1)
    plt.rcParams.update({
        'font.family'      : 'DejaVu Sans',
        'axes.unicode_minus': False,
        'axes.spines.top'  : False,
        'axes.spines.right': False,
    })

    # Màu pastel thanh lịch
    PALETTE = ['#4C78B5', '#74B87C', '#E07B54']   # blue, green, orange
    MEDIAN_COLOR = '#2C2C2C'

    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    fig.suptitle(
        'Phân tích độ bất định hệ thống đèn giao thông mạng lưới 26 nút\n'
        'Mô phỏng Monte Carlo (N = 50, nhiễu chuẩn σ = 5%)',
        fontsize=13, fontweight='bold', y=1.02
    )

    datasets = [
        (f1_results, 'f₁ – Tổng độ trễ',      'Tổng độ trễ (xe·h)',       PALETTE[0]),
        (f2_results, 'f₂ – Tổng hàng chờ',     'Chiều dài hàng chờ (xe)',  PALETTE[1]),
        (f3_results, 'f₃ – Tổng số lần dừng',  'Số lần dừng xe (lượt/h)', PALETTE[2]),
    ]

    box_props = dict(
        boxprops    = dict(linewidth=1.5),
        whiskerprops= dict(linewidth=1.3, linestyle='--'),
        capprops    = dict(linewidth=1.5),
        medianprops = dict(linewidth=2.2, color=MEDIAN_COLOR),
        flierprops  = dict(marker='o', markersize=5, linestyle='none',
                           markerfacecolor='#D62728', markeredgewidth=0.8,
                           alpha=0.7),
    )

    for ax, (data, title, ylabel, color) in zip(axes, datasets):
        # Boxplot chính
        bp = ax.boxplot(data, patch_artist=True,
                        widths=0.45, **box_props)
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.72)

        # Jitter strip (phân bố từng điểm)
        x_jitter = np.random.normal(1, 0.055, size=len(data))
        ax.scatter(x_jitter, data, alpha=0.35, s=18,
                   color=color, zorder=3, edgecolors='white', linewidth=0.4)

        # Thống kê nhanh
        mu  = np.mean(data)
        std = np.std(data, ddof=1)
        ax.axhline(mu, color=MEDIAN_COLOR, linestyle=':', linewidth=1.4,
                   label=f'Mean = {mu:.2f}')

        # Nhãn thống kê trong plot
        y_range = np.max(data)-np.min(data)
        ax.text(1.46, np.percentile(data,75),
                f'Q3={np.percentile(data,75):.1f}',
                va='center', ha='left', fontsize=8.5, color='#444')
        ax.text(1.46, np.percentile(data,25),
                f'Q1={np.percentile(data,25):.1f}',
                va='center', ha='left', fontsize=8.5, color='#444')
        ax.text(1.46, mu,
                f'μ={mu:.1f}\nσ={std:.1f}',
                va='center', ha='left', fontsize=8.5,
                color=MEDIAN_COLOR, fontweight='bold')

        # Định dạng axes
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel(ylabel, fontsize=10.5)
        ax.set_xticks([])
        ax.yaxis.grid(True, linestyle='--', linewidth=0.7,
                      color='#BBBBBB', alpha=0.9)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', labelsize=9.5)

        # Legend nhỏ
        mean_patch = mpatches.Patch(facecolor=color, alpha=0.72,
                                    edgecolor='black', linewidth=0.8,
                                    label=f'N=50  μ±σ: {mu:.1f}±{std:.1f}')
        ax.legend(handles=[mean_patch], loc='upper left',
                  fontsize=8, framealpha=0.85)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(outfile, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n[✓] Đã lưu biểu đồ: {outfile}")


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 9 – MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  PHÂN TÍCH ĐỘ BẤT ĐỊNH – MẠNG LƯỚI 26 NÚT")
    print("  Monte Carlo N=50 | Nhiễu σ=5% | Nghiệm tối ưu NSGA-II")
    print("=" * 65)

    # 1. Nạp dữ liệu
    load_data_from_excel(
        path_nut='Du_lieu_nut_mang_luoi.xlsx',
        path_kc ='Khoang_Cach_mang_luoi.xlsx'
    )

    # 2. Lấy bộ nghiệm tối ưu
    best_individual = get_best_solution()
    print(f"\n[+] Nghiệm tối ưu: C* = {best_individual[-1]}s | 26 nút")

    # 3. Tính fitness gốc (không nhiễu) để so sánh
    f1_det, f2_det, f3_det = evaluate_individual(best_individual)
    print(f"\n[+] Kết quả tất định (không nhiễu):")
    print(f"    f1 = {f1_det:.2f} xe.h  |  f2 = {f2_det:.2f} xe  |  f3 = {f3_det:.0f} lượt/h")

    # 4. Chạy Monte Carlo
    f1_results, f2_results, f3_results = run_monte_carlo(
        best_individual, N=50, sigma=0.05, seed=42
    )

    # 5. In thống kê terminal
    print_statistics(f1_results, f2_results, f3_results)

    # In thêm dạng yêu cầu: "Tên: Trung bình ± Độ lệch chuẩn"
    print("\n  ── Tóm tắt (định dạng báo cáo) ──")
    print(f"  Tổng độ trễ    : {np.mean(f1_results):.2f} ± {np.std(f1_results, ddof=1):.2f} xe.h")
    print(f"  Hàng chờ       : {np.mean(f2_results):.2f} ± {np.std(f2_results, ddof=1):.2f} xe")
    print(f"  Số lần dừng xe : {np.mean(f3_results):.0f} ± {np.std(f3_results, ddof=1):.0f} lượt/h")

    # 6. Vẽ biểu đồ
    print("\n[+] Đang vẽ biểu đồ Boxplot...")
    plot_uncertainty_boxplots(f1_results, f2_results, f3_results)

    print("\n[✓] Hoàn tất phân tích độ bất định.")
    return f1_results, f2_results, f3_results


if __name__ == '__main__':
    f1_r, f2_r, f3_r = main()
