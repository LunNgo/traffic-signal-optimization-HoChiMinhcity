import pandas as pd
import numpy as np

# 1. Tên file dữ liệu cơ sở
file_goc = 'Data_co_so.xlsx'

try:
    df_base = pd.read_excel(file_goc, skiprows=3)
    print(f"✓ Đã đọc thành công dữ liệu cơ sở gồm {len(df_base)} nút giao.")
except Exception as e:
    print("Lỗi đọc file:", e)
    exit()

# 2. Làm sạch tên cột
df_base.columns = [str(col).replace('\n', ' ').strip() for col in df_base.columns]

# Tìm cột q_total và qb
col_q = [col for col in df_base.columns if 'q_total' in col][0]
col_qb = [col for col in df_base.columns if 'qb' in col][0]

# Hàm tạo kịch bản tự động
def tao_kich_ban(df, ty_le, ten_file, ten_kich_ban):
    df_moi = df.copy()
    
    # Nếu tỷ lệ khác 0 thì mới tính toán lại
    if ty_le != 0:
        # Thay đổi lưu lượng q
        df_moi[col_q] = df_moi[col_q] * (1 + ty_le)
        
        # Thay đổi qb (hàng chờ ban đầu)
        if ty_le > 0:
            df_moi[col_qb] = np.ceil(df_moi[col_qb] * (1 + ty_le)).astype(int)
        else:
            df_moi[col_qb] = np.floor(df_moi[col_qb] * (1 + ty_le)).astype(int)
            
        # Đảm bảo hàng chờ không bị âm
        df_moi[col_qb] = df_moi[col_qb].apply(lambda x: max(0, x))
        
    # Xuất file
    df_moi.to_excel(ten_file, index=False)
    print(f"  -> Đã tạo {ten_kich_ban}: {ten_file} (Tỷ lệ thay đổi: {ty_le*100:+.0f}%)")

print("\nĐANG TẠO CÁC KỊCH BẢN MÔ PHỎNG...")

# Các kịch bản giảm lưu lượng (Đã bỏ -20%)
tao_kich_ban(df_base, -0.30, 'Data_Giam_30.xlsx', 'Giờ thấp điểm (Trưa)')
tao_kich_ban(df_base, -0.10, 'Data_Giam_10.xlsx', 'Giờ chuyển tiếp (Cận cao điểm)')

# TẠO FILE CAO ĐIỂM TỪ DATA CƠ SỞ (Giữ nguyên 100% dữ liệu gốc)
tao_kich_ban(df_base, 0.0, 'Data_Cao_Diem.xlsx', 'Giờ CAO ĐIỂM (Base)')

# Kịch bản tăng lưu lượng
tao_kich_ban(df_base, 0.10, 'Data_Tang_10.xlsx', 'Giờ kẹt xe (Siêu cao điểm)')

print("\n=> Hoàn tất sinh dữ liệu!")