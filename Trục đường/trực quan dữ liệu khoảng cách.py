import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu
df = pd.read_excel('Khoang_Cach.xlsx')
df['Thoi_gian_chuyen_tl'] = df['Thoi_gian_chuyen_tl'].round(2)

# Tạo nhãn đoạn đường ngắn gọn
df['Segment'] = df['Tu_nut'].astype(str) + '→' + df['Den_nut'].astype(str)

# ===================== BIỂU ĐỒ SAU KHI SỬA =====================
fig, ax1 = plt.subplots(figsize=(15, 9))

color_bar = '#1E88E5'      # Xanh da trời cho cột khoảng cách
color_line = '#66BB6A'     # XANH LÁ NHẠT cho đường thời gian

# Vẽ cột khoảng cách
bars = ax1.bar(df['Segment'], df['Khoang_cach_D'],
               color=color_bar, alpha=0.85, label='Khoảng cách (m)',
               edgecolor='white', linewidth=0.8, width=0.65)

ax1.set_xlabel('Đoạn đường giữa các nút', fontsize=12, fontweight='bold')
ax1.set_ylabel('Khoảng cách (mét)', fontsize=12, fontweight='bold', color=color_bar)
ax1.tick_params(axis='y', labelcolor=color_bar)

# Vẽ đường thời gian (màu xanh lá nhạt)
ax2 = ax1.twinx()
line = ax2.plot(df['Segment'], df['Thoi_gian_chuyen_tl'],
                color=color_line, marker='o', linewidth=4, markersize=10,
                markerfacecolor='white', markeredgecolor=color_line,
                markeredgewidth=2.5, label='Thời gian chuyển lý tưởng (giây)')

ax2.set_ylabel('Thời gian chuyển (giây)', fontsize=12, fontweight='bold', color=color_line)
ax2.tick_params(axis='y', labelcolor=color_line)

# Tiêu đề
plt.title('KHOẢNG CÁCH VÀ THỜI GIAN CHUYỂN LÝ TƯỞNG GIỮA CÁC NÚT\n'
          '(Mạng lưới 10 nút - Trục đường)',
          fontsize=15, fontweight='bold', pad=25)

# Giá trị trên cột khoảng cách
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 12,
             f'{int(height)}', ha='center', va='bottom',
             fontsize=10.5, fontweight='bold', color='#0D47A1')

# Giá trị thời gian
for i, txt in enumerate(df['Thoi_gian_chuyen_tl']):
    ax2.annotate(f'{txt:.2f}',
                 (i, txt),
                 textcoords="offset points",
                 xytext=(0, 15),
                 ha='center',
                 fontsize=10.5,
                 fontweight='bold',
                 color='#2E7D32')   # Xanh lá đậm hơn cho chữ dễ đọc

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc='upper left', fontsize=11, frameon=True)

plt.xticks(rotation=45, ha='right', fontsize=11)
plt.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('outputs/Khoang_Cach_Combined_XanhLaNhat.png', dpi=250, bbox_inches='tight')
plt.show()

print("✅ Đã thay đổi đường thành màu xanh lá nhạt thành công!")