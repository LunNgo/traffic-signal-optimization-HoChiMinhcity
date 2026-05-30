# Mô Hình Tối Ưu Hóa Lập Lịch Vận Hành Đèn Giao Thông

## Giới thiệu dự án
Dự án xây dựng một mô hình toán học nhằm tối ưu hóa chu kỳ đèn giao thông, với mục tiêu cốt lõi là giảm thiểu tình trạng kẹt xe và tối ưu hóa thời gian lưu thông trung bình tại các ngã tư, đặc biệt tập trung vào các "điểm nóng" tại TP.HCM. 

Mô hình áp dụng kỹ thuật tối ưu hóa đa mục tiêu để cân bằng giữa ba chỉ số:
*   Tổng thời gian chậm trễ ($f_1$).
*   Tổng chiều dài hàng chờ trung bình ($f_2$).
*   Tổng số lần dừng xe ($f_3$) nhằm tạo "làn sóng xanh".

## Công nghệ và Thuật toán
*   **Ngôn ngữ & Môi trường:** Python trên VSCode.
*   **Thư viện xử lý & Trực quan hóa:** NumPy, Pandas, Matplotlib, Openpyxl.
*   **Hệ thống Mô phỏng:** SUMO (Simulation of Urban Mobility) để đánh giá và đối chiếu trực quan hiệu năng vi mô.
*   **Thuật toán cốt lõi:** Giải thuật tiến hóa lai kết hợp giữa **NSGA-II** và tìm kiếm cục bộ (**Local Search**).
*   **Phương pháp ra quyết định:** Phân tích thứ bậc **AHP** (Analytic Hierarchy Process) và lựa chọn nghiệm bằng **Weighted Sum** có chuẩn hóa tuyến tính Min-Max.

## Phạm vi dữ liệu & Thử nghiệm
Mô hình đã được triển khai và kiểm chứng trên hai hệ thống mạng lưới giao thông:
1.  **Bài toán trục đường (10 nút giao):** Áp dụng trên trục Cách mạng Tháng Tám - Đinh Tiên Hoàng với các tập dữ liệu lưu lượng vào giờ cao điểm và thấp điểm.
2.  **Bài toán mạng lưới mở rộng (26 nút giao):** Bao phủ các trục đường trọng điểm như Điện Biên Phủ, Võ Thị Sáu và Lý Chính Thắng để kiểm chứng khả năng mở rộng của thuật toán.

Hệ thống tự động thiết lập thời gian đèn xanh ($g_{k,i}$), chu kỳ chung ($C$) và độ lệch pha ($o_k$) tương ứng cho từng điều kiện lưu lượng.

## 👥 Nhóm phát triển
Sinh viên thực hiện thuộc Khoa Toán - Tin học, Trường Đại học Khoa học Tự nhiên (ĐHQG-HCM):
*   **Ngô Hoàng Lực** - 24110031
*   **Huỳnh Tuấn Kiệt** - 24110032
*   **Nguyễn Nhật Tuân** - 24110069
*   **Cao Nguyễn Kỳ Duyên** - 24110079
