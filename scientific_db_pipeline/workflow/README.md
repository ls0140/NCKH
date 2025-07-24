# 📊 Workflow Visualization Package

## Mô tả
Thư mục này chứa các mã nguồn để tạo sơ đồ quy trình (workflow diagram) cho dự án **Phân tích Bài báo Khoa học ROT**.

## Cấu trúc Thư mục
```
workflow/
├── README.md                    # Hướng dẫn này
├── matplotlib_workflow.py       # Sơ đồ sử dụng Matplotlib
├── plotly_workflow.py          # Sơ đồ sử dụng Plotly (thay thế)
├── colab_notebook.ipynb        # Notebook Google Colab
└── requirements.txt            # Thư viện cần thiết
```

## Cách sử dụng

### 1. Chạy trên Google Colab (Khuyến nghị)
1. Mở [Google Colab](https://colab.research.google.com)
2. Tạo notebook mới
3. Copy code từ `colab_notebook.ipynb`
4. Chạy từng ô theo thứ tự

### 2. Chạy trên máy local
```bash
# Cài đặt thư viện
pip install -r requirements.txt

# Chạy sơ đồ Matplotlib
python matplotlib_workflow.py

# Chạy sơ đồ Plotly
python plotly_workflow.py
```

## Sơ đồ Quy trình

### Giai đoạn 1: Thu thập Dữ liệu
- **ArXiv API**: Thu thập bài báo AI/ML
- **PubMed API**: Thu thập bài báo Y sinh
- **Google Scholar**: Dữ liệu trích dẫn

### Giai đoạn 2: Xử lý và Phân tích
- **Trích xuất đặc trưng**: Số từ, độ dễ đọc, điểm thuật ngữ
- **Phân tích văn bản**: TF-IDF, cảm xúc, từ khóa
- **AI phân tích**: Tích hợp OpenAI GPT
- **Tính toán ROT**: Tỷ lệ trích dẫn = Trích dẫn/Tuổi
- **Phân loại 5 nhóm**: Rất thấp → Rất cao ROT

### Giai đoạn 3: Cơ sở Dữ liệu
- **Bảng Papers**: Thông tin bài báo, điểm ROT
- **Bảng Authors**: Thông tin tác giả
- **Bảng Features**: Đặc trưng trích xuất
- **Bảng Citation History**: Dữ liệu thời gian

### Giai đoạn 4: Đầu ra và Ứng dụng
- **Kết quả phân tích**: CSV, thống kê, phân bố ROT
- **Trực quan hóa**: Biểu đồ, đồ thị, bản đồ nhiệt
- **Ứng dụng Web**: Giao diện người dùng, tìm kiếm, lọc
- **API Endpoints**: RESTful API cho truy cập bên ngoài
- **Đề xuất**: Dựa trên SVM, phân tích xu hướng

## Xử lý Sự cố

### Hình ảnh trắng trong Colab
1. **Giải pháp 1**: Sử dụng backend 'Agg' trong Matplotlib
2. **Giải pháp 2**: Chuyển sang Plotly (khuyến nghị)
3. **Giải pháp 3**: Restart runtime Colab

### Lỗi thư viện
```bash
pip install --upgrade matplotlib plotly seaborn
```

## Tùy chỉnh

### Thay đổi màu sắc
Chỉnh sửa dictionary `colors` trong file Python:
```python
colors = {
    'input': '#E8F4FD',      # Xanh nhạt
    'process': '#FFF2CC',    # Vàng nhạt
    'database': '#D5E8D4',   # Xanh lá nhạt
    'output': '#F8CECC',     # Đỏ nhạt
    'ai': '#E1D5E7'          # Tím nhạt
}
```

### Thay đổi kích thước
Chỉnh sửa `figsize` trong `plt.subplots()`:
```python
fig, ax = plt.subplots(1, 1, figsize=(20, 14))  # (rộng, cao)
```

## Tác giả
Dự án Phân tích Bài báo Khoa học ROT - Ứng dụng AI trong đánh giá chất lượng nghiên cứu

## Phiên bản
v1.0 - Tháng 7/2025 