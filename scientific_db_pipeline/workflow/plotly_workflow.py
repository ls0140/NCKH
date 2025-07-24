#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sơ đồ Quy trình Phân tích Bài báo Khoa học ROT
Sử dụng Plotly để tạo sơ đồ workflow tương tác
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

def create_interactive_workflow():
    """Tạo sơ đồ quy trình tương tác với Plotly"""
    
    # Tạo figure
    fig = go.Figure()
    
    # Bảng màu
    colors = {
        'input': '#E8F4FD',      # Xanh nhạt
        'process': '#FFF2CC',    # Vàng nhạt
        'database': '#D5E8D4',   # Xanh lá nhạt
        'output': '#F8CECC',     # Đỏ nhạt
        'ai': '#E1D5E7'          # Tím nhạt
    }
    
    # Định nghĩa các hình dạng cho workflow
    shapes = [
        # === PHẦN ĐẦU VÀO ===
        # Nguồn dữ liệu
        dict(type="rect", x0=0, y0=10, x1=4, y1=12, fillcolor=colors['input'], 
             line=dict(color="black", width=2), name="arXiv API"),
        dict(type="rect", x0=0, y0=8, x1=4, y1=10, fillcolor=colors['input'], 
             line=dict(color="black", width=2), name="PubMed API"),
        dict(type="rect", x0=0, y0=6, x1=4, y1=8, fillcolor=colors['input'], 
             line=dict(color="black", width=2), name="Google Scholar"),
        
        # Dữ liệu bài báo
        dict(type="rect", x0=0, y0=3, x1=4, y1=6, fillcolor=colors['input'], 
             line=dict(color="black", width=2), name="Siêu dữ liệu"),
        dict(type="rect", x0=0, y0=0, x1=4, y1=3, fillcolor=colors['input'], 
             line=dict(color="black", width=2), name="Thông tin Trích dẫn"),
        
        # === PHẦN XỬ LÝ ===
        # Trích xuất đặc trưng
        dict(type="rect", x0=6, y0=10, x1=12, y1=12, fillcolor=colors['process'], 
             line=dict(color="black", width=2), name="Trích xuất Đặc trưng"),
        dict(type="rect", x0=6, y0=8, x1=12, y1=10, fillcolor=colors['process'], 
             line=dict(color="black", width=2), name="Phân tích Văn bản"),
        
        # Phân tích AI
        dict(type="rect", x0=6, y0=6, x1=12, y1=8, fillcolor=colors['ai'], 
             line=dict(color="black", width=2), name="Phân tích AI"),
        dict(type="rect", x0=6, y0=4, x1=12, y1=6, fillcolor=colors['ai'], 
             line=dict(color="black", width=2), name="Đánh giá Chất lượng"),
        
        # Tính toán ROT
        dict(type="rect", x0=6, y0=2, x1=12, y1=4, fillcolor=colors['process'], 
             line=dict(color="black", width=2), name="Tính toán ROT"),
        dict(type="rect", x0=6, y0=0, x1=12, y1=2, fillcolor=colors['process'], 
             line=dict(color="black", width=2), name="Phân loại 5 Nhóm"),
        
        # === PHẦN CƠ SỞ DỮ LIỆU ===
        dict(type="rect", x0=14, y0=10, x1=18, y1=12, fillcolor=colors['database'], 
             line=dict(color="black", width=2), name="Bảng Bài báo"),
        dict(type="rect", x0=14, y0=8, x1=18, y1=10, fillcolor=colors['database'], 
             line=dict(color="black", width=2), name="Bảng Tác giả"),
        dict(type="rect", x0=14, y0=6, x1=18, y1=8, fillcolor=colors['database'], 
             line=dict(color="black", width=2), name="Bảng Đặc trưng"),
        dict(type="rect", x0=14, y0=4, x1=18, y1=6, fillcolor=colors['database'], 
             line=dict(color="black", width=2), name="Lịch sử Trích dẫn"),
        
        # === PHẦN ĐẦU RA ===
        dict(type="rect", x0=0, y0=-6, x1=4, y1=-3, fillcolor=colors['output'], 
             line=dict(color="black", width=2), name="Kết quả Phân tích"),
        dict(type="rect", x0=0, y0=-9, x1=4, y1=-6, fillcolor=colors['output'], 
             line=dict(color="black", width=2), name="Trực quan hóa"),
        
        dict(type="rect", x0=6, y0=-6, x1=12, y1=-3, fillcolor=colors['output'], 
             line=dict(color="black", width=2), name="Ứng dụng Web"),
        dict(type="rect", x0=6, y0=-9, x1=12, y1=-6, fillcolor=colors['output'], 
             line=dict(color="black", width=2), name="API Endpoints"),
        
        dict(type="rect", x0=14, y0=-6, x1=18, y1=-3, fillcolor=colors['output'], 
             line=dict(color="black", width=2), name="Đề xuất Bài báo"),
        dict(type="rect", x0=14, y0=-9, x1=18, y1=-6, fillcolor=colors['output'], 
             line=dict(color="black", width=2), name="Phân tích Xu hướng"),
    ]
    
    # Định nghĩa các mũi tên
    arrows = [
        # Đầu vào → Xử lý
        dict(type="line", x0=4, y0=11, x1=6, y1=11, line=dict(color="black", width=3)),
        dict(type="line", x0=4, y0=9, x1=6, y1=9, line=dict(color="black", width=3)),
        dict(type="line", x0=4, y0=7, x1=6, y1=7, line=dict(color="black", width=3)),
        dict(type="line", x0=4, y0=4.5, x1=6, y1=4.5, line=dict(color="black", width=3)),
        dict(type="line", x0=4, y0=1.5, x1=6, y1=1.5, line=dict(color="black", width=3)),
        
        # Xử lý → Cơ sở dữ liệu
        dict(type="line", x0=12, y0=11, x1=14, y1=11, line=dict(color="black", width=3)),
        dict(type="line", x0=12, y0=9, x1=14, y1=9, line=dict(color="black", width=3)),
        dict(type="line", x0=12, y0=7, x1=14, y1=7, line=dict(color="black", width=3)),
        dict(type="line", x0=12, y0=5, x1=14, y1=5, line=dict(color="black", width=3)),
        
        # Cơ sở dữ liệu → Đầu ra
        dict(type="line", x0=16, y0=11, x1=16, y1=-3, line=dict(color="black", width=3)),
        dict(type="line", x0=16, y0=9, x1=16, y1=-3, line=dict(color="black", width=3)),
        dict(type="line", x0=16, y0=7, x1=16, y1=-3, line=dict(color="black", width=3)),
        dict(type="line", x0=16, y0=5, x1=16, y1=-3, line=dict(color="black", width=3)),
        
        # Xử lý → Đầu ra
        dict(type="line", x0=9, y0=0, x1=9, y1=-3, line=dict(color="black", width=3)),
    ]
    
    # Cập nhật layout
    fig.update_layout(
        title={
            'text': 'Hệ Thống Phân Tích Bài Báo Khoa Học ROT<br><sub>Ứng dụng AI trong Phân tích và Đánh giá Chất lượng Bài viết Nghiên cứu Khoa học</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#2E86AB'}
        },
        shapes=shapes + arrows,
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            range=[-1, 19]
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            range=[-10, 13]
        ),
        width=1200,
        height=800,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Thêm chú thích văn bản
    annotations = [
        # Tiêu đề các phần
        dict(x=2, y=12.5, text="ĐẦU VÀO", showarrow=False, font=dict(size=16, color="black", family="Arial Black")),
        dict(x=9, y=12.5, text="XỬ LÝ", showarrow=False, font=dict(size=16, color="black", family="Arial Black")),
        dict(x=16, y=12.5, text="CƠ SỞ DỮ LIỆU", showarrow=False, font=dict(size=16, color="black", family="Arial Black")),
        dict(x=9, y=-1.5, text="ĐẦU RA", showarrow=False, font=dict(size=16, color="black", family="Arial Black")),
        
        # Chú thích chi tiết
        dict(x=2, y=11, text="arXiv API<br>(CS.LG, CS.AI, CS.CV)", showarrow=False, font=dict(size=10, color="black")),
        dict(x=2, y=9, text="PubMed API<br>(Bài báo Y sinh)", showarrow=False, font=dict(size=10, color="black")),
        dict(x=2, y=7, text="Google Scholar<br>(Dữ liệu Trích dẫn)", showarrow=False, font=dict(size=10, color="black")),
        dict(x=2, y=4.5, text="Siêu dữ liệu Bài báo<br>(Tiêu đề, Tóm tắt, Tác giả)", showarrow=False, font=dict(size=10, color="black")),
        dict(x=2, y=1.5, text="Thông tin Trích dẫn<br>(Số trích dẫn, Lịch sử)", showarrow=False, font=dict(size=10, color="black")),
        
        dict(x=9, y=11, text="Trích xuất Đặc trưng<br>(Số từ, Độ dễ đọc)", showarrow=False, font=dict(size=10, color="black")),
        dict(x=9, y=9, text="Phân tích Văn bản<br>(TF-IDF, Cảm xúc)", showarrow=False, font=dict(size=10, color="black")),
        dict(x=9, y=7, text="Phân tích AI<br>(OpenAI GPT)", showarrow=False, font=dict(size=10, color="black")),
        dict(x=9, y=5, text="Đánh giá Chất lượng<br>(Điểm chất lượng)", showarrow=False, font=dict(size=10, color="black")),
        dict(x=9, y=3, text="Tính toán ROT<br>(Trích dẫn/Tuổi)", showarrow=False, font=dict(size=10, color="black")),
        dict(x=9, y=1, text="Phân loại 5 Nhóm<br>(Rất thấp → Rất cao)", showarrow=False, font=dict(size=10, color="black")),
        
        dict(x=16, y=11, text="Bảng Bài báo<br>(Thông tin, Điểm ROT)", showarrow=False, font=dict(size=10, color="black")),
        dict(x=16, y=9, text="Bảng Tác giả<br>(Thông tin Tác giả)", showarrow=False, font=dict(size=10, color="black")),
        dict(x=16, y=7, text="Bảng Đặc trưng<br>(Đặc trưng trích xuất)", showarrow=False, font=dict(size=10, color="black")),
        dict(x=16, y=5, text="Lịch sử Trích dẫn<br>(Dữ liệu Thời gian)", showarrow=False, font=dict(size=10, color="black")),
        
        dict(x=2, y=-4.5, text="Kết quả Phân tích<br>(CSV, Thống kê)", showarrow=False, font=dict(size=10, color="black")),
        dict(x=2, y=-7.5, text="Trực quan hóa<br>(Biểu đồ, Đồ thị)", showarrow=False, font=dict(size=10, color="black")),
        dict(x=9, y=-4.5, text="Ứng dụng Web<br>(Giao diện, Tìm kiếm)", showarrow=False, font=dict(size=10, color="black")),
        dict(x=9, y=-7.5, text="API Endpoints<br>(RESTful API)", showarrow=False, font=dict(size=10, color="black")),
        dict(x=16, y=-4.5, text="Đề xuất Bài báo<br>(Dựa trên SVM)", showarrow=False, font=dict(size=10, color="black")),
        dict(x=16, y=-7.5, text="Phân tích Xu hướng<br>(Lĩnh vực Mới)", showarrow=False, font=dict(size=10, color="black")),
        
        # Chỉ báo giai đoạn
        dict(x=2, y=13, text="Giai đoạn 1", showarrow=False, font=dict(size=12, color="white"), 
             bgcolor="lightblue", bordercolor="black", borderwidth=1),
        dict(x=9, y=13, text="Giai đoạn 2", showarrow=False, font=dict(size=12, color="white"), 
             bgcolor="lightgreen", bordercolor="black", borderwidth=1),
        dict(x=16, y=13, text="Giai đoạn 3", showarrow=False, font=dict(size=12, color="white"), 
             bgcolor="lightcoral", bordercolor="black", borderwidth=1),
        dict(x=9, y=-10, text="Giai đoạn 4", showarrow=False, font=dict(size=12, color="white"), 
             bgcolor="lightyellow", bordercolor="black", borderwidth=1),
    ]
    
    fig.update_layout(annotations=annotations)
    
    return fig

def create_legend():
    """Tạo chú thích cho sơ đồ"""
    colors = {
        'input': '#E8F4FD',
        'process': '#FFF2CC', 
        'database': '#D5E8D4',
        'output': '#F8CECC',
        'ai': '#E1D5E7'
    }
    
    legend_fig = go.Figure()
    
    # Thêm các mục chú thích
    legend_fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(size=20, color=colors['input']),
        name='Dữ liệu Đầu vào',
        showlegend=True
    ))
    
    legend_fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers', 
        marker=dict(size=20, color=colors['process']),
        name='Xử lý',
        showlegend=True
    ))
    
    legend_fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(size=20, color=colors['ai']),
        name='Phân tích AI',
        showlegend=True
    ))
    
    legend_fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(size=20, color=colors['database']),
        name='Cơ sở Dữ liệu',
        showlegend=True
    ))
    
    legend_fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(size=20, color=colors['output']),
        name='Đầu ra',
        showlegend=True
    ))
    
    legend_fig.update_layout(
        title="Chú thích",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=300,
        height=200,
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return legend_fig

def main():
    """Hàm chính để tạo và lưu sơ đồ"""
    print("Đang tạo sơ đồ quy trình tương tác...")
    
    # Tạo sơ đồ chính
    fig = create_interactive_workflow()
    
    # Hiển thị sơ đồ
    fig.show()
    
    # Lưu sơ đồ
    fig.write_html("scientific_paper_rot_workflow_interactive.html")
    print("✅ Sơ đồ tương tác đã được lưu thành 'scientific_paper_rot_workflow_interactive.html'")
    
    # Tạo và lưu chú thích
    legend_fig = create_legend()
    legend_fig.write_html("workflow_legend.html")
    print("✅ Chú thích đã được lưu thành 'workflow_legend.html'")
    
    # Lưu dưới dạng ảnh tĩnh
    fig.write_image("scientific_paper_rot_workflow_plotly.png", width=1200, height=800)
    print("✅ Sơ đồ đã được lưu thành 'scientific_paper_rot_workflow_plotly.png'")

if __name__ == "__main__":
    main() 