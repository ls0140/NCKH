#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sơ đồ Quy trình Phân tích Bài báo Khoa học ROT
Sử dụng Matplotlib để tạo sơ đồ workflow
"""

import matplotlib
matplotlib.use('Agg')  # Sử dụng backend không tương tác
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import seaborn as sns

def create_workflow_diagram():
    """Tạo sơ đồ quy trình hoàn chỉnh"""
    
    # Thiết lập figure
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis('off')

    # Bảng màu
    colors = {
        'input': '#E8F4FD',      # Xanh nhạt
        'process': '#FFF2CC',    # Vàng nhạt
        'database': '#D5E8D4',   # Xanh lá nhạt
        'output': '#F8CECC',     # Đỏ nhạt
        'ai': '#E1D5E7'          # Tím nhạt
    }

    # Hàm tạo hộp bo tròn
    def create_box(ax, x, y, width, height, text, color, fontsize=10, alpha=0.8):
        box = FancyBboxPatch((x, y), width, height,
                            boxstyle="round,pad=0.1",
                            facecolor=color, alpha=alpha, edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x + width/2, y + height/2, text, ha='center', va='center', 
                fontsize=fontsize, fontweight='bold', wrap=True)

    # Hàm tạo mũi tên
    def create_arrow(ax, start_x, start_y, end_x, end_y, color='black', style='->'):
        arrow = ConnectionPatch((start_x, start_y), (end_x, end_y), "data", "data",
                               arrowstyle=style, shrinkA=5, shrinkB=5, 
                               mutation_scale=20, fc=color, ec=color, linewidth=2)
        ax.add_patch(arrow)

    # Tiêu đề chính
    ax.text(10, 13.5, 'Hệ Thống Phân Tích Bài Báo Khoa Học ROT', 
            ha='center', va='center', fontsize=24, fontweight='bold', color='#2E86AB')

    # PHẦN ĐẦU VÀO
    ax.text(3, 12.5, 'ĐẦU VÀO', ha='center', va='center', fontsize=16, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['input'], alpha=0.9))

    # Nguồn dữ liệu
    create_box(ax, 0.5, 11, 4, 0.8, 'arXiv API\n(CS.LG, CS.AI, CS.CV)', colors['input'], 9)
    create_box(ax, 0.5, 10, 4, 0.8, 'PubMed API\n(Bài báo Y sinh)', colors['input'], 9)
    create_box(ax, 0.5, 9, 4, 0.8, 'Google Scholar\n(Dữ liệu Trích dẫn)', colors['input'], 9)

    # Dữ liệu bài báo
    create_box(ax, 0.5, 7.5, 4, 1, 'Siêu dữ liệu Bài báo\n(Tiêu đề, Tóm tắt, Tác giả,\nNăm xuất bản, DOI)', colors['input'], 9)
    create_box(ax, 0.5, 6, 4, 1, 'Thông tin Trích dẫn\n(Số trích dẫn, Lịch sử\nTrích dẫn, Hệ số Ảnh hưởng)', colors['input'], 9)

    # PHẦN XỬ LÝ
    ax.text(10, 12.5, 'XỬ LÝ', ha='center', va='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['process'], alpha=0.9))

    # Trích xuất đặc trưng
    create_box(ax, 7, 11, 6, 0.8, 'Trích xuất Đặc trưng\n(Số từ, Độ dễ đọc, Điểm thuật ngữ)', colors['process'], 9)
    create_box(ax, 7, 10, 6, 0.8, 'Phân tích Văn bản\n(TF-IDF, Cảm xúc, Từ khóa)', colors['process'], 9)

    # Phân tích AI
    create_box(ax, 7, 8.5, 6, 0.8, 'Phân tích AI\n(Tích hợp OpenAI GPT)', colors['ai'], 9)
    create_box(ax, 7, 7.5, 6, 0.8, 'Đánh giá Chất lượng\n(Điểm chất lượng bài báo)', colors['ai'], 9)

    # Tính toán ROT
    create_box(ax, 7, 6.5, 6, 0.8, 'Tính toán ROT\n(Tỷ lệ Trích dẫn = Trích dẫn/Tuổi)', colors['process'], 9)
    create_box(ax, 7, 5.5, 6, 0.8, 'Phân loại 5 Nhóm\n(Rất thấp → Rất cao ROT)', colors['process'], 9)

    # PHẦN CƠ SỞ DỮ LIỆU
    ax.text(16.5, 12.5, 'CƠ SỞ DỮ LIỆU', ha='center', va='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['database'], alpha=0.9))

    # Bảng cơ sở dữ liệu
    create_box(ax, 14.5, 11, 4, 0.8, 'Bảng Bài báo\n(Thông tin, Điểm ROT)', colors['database'], 9)
    create_box(ax, 14.5, 10, 4, 0.8, 'Bảng Tác giả\n(Thông tin Tác giả)', colors['database'], 9)
    create_box(ax, 14.5, 9, 4, 0.8, 'Bảng Đặc trưng\n(Đặc trưng trích xuất)', colors['database'], 9)
    create_box(ax, 14.5, 8, 4, 0.8, 'Lịch sử Trích dẫn\n(Dữ liệu Thời gian)', colors['database'], 9)

    # PHẦN ĐẦU RA
    ax.text(10, 4, 'ĐẦU RA', ha='center', va='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['output'], alpha=0.9))

    # Kết quả phân tích
    create_box(ax, 0.5, 2.5, 4, 1, 'Kết quả Phân tích\n(Xuất CSV, Thống kê,\nPhân bố ROT)', colors['output'], 9)
    create_box(ax, 0.5, 1, 4, 1, 'Trực quan hóa Dữ liệu\n(Biểu đồ, Đồ thị, Bản đồ nhiệt)', colors['output'], 9)

    # Ứng dụng Web
    create_box(ax, 7, 2.5, 6, 1, 'Ứng dụng Web\n(Giao diện người dùng, Tìm kiếm,\nLọc, Đề xuất)', colors['output'], 9)
    create_box(ax, 7, 1, 6, 1, 'API Endpoints\n(RESTful API cho Truy cập\nBên ngoài)', colors['output'], 9)

    # Đề xuất
    create_box(ax, 14.5, 2.5, 4, 1, 'Đề xuất Bài báo\n(Đề xuất dựa trên SVM)', colors['output'], 9)
    create_box(ax, 14.5, 1, 4, 1, 'Phân tích Xu hướng\n(Lĩnh vực Nghiên cứu Mới)', colors['output'], 9)

    # MŨI TÊN - Đầu vào đến Xử lý
    create_arrow(ax, 4.5, 11.4, 7, 11.4)
    create_arrow(ax, 4.5, 10.4, 7, 10.4)
    create_arrow(ax, 4.5, 9.4, 7, 9.4)
    create_arrow(ax, 4.5, 8, 7, 8)
    create_arrow(ax, 4.5, 6.5, 7, 6.5)

    # MŨI TÊN - Xử lý đến Cơ sở dữ liệu
    create_arrow(ax, 13, 11.4, 14.5, 11.4)
    create_arrow(ax, 13, 10.4, 14.5, 10.4)
    create_arrow(ax, 13, 9.4, 14.5, 9.4)
    create_arrow(ax, 13, 8, 14.5, 8)

    # MŨI TÊN - Cơ sở dữ liệu đến Đầu ra
    create_arrow(ax, 16.5, 11.4, 16.5, 3.5)
    create_arrow(ax, 16.5, 10.4, 16.5, 3.5)
    create_arrow(ax, 16.5, 9.4, 16.5, 3.5)
    create_arrow(ax, 16.5, 8.4, 16.5, 3.5)

    # MŨI TÊN - Xử lý đến Đầu ra
    create_arrow(ax, 10, 5.5, 10, 3.5)

    # Thêm chỉ báo giai đoạn
    ax.text(1, 13.8, 'Giai đoạn 1', ha='center', va='center', fontsize=12, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue', alpha=0.7))
    ax.text(10, 13.8, 'Giai đoạn 2', ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen', alpha=0.7))
    ax.text(16.5, 13.8, 'Giai đoạn 3', ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='lightcoral', alpha=0.7))
    ax.text(10, 0.3, 'Giai đoạn 4', ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='lightyellow', alpha=0.7))

    # Thêm chú thích
    legend_elements = [
        patches.Patch(color=colors['input'], label='Dữ liệu Đầu vào'),
        patches.Patch(color=colors['process'], label='Xử lý'),
        patches.Patch(color=colors['ai'], label='Phân tích AI'),
        patches.Patch(color=colors['database'], label='Cơ sở Dữ liệu'),
        patches.Patch(color=colors['output'], label='Đầu ra')
    ]

    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    # Thêm mô tả chi tiết
    ax.text(10, 13.2, 'Ứng dụng AI trong Phân tích và Đánh giá Chất lượng Bài viết Nghiên cứu Khoa học', 
            ha='center', va='center', fontsize=14, style='italic', color='#666666')

    # Thêm nhãn luồng dữ liệu
    ax.text(5.5, 12.2, 'Thu thập Dữ liệu', ha='center', va='center', fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    ax.text(15.5, 12.2, 'Lưu trữ Dữ liệu', ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    ax.text(10, 3.2, 'Kết quả & Giao diện', ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig

def main():
    """Hàm chính để tạo và lưu sơ đồ"""
    print("Đang tạo sơ đồ quy trình...")
    
    # Tạo sơ đồ
    fig = create_workflow_diagram()
    
    # Hiển thị sơ đồ
    plt.show()
    
    # Lưu sơ đồ
    fig.savefig('scientific_paper_rot_workflow.png', dpi=300, bbox_inches='tight')
    print("✅ Sơ đồ quy trình đã được lưu thành 'scientific_paper_rot_workflow.png'")
    
    # Lưu thêm định dạng SVG cho chất lượng cao
    fig.savefig('scientific_paper_rot_workflow.svg', bbox_inches='tight')
    print("✅ Sơ đồ quy trình đã được lưu thành 'scientific_paper_rot_workflow.svg'")

if __name__ == "__main__":
    main() 