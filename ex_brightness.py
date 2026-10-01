import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt


# ========================================================
# 1. 그레이스케일 영상 불러오기 및 히스토그램 비교
# ========================================================
src_gray = cv2.imread('3.jpg', cv2.IMREAD_GRAYSCALE)

if src_gray is None:
    print('Image load failed!')
    sys.exit()

# 밝기 50 증가
dst_gray = cv2.add(src_gray, 50)
# dst_gray = np.clip(src_gray + 50., 0, 255).astype(np.uint8)

# 히스토그램 계산
hist_src_gray = cv2.calcHist([src_gray], [0], None, [256], [0, 256])
hist_dst_gray = cv2.calcHist([dst_gray], [0], None, [256], [0, 256])

# 영상 출력
cv2.imshow('src_gray', src_gray)
cv2.imshow('dst_gray (+50)', dst_gray)
cv2.waitKey(1)

# 그레이스케일 히스토그램 시각화 (원본 vs 수정 후)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.canvas.manager.set_window_title('Grayscale Histogram Comparison')

# 255 포화(saturation)로 인해 Y축이 왜곡되는 것을 방지하기 위해 Y축 상한선 통일
max_val = max(hist_src_gray.max(), hist_dst_gray[:255].max()) * 1.15

axes[0].plot(hist_src_gray, color='black', label='Original')
axes[0].set_title('Original Gray Histogram')
axes[0].set_xlabel('Pixel Value (0~255)')
axes[0].set_ylabel('Pixel Count')
axes[0].set_xlim([0, 256])
axes[0].set_ylim([0, max_val])
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend()

axes[1].plot(hist_dst_gray, color='red', label='Brightness +50')
axes[1].set_title('Modified Gray Histogram (+50) [Y-scale synchronized]')
axes[1].set_xlabel('Pixel Value (0~255)')
axes[1].set_ylabel('Pixel Count')
axes[1].set_xlim([0, 256])
axes[1].set_ylim([0, max_val])
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].legend()

# 255 포화 지점에 설명 텍스트 표시
sat_count = int(hist_dst_gray.ravel()[255])
axes[1].annotate(f'255 Saturation\n({sat_count:,} px)',
                 xy=(255, max_val * 0.95), xytext=(170, max_val * 0.8),
                 arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6),
                 fontsize=9, color='darkred', fontweight='bold')

plt.tight_layout()
print("그레이스케일 히스토그램 그래프 창을 닫으면 컬러 영상 단계로 진행합니다.")
plt.show()

cv2.destroyAllWindows()


# ========================================================
# 2. 컬러 영상 불러오기 및 히스토그램 비교
# ========================================================
src_color = cv2.imread('3.jpg')

if src_color is None:
    print('Image load failed!')
    sys.exit()

# 밝기 50 증가 (B, G, R 각각 50씩 증가)
dst_color = cv2.add(src_color, (50, 50, 50, 0))
# dst_color = np.clip(src_color + 50., 0, 255).astype(np.uint8)

# 영상 출력
cv2.imshow('src_color', src_color)
cv2.imshow('dst_color (+50)', dst_color)
cv2.waitKey(1)

# 컬러 히스토그램 계산
colors = ('b', 'g', 'r')
channel_names = ('Blue', 'Green', 'Red')

src_hists = [cv2.calcHist([src_color], [i], None, [256], [0, 256]) for i in range(3)]
dst_hists = [cv2.calcHist([dst_color], [i], None, [256], [0, 256]) for i in range(3)]

# 컬러 히스토그램 시각화 (B, G, R 채널별)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.canvas.manager.set_window_title('Color Histogram Comparison')

# 255 제외 최대값을 기준으로 Y축 상한선 통일
color_max_val = max(max(h.max() for h in src_hists), max(h[:255].max() for h in dst_hists)) * 1.15

# 원본 컬러 히스토그램
for i, col in enumerate(colors):
    axes[0].plot(src_hists[i], color=col, label=channel_names[i])
axes[0].set_title('Original Color Histogram')
axes[0].set_xlabel('Pixel Value (0~255)')
axes[0].set_ylabel('Pixel Count')
axes[0].set_xlim([0, 256])
axes[0].set_ylim([0, color_max_val])
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend()

# 수정 후 컬러 히스토그램
for i, col in enumerate(colors):
    axes[1].plot(dst_hists[i], color=col, label=channel_names[i])
axes[1].set_title('Modified Color Histogram (+50) [Y-scale synchronized]')
axes[1].set_xlabel('Pixel Value (0~255)')
axes[1].set_ylabel('Pixel Count')
axes[1].set_xlim([0, 256])
axes[1].set_ylim([0, color_max_val])
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].legend()

plt.tight_layout()
print("컬러 히스토그램 그래프 창을 닫으면 프로그램이 종료됩니다.")
plt.show()

cv2.destroyAllWindows()