import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt


# ========================================================
# 1. 그레이스케일 영상의 히스토그램 평활화 및 히스토그램 비교
# ========================================================
src_gray = cv2.imread('3.jpg', cv2.IMREAD_GRAYSCALE)

if src_gray is None:
    print('Image load failed!')
    sys.exit()

# 히스토그램 평활화 수행
dst_gray = cv2.equalizeHist(src_gray)

# 히스토그램 계산 (평활화 전 / 후)
hist_src_gray = cv2.calcHist([src_gray], [0], None, [256], [0, 256])
hist_dst_gray = cv2.calcHist([dst_gray], [0], None, [256], [0, 256])

# 영상 출력
cv2.imshow('src_gray', src_gray)
cv2.imshow('dst_gray (Equalized)', dst_gray)
cv2.waitKey(1)

# 히스토그램 시각화 (원본 vs 평활화 후)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.canvas.manager.set_window_title('Grayscale Histogram Equalization Comparison')

# Y축 상한선 계산
max_val_gray = max(hist_src_gray.max(), hist_dst_gray.max()) * 1.15

axes[0].plot(hist_src_gray, color='black', label='Original')
axes[0].set_title('Original Gray Histogram')
axes[0].set_xlabel('Pixel Value (0~255)')
axes[0].set_ylabel('Pixel Count')
axes[0].set_xlim([0, 256])
axes[0].set_ylim([0, max_val_gray])
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend()

axes[1].plot(hist_dst_gray, color='blue', label='Equalized')
axes[1].set_title('Equalized Gray Histogram')
axes[1].set_xlabel('Pixel Value (0~255)')
axes[1].set_ylabel('Pixel Count')
axes[1].set_xlim([0, 256])
axes[1].set_ylim([0, max_val_gray])
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].legend()

plt.tight_layout()
print("그레이스케일 히스토그램 그래프 창을 닫으면 컬러 영상 단계로 진행합니다.")
plt.show()

cv2.destroyAllWindows()


# ========================================================
# 2. 컬러 영상의 히스토그램 평활화 및 히스토그램 비교
# ========================================================
src_color = cv2.imread('3.jpg')

if src_color is None:
    print('Image load failed!')
    sys.exit()

# BGR -> YCrCb 변환 후 밝기(Y) 채널에 대해서만 히스토그램 평활화 수행
src_ycrcb = cv2.cvtColor(src_color, cv2.COLOR_BGR2YCrCb)
ycrcb_planes = list(cv2.split(src_ycrcb))

# 평활화 전 Y 채널 복사본 보관 (히스토그램 비교용)
src_y = ycrcb_planes[0].copy()

# 밝기(Y) 성분에 대해서만 히스토그램 평활화 수행
ycrcb_planes[0] = cv2.equalizeHist(ycrcb_planes[0])
dst_y = ycrcb_planes[0]

dst_ycrcb = cv2.merge(ycrcb_planes)
dst_color = cv2.cvtColor(dst_ycrcb, cv2.COLOR_YCrCb2BGR)

# 영상 출력
cv2.imshow('src_color', src_color)
cv2.imshow('dst_color (Equalized)', dst_color)
cv2.waitKey(1)

# 컬러 히스토그램 계산 (BGR 채널별 및 Y 채널)
colors = ('b', 'g', 'r')
channel_names = ('Blue', 'Green', 'Red')

src_hists = [cv2.calcHist([src_color], [i], None, [256], [0, 256]) for i in range(3)]
dst_hists = [cv2.calcHist([dst_color], [i], None, [256], [0, 256]) for i in range(3)]

# 히스토그램 시각화 (원본 컬러 vs 평활화 후 컬러)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.canvas.manager.set_window_title('Color Histogram Equalization Comparison')

color_max_val = max(max(h.max() for h in src_hists), max(h.max() for h in dst_hists)) * 1.15

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

# 평활화 후 컬러 히스토그램
for i, col in enumerate(colors):
    axes[1].plot(dst_hists[i], color=col, label=channel_names[i])
axes[1].set_title('Equalized Color Histogram')
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
