"""
화면 캡처 및 좌표 정규화 모듈 (Screen Capture & Coordinate Normalization)
- mss 기반 초고속 화면 캡처
- Mac Retina / HiDPI 디스플레이 스케일 팩터 자동 감지 및 보정
- 0~1000 정규화 상대좌표 <-> 실제 화면 물리 픽셀 변환
- OpenCV 기반 시각 보조 그리드 오버레이 지원
"""

import os
import time
from typing import Tuple, Optional
import cv2
import numpy as np
import mss
import pyautogui
from PIL import Image


class ScreenCapturer:
    def __init__(self, monitor_idx: int = 1):
        """
        monitor_idx: 주 모니터 (1: 메인 디스플레이)
        """
        self.sct = mss.mss()
        self.monitor_idx = monitor_idx
        self.monitor = self.sct.monitors[self.monitor_idx]

        # 물리 화면 해상도 (mss 기준)
        self.raw_width = self.monitor["width"]
        self.raw_height = self.monitor["height"]

        # pyautogui 논리 화면 해상도
        logical_size = pyautogui.size()
        self.logical_width = logical_size.width
        self.logical_height = logical_size.height

        # Retina / HiDPI 스케일 팩터 계산 (Mac의 경우 2.0 등)
        self.scale_x = self.raw_width / self.logical_width
        self.scale_y = self.raw_height / self.logical_height

        print(f"[ScreenCapturer] Physical: {self.raw_width}x{self.raw_height}, "
              f"Logical: {self.logical_width}x{self.logical_height}, "
              f"Scale Factor: ({self.scale_x:.2f}, {self.scale_y:.2f})")

    def capture(self, resize_to: Optional[Tuple[int, int]] = (1280, 800), add_grid: bool = False) -> Tuple[np.ndarray, Image.Image]:
        """
        현재 화면을 캡처하고 OpenCV BGR 이미지 및 PIL Image로 반환.
        - resize_to: Gemini 모델에 전달할 최적 크기 (None이면 원본 크기)
        - add_grid: 100단위 눈금선 오버레이 여부 (Computer Use 인식률 향상)
        """
        # mss 캡처 (BGRA 포맷)
        sct_img = self.sct.grab(self.monitor)
        frame = np.array(sct_img)
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        if add_grid:
            bgr = self.draw_grid_overlay(bgr)

        # 모델 전달용 리사이즈
        if resize_to:
            processed = cv2.resize(bgr, resize_to, interpolation=cv2.INTER_AREA)
        else:
            processed = bgr

        # PIL 변환
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        return bgr, pil_image

    def draw_grid_overlay(self, image: np.ndarray, step: int = 100) -> np.ndarray:
        """캡처 이미지 위에 0~1000 상대좌표 기준 반투명 그리드 및 좌표 텍스트 렌더링"""
        overlay = image.copy()
        h, w = image.shape[:2]

        for norm_coord in range(step, 1000, step):
            # 수직선
            px_x = int((norm_coord / 1000.0) * w)
            cv2.line(overlay, (px_x, 0), (px_x, h), (0, 255, 255), 1)
            cv2.putText(overlay, str(norm_coord), (px_x + 3, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

            # 수평선
            px_y = int((norm_coord / 1000.0) * h)
            cv2.line(overlay, (0, px_y), (w, px_y), (0, 255, 255), 1)
            cv2.putText(overlay, str(norm_coord), (5, px_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

        # 가중 합성 (0.75 원본 + 0.25 그리드)
        return cv2.addWeighted(overlay, 0.35, image, 0.65, 0)

    def norm_to_logical(self, norm_x: float, norm_y: float) -> Tuple[int, int]:
        """
        0~1000 상대 정규화 좌표를 pyautogui에서 사용하는 논리 좌표로 변환
        """
        # 범위 클리핑
        clamped_x = max(0.0, min(1000.0, float(norm_x)))
        clamped_y = max(0.0, min(1000.0, float(norm_y)))

        logical_x = int((clamped_x / 1000.0) * self.logical_width)
        logical_y = int((clamped_y / 1000.0) * self.logical_height)

        return logical_x, logical_y

    def logical_to_norm(self, logical_x: int, logical_y: int) -> Tuple[float, float]:
        """논리 화면 좌표를 0~1000 상대 좌표로 변환"""
        norm_x = (logical_x / self.logical_width) * 1000.0
        norm_y = (logical_y / self.logical_height) * 1000.0
        return round(norm_x, 1), round(norm_y, 1)

    @staticmethod
    def compute_diff_ratio(img1_bgr: np.ndarray, img2_bgr: np.ndarray, threshold: int = 25) -> float:
        """
        두 화면 BGR 프레임 간의 차분(Frame Diff) 비율을 계산 (0.0 ~ 1.0)
        클릭/동작 후 화면이 실제로 갱신되었는지 검증할 때 사용
        """
        if img1_bgr is None or img2_bgr is None:
            return 0.0
        if img1_bgr.shape != img2_bgr.shape:
            # 해상도가 다르면 리사이즈 후 비교
            img2_bgr = cv2.resize(img2_bgr, (img1_bgr.shape[1], img1_bgr.shape[0]))

        gray1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        non_zero = np.count_nonzero(thresh)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        return float(non_zero) / float(total_pixels)


if __name__ == "__main__":
    capturer = ScreenCapturer()
    print("Testing capture...")
    raw_img, pil_img = capturer.capture(add_grid=True)
    out_path = "test_screen_grid.jpg"
    pil_img.save(out_path)
    print(f"Captured screen saved with grid: {out_path} ({pil_img.size})")

    # Center coordinate test
    lx, ly = capturer.norm_to_logical(500, 500)
    print(f"Center normalized (500, 500) -> Logical: ({lx}, {ly})")
