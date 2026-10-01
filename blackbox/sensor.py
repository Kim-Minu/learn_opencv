import cv2
import numpy as np
import time
import math
from collections import deque

class AccelerometerSensor:
    """
    차량용 3축(X, Y, Z) 가속도 센서(G-Sensor) 물리 모델링 시뮬레이터.
    - X축: 좌/우 횡가속도 (코너링)
    - Y축: 전/후 종가속도 (가속/감속)
    - Z축: 수직 가속도 (기본 중력 1.0G + 노면 요철)
    - Total G: sqrt(X^2 + Y^2 + Z^2)
    """
    def __init__(self, threshold_g=2.5, history_len=120):
        self.threshold_g = threshold_g
        self.history_len = history_len
        
        # 기본 가속도 (Z축은 지구 중력가속도 1.0G)
        self.ax = 0.0
        self.ay = 0.0
        self.az = 1.0
        self.total_g = 1.0
        
        # 충격 물리 모델링 상태 변수
        self.impact_active = False
        self.impact_start_time = 0.0
        self.impact_peak_g = 0.0
        self.impact_duration = 0.8  # 충격 지속 및 감쇠 시간 (초)
        self.impact_freq = 15.0     # 충격 진동 주파수 (Hz)
        self.impact_damping = 4.5   # 감쇠 계수
        self.impact_dir = (0.0, 1.0, 0.5) # 충격 방향 벡터
        
        # 이력 데이터 (그래프 시각화용)
        self.history_total = deque([1.0] * history_len, maxlen=history_len)
        self.history_x = deque([0.0] * history_len, maxlen=history_len)
        self.history_y = deque([0.0] * history_len, maxlen=history_len)
        self.history_z = deque([1.0] * history_len, maxlen=history_len)
        
        # 마지막 충격 이벤트 발생 시각
        self.last_event_time = 0.0

    def trigger_impact(self, peak_g=4.2, direction="front"):
        """
        사용자 버튼 클릭 시 충격 모델링 함수 발동
        - peak_g: 최대 가속도 (G)
        - direction: 'front'(정면), 'rear'(후방), 'side'(측면)
        """
        self.impact_active = True
        self.impact_start_time = time.time()
        self.impact_peak_g = peak_g
        self.last_event_time = time.time()
        
        if direction == "front":
            self.impact_dir = (0.2, -1.0, 0.4)
        elif direction == "rear":
            self.impact_dir = (0.1, 1.0, 0.4)
        elif direction == "side":
            self.impact_dir = (1.0, 0.2, 0.3)
        else:
            self.impact_dir = (0.3, -0.8, 0.5)
            
        print(f"\n[G-Sensor] 🚨 충격 이벤트 발생! (Peak: {peak_g:.2f}G, Direction: {direction})")

    def update(self):
        """
        매 프레임 호출되어 물리 모델에 따라 가속도 값을 계산하고 갱신합니다.
        """
        now = time.time()
        
        # 1. 평상시 노면 주행 진동 노이즈 모델링 (미세 가우시안 노이즈)
        noise_x = np.random.normal(0, 0.03)
        noise_y = np.random.normal(0, 0.04)
        noise_z = np.random.normal(0, 0.05)
        
        base_x = noise_x
        base_y = noise_y
        base_z = 1.0 + noise_z
        
        # 2. 충격 발생 시 물리적 감쇠 진동 (Damped Harmonic Oscillation) 모델링
        # Formula: a(t) = Peak * e^(-damping * t) * cos(omega * t)
        if self.impact_active:
            t = now - self.impact_start_time
            if t <= self.impact_duration:
                # 감쇠 진동 함수
                decay = math.exp(-self.impact_damping * t)
                osc = math.cos(2 * math.pi * self.impact_freq * t)
                wave = decay * osc
                
                # 방향 벡터 반영
                norm = math.sqrt(sum(d**2 for d in self.impact_dir)) or 1.0
                dx = self.impact_dir[0] / norm
                dy = self.impact_dir[1] / norm
                dz = self.impact_dir[2] / norm
                
                imp_acc = self.impact_peak_g * wave
                base_x += imp_acc * dx
                base_y += imp_acc * dy
                base_z += imp_acc * dz
            else:
                self.impact_active = False
                
        self.ax = base_x
        self.ay = base_y
        self.az = base_z
        self.total_g = math.sqrt(self.ax**2 + self.ay**2 + self.az**2)
        
        # 이력 갱신
        self.history_total.append(self.total_g)
        self.history_x.append(self.ax)
        self.history_y.append(self.ay)
        self.history_z.append(self.az)
        
        # 충격 감지 여부
        is_shock = (self.total_g >= self.threshold_g)
        return self.total_g, is_shock

class SensorControlPanel:
    """
    OpenCV 전용 가속도 센서 컨트롤 패널 및 버튼 UI 창.
    - 사용자가 마우스로 클릭할 수 있는 충격 발생 버튼
    - 실시간 G-Force 오실로스코프 그래프 및 수치 게이지 표시
    """
    def __init__(self, sensor: AccelerometerSensor, window_name="Blackbox Sensor Controller"):
        self.sensor = sensor
        self.window_name = window_name
        self.width = 620
        self.height = 360
        
        # 버튼 영역 정의 [x, y, w, h, label, bg_color, hover_color, action_type]
        self.buttons = [
            {
                "id": "btn_impact_heavy",
                "rect": (25, 40, 260, 48),
                "label": "🚨 충격 발생 (Heavy Crash: 4.5G)",
                "color": (40, 40, 200),
                "hover_color": (60, 60, 240),
                "action": lambda: self.sensor.trigger_impact(peak_g=4.5, direction="front")
            },
            {
                "id": "btn_impact_minor",
                "rect": (305, 40, 285, 48),
                "label": "⚠️ 둔턱/경미 충격 (Bump: 2.8G)",
                "color": (20, 120, 200),
                "hover_color": (40, 150, 240),
                "action": lambda: self.sensor.trigger_impact(peak_g=2.8, direction="front")
            },
            {
                "id": "btn_impact_side",
                "rect": (25, 98, 260, 42),
                "label": "⚡ 측면 충돌 (Side Impact: 3.8G)",
                "color": (140, 60, 180),
                "hover_color": (170, 80, 210),
                "action": lambda: self.sensor.trigger_impact(peak_g=3.8, direction="side")
            },
            {
                "id": "btn_reset",
                "rect": (305, 98, 285, 42),
                "label": "🔄 센서 초기화 (Reset)",
                "color": (60, 60, 60),
                "hover_color": (90, 90, 90),
                "action": self.reset_sensor
            }
        ]
        
        self.mouse_x = -1
        self.mouse_y = -1
        self.clicked_button_id = None
        self.click_effect_timer = 0
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

    def reset_sensor(self):
        self.sensor.impact_active = False
        print("[G-Sensor] 🔄 가속도 센서 안정화 초기화 완료")

    def _on_mouse(self, event, x, y, flags, param):
        self.mouse_x = x
        self.mouse_y = y
        
        if event == cv2.EVENT_LBUTTONDOWN:
            for btn in self.buttons:
                bx, by, bw, bh = btn["rect"]
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self.clicked_button_id = btn["id"]
                    self.click_effect_timer = time.time()
                    btn["action"]()
                    break

    def render(self):
        """컨트롤 패널 UI 렌더링"""
        panel = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        panel[:] = (28, 28, 30) # 모던 다크 테마 배경
        
        # 1. 상단 타이틀 바
        cv2.putText(panel, "G-SENSOR CONTROL PANEL & MONITOR", (25, 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (220, 220, 220), 1)
        
        # 충격 감지 상태 인디케이터
        now = time.time()
        is_recent_impact = (now - self.sensor.last_event_time) < 1.5
        status_text = "🚨 IMPACT EVENT DETECTED!" if is_recent_impact else "● NORMAL DRIVING"
        status_color = (60, 60, 255) if is_recent_impact else (80, 220, 100)
        cv2.putText(panel, status_text, (350, 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, status_color, 1 if not is_recent_impact else 2)
        
        # 2. 버튼 렌더링
        for btn in self.buttons:
            bx, by, bw, bh = btn["rect"]
            is_hover = (bx <= self.mouse_x <= bx + bw and by <= self.mouse_y <= by + bh)
            is_active_click = (self.clicked_button_id == btn["id"] and (now - self.click_effect_timer) < 0.15)
            
            # 버튼 배경색 결정
            if is_active_click:
                color = (255, 255, 255)
                text_color = (0, 0, 0)
            elif is_hover:
                color = btn["hover_color"]
                text_color = (255, 255, 255)
            else:
                color = btn["color"]
                text_color = (240, 240, 240)
                
            # 버튼 둥근 사각형 효과
            cv2.rectangle(panel, (bx, by), (bx + bw, by + bh), color, -1)
            cv2.rectangle(panel, (bx, by), (bx + bw, by + bh), (180, 180, 180), 1)
            
            # 텍스트 가운데 정렬
            text_size = cv2.getTextSize(btn["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0]
            tx = bx + (bw - text_size[0]) // 2
            ty = by + (bh + text_size[1]) // 2
            cv2.putText(panel, btn["label"], (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, text_color, 1)

        # 3. 실시간 가속도 수치 표시
        val_y = 175
        cv2.putText(panel, f"Total G: {self.sensor.total_g:5.2f} G  (Threshold: {self.sensor.threshold_g:.1f} G)",
                    (25, val_y), cv2.FONT_HERSHEY_DUPLEX, 0.6,
                    (60, 60, 255) if self.sensor.total_g >= self.sensor.threshold_g else (100, 255, 100), 1)
        
        axis_info = f"Ax: {self.sensor.ax:+5.2f}G | Ay: {self.sensor.ay:+5.2f}G | Az: {self.sensor.az:+5.2f}G"
        cv2.putText(panel, axis_info, (25, val_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
                    
        # 4. 실시간 G-Force 오실로스코프 그래프 영역
        gx, gy, gw, gh = 25, 220, 565, 110
        cv2.rectangle(panel, (gx, gy), (gx + gw, gy + gh), (15, 15, 18), -1)
        cv2.rectangle(panel, (gx, gy), (gx + gw, gy + gh), (70, 70, 75), 1)
        
        # 기준선 (1.0G = 평상시 중력)
        # Y 스케일: 0.0G ~ 5.0G
        max_g = 5.0
        y_1g = int(gy + gh - (1.0 / max_g) * gh)
        cv2.line(panel, (gx, y_1g), (gx + gw, y_1g), (60, 60, 60), 1)
        cv2.putText(panel, "1.0G (Normal)", (gx + 5, y_1g - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
        
        # 임계값 기준선 (2.5G)
        y_thresh = int(gy + gh - (self.sensor.threshold_g / max_g) * gh)
        cv2.line(panel, (gx, y_thresh), (gx + gw, y_thresh), (0, 0, 180), 1)
        cv2.putText(panel, f"Event Threshold ({self.sensor.threshold_g:.1f}G)",
                    (gx + 5, y_thresh - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 220), 1)
        
        # 실시간 G-force 파형 그리기
        history = list(self.sensor.history_total)
        num_pts = len(history)
        if num_pts > 1:
            pts = []
            for i, val in enumerate(history):
                px = int(gx + (i / (self.sensor.history_len - 1)) * gw)
                clipped_val = min(max(val, 0.0), max_g)
                py = int(gy + gh - (clipped_val / max_g) * gh)
                pts.append((px, py))
                
            for i in range(len(pts) - 1):
                pt1 = pts[i]
                pt2 = pts[i + 1]
                val = history[i]
                line_color = (0, 0, 255) if val >= self.sensor.threshold_g else (50, 220, 120)
                cv2.line(panel, pt1, pt2, line_color, 2)
                
        # 하단 조작 가이드
        cv2.putText(panel, "Click buttons with mouse OR press 'I' on preview to trigger shock",
                    (25, self.height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)

        cv2.imshow(self.window_name, panel)
        return panel
