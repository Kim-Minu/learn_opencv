import cv2
import os
import sys
import time
import argparse
from datetime import datetime
from sensor import AccelerometerSensor, SensorControlPanel

# 지원 코덱 및 확장자 매핑
CODEC_CONFIG = {
    "DIVX": {"fourcc": "DIVX", "ext": "avi", "desc": "DivX 코덱 (AVI)"},
    "XVID": {"fourcc": "XVID", "ext": "avi", "desc": "Xvid 코덱 (AVI)"},
    "MP4":  {"fourcc": "mp4v", "ext": "mp4", "desc": "MPEG-4 코덱 (MP4)"},
    "MP4V": {"fourcc": "mp4v", "ext": "mp4", "desc": "MPEG-4 코덱 (MP4)"},
    "AVC1": {"fourcc": "avc1", "ext": "mp4", "desc": "H.264/AVC 코덱 (MP4)"},
    "H264": {"fourcc": "avc1", "ext": "mp4", "desc": "H.264 코덱 (MP4)"}
}

class BlackboxRecorder:
    def __init__(self, camera_id=0, width=1920, height=1080, fps=30.0,
                 codec="MP4", split_seconds=60, save_dir="recordings",
                 test_mode=False, threshold_g=2.5):
        self.camera_id = camera_id
        self.req_width = width
        self.req_height = height
        self.req_fps = fps
        self.split_seconds = split_seconds
        self.save_dir = save_dir
        self.test_mode = test_mode
        
        # 저장 디렉토리 분리 (일반 상시 녹화 / 충격 이벤트 녹화)
        self.normal_dir = os.path.join(self.save_dir, "normal")
        self.event_dir = os.path.join(self.save_dir, "event")
        os.makedirs(self.normal_dir, exist_ok=True)
        os.makedirs(self.event_dir, exist_ok=True)
        
        # 가속도 센서 모델링 인스턴스 초기화
        self.sensor = AccelerometerSensor(threshold_g=threshold_g)
        self.control_panel = None
        self.is_event_segment = False  # 현재 녹화 중인 세그먼트에 충격 이벤트가 포함되었는지 여부
        
        # 코덱 설정 확인
        codec_key = codec.upper()
        if codec_key not in CODEC_CONFIG:
            print(f"[경고] 알 수 없는 코덱 '{codec}'. 기본값 'MP4' (mp4v)로 대체합니다.")
            codec_key = "MP4"
            
        self.codec_info = CODEC_CONFIG[codec_key]
        self.fourcc_str = self.codec_info["fourcc"]
        self.ext = self.codec_info["ext"]
        self.fourcc = cv2.VideoWriter_fourcc(*self.fourcc_str)
        
        # 상태 변수
        self.cap = None
        self.out = None
        self.actual_width = width
        self.actual_height = height
        self.actual_fps = fps
        self.current_filename = None
        self.segment_start_time = None
        self.recorded_files = []
        self.is_running = False

    def initialize_camera(self):
        if self.test_mode:
            print("[모드] 🧪 테스트 시뮬레이션 모드 활성화 (카메라 없이 가상 주행 영상 생성)")
            self.actual_width = self.req_width
            self.actual_height = self.req_height
            self.actual_fps = self.req_fps
            return True
            
        print(f"[카메라] 카메라 ID {self.camera_id} 연결 시도 중...")
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"[에러] 카메라 {self.camera_id}를 열 수 없습니다.")
            return False
            
        # 해상도 및 FPS 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.req_fps)
        
        # 실제 카메라 세팅값 확인
        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cam_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.actual_fps = cam_fps if (cam_fps and cam_fps > 0) else self.req_fps
        
        print(f"[카메라] 설정 완료:")
        print(f"  - 요청 사양 : {self.req_width}x{self.req_height} @ {self.req_fps:.1f} FPS")
        print(f"  - 실제 하드웨어: {self.actual_width}x{self.actual_height} @ {self.actual_fps:.1f} FPS")
        print(f"  - 지정 코덱 : {self.fourcc_str} ({self.codec_info['desc']})")
        print(f"  - 파일 분할 : {self.split_seconds}초 (1분 단위 상시 녹화)")
        print(f"  - 상시 저장소: {os.path.abspath(self.normal_dir)}")
        print(f"  - 이벤트 저장소: {os.path.abspath(self.event_dir)}")
        return True

    def start_new_segment(self):
        # 기존 파일 정상 닫기 및 이벤트 발생 시 이벤트 폴더로 이동/보존
        if self.out is not None:
            self.out.release()
            self.out = None
            if self.current_filename and os.path.exists(self.current_filename):
                final_path = self.current_filename
                
                # 이벤트가 발생했던 세그먼트라면 파일명 변경 및 이벤트 폴더로 이동
                if self.is_event_segment:
                    base_name = os.path.basename(self.current_filename)
                    event_filename = base_name.replace("_NORMAL_", "_EVENT_")
                    final_path = os.path.join(self.event_dir, event_filename)
                    try:
                        os.rename(self.current_filename, final_path)
                    except Exception as e:
                        print(f"[이동 오류] {e}")
                        final_path = self.current_filename
                        
                size_mb = os.path.getsize(final_path) / (1024 * 1024)
                tag = "🚨 [이벤트 녹화 저장]" if self.is_event_segment else "💾 [상시 녹화 저장]"
                print(f"{tag} {os.path.basename(final_path)} ({size_mb:.2f} MB)")
                self.recorded_files.append(final_path)
        
        # 새 세그먼트 상태 초기화
        self.is_event_segment = False
        
        # 새 파일명 생성 (예: recordings/normal/20261001_093000_NORMAL_1920x1080_mp4v.mp4)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_NORMAL_{self.actual_width}x{self.actual_height}_{self.fourcc_str}.{self.ext}"
        filepath = os.path.join(self.normal_dir, filename)
        
        self.out = cv2.VideoWriter(
            filepath,
            self.fourcc,
            self.actual_fps,
            (self.actual_width, self.actual_height)
        )
        
        if not self.out.isOpened():
            print(f"[에러] 비디오 파일 생성 실패: {filepath}")
            print(f"      FourCC '{self.fourcc_str}' 코덱 초기화에 실패했습니다.")
            return False
            
        self.current_filename = filepath
        self.segment_start_time = time.time()
        print(f"\n[녹화 시작] 🔴 새 세그먼트: {filename}")
        return True

    def draw_osd(self, frame, elapsed_seg):
        """블랙박스 OSD(On-Screen Display) 정보 오버레이"""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        h, w = frame.shape[:2]
        now = time.time()
        
        # 최근 충격 이벤트 발생 여부 (1.5초간 화면에 시각 경고 유지)
        is_shock_warning = (now - self.sensor.last_event_time) < 1.5
        
        # 반투명 상단/하단 바 (가독성 향상)
        overlay = frame.copy()
        top_bar_color = (10, 10, 180) if is_shock_warning else (20, 20, 20)
        cv2.rectangle(overlay, (0, 0), (w, 55), top_bar_color, -1)
        cv2.rectangle(overlay, (0, h - 45), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        
        # 1. 상단 정보: REC 인디케이터 + 현재 시각
        # 깜빡이는 REC 점 (1초 주기)
        if int(time.time() * 2) % 2 == 0:
            rec_color = (0, 0, 255) if not is_shock_warning else (0, 255, 255)
            cv2.circle(frame, (30, 28), 9, rec_color, -1)
            
        cv2.putText(frame, "REC", (48, 35), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(frame, now_str, (125, 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
        
        # 충격 감지 시 OSD 중앙 팝업 경고 배너
        if is_shock_warning:
            cv2.rectangle(frame, (w // 2 - 320, 8), (w // 2 + 320, 50), (0, 0, 220), -1)
            cv2.putText(frame, "🚨 IMPACT EVENT DETECTED! (EVENT RECORDING)",
                        (w // 2 - 300, 37), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
        else:
            if self.is_event_segment:
                cv2.putText(frame, "[EVENT TAGGED]", (w // 2 - 80, 35),
                            cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 165, 255), 2)
        
        # 코덱 및 해상도 표시
        codec_text = f"CAM: FRONT | {w}x{h} @ {self.actual_fps:.0f}FPS | CODEC: {self.fourcc_str}"
        cv2.putText(frame, codec_text, (w - 560, 35), cv2.FONT_HERSHEY_DUPLEX, 0.58, (200, 255, 200), 1)
        
        # 2. 하단 정보: 현재 파일 녹화 시간 / G-센서 실시간 가속도 정보
        seg_type = "EVENT" if self.is_event_segment else "NORMAL"
        seg_info = f"[{seg_type}] Elapsed: {elapsed_seg:04.1f}s / {self.split_seconds}s | Seg: {os.path.basename(self.current_filename or '')}"
        cv2.putText(frame, seg_info, (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
        
        # 우측 하단: G-Sensor 수치 실시간 표시
        g_color = (0, 0, 255) if self.sensor.total_g >= self.sensor.threshold_g else (120, 255, 120)
        sensor_text = f"G-Sensor: {self.sensor.total_g:4.2f}G (X:{self.sensor.ax:+.1f} Y:{self.sensor.ay:+.1f} Z:{self.sensor.az:+.1f})"
        cv2.putText(frame, sensor_text, (w - 490, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.52, g_color, 1)

    def generate_dummy_frame(self, frame_idx):
        """테스트 시뮬레이션용 가상 주행 영상 생성"""
        import numpy as np
        frame = np.zeros((self.actual_height, self.actual_width, 3), dtype=np.uint8)
        
        # 도로/하늘 배경 시뮬레이션
        horizon = self.actual_height // 2
        frame[:horizon, :] = [60, 40, 20]     # 하늘
        frame[horizon:, :] = [40, 40, 40]     # 아스팔트 도로
        
        # 도로 차선 애니메이션
        offset = (frame_idx * 15) % 80
        center_x = self.actual_width // 2
        for y in range(horizon + offset, self.actual_height, 80):
            cv2.line(frame, (center_x - 10, y), (center_x + 10, y + 40), (0, 255, 255), 5)
            
        cv2.putText(frame, "[VIRTUAL BLACKBOX CAMERA SIMULATION]", 
                    (self.actual_width // 2 - 350, horizon - 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
        return frame

    def run(self, show_preview=True, max_duration=None):
        if not self.initialize_camera():
            return False
            
        if not self.start_new_segment():
            return False
            
        self.is_running = True
        total_start_time = time.time()
        frame_idx = 0
        
        # 가속도 센서 컨트롤 패널 (버튼 UI 창) 초기화
        if show_preview:
            self.control_panel = SensorControlPanel(self.sensor)
            
        print("\n" + "=" * 65)
        print("  [블랙박스 상시 녹화 및 가속도 센서 감지 작동 중]")
        print("  - 조작: 센서 제어창의 [충격 발생] 버튼 클릭 또는 키보드 [I] 누름")
        print("  - 종료: 프리뷰 창에서 'q' 또는 터미널 Ctrl+C")
        print("=" * 65)
        
        try:
            while self.is_running:
                loop_start = time.time()
                
                # 1. 가속도 센서 물리 모델 업데이트
                total_g, is_shock = self.sensor.update()
                
                # 충격 발생 시 이벤트 녹화 태그 활성화
                if is_shock:
                    if not self.is_event_segment:
                        self.is_event_segment = True
                        print(f"\n[EVENT] 🚨 충격 감지! 현재 녹화 세그먼트를 '이벤트 녹화'로 전환합니다. ({total_g:.2f}G)")
                
                # 2. 카메라 또는 시뮬레이션 프레임 획득
                if self.test_mode:
                    frame = self.generate_dummy_frame(frame_idx)
                    ret = True
                else:
                    ret, frame = self.cap.read()
                    if not ret:
                        time.sleep(0.05)
                        continue
                        
                frame_idx += 1
                now = time.time()
                elapsed_seg = now - self.segment_start_time
                
                # 3. 1분(설정 시간) 경과 시 파일 롤링(새 파일로 분할)
                if elapsed_seg >= self.split_seconds:
                    if not self.start_new_segment():
                        break
                    elapsed_seg = 0.0
                    
                # 4. OSD 오버레이 그리기
                self.draw_osd(frame, elapsed_seg)
                
                # 5. 비디오 파일에 프레임 쓰기
                self.out.write(frame)
                
                # 6. 최대 지속 시간 검사 (테스트 자동화용)
                if max_duration and (now - total_start_time) >= max_duration:
                    print(f"\n[알림] 최대 테스트 실행 시간({max_duration}초)에 도달하여 자동 종료합니다.")
                    break
                
                # 7. 화면 프리뷰 및 센서 컨트롤 패널 표시
                if show_preview:
                    # 메인 카메라 프리뷰
                    preview_frame = cv2.resize(frame, (960, 540))
                    cv2.imshow("Car Blackbox (Front FHD 1080p)", preview_frame)
                    
                    # 가속도 센서 버튼 UI 창 렌더링
                    if self.control_panel is not None:
                        self.control_panel.render()
                        
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        print("\n[사용자 종료] 사용자가 녹화를 종료했습니다.")
                        break
                    elif key in [ord('i'), ord('I'), ord(' ')]:  # 단축키로 충격 발생
                        self.sensor.trigger_impact(peak_g=4.2, direction="front")
                else:
                    frame_delay = 1.0 / self.actual_fps
                    process_time = time.time() - loop_start
                    if process_time < frame_delay:
                        time.sleep(frame_delay - process_time)
                        
        except KeyboardInterrupt:
            print("\n[인터럽트] 녹화를 중단합니다...")
        finally:
            self.cleanup()
            
        return True

    def cleanup(self):
        print("\n[정리 중] 비디오 저장 및 리소스 해제 중...")
        if self.out is not None:
            self.out.release()
            self.out = None
            if self.current_filename and os.path.exists(self.current_filename):
                final_path = self.current_filename
                if self.is_event_segment:
                    base_name = os.path.basename(self.current_filename)
                    event_filename = base_name.replace("_NORMAL_", "_EVENT_")
                    final_path = os.path.join(self.event_dir, event_filename)
                    try:
                        os.rename(self.current_filename, final_path)
                    except Exception:
                        final_path = self.current_filename
                        
                size_mb = os.path.getsize(final_path) / (1024 * 1024)
                tag = "🚨 [이벤트 녹화 저장]" if self.is_event_segment else "💾 [상시 녹화 저장]"
                print(f"{tag} {os.path.basename(final_path)} ({size_mb:.2f} MB)")
                self.recorded_files.append(final_path)
                
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        cv2.destroyAllWindows()
        print("=" * 65)
        print("  [녹화 완료된 파일 목록]")
        for f in self.recorded_files:
            if os.path.exists(f):
                tag = "🚨 [EVENT]" if "_EVENT_" in f else "📁 [NORMAL]"
                print(f"  {tag} {f} ({os.path.getsize(f) / (1024*1024):.2f} MB)")
        print("=" * 65)

def main():
    parser = argparse.ArgumentParser(description="차량용 블랙박스 상시/충격 이벤트 녹화 프로그램 (OpenCV)")
    parser.add_argument("--camera", type=int, default=0, help="카메라 장치 번호 (기본: 0)")
    parser.add_argument("--width", type=int, default=1920, help="전방 카메라 가로 해상도 (기본: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="전방 카메라 세로 해상도 (기본: 1080)")
    parser.add_argument("--fps", type=float, default=30.0, help="녹화 프레임레이트 (기본: 30.0)")
    parser.add_argument("--codec", type=str, default="MP4", choices=["DIVX", "XVID", "MP4", "MP4V", "AVC1", "H264"],
                        help="동영상 코덱 선택 (기본: MP4 [DIVX, XVID, MP4/MP4V, AVC1])")
    parser.add_argument("--split-sec", type=int, default=60, help="파일 분할 단위 초 (기본: 60초 = 1분)")
    parser.add_argument("--save-dir", type=str, default="recordings", help="녹화 파일 저장 경로 (기본: recordings)")
    parser.add_argument("--threshold", type=float, default=2.5, help="충격 감지 G-Force 임계값 (기본: 2.5G)")
    parser.add_argument("--no-preview", action="store_true", help="화면 프리뷰 창 비활성화 (헤드리스/백그라운드 녹화)")
    parser.add_argument("--test-mode", action="store_true", help="가상 블랙박스 주행 시뮬레이션 모드 (카메라 미연결 시 사용)")
    parser.add_argument("--duration", type=int, default=None, help="테스트용 최대 녹화 시간(초)")
    
    args = parser.parse_args()
    
    recorder = BlackboxRecorder(
        camera_id=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        codec=args.codec,
        split_seconds=args.split_sec,
        save_dir=args.save_dir,
        test_mode=args.test_mode,
        threshold_g=args.threshold
    )
    
    recorder.run(show_preview=not args.no_preview, max_duration=args.duration)

if __name__ == "__main__":
    main()
