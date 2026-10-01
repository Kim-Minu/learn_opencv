import cv2
import os
import sys
import glob
import time
import argparse

class BlackboxViewer:
    def __init__(self, target=None, directory="recordings", scale=0.6):
        self.directory = directory
        self.scale = scale
        self.file_list = []
        self.current_file_idx = 0
        
        # 파일 목록 구성
        if target and os.path.isfile(target):
            self.file_list = [os.path.abspath(target)]
            self.directory = os.path.dirname(os.path.abspath(target)) or "."
        else:
            search_dir = target if (target and os.path.isdir(target)) else directory
            self.file_list = self._scan_video_files(search_dir)
            
        if not self.file_list:
            print(f"[오류] 재생할 동영상 파일을 찾을 수 없습니다. (디렉토리: {directory})")
            sys.exit(1)
            
        print(f"\n📂 총 {len(self.file_list)}개의 녹화 파일이 발견되었습니다 (최신순):")
        for i, fp in enumerate(self.file_list[:10]):
            fn = os.path.basename(fp)
            tag = "🚨 [EVENT]" if ("_EVENT_" in fn or "/event/" in fp) else "📁 [NORMAL]"
            print(f"  {i + 1:2d}. {tag} {fn}")
        if len(self.file_list) > 10:
            print(f"  ... 외 {len(self.file_list) - 10}개 파일")
            
        # 재생 상태 제어 변수
        self.cap = None
        self.total_frames = 0
        self.fps = 30.0
        self.orig_width = 1920
        self.orig_height = 1080
        self.is_paused = False
        self.playback_speed = 1.0  # 0.5x, 1.0x, 2.0x, 4.0x
        self.show_help = True
        self.window_name = "Blackbox Video Player"
        self.trackbar_moving = False

    def _scan_video_files(self, search_dir, latest_first=True):
        valid_exts = {".avi", ".mp4", ".mkv", ".mov"}
        files = []
        for root, _, filenames in os.walk(search_dir):
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext in valid_exts:
                    files.append(os.path.join(root, fn))
        # 파일명(타임스탬프) 기준 최신순 정렬
        files.sort(key=lambda x: os.path.basename(x), reverse=latest_first)
        return files

    def load_video(self, idx):
        if not (0 <= idx < len(self.file_list)):
            return False
            
        if self.cap is not None:
            self.cap.release()
            
        self.current_file_idx = idx
        filepath = self.file_list[self.current_file_idx]
        self.cap = cv2.VideoCapture(filepath)
        
        if not self.cap.isOpened():
            print(f"[오류] 동영상 파일을 열 수 없습니다: {filepath}")
            return False
            
        self.orig_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cam_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = cam_fps if (cam_fps and cam_fps > 0) else 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames <= 0:
            self.total_frames = 1
            
        print(f"\n[재생 로드] [{self.current_file_idx + 1}/{len(self.file_list)}] {os.path.basename(filepath)}")
        print(f"  - 해상도: {self.orig_width}x{self.orig_height}, FPS: {self.fps:.1f}, 총 프레임: {self.total_frames}")
        
        # 트랙바 범위 갱신
        try:
            cv2.setTrackbarMax("Frame", self.window_name, self.total_frames - 1)
            cv2.setTrackbarPos("Frame", self.window_name, 0)
        except Exception:
            pass
            
        return True

    def on_trackbar_change(self, pos):
        if self.trackbar_moving:
            return
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

    def draw_hud(self, frame, current_frame):
        """동영상 화면에 재생 제어 및 비디오 정보 HUD 오버레이"""
        h, w = frame.shape[:2]
        filepath = self.file_list[self.current_file_idx]
        filename = os.path.basename(filepath)
        
        # 현재 재생 시각 / 전체 시각 계산
        curr_sec = current_frame / self.fps if self.fps > 0 else 0
        total_sec = self.total_frames / self.fps if self.fps > 0 else 0
        curr_str = f"{int(curr_sec // 60):02d}:{int(curr_sec % 60):02d}.{int((curr_sec % 1) * 10)}"
        total_str = f"{int(total_sec // 60):02d}:{int(total_sec % 60):02d}"
        
        # 상단/하단 반투명 오버레이 바
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
        cv2.rectangle(overlay, (0, h - 50), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        
        # 상단 정보: 파일명, 번호, 원본 사양
        status_text = "PAUSED ⏸️" if self.is_paused else f"PLAYING ▶️ ({self.playback_speed}x)"
        status_color = (100, 200, 255) if self.is_paused else (100, 255, 100)
        
        is_event = ("_EVENT_" in filename) or ("/event/" in filepath)
        type_badge = "[🚨 EVENT]" if is_event else "[📁 NORMAL]"
        badge_color = (60, 60, 255) if is_event else (220, 180, 80)
        
        cv2.putText(frame, f"{type_badge} [{self.current_file_idx + 1}/{len(self.file_list)}] {filename}",
                    (15, 24), cv2.FONT_HERSHEY_DUPLEX, 0.58, badge_color, 1)
        cv2.putText(frame, f"SPEC: {self.orig_width}x{self.orig_height} @ {self.fps:.0f}fps",
                    (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
        cv2.putText(frame, status_text, (w - 220, 32), cv2.FONT_HERSHEY_DUPLEX, 0.65, status_color, 2)
        
        # 하단 정보: 재생 시간, 프레임 번호, 단축키 도움말 안내
        progress_text = f"Time: {curr_str} / {total_str}  (Frame: {current_frame}/{self.total_frames})"
        cv2.putText(frame, progress_text, (15, h - 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        guide_text = "[H] Toggle Help | [Space] Play/Pause | [Q] Exit"
        cv2.putText(frame, guide_text, (w - 430, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 240, 255), 1)
        
        # 도움말 패널 오버레이 (H 키로 토글)
        if self.show_help:
            help_w, help_h = 420, 210
            x1, y1 = 20, 70
            sub_over = frame.copy()
            cv2.rectangle(sub_over, (x1, y1), (x1 + help_w, y1 + help_h), (10, 10, 10), -1)
            cv2.addWeighted(sub_over, 0.75, frame, 0.25, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x1 + help_w, y1 + help_h), (80, 160, 240), 1)
            
            help_lines = [
                "--- Blackbox Viewer Shortcuts ---",
                "Space      : Play / Pause",
                "Left/Right : Seek -5s / +5s (or Frame step)",
                "N / P ([,]): Next / Previous video file",
                "Up / Down  : Speed 0.5x, 1x, 2x, 4x",
                "R          : Restart current video",
                "H          : Show / Hide this Help",
                "Q / ESC    : Quit Viewer"
            ]
            for i, line in enumerate(help_lines):
                color = (100, 220, 255) if i == 0 else (230, 230, 230)
                cv2.putText(frame, line, (x1 + 15, y1 + 25 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        
        if not self.load_video(self.current_file_idx):
            return
            
        cv2.createTrackbar("Frame", self.window_name, 0, self.total_frames - 1, self.on_trackbar_change)
        
        current_frame = 0
        speed_list = [0.5, 1.0, 1.5, 2.0, 4.0]
        
        print("\n" + "=" * 65)
        print("         [블랙박스 동영상 뷰어 실행]")
        print("  - 단축키: [Space] 일시정지/재생, [Left/Right] 이동, [N/P] 파일 변경")
        print("=" * 65)
        
        while True:
            start_loop = time.time()
            
            if not self.is_paused:
                ret, frame = self.cap.read()
                if not ret:
                    # 영상 끝 도달 시 일시정지 상태로 마지막 프레임 유지 또는 정지
                    self.is_paused = True
                    current_frame = self.total_frames - 1
                else:
                    current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    # 트랙바 위치 동기화
                    self.trackbar_moving = True
                    cv2.setTrackbarPos("Frame", self.window_name, current_frame)
                    self.trackbar_moving = False
            else:
                # 일시정지 시 현재 프레임 유지
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = self.cap.read()
                if not ret:
                    frame = 128 * (frame is not None)
            
            if frame is not None:
                # 화면 스케일 조정 (예: FHD 1080p -> 모니터용 리사이징)
                disp_w = int(frame.shape[1] * self.scale)
                disp_h = int(frame.shape[0] * self.scale)
                disp_frame = cv2.resize(frame, (disp_w, disp_h))
                
                # HUD 그리기
                self.draw_hud(disp_frame, current_frame)
                cv2.imshow(self.window_name, disp_frame)
            
            # 지연 시간 계산 (재생 배속 반영)
            base_delay_ms = int(1000 / (self.fps * self.playback_speed))
            wait_time = max(1, base_delay_ms if not self.is_paused else 30)
            
            key = cv2.waitKey(wait_time) & 0xFF
            
            if key == 255:  # no key
                continue
                
            # 종료
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
                
            # 재생 / 일시정지
            elif key == ord(' '):
                self.is_paused = not self.is_paused
                # 끝에서 일시정지 해제 시 처음부터 재시작
                if not self.is_paused and current_frame >= self.total_frames - 1:
                    current_frame = 0
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    
            # 처음부터 다시 재생 (R)
            elif key == ord('r') or key == ord('R'):
                current_frame = 0
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.is_paused = False
                
            # 도움말 토글 (H)
            elif key == ord('h') or key == ord('H'):
                self.show_help = not self.show_help
                
            # 배속 조절 (위/아래 화살표 or + / -)
            elif key in [ord('+'), ord('='), 0, 82]:  # Up arrow or +
                cur_i = speed_list.index(self.playback_speed) if self.playback_speed in speed_list else 1
                if cur_i < len(speed_list) - 1:
                    self.playback_speed = speed_list[cur_i + 1]
            elif key in [ord('-'), ord('_'), 1, 84]:  # Down arrow or -
                cur_i = speed_list.index(self.playback_speed) if self.playback_speed in speed_list else 1
                if cur_i > 0:
                    self.playback_speed = speed_list[cur_i - 1]
                    
            # 탐색: 이전/다음 프레임 또는 5초 이동 (Left / Right or A / D or J / L)
            elif key in [ord('a'), ord('A'), ord('j'), ord('J'), 2, 81]:  # Left
                jump_frames = int(self.fps * 5) if not self.is_paused else 1
                current_frame = max(0, current_frame - jump_frames)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                self.trackbar_moving = True
                cv2.setTrackbarPos("Frame", self.window_name, current_frame)
                self.trackbar_moving = False
            elif key in [ord('d'), ord('D'), ord('l'), ord('L'), 3, 83]:  # Right
                jump_frames = int(self.fps * 5) if not self.is_paused else 1
                current_frame = min(self.total_frames - 1, current_frame + jump_frames)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                self.trackbar_moving = True
                cv2.setTrackbarPos("Frame", self.window_name, current_frame)
                self.trackbar_moving = False
                
            # 다음 / 이전 비디오 파일 (N / P or [ / ])
            elif key in [ord('n'), ord('N'), ord(']')]:  # Next file
                if self.current_file_idx < len(self.file_list) - 1:
                    self.load_video(self.current_file_idx + 1)
                    current_frame = 0
                    self.is_paused = False
            elif key in [ord('p'), ord('P'), ord('[')]:  # Prev file
                if self.current_file_idx > 0:
                    self.load_video(self.current_file_idx - 1)
                    current_frame = 0
                    self.is_paused = False

        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("\n[뷰어 종료] 뷰어를 정상적으로 종료했습니다.")

def main():
    parser = argparse.ArgumentParser(description="블랙박스 녹화 동영상 전용 뷰어")
    parser.add_argument("target", nargs="?", default=None,
                        help="재생할 동영상 파일 경로 또는 디렉토리 (기본값: ./recordings)")
    parser.add_argument("--dir", type=str, default="recordings",
                        help="동영상 폴더 지정 (기본값: recordings)")
    parser.add_argument("--scale", type=float, default=0.55,
                        help="화면 표시 배율 (기본값 0.55, FHD 1080p -> 약 1056x594)")
    
    args = parser.parse_args()
    
    viewer = BlackboxViewer(
        target=args.target,
        directory=args.dir,
        scale=args.scale
    )
    viewer.run()

if __name__ == "__main__":
    main()
