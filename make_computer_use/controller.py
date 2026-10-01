"""
OS 마우스 및 키보드 조작 실행기 모듈 (OS Action Controller)
- pyautogui 기반 마우스/키보드 에뮬레이션
- 좌클릭, 우클릭, 더블클릭, 단축키 지원
- Mac 접근성 권한 및 Fail-Safe (화면 모서리 이동 시 비상정지)
- 한글 타이핑 클립보드(pyperclip) 우회 지원
"""

import time
from typing import List, Optional
import pyautogui
import pyperclip

# 비상 정지: 마우스를 화면 모서리로 홱 이동하면 즉시 중단
pyautogui.FAILSAFE = True
# 액션 후 기본 지연 (초)
pyautogui.PAUSE = 0.15


class OSController:
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()
        print(f"[OSController] Initialized for screen size: {self.screen_width}x{self.screen_height}")

    def click(self, x: int, y: int, click_type: str = "single", duration: float = 0.15):
        """
        논리 픽셀 (x, y) 위치 클릭
        - click_type: 'single' (좌클릭), 'right' (우클릭), 'double' (더블클릭)
        """
        # 화면 범위 제한
        clamped_x = max(0, min(self.screen_width - 1, x))
        clamped_y = max(0, min(self.screen_height - 1, y))

        print(f"[OSController] Moving to ({clamped_x}, {clamped_y}) and executing '{click_type}' click...")
        pyautogui.moveTo(clamped_x, clamped_y, duration=duration)

        if click_type == "right":
            pyautogui.rightClick()
        elif click_type == "double":
            pyautogui.doubleClick()
        else:
            pyautogui.click()

    def type_text(self, text: str):
        """
        텍스트 입력. 한글이나 특수문자 깨짐을 방지하기 위해 클립보드 복사 후 Cmd/Ctrl + V 실행.
        """
        print(f"[OSController] Typing text via clipboard: '{text}'")
        pyperclip.copy(text)
        time.sleep(0.05)
        # Mac에서는 command, 기타 OS에서는 ctrl
        pyautogui.hotkey('command', 'v')

    def press_hotkey(self, keys: List[str]):
        """단축키 실행 (예: ['command', 'space'])"""
        print(f"[OSController] Pressing hotkey: {keys}")
        pyautogui.hotkey(*keys)

    def scroll(self, direction: str = "down", amount: int = 5):
        """스크롤"""
        clicks = amount if direction == "up" else -amount
        print(f"[OSController] Scrolling {direction} by {amount} units")
        pyautogui.scroll(clicks)

    def wait(self, seconds: float):
        """대기"""
        print(f"[OSController] Waiting {seconds:.1f} seconds...")
        time.sleep(max(0.1, seconds))


if __name__ == "__main__":
    controller = OSController()
    print("Testing controller...")
    cx, cy = controller.screen_width // 2, controller.screen_height // 2
    print(f"Current mouse position: {pyautogui.position()}")
