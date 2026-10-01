"""
Minesweeper Computer Use Agent Launcher
- 브라우저에서 minesweeper.html 자동 실행
- 사용자 카운트다운 후 Gemini 비전 에이전트 자율 플레이 루프 시작
- 실시간 콘솔 모니터링 및 비상 정지 안내
"""

import os
import sys
import time
import webbrowser
from agent import MinesweeperAgent


def main():
    print("=" * 65)
    print("   🎮 Gemini Computer Use Minesweeper Autonomous Runner    ")
    print("=" * 65)

    html_path = os.path.abspath("minesweeper.html")
    if not os.path.exists(html_path):
        print(f"❌ Error: {html_path} 파일을 찾을 수 없습니다.")
        sys.exit(1)

    print(f"\n1. 지뢰찾기 웹 앱을 기본 브라우저에서 실행합니다: \n   file://{html_path}")
    webbrowser.open(f"file://{html_path}")

    print("\n2. [주의] 지뢰찾기 브라우저 창이 화면 중앙에 보이도록 유지해주세요.")
    print("   - 비상 정지: 마우스 커서를 화면 모서리로 빠르게 이동 (PyAutoGUI Fail-Safe)")
    print("   - 또는 터미널에서 Ctrl + C를 눌러 중단할 수 있습니다.")

    print("\n3. 에이전트 준비 중 (5초 카운트다운)...")
    for sec in range(5, 0, -1):
        print(f"   ⏳ {sec}초 후 에이전트가 화면 제어를 시작합니다...", end="\r", flush=True)
        time.sleep(1.0)
    print("\n   🚀 에이전트 루프 가동!\n")

    agent = MinesweeperAgent(model_name="gemini-3.8-flash")

    # 기본 최대 20스텝 자율 플레이
    try:
        agent.play(max_steps=20)
    except KeyboardInterrupt:
        print("\n\n🛑 [User Interrupt] 사용자에 의해 에이전트가 안전하게 중단되었습니다.")
    except Exception as e:
        print(f"\n\n❌ [Runtime Error] 실행 중 오류 발생: {e}")


if __name__ == "__main__":
    main()
