"""
Gemini Computer Use Minesweeper Agent
- Gemini 멀티모달 모델을 통해 화면 캡처 이미지를 분석하고 다음 지뢰찾기 수를 결정
- Function Calling 기반 자율 마우스 제어
- 승리/패배 상태 감지 및 무한루프 방지 슬라이딩 윈도우 문맥 관리
"""

import os
import sys
import time
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# 로컬 모듈
from screen_capture import ScreenCapturer
from controller import OSController

load_dotenv()

# Gemini SDK 임포트 (google-genai 우선, google.generativeai 차선)
GEMINI_CLIENT_TYPE = None
try:
    from google import genai
    from google.genai import types
    GEMINI_CLIENT_TYPE = "google-genai"
except ImportError:
    try:
        import google.generativeai as genai_legacy
        GEMINI_CLIENT_TYPE = "legacy"
    except ImportError:
        GEMINI_CLIENT_TYPE = None


MINESWEEPER_SYSTEM_PROMPT = """
당신은 컴퓨터 화면을 관측하고 마우스를 직접 조작하여 지뢰찾기(Minesweeper) 게임을 클리어하는 전문 AI 에이전트입니다.

[동작 지침]
1. 당신에게는 실시간으로 캡처된 현재 컴퓨터 화면 이미지가 제공됩니다.
2. 화면에서 'Minesweeper' 게임 보드를 찾고, 각 칸(Cell)의 상태를 분석하세요:
   - 미개방 셀 (어두운 회색 닫힌 칸)
   - 열린 셀 (숫자 1~8 또는 빈 칸)
   - 깃발 셀 (🚩)
3. 지뢰찾기 논리 규칙에 따라 가장 안전하거나 확실한 액션을 단 하나 선택하여 도구를 호출하세요:
   - 게임 시작 직후(모든 칸이 닫힌 상태): 보드의 정중앙 셀을 'single'(좌클릭)하여 넓은 안전 영역을 확보합니다.
   - 열린 칸의 숫자 N과 인접한 닫힌 칸의 개수가 정확히 일치하면: 그 닫힌 칸들은 지뢰이므로 click_type='right'(우클릭)으로 깃발을 꽂습니다.
   - 숫자 N 주위에 이미 N개의 깃발이 꽂혀 있다면: 나머지 인접한 닫힌 칸들은 안전하므로 click_type='single'(좌클릭)하여 엽니다.
4. 좌표계:
   - 화면 전체는 가로 0~1000, 세로 0~1000의 정규화 좌표계입니다.
   - 타겟 셀의 정중앙 좌표(x, y)를 정확하게 측정하여 입력하세요.
5. 종료 조건:
   - 화면에 승리 축하 배너(🎉)나 스마일 버튼이 선글라스(😎)로 바뀌면: task_finish(status='SUCCESS') 호출
   - 지뢰가 폭발(💥)하거나 스마일 버튼이 기절(😵)로 바뀌면: task_finish(status='FAILED') 호출
"""

# Tool 정의
TOOLS_SCHEMA = [
    {
        "name": "click_coordinate",
        "description": "지정한 화면의 정규화 좌표(0~1000)를 마우스로 클릭합니다.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "x": {"type": "NUMBER", "description": "가로 좌표 비율 (0: 왼쪽 끝, 1000: 오른쪽 끝)"},
                "y": {"type": "NUMBER", "description": "세로 좌표 비율 (0: 상단 끝, 1000: 하단 끝)"},
                "click_type": {
                    "type": "STRING",
                    "enum": ["single", "right"],
                    "description": "클릭 종류 ('single': 좌클릭으로 칸 열기, 'right': 우클릭으로 깃발 꽂기)"
                },
                "reason": {"type": "STRING", "description": "해당 칸을 선택한 이유 및 추론 근거"}
            },
            "required": ["x", "y", "click_type", "reason"]
        }
    },
    {
        "name": "wait_seconds",
        "description": "화면 렌더링 및 애니메이션을 위해 지정된 시간(초) 동안 대기합니다.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "seconds": {"type": "NUMBER", "description": "대기 시간(초), 보통 0.5~1.5초"}
            },
            "required": ["seconds"]
        }
    },
    {
        "name": "task_finish",
        "description": "지뢰찾기 게임이 종료(승리 또는 패배)되었을 때 호출합니다.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "status": {"type": "STRING", "enum": ["SUCCESS", "FAILED"], "description": "게임 결과"},
                "result_message": {"type": "STRING", "description": "종료 원인 및 최종 리포트"}
            },
            "required": ["status", "result_message"]
        }
    }
]


class MinesweeperAgent:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-3.8-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name
        self.capturer = ScreenCapturer()
        self.controller = OSController()
        self.history_actions: List[Dict[str, Any]] = []

        if not self.api_key:
            print("[Warning] GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")

        self.init_client()

    def init_client(self):
        if GEMINI_CLIENT_TYPE == "google-genai":
            self.client = genai.Client(api_key=self.api_key)
            print(f"[Agent] Initialized with new google-genai SDK ({self.model_name})")
        elif GEMINI_CLIENT_TYPE == "legacy":
            genai_legacy.configure(api_key=self.api_key)
            self.client = genai_legacy.GenerativeModel(
                model_name=self.model_name,
                system_instruction=MINESWEEPER_SYSTEM_PROMPT
            )
            print(f"[Agent] Initialized with legacy google.generativeai SDK ({self.model_name})")
        else:
            self.client = None
            print("[Agent] Gemini SDK not installed or missing. Standalone/Dry-run mode available.")

    def run_step(self, step_idx: int) -> bool:
        """
        1스텝 실행: 화면 캡처 -> Gemini 분석 -> Tool 호출 실행
        반환값: 계속 진행할지 여부 (False면 루프 종료)
        """
        print(f"\n================ [Step {step_idx}] Observation & Decision ================")
        # 1. 화면 캡처 (그리드 보조선 추가)
        bgr_frame, pil_image = self.capturer.capture(resize_to=(1280, 800), add_grid=True)
        # 임시 디버그 스크린샷 저장
        debug_img_path = f"step_{step_idx}_screen.jpg"
        pil_image.save(debug_img_path)
        print(f"[Step {step_idx}] Screen captured and saved: {debug_img_path}")

        # 2. Gemini 추론 요청
        prompt_text = f"""
현재 지뢰찾기 화면(스텝 {step_idx})입니다.
최근 수행한 이전 액션 및 시각 피드백:
{json.dumps(self.history_actions[-3:], ensure_ascii=False, indent=2)}

화면을 정밀하게 분석하여 다음 행동 도구를 하나만 호출하세요.
"""

        if not self.client:
            print("[Dry-Run] Gemini 클라이언트가 없으므로 화면 중앙(500, 500) 모의 클릭을 수행합니다.")
            tool_call = {
                "name": "click_coordinate",
                "args": {"x": 500, "y": 500, "click_type": "single", "reason": "Dry-run center click"}
            }
            return self.execute_tool(tool_call, bgr_before=bgr_frame)

        try:
            # SDK별 호출
            if GEMINI_CLIENT_TYPE == "google-genai":
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[pil_image, prompt_text],
                    config=types.GenerateContentConfig(
                        system_instruction=MINESWEEPER_SYSTEM_PROMPT,
                        tools=[types.Tool(function_declarations=TOOLS_SCHEMA)],
                        temperature=0.2
                    )
                )

                # Tool Call 파싱
                tool_calls = response.function_calls
                if not tool_calls:
                    print(f"[Step {step_idx}] 모델이 텍스트로 응답했습니다: {response.text}")
                    # 만약 텍스트로 나왔다면 다시 재시도하거나 기본 대기
                    return True

                first_call = tool_calls[0]
                tool_data = {
                    "name": first_call.name,
                    "args": dict(first_call.args)
                }
                return self.execute_tool(tool_data, bgr_before=bgr_frame)

            elif GEMINI_CLIENT_TYPE == "legacy":
                # legacy SDK 호출
                response = self.client.generate_content([pil_image, prompt_text])
                print(f"[Step {step_idx}] Response: {response.text}")
                return True

        except Exception as e:
            print(f"[Error in Step {step_idx}] Gemini API 호출 실패: {e}")
            return False

        return True

    def execute_tool(self, tool_call: Dict[str, Any], bgr_before: Optional[Any] = None) -> bool:
        """도구 실행 및 OpenCV 시각 차분(Frame Diff) 피드백"""
        name = tool_call.get("name")
        args = tool_call.get("args", {})
        print(f"[Action] Executing Tool: {name}({json.dumps(args, ensure_ascii=False)})")

        if name == "click_coordinate":
            norm_x = args.get("x", 500)
            norm_y = args.get("y", 500)
            click_type = args.get("click_type", "single")
            reason = args.get("reason", "")
            print(f"👉 Target Normalized: ({norm_x}, {norm_y}), Type: {click_type}")
            print(f"💡 Reason: {reason}")

            # 0~1000 상대좌표 -> OS 논리 픽셀 변환
            lx, ly = self.capturer.norm_to_logical(norm_x, norm_y)
            self.controller.click(lx, ly, click_type=click_type)
            time.sleep(0.4)

            # OpenCV 비전 차분 검증 (액션 후 화면 변화 감지)
            if bgr_before is not None:
                bgr_after, _ = self.capturer.capture(resize_to=(1280, 800), add_grid=False)
                diff_ratio = ScreenCapturer.compute_diff_ratio(bgr_before, bgr_after)
                percent = diff_ratio * 100.0
                print(f"👁️ [Vision Feedback] Screen changed: {percent:.2f}%")
                tool_call["feedback"] = f"Screen changed by {percent:.2f}%"
                if diff_ratio < 0.0005:
                    print("⚠️ [Warning] 화면에 거의 변화가 없습니다. 클릭 위치가 어긋났거나 로딩 중일 수 있습니다.")
                    tool_call["feedback"] += " (WARNING: No visible change detected)"
            
            self.history_actions.append(tool_call)
            return True

        elif name == "wait_seconds":
            sec = args.get("seconds", 1.0)
            self.controller.wait(sec)
            self.history_actions.append(tool_call)
            return True

        elif name == "task_finish":
            status = args.get("status")
            msg = args.get("result_message")
            print(f"\n🏁 [Game Finished] Status: {status} | Message: {msg}")
            self.history_actions.append(tool_call)
            return False

        self.history_actions.append(tool_call)
        return True

    def play(self, max_steps: int = 15):
        """메인 에이전트 루프"""
        print(f"\n🚀 [Agent Started] Starting Minesweeper Computer Use Loop (Max Steps: {max_steps})")
        print("안내: 비상시 마우스를 모서리로 빠르게 이동시키면 Fail-Safe로 즉시 중단됩니다.")
        time.sleep(2.0)  # 사용자가 브라우저 창을 띄울 시간 부여

        for step in range(1, max_steps + 1):
            should_continue = self.run_step(step)
            if not should_continue:
                break
            time.sleep(1.0)

        print("\n✨ [Agent Finished] 루프가 종료되었습니다.")


if __name__ == "__main__":
    agent = MinesweeperAgent()
    # 단독 테스트 실행 시 1회 테스트
    agent.run_step(1)
