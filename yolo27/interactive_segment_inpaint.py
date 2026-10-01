import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError("google-genai 패키지가 필요합니다. 'pip install google-genai'를 실행해주세요.")

# ==========================================
# 1. Pydantic 스키마 정의 (Gemini 3.8 Flash 세그멘테이션)
# ==========================================
class Point(BaseModel):
    x: float = Field(description="가로 X 좌표 (0.0 ~ 1.0)")
    y: float = Field(description="세로 Y 좌표 (0.0 ~ 1.0)")


class DetectedInstance(BaseModel):
    id: int = Field(description="1부터 시작하는 고유 번호")
    label: str = Field(description="객체 라벨 (예: person, bus, car 등)")
    description: str = Field(description="객체에 대한 시각적 묘사 (의상, 색상, 위치)")
    ymin: float = Field(description="상단 Y 좌표 (0.0 ~ 1.0)")
    xmin: float = Field(description="좌측 X 좌표 (0.0 ~ 1.0)")
    ymax: float = Field(description="하단 Y 좌표 (0.0 ~ 1.0)")
    xmax: float = Field(description="우측 X 좌표 (0.0 ~ 1.0)")
    polygon: List[Point] = Field(description="객체 외곽선을 촘촘히 둘러싸는 다각형 점 리스트 (최소 15점 이상)")


class SegmentationResult(BaseModel):
    instances: List[DetectedInstance] = Field(description="검출된 모든 개별 인스턴스")
    scene_description: str = Field(description="전체 장면 묘사")


# 인스턴스별 고유 컬러 팔레트 (BGR)
PALETTE = [
    (0, 230, 115),   # 민트 그린
    (230, 115, 0),   # 블루
    (0, 165, 255),   # 오렌지
    (204, 51, 204),  # 마젠타
    (0, 215, 255),   # 옐로우골드
    (255, 102, 102), # 라이트 시안
    (102, 102, 255), # 라이트 레드
]


# ==========================================
# 2. 강력한 인터랙티브 뷰어 (호버 + 클릭 + 키보드 숫자 선택)
# ==========================================
class InteractiveViewer:
    def __init__(self, image_bgr: np.ndarray, instances: List[DetectedInstance], max_display_h: int = 760):
        self.orig_img = image_bgr.copy()
        self.instances = instances
        self.orig_h, self.orig_w = image_bgr.shape[:2]

        # 디스플레이 스케일 계산 (노트북 화면에 맞춤)
        self.scale = min(1.0, max_display_h / self.orig_h)
        self.disp_w = int(self.orig_w * self.scale)
        self.disp_h = int(self.orig_h * self.scale)

        self.selected_id: Optional[int] = None
        self.hovered_id: Optional[int] = None

        # 원본 좌표계 기준 폴리곤과 바운딩 박스
        self.orig_polygons = []
        for inst in instances:
            pts = np.array(
                [[int(np.clip(p.x * self.orig_w, 0, self.orig_w - 1)),
                  int(np.clip(p.y * self.orig_h, 0, self.orig_h - 1))]
                 for p in inst.polygon],
                dtype=np.int32
            )
            # 폴리곤 점이 3개 미만이면 바운딩 박스로 사각형 대체
            if len(pts) < 3:
                bx1, by1 = int(inst.xmin * self.orig_w), int(inst.ymin * self.orig_h)
                bx2, by2 = int(inst.xmax * self.orig_w), int(inst.ymax * self.orig_h)
                pts = np.array([[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]], dtype=np.int32)
            self.orig_polygons.append(pts)

    def render(self) -> np.ndarray:
        """선명한 반투명 마스크, 외곽선, 뱃지, 하단 안내바를 렌더링합니다."""
        base = self.orig_img.copy()
        mask_layer = base.copy()

        # 1. 반투명 마스크 칠하기
        for idx, (inst, pts) in enumerate(zip(self.instances, self.orig_polygons)):
            is_selected = (self.selected_id == inst.id)
            is_hovered = (self.hovered_id == inst.id)

            if is_selected:
                fill_color = (0, 0, 255)      # 선택: 강렬한 빨간색
            elif is_hovered:
                fill_color = (0, 255, 255)    # 호버: 밝은 노란색
            else:
                fill_color = PALETTE[idx % len(PALETTE)]

            cv2.fillPoly(mask_layer, [pts], fill_color)

        # 2. 알파 블렌딩 (마스크 합성)
        cv2.addWeighted(mask_layer, 0.45, base, 0.55, 0, base)

        # 3. 선명한 외곽선 및 라벨 뱃지 그리기 (블렌딩 위에 그려야 또렷함)
        for idx, (inst, pts) in enumerate(zip(self.instances, self.orig_polygons)):
            is_selected = (self.selected_id == inst.id)
            is_hovered = (self.hovered_id == inst.id)

            if is_selected:
                line_color = (0, 0, 255)
                line_thick = 4
            elif is_hovered:
                line_color = (0, 255, 255)
                line_thick = 3
            else:
                line_color = (255, 255, 255)
                line_thick = 2

            cv2.polylines(base, [pts], True, line_color, line_thick, cv2.LINE_AA)

            # 무게중심 또는 최상단 위치 계산
            M = cv2.moments(pts)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = pts[0][0], pts[0][1]

            # 뱃지 박스 그리기
            badge_text = f"[{inst.id}] {inst.label}"
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            bx1, by1 = cx - tw // 2 - 6, cy - th // 2 - 4
            bx2, by2 = bx1 + tw + 12, by1 + th + 8

            badge_bg = (0, 0, 220) if is_selected else ((30, 30, 30) if not is_hovered else (0, 180, 220))
            cv2.rectangle(base, (bx1, by1), (bx2, by2), badge_bg, -1)
            cv2.rectangle(base, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
            cv2.putText(base, badge_text, (bx1 + 6, by1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # 4. 화면 크기에 맞게 리사이즈
        disp_img = cv2.resize(base, (self.disp_w, self.disp_h), interpolation=cv2.INTER_AREA)

        # 5. 하단 상태 안내바 추가
        bar_h = 50
        bar = np.zeros((bar_h, self.disp_w, 3), dtype=np.uint8)

        if self.selected_id is not None:
            sel_inst = next((i for i in self.instances if i.id == self.selected_id), None)
            desc_sub = sel_inst.description[:40] + ("..." if len(sel_inst.description) > 40 else "")
            status_text = f"[선택: #{sel_inst.id} {sel_inst.label} - {desc_sub}]"
            cmd_text = "'Enter' / 'Space': 나노바나나2로 삭제 및 인페인팅 | 'c': 취소"
            cv2.putText(bar, status_text, (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(bar, cmd_text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 1, cv2.LINE_AA)
        else:
            hint1 = "마우스 클릭 또는 키보드 숫자(1~9)로 삭제할 인스턴스를 선택하세요."
            hint2 = "종료: 'q' 또는 ESC"
            cv2.putText(bar, hint1, (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(bar, hint2, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

        return np.vstack([disp_img, bar])

    def screen_to_orig_coords(self, sx: int, sy: int) -> Tuple[int, int]:
        """표시 화면 좌표를 원본 이미지 좌표로 변환"""
        ox = int(sx / self.scale)
        oy = int(sy / self.scale)
        return ox, oy

    def find_instance_at(self, ox: int, oy: int) -> Optional[DetectedInstance]:
        """원본 좌표 (ox, oy)에 위치한 인스턴스를 검색합니다."""
        # 1. 폴리곤 내부 검사
        for inst, pts in zip(self.instances, self.orig_polygons):
            if cv2.pointPolygonTest(pts, (ox, oy), False) >= 0:
                return inst

        # 2. 바운딩 박스 검사 (폴리곤 미세 오차 대비)
        for inst in self.instances:
            bx1, by1 = int(inst.xmin * self.orig_w), int(inst.ymin * self.orig_h)
            bx2, by2 = int(inst.xmax * self.orig_w), int(inst.ymax * self.orig_h)
            if bx1 <= ox <= bx2 and by1 <= oy <= by2:
                return inst
        return None

    def on_mouse(self, event, x, y, flags, param):
        """마우스 이동(호버) 및 클릭 이벤트 핸들러"""
        if y >= self.disp_h:
            return  # 상태바 영역

        ox, oy = self.screen_to_orig_coords(x, y)

        if event == cv2.EVENT_MOUSEMOVE:
            inst = self.find_instance_at(ox, oy)
            self.hovered_id = inst.id if inst else None

        elif event == cv2.EVENT_LBUTTONDOWN:
            inst = self.find_instance_at(ox, oy)
            if inst:
                self.selected_id = inst.id
                print(f"[선택됨] #{inst.id} {inst.label} ({inst.description})")
                print("   -> 'Enter' 또는 'Space'를 누르면 나노바나나 2로 삭제 후 배경을 채웁니다.")
            else:
                self.selected_id = None
                print("[선택 해제] 빈 영역 클릭")


# ==========================================
# 3. Gemini 3.8 Flash 인스턴스 세그멘테이션
# ==========================================
def segment_with_gemini(client: genai.Client, image_path: Path) -> SegmentationResult:
    print(f"\n[INFO] Gemini 3.8 Flash 모델로 인스턴스 세그멘테이션 실행 중...")
    pil_img = Image.open(image_path)

    prompt = """
You are an expert in computer vision and instance segmentation.
Detect and segment all distinct foreground and middleground objects (people, bus, vehicles, prominent items) in this image.
For each instance, return:
1. 'id': integer starting from 1
2. 'label': category label (e.g., 'bus', 'person', 'car')
3. 'description': detailed visual description (clothing, color, posture, position)
4. 'ymin', 'xmin', 'ymax', 'xmax': normalized bounding box coordinates (0.0 to 1.0)
5. 'polygon': a sequence of at least 20 closed boundary points [x, y] in normalized (0.0 to 1.0) coordinates tracing the silhouette contour precisely.
"""

    resp = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=[pil_img, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SegmentationResult,
            temperature=0.1,
        ),
    )
    result = SegmentationResult.model_validate_json(resp.text)
    print(f"[INFO] Gemini 3.8 Flash 세그멘테이션 완료! 총 {len(result.instances)}개 객체 검출:")
    for inst in result.instances:
        print(f"  [{inst.id}] {inst.label:<10} | 설명: {inst.description} | 외곽선 정점: {len(inst.polygon)}개")
    return result


# ==========================================
# 4. 나노바나나 2 (gemini-3.1-flash-image) 인페인팅
# ==========================================
def inpaint_with_nanobanana2(
    client: genai.Client,
    image_path: Path,
    target_instance: DetectedInstance,
    output_dir: Path,
) -> np.ndarray:
    print(f"\n{'='*65}")
    print(f"[INFO] 나노바나나 2 (gemini-3.1-flash-image) 배경 인페인팅 시작")
    print(f" - 삭제 대상: #{target_instance.id} {target_instance.label}")
    print(f" - 묘사: {target_instance.description}")
    print(f" - 위치: Y({target_instance.ymin:.2f}~{target_instance.ymax:.2f}), X({target_instance.xmin:.2f}~{target_instance.xmax:.2f})")
    print(f"{'='*65}")

    pil_img = Image.open(image_path)

    inpaint_prompt = f"""
Image inpainting and seamless object removal:
Completely remove the specified object from the image:
- Target object label: {target_instance.label}
- Description: {target_instance.description}
- Bounding box region: top={target_instance.ymin*100:.1f}%, left={target_instance.xmin*100:.1f}%, bottom={target_instance.ymax*100:.1f}%, right={target_instance.xmax*100:.1f}%

Inpaint and reconstruct the background in place of the removed object:
- Seamlessly extend the street pavement, sidewalk tiles, road texture, bus vehicle body, and surrounding background.
- Ensure no ghost shadows, blur, or boundary seam artifacts remain.
- Keep the overall photo style, perspective, and lighting photorealistic and completely natural.
"""

    resp = client.models.generate_content(
        model="gemini-3.1-flash-image",
        contents=[pil_img, inpaint_prompt],
    )

    image_bytes = None
    for part in resp.candidates[0].content.parts:
        if part.inline_data and part.inline_data.data:
            image_bytes = part.inline_data.data
            break

    if image_bytes is None:
        raise RuntimeError("나노바나나 2 모델로부터 인페인팅 결과 이미지를 수신하지 못했습니다.")

    nparr = np.frombuffer(image_bytes, np.uint8)
    inpainted_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    orig_bgr = cv2.imread(str(image_path))
    if inpainted_bgr.shape[:2] != orig_bgr.shape[:2]:
        inpainted_bgr = cv2.resize(inpainted_bgr, (orig_bgr.shape[1], orig_bgr.shape[0]))

    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{image_path.stem}_inpainted_id{target_instance.id}.jpg"
    cv2.imwrite(str(save_path), inpainted_bgr)
    print(f"[INFO] 인페인팅 결과 저장 완료 -> {save_path.resolve()}")

    compare_img = np.hstack([orig_bgr, inpainted_bgr])
    compare_path = output_dir / f"{image_path.stem}_compare_id{target_instance.id}.jpg"
    cv2.imwrite(str(compare_path), compare_img)
    print(f"[INFO] 원본 vs 결과 비교 이미지 저장 완료 -> {compare_path.resolve()}")

    return inpainted_bgr


# ==========================================
# 5. 메인 함수
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Gemini 3.8 Flash Instance Segmentation + Nano Banana 2 Inpainting")
    parser.add_argument("--image", default="bus.jpg", help="대상 이미지 경로")
    parser.add_argument("--output-dir", default="runs/inpaint", help="결과 저장 디렉토리")
    parser.add_argument("--auto-remove-id", type=int, default=None, help="마우스 대신 자동 지정하여 삭제할 인스턴스 ID")

    args = parser.parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[ERROR] 이미지를 찾을 수 없습니다: {image_path}")
        return

    client = genai.Client()
    out_dir = Path(args.output_dir)

    # 1. Gemini 3.8 Flash 세그멘테이션 실행
    seg_result = segment_with_gemini(client, image_path)
    if not seg_result.instances:
        print("[WARNING] 감지된 인스턴스가 없습니다.")
        return

    orig_bgr = cv2.imread(str(image_path))

    # 2. 자동 삭제 모드 (지정된 경우)
    if args.auto_remove_id is not None:
        target = next((inst for inst in seg_result.instances if inst.id == args.auto_remove_id), None)
        if target is None:
            print(f"[ERROR] ID {args.auto_remove_id}를 찾을 수 없습니다. (가능한 ID: {[i.id for i in seg_result.instances]})")
            return
        inpaint_with_nanobanana2(client, image_path, target, out_dir)
        return

    # 3. 마우스 인터랙티브 선택 모드
    viewer = InteractiveViewer(orig_bgr, seg_result.instances)
    win_name = "Gemini 3.8 Flash Segmentation (Click or Type 1~9)"

    # 먼저 초기 렌더링 이미지를 파일로도 저장 (확인용)
    initial_render = viewer.render()
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_preview_path = out_dir / f"{image_path.stem}_segmented_preview.jpg"
    cv2.imwrite(str(mask_preview_path), initial_render)
    print(f"[INFO] 세그멘테이션 마스크 미리보기 이미지가 저장되었습니다: {mask_preview_path.resolve()}")

    try:
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win_name, viewer.on_mouse)

        print("\n" + "="*60)
        print("[사용법 안내]")
        print("  1. 마우스로 객체를 클릭하거나, 키보드 숫자(1, 2, 3...)를 누르면 인스턴스가 선택됩니다.")
        print("  2. 선택 후 'Enter' 또는 'Space'를 누르면 나노바나나 2 모델로 객체를 삭제하고 배경을 채웁니다.")
        print("  3. 'c': 선택 취소  |  'q' 또는 ESC: 프로그램 종료")
        print("="*60 + "\n")

        while True:
            display_img = viewer.render()
            cv2.imshow(win_name, display_img)
            key = cv2.waitKey(25) & 0xFF

            # 키보드 숫자 1~9 로 인스턴스 바로 선택 지원
            if ord('1') <= key <= ord('9'):
                target_num = key - ord('0')
                matched = next((i for i in seg_result.instances if i.id == target_num), None)
                if matched:
                    viewer.selected_id = matched.id
                    print(f"[키보드 선택] #{matched.id} {matched.label} ({matched.description})")
                else:
                    print(f"[안내] ID {target_num}번 인스턴스는 존재하지 않습니다.")

            # Enter (13) 또는 Space (32): 인페인팅 실행
            elif key in [13, 32]:
                if viewer.selected_id is not None:
                    target_inst = next((i for i in seg_result.instances if i.id == viewer.selected_id), None)
                    cv2.destroyWindow(win_name)

                    # 나노바나나 2 인페인팅 수행
                    inpainted = inpaint_with_nanobanana2(client, image_path, target_inst, out_dir)

                    # 완료 비교 창
                    res_win = f"Nano Banana 2 Result (Removed #{target_inst.id} {target_inst.label})"
                    compare_disp = np.hstack([orig_bgr, inpainted])
                    # 화면 크기에 맞춤
                    disp_scale = min(1.0, 760 / compare_disp.shape[0])
                    compare_disp_scaled = cv2.resize(
                        compare_disp,
                        (int(compare_disp.shape[1] * disp_scale), int(compare_disp.shape[0] * disp_scale))
                    )
                    cv2.imshow(res_win, compare_disp_scaled)
                    print("\n[INFO] 인페인팅 결과 창이 열렸습니다. 창을 닫으려면 아무 키나 누르세요.")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    break
                else:
                    print("[안내] 먼저 마우스 클릭 또는 숫자 키로 삭제할 인스턴스를 선택해주세요.")

            elif key == ord('c'):
                viewer.selected_id = None
                print("[선택 취소]")

            elif key == ord('q') or key == 27:
                print("[종료]")
                cv2.destroyAllWindows()
                break

    except cv2.error as e:
        print(f"[GUI 오류] 화면을 열 수 없는 환경입니다({e}).")
        print(f"터미널에서 '--auto-remove-id 2' 옵션을 주어 실행할 수 있습니다.")


if __name__ == "__main__":
    main()
