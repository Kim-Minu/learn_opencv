import argparse
import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError("google-genai 패키지가 필요합니다. 'pip install google-genai'를 실행해주세요.")

# 스켈레톤 키포인트 연결 정의 (연결선) - 손(hand) 포함
SKELETON_CONNECTIONS = [
    # 얼굴
    ("left_ear", "left_eye"),
    ("left_eye", "nose"),
    ("nose", "right_eye"),
    ("right_eye", "right_ear"),
    # 몸통
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    # 좌측 팔 & 손
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_hand"),
    # 우측 팔 & 손
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_hand"),
    # 좌측 다리
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    # 우측 다리
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

# 한글 라벨 맵 (손 포함 19개 키포인트)
KOREAN_NAMES = {
    "nose": "코 (Nose)",
    "left_eye": "왼쪽 눈 (Left Eye)",
    "right_eye": "오른쪽 눈 (Right Eye)",
    "left_ear": "왼쪽 귀 (Left Ear)",
    "right_ear": "오른쪽 귀 (Right Ear)",
    "left_shoulder": "왼쪽 어깨 (Left Shoulder)",
    "right_shoulder": "오른쪽 어깨 (Right Shoulder)",
    "left_elbow": "왼쪽 팔꿈치 (Left Elbow)",
    "right_elbow": "오른쪽 팔꿈치 (Right Elbow)",
    "left_wrist": "왼쪽 손목 (Left Wrist)",
    "right_wrist": "오른쪽 손목 (Right Wrist)",
    "left_hand": "왼손 (Left Hand)",
    "right_hand": "오른손 (Right Hand)",
    "left_hip": "왼쪽 골반 (Left Hip)",
    "right_hip": "오른쪽 골반 (Right Hip)",
    "left_knee": "왼쪽 무릎 (Left Knee)",
    "right_knee": "오른쪽 무릎 (Right Knee)",
    "left_ankle": "왼쪽 발목 (Left Ankle)",
    "right_ankle": "오른쪽 발목 (Right Ankle)",
}

# 키포인트별 고유 색상 정의 (BGR 포맷)
KEYPOINT_COLORS = {
    # [얼굴] 핑크 / 마젠타 / 바이올렛 계열
    "nose": (255, 105, 180),        # 핫핑크
    "left_eye": (255, 0, 255),       # 마젠타
    "right_eye": (180, 105, 255),    # 핑크-퍼플
    "left_ear": (211, 85, 186),      # 미디엄 오키드
    "right_ear": (130, 0, 200),      # 퍼플
    # [상체 몸통] 노란색 / 금색 계열
    "left_shoulder": (0, 255, 255),  # 옐로우
    "right_shoulder": (0, 215, 255), # 골드
    "left_hip": (0, 165, 255),       # 오렌지
    "right_hip": (0, 140, 255),      # 다크 오렌지
    # [좌측 팔 & 손] 밝은 녹색 / 라임 계열
    "left_elbow": (50, 205, 50),     # 라임 그린
    "left_wrist": (0, 255, 127),     # 스프링 그린
    "left_hand": (120, 255, 120),    # 라이트 그린 (손)
    # [우측 팔 & 손] 시안 / 하늘색 계열
    "right_elbow": (255, 200, 50),   # 스카이블루
    "right_wrist": (255, 140, 0),    # 딥 스카이블루
    "right_hand": (255, 180, 80),    # 라이트 블루 (손)
    # [좌측 다리] 다홍 / 오렌지레드 계열
    "left_knee": (0, 120, 255),      # 오렌지레드
    "left_ankle": (0, 60, 255),      # 진한 오렌지
    # [우측 다리] 빨강 / 크림슨 계열
    "right_knee": (50, 50, 255),     # 밝은 레드
    "right_ankle": (0, 0, 255),      # 퓨어 레드
}

# 연결선(뼈대)별 색상 매핑 (신체 부위별 구분)
CONNECTION_COLORS = {
    # 얼굴
    ("left_ear", "left_eye"): (220, 100, 220),
    ("left_eye", "nose"): (255, 120, 200),
    ("nose", "right_eye"): (220, 120, 255),
    ("right_eye", "right_ear"): (180, 50, 220),
    # 몸통
    ("left_shoulder", "right_shoulder"): (0, 255, 255),
    ("left_shoulder", "left_hip"): (0, 215, 255),
    ("right_shoulder", "right_hip"): (0, 215, 255),
    ("left_hip", "right_hip"): (0, 165, 255),
    # 좌측 팔 & 손
    ("left_shoulder", "left_elbow"): (0, 255, 100),
    ("left_elbow", "left_wrist"): (50, 220, 50),
    ("left_wrist", "left_hand"): (100, 255, 100),
    # 우측 팔 & 손
    ("right_shoulder", "right_elbow"): (255, 200, 50),
    ("right_elbow", "right_wrist"): (255, 140, 0),
    ("right_wrist", "right_hand"): (255, 180, 50),
    # 좌측 다리
    ("left_hip", "left_knee"): (0, 140, 255),
    ("left_knee", "left_ankle"): (0, 80, 255),
    # 우측 다리
    ("right_hip", "right_knee"): (50, 50, 255),
    ("right_knee", "right_ankle"): (0, 0, 255),
}

DEFAULT_COLOR = (0, 255, 0)


# Gemini Structured Outputs 스키마 정의
class Keypoint(BaseModel):
    name: str = Field(
        description="키포인트 표준 영문명: nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hand, right_hand, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle"
    )
    x: float = Field(
        description="정규화된 가로 X 좌표 (0.0=가장 왼쪽, 1.0=가장 오른쪽)"
    )
    y: float = Field(
        description="정규화된 세로 Y 좌표 (0.0=가장 위쪽, 1.0=가장 아래쪽)"
    )
    visible: bool = Field(description="해당 관절/손이 이미지에 보이는지 여부")


class PersonPose(BaseModel):
    person_id: int = Field(description="사람 식별 번호 (1부터 시작)")
    pose_description: str = Field(
        description="인물의 전체적인 동작 및 자세에 대한 상세 설명 (한국어)"
    )
    keypoints: List[Keypoint] = Field(
        description="손(left_hand, right_hand)을 포함한 주요 관절 키포인트 리스트"
    )


class PoseEstimationResult(BaseModel):
    people: List[PersonPose] = Field(description="감지된 모든 사람의 포즈 데이터")
    overall_scene: str = Field(description="이미지 속 전체적인 장면 및 활동 요약 (한국어)")


def draw_legend(img: np.ndarray):
    """우측 상단에 신체 부위별 색상 범례 박스를 반투명하게 그립니다."""
    legend_items = [
        ("Face", (255, 105, 180)),
        ("Body", (0, 255, 255)),
        ("Left Arm & Hand", (0, 255, 127)),
        ("Right Arm & Hand", (255, 180, 0)),
        ("Left Leg", (0, 100, 255)),
        ("Right Leg", (0, 0, 255)),
    ]

    h, w = img.shape[:2]
    box_w, box_h = 175, 25 + len(legend_items) * 20
    x1, y1 = w - box_w - 15, 15
    x2, y2 = x1 + box_w, y1 + box_h

    # 반투명 배경 박스
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), (180, 180, 180), 1)

    cv2.putText(
        img, "Part Colors (Legend)", (x1 + 10, y1 + 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA
    )

    for i, (label, color) in enumerate(legend_items):
        item_y = y1 + 34 + i * 18
        cv2.circle(img, (x1 + 16, item_y - 4), 5, color, -1)
        cv2.circle(img, (x1 + 16, item_y - 4), 6, (255, 255, 255), 1)
        cv2.putText(
            img, label, (x1 + 30, item_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA
        )


def draw_pose_on_image(image_path: Path, pose_data: PoseEstimationResult) -> np.ndarray:
    """Gemini가 예측한 관절 및 손 좌표를 부위별/키포인트별 고유 색상으로 시각화합니다."""
    img_bgr = cv2.imread(str(image_path))
    h, w = img_bgr.shape[:2]

    for person in pose_data.people:
        kpt_dict = {}

        # 1. 키포인트 정규화 좌표를 픽셀 좌표로 변환
        for kpt in person.keypoints:
            kpt_name = kpt.name.lower().strip()
            px = int(np.clip(kpt.x * w, 0, w - 1))
            py = int(np.clip(kpt.y * h, 0, h - 1))
            kpt_dict[kpt_name] = (px, py, kpt.visible)

        # 2. 스켈레톤 연결선(뼈대) 먼저 그리기 (신체 부위별 고유 색상)
        for p1_name, p2_name in SKELETON_CONNECTIONS:
            if p1_name in kpt_dict and p2_name in kpt_dict:
                pt1, pt2 = kpt_dict[p1_name], kpt_dict[p2_name]
                if pt1[2] and pt2[2]:  # 둘 다 visible 일 때만 연결선 표시
                    line_color = CONNECTION_COLORS.get(
                        (p1_name, p2_name),
                        CONNECTION_COLORS.get((p2_name, p1_name), DEFAULT_COLOR),
                    )
                    cv2.line(img_bgr, (pt1[0], pt1[1]), (pt2[0], pt2[1]), line_color, 2, cv2.LINE_AA)

        # 3. 키포인트 원 그리기 (관절별 고유 색상 + 테두리)
        for kpt in person.keypoints:
            kpt_name = kpt.name.lower().strip()
            if kpt_name in kpt_dict:
                px, py, visible = kpt_dict[kpt_name]
                kpt_color = KEYPOINT_COLORS.get(kpt_name, DEFAULT_COLOR)

                if visible:
                    # 손 키포인트는 약간 더 큰 원(반지름 8)으로 강조
                    radius = 8 if "hand" in kpt_name else 6
                    cv2.circle(img_bgr, (px, py), radius + 1, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(img_bgr, (px, py), radius, kpt_color, -1, cv2.LINE_AA)
                else:
                    # 가려진 관절
                    cv2.circle(img_bgr, (px, py), 5, (100, 100, 100), 1, cv2.LINE_AA)

    # 4. 부위별 색상 범례 표시
    draw_legend(img_bgr)

    return img_bgr


def run_gemini_pose_estimation(
    image_paths: List[str],
    model_name: str = "gemini-3.8-flash",
    output_dir: str = "runs/pose_gemini",
    show: bool = False,
):
    """Gemini 모델을 호출하여 손을 포함한 포즈를 추정하고 결과를 저장/표시합니다."""
    client = genai.Client()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt = """
당신은 최고의 컴퓨터 비전 및 인체 해부학 전문가입니다.
제공된 이미지 속 인물들의 포즈(자세)를 정밀하게 분석해주세요.
각 인물에 대해 다음을 수행합니다:
1. 표준 COCO 17개 관절 키포인트에 더해 손 키포인트를 포함한 총 19개 키포인트의 정확한 정규화 좌표 (x: 0.0~1.0, y: 0.0~1.0)를 추출합니다:
   - 얼굴: nose, left_eye, right_eye, left_ear, right_ear
   - 상체/몸통: left_shoulder, right_shoulder, left_hip, right_hip
   - 팔 및 손: left_elbow, right_elbow, left_wrist, right_wrist, left_hand (왼손 중심/손끝), right_hand (오른손 중심/손끝)
   - 다리: left_knee, right_knee, left_ankle, right_ankle
2. 관절이나 손이 가려져 있거나 이미지 밖에 있는 경우 visible=false로 표기하고 대략적인 위치를 추정합니다.
3. 인물의 동작과 자세에 대한 설명을 한국어로 상세히 작성합니다.
"""

    for img_path_str in image_paths:
        img_path = Path(img_path_str)
        if not img_path.exists():
            print(f"[ERROR] 파일이 존재하지 않습니다: {img_path}")
            continue

        print(f"\n{'='*60}")
        print(f"[INFO] Gemini ({model_name}) 포즈 & 손 추정 시작: {img_path.name}")
        print(f"{'='*60}")

        pil_img = Image.open(img_path)

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[pil_img, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=PoseEstimationResult,
                    temperature=0.1,
                ),
            )

            # 결과 객체 역직렬화
            pose_result = PoseEstimationResult.model_validate_json(response.text)

            print(f"[장면 요약] {pose_result.overall_scene}")
            print(f"[감지된 인물 수] {len(pose_result.people)}명\n")

            for person in pose_result.people:
                print(f"  [인물 #{person.person_id}]")
                print(f"  - 자세 분석: {person.pose_description}")
                print(f"  - 키포인트 좌표 (손 포함):")

                for kpt in person.keypoints:
                    korean_name = KOREAN_NAMES.get(kpt.name.lower(), kpt.name)
                    vis_mark = "✓" if kpt.visible else " 가려짐 "
                    print(
                        f"    [{vis_mark}] {korean_name:<20} : "
                        f"X={kpt.x:6.3f}, Y={kpt.y:6.3f}"
                    )

            # 시각화 이미지 생성 및 저장
            annotated_img = draw_pose_on_image(img_path, pose_result)
            save_path = out_dir / f"{img_path.stem}_gemini_pose.jpg"
            cv2.imwrite(str(save_path), annotated_img)
            print(f"\n[INFO] 포즈 & 손 시각화 결과 저장 완료 -> {save_path.resolve()}")

            # 화면 표시
            if show:
                try:
                    cv2.imshow(f"Gemini Pose & Hand - {img_path.name}", annotated_img)
                    print("[INFO] 아무 키를 누르면 다음 이미지로 이동합니다.")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                except cv2.error as e:
                    print(f"[WARNING] GUI 에러({e}).")

        except Exception as e:
            print(f"[ERROR] Gemini API 호출 실패: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini 3.8 Flash Pose & Hand Estimation")
    parser.add_argument(
        "--images",
        nargs="+",
        default=["images/Pose1.png", "images/Pose2.png"],
        help="분석할 이미지 경로 목록",
    )
    parser.add_argument(
        "--model",
        default="gemini-3.8-flash",
        help="사용할 Gemini 모델명",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/pose_gemini",
        help="시각화 이미지 저장 경로",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="화면 팝업 표시 여부",
    )

    args = parser.parse_args()
    run_gemini_pose_estimation(
        image_paths=args.images,
        model_name=args.model,
        output_dir=args.output_dir,
        show=args.show,
    )
