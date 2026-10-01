from pathlib import Path
import cv2
from ultralytics import YOLO

# COCO 17개 포즈 키포인트 이름
KEYPOINT_NAMES = [
    "코 (Nose)",
    "왼쪽 눈 (Left Eye)",
    "오른쪽 눈 (Right Eye)",
    "왼쪽 귀 (Left Ear)",
    "오른쪽 귀 (Right Ear)",
    "왼쪽 어깨 (Left Shoulder)",
    "오른쪽 어깨 (Right Shoulder)",
    "왼쪽 팔꿈치 (Left Elbow)",
    "오른쪽 팔꿈치 (Right Elbow)",
    "왼쪽 손목 (Left Wrist)",
    "오른쪽 손목 (Right Wrist)",
    "왼쪽 골반 (Left Hip)",
    "오른쪽 골반 (Right Hip)",
    "왼쪽 무릎 (Left Knee)",
    "오른쪽 무릎 (Right Knee)",
    "왼쪽 발목 (Left Ankle)",
    "오른쪽 발목 (Right Ankle)",
]


def run_pose_estimation(image_paths, model_path="yolo26m-pose.pt", output_dir="runs/pose", show=False):
    """
    YOLO pose 모델을 사용하여 이미지들에서 포즈(관절 키포인트)를 추정하고 결과를 저장/표시합니다.
    """
    # 1. 모델 로드
    print(f"[INFO] YOLO Pose 모델 로드 중: {model_path}")
    model = YOLO(model_path)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for img_path_str in image_paths:
        img_path = Path(img_path_str)
        if not img_path.exists():
            print(f"[ERROR] 이미지 파일을 찾을 수 없습니다: {img_path}")
            continue

        print(f"\n==================================================")
        print(f"[INFO] 포즈 추정 시작: {img_path.name}")
        print(f"==================================================")

        # 2. 포즈 추정 수행
        results = model(str(img_path))

        for result in results:
            keypoints = result.keypoints
            boxes = result.boxes
            num_persons = len(boxes) if boxes is not None else 0
            print(f"[INFO] 감지된 사람 수: {num_persons}명")

            if keypoints is not None and keypoints.xy is not None:
                # person별 키포인트 좌표 및 신뢰도
                xy_data = keypoints.xy.cpu().numpy()  # (N, 17, 2)
                conf_data = (
                    keypoints.conf.cpu().numpy()
                    if keypoints.conf is not None
                    else None
                )

                for person_idx in range(len(xy_data)):
                    box = boxes[person_idx]
                    box_conf = float(box.conf.item())
                    print(f"\n  [사람 #{person_idx + 1}] (검출 신뢰도: {box_conf:.2%})")

                    for kpt_idx, (x, y) in enumerate(xy_data[person_idx]):
                        kpt_conf = (
                            float(conf_data[person_idx, kpt_idx])
                            if conf_data is not None
                            else 0.0
                        )
                        # 신뢰도 0.5 이상인 유효 키포인트 위주로 표시
                        valid_mark = "✓" if kpt_conf >= 0.5 else " "
                        print(
                            f"    [{valid_mark}] {KEYPOINT_NAMES[kpt_idx]:<20} : "
                            f"X={x:6.1f}, Y={y:6.1f} (신뢰도: {kpt_conf:.2%})"
                        )

            # 3. 결과 이미지 저장
            save_file = out_path / f"{img_path.stem}_pose_result.jpg"
            result.save(filename=str(save_file))
            print(f"\n[INFO] 결과 이미지 저장 완료 -> {save_file.resolve()}")

            # 4. 결과 화면 표시 (옵션)
            if show:
                annotated_img = result.plot()
                try:
                    cv2.imshow(f"Pose Estimation - {img_path.name}", annotated_img)
                    print("[INFO] 아무 키를 누르면 다음 이미지로 넘어갑니다.")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                except cv2.error as e:
                    print(f"[WARNING] GUI 환경 에러({e}). 기본 뷰어로 표시합니다.")
                    result.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLO26 Pose Estimation")
    parser.add_argument(
        "--images",
        nargs="+",
        default=["images/Pose1.png", "images/Pose2.png"],
        help="추정할 이미지 경로 목록",
    )
    parser.add_argument(
        "--model",
        default="yolo26m-pose.pt",
        help="YOLO Pose 모델 가중치 파일",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/pose",
        help="결과 저장 폴더",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="결과 이미지를 화면에 표시할지 여부",
    )

    args = parser.parse_args()
    run_pose_estimation(
        image_paths=args.images,
        model_path=args.model,
        output_dir=args.output_dir,
        show=args.show,
    )
