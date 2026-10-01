import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


def run_instance_segmentation(
    image_paths,
    model_path="yolo26n-seg.pt",
    output_dir="runs/segment",
    conf_threshold=0.25,
    show=False,
):
    """
    YOLO26 Instance Segmentation 모델을 사용하여
    객체별 바운딩 박스와 픽셀 단위 세그멘테이션 마스크를 검출하고 시각화합니다.
    """
    # 1. 모델 로드
    print(f"[INFO] YOLO26 Instance Segmentation 모델 로드: {model_path}")
    model = YOLO(model_path)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for img_path_str in image_paths:
        img_path = Path(img_path_str)
        if not img_path.exists():
            print(f"[ERROR] 파일이 존재하지 않습니다: {img_path}")
            continue

        print(f"\n{'='*60}")
        print(f"[INFO] 인스턴스 세그멘테이션 분석 시작: {img_path.name}")
        print(f"{'='*60}")

        # 2. 인스턴스 세그멘테이션 추론 실행
        results = model(str(img_path), conf=conf_threshold)

        for result in results:
            boxes = result.boxes
            masks = result.masks
            names = result.names

            num_instances = len(boxes) if boxes is not None else 0
            print(f"[INFO] 감지된 객체(인스턴스) 수: {num_instances}개\n")

            if num_instances == 0:
                print("  - 감지된 객체가 없습니다.")
                continue

            # 인스턴스 상세 정보 출력
            for idx in range(num_instances):
                cls_id = int(boxes.cls[idx].item())
                cls_name = names[cls_id]
                conf = float(boxes.conf[idx].item())
                xyxy = [round(c, 1) for c in boxes.xyxy[idx].tolist()]

                polygon_pts_count = 0
                if masks is not None and len(masks.xy) > idx:
                    polygon_pts_count = len(masks.xy[idx])

                print(
                    f"  [{idx + 1:2d}] {cls_name:<12} | "
                    f"신뢰도: {conf:.2%} | "
                    f"박스: {xyxy} | "
                    f"마스크 폴리곤 점 개수: {polygon_pts_count}개"
                )

            # 3. 결과 시각화 및 저장
            # 3-1. YOLO 내장 시각화 이미지 획득 (반투명 컬러 마스크 + 박스 + 라벨)
            annotated_frame = result.plot()

            # 3-2. 이미지 파일 저장
            save_file = out_path / f"{img_path.stem}_segment_result.jpg"
            cv2.imwrite(str(save_file), annotated_frame)
            print(f"\n[INFO] 세그멘테이션 결과 이미지 저장 완료 -> {save_file.resolve()}")

            # 4. 화면 표시 (옵션)
            if show:
                try:
                    cv2.imshow(f"Instance Segmentation - {img_path.name}", annotated_frame)
                    print("[INFO] 아무 키나 누르면 다음 이미지로 넘어갑니다.")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                except cv2.error as e:
                    print(f"[WARNING] GUI 환경 에러({e}). result.show()를 호출합니다.")
                    result.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO26 Instance Segmentation")
    parser.add_argument(
        "--images",
        nargs="+",
        default=["bus.jpg", "images/Pose1.png", "images/Pose2.png"],
        help="세그멘테이션을 수행할 이미지 파일 경로 목록",
    )
    parser.add_argument(
        "--model",
        default="yolo26n-seg.pt",
        help="사용할 YOLO 세그멘테이션 모델 가중치 파일",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/segment",
        help="결과 이미지를 저장할 디렉토리",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="검출 신뢰도 임계값 (기본값: 0.25)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="결과 이미지를 화면 팝업 창으로 표시할지 여부",
    )

    args = parser.parse_args()
    run_instance_segmentation(
        image_paths=args.images,
        model_path=args.model,
        output_dir=args.output_dir,
        conf_threshold=args.conf,
        show=args.show,
    )
