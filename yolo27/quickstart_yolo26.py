from pathlib import Path
import cv2
from ultralytics import YOLO


def main():
    # 1. 모델 로드 (가중치 파일)
    # yolo26n.pt 모델을 불러옵니다. (로컬에 없으면 자동 다운로드 또는 yolo11n.pt/yolov8n.pt 등 호환 모델 사용)
    model = YOLO("yolo26n.pt")

    # 2. 탐지할 이미지 소스 (로컬 파일 경로 또는 URL)
    source_img = "https://ultralytics.com/images/bus.jpg"

    print(f"[INFO] 객체 탐지 시작: {source_img}")
    results = model(source_img)

    # 3. 결과 저장 경로 설정
    output_dir = Path("runs/detect")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "detected_result.jpg"

    # 4. 탐지 결과 확인, 저장 및 시각화
    for result in results:
        boxes = result.boxes
        print(f"\n[INFO] 총 {len(boxes)}개의 객체가 탐지되었습니다.")

        for idx, box in enumerate(boxes):
            cls_id = int(box.cls.item())
            cls_name = result.names[cls_id]
            conf = float(box.conf.item())
            xyxy = [round(coord, 2) for coord in box.xyxy[0].tolist()]  # [x1, y1, x2, y2]
            print(f"  - [{idx + 1}] {cls_name:<10} 신뢰도: {conf:.2%} | 좌표: {xyxy}")

        # 4-1. 탐지된 결과 이미지 저장 (result.save)
        result.save(filename=str(save_path))
        print(f"\n[INFO] 탐지 결과 이미지가 저장되었습니다 -> {save_path.resolve()}")

        # 4-2. OpenCV를 통한 결과 표시
        annotated_frame = result.plot()  # 바운딩 박스와 라벨이 그려진 BGR numpy 배열 반환

        try:
            cv2.imshow("YOLO Detection Result", annotated_frame)
            print("[INFO] 결과 창이 열렸습니다. 아무 키나 누르면 종료됩니다.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as e:
            # GUI 창을 띄울 수 없는 환경(Headless 등)일 경우 대비
            print(f"[WARNING] GUI 환경 에러({e}). 기본 뷰어로 결과를 표시합니다.")
            result.show()


if __name__ == "__main__":
    main()