# learn_opencv

OpenCV 기초 예제부터 블랙박스 녹화기, YOLO26 비전 실습, Gemini 기반 Computer Use 에이전트까지 담은 학습용 저장소입니다.

## 📁 구성

```
learn_opencv/
├── ex_*.py              # OpenCV 기초 예제
├── webp_to_jpg.py       # WebP → JPG 변환 유틸리티
├── blackbox/            # 자동차 블랙박스 상시/이벤트 녹화 프로그램
├── yolo27/              # YOLO26 탐지·세그멘테이션·포즈 추정 실습
└── make_computer_use/   # Gemini 기반 Computer Use GUI 에이전트 (지뢰찾기)
```

## 🖼️ OpenCV 기초 예제

| 파일 | 내용 | 입력 |
|------|------|------|
| `ex_crop.py` | ROI 슬라이싱으로 영상 자르기 | `2.jpg` |
| `ex_resize.py` | `pyrUp` / `pyrDown` 이미지 피라미드 | `2.jpg` |
| `ex_brightness.py` | 밝기 조절과 히스토그램 비교 | `3.jpg` |
| `ex_equalize.py` | 히스토그램 평활화 (그레이스케일 / 컬러) | `3.jpg` |
| `ex_inrange2.py` | HSV `inRange` 색상 영역 추출 (트랙바) | `candies.png` |
| `ex_chroma_key.py` | 녹색 배경 크로마키 합성 | `woman.mp4`, `raining.mp4` |
| `ex_docuscan.py` | 마우스로 꼭짓점 지정 후 투시 변환 문서 스캔 | `road.png` |

```bash
pip install opencv-python numpy matplotlib
python ex_equalize.py
```

`webp_to_jpg.py`는 파일·폴더·URL의 WebP 이미지를 JPG로 변환합니다.

```bash
python webp_to_jpg.py /path/to/folder
python webp_to_jpg.py --url https://example.com/sample.webp
```

## 🚗 blackbox — 블랙박스 녹화 프로그램

FHD(1920×1080) 30FPS 전방 카메라 녹화기입니다.

- DIVX / XVID / MP4(mp4v) / AVC1 코덱 지원
- 1분 단위 상시 분할 녹화
- 가속도 센서(G-Sensor) 시뮬레이션: 2.5G 이상 충격 시 이벤트 녹화로 전환 (`recordings/event/`에 보존)
- 녹화 파일 뷰어와 검증 스크립트 포함

```bash
cd blackbox
python codec_checker.py          # 사용 가능한 코덱 확인
python blackbox.py --codec MP4   # 녹화 시작
python viewer.py                 # 녹화 영상 재생
```

자세한 사용법은 [blackbox/README.md](blackbox/README.md)를 참고하세요. 녹화 결과물(`blackbox/recordings/`)은 저장소에 포함하지 않습니다.

## 🎯 yolo27 — YOLO26 비전 실습

| 파일 | 내용 |
|------|------|
| `quickstart_yolo26.py` | YOLO26 객체 탐지 시작 예제 |
| `instance_segmentation_yolo26.py` | 인스턴스 세그멘테이션 |
| `pose_estimation_yolo26.py` | COCO 17 키포인트 포즈 추정 |
| `pose_estimation_gemini.py` | Gemini를 이용한 포즈 추정 비교 |
| `interactive_segment_inpaint.py` | 클릭 기반 세그멘테이션과 Gemini 인페인팅 |

```bash
cd yolo27
pip install -r requirements.txt
python quickstart_yolo26.py
```

모델 가중치(`*.pt`)는 저장소에 포함하지 않습니다. 처음 실행할 때 Ultralytics가 자동으로 내려받습니다.

## 🤖 make_computer_use — Gemini Computer Use 에이전트

화면 캡처 → Gemini 멀티모달 판단 → 마우스·키보드 조작을 반복하는 **Sense-Think-Act** 루프 에이전트입니다. 데모로 브라우저 지뢰찾기(`minesweeper.html`)를 스스로 플레이합니다.

- `screen_capture.py`: `mss` 기반 화면 캡처 및 전처리
- `agent.py`: Gemini Function Calling 기반 판단
- `controller.py`: `pyautogui` 기반 OS 입력 제어
- `run_minesweeper_agent.py`: 지뢰찾기 데모 실행기

```bash
cd make_computer_use
pip install -r requirements.txt
cp .env.example .env              # GEMINI_API_KEY 입력
python run_minesweeper_agent.py
```

설계 문서는 [GEMINI_COMPUTER_USE_PLAN.md](make_computer_use/GEMINI_COMPUTER_USE_PLAN.md)에 있습니다.

> ⚠️ 에이전트가 실제 마우스와 키보드를 제어합니다. 비상 정지가 필요하면 마우스를 화면 모서리로 옮기세요 (pyautogui FAILSAFE).

## 🔧 환경

- Python 3.12
- OpenCV 4.8 이상
- Gemini 기능을 쓰려면 `GEMINI_API_KEY`가 필요합니다.
