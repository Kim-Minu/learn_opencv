# 🚗 자동차용 블랙박스 상시 녹화 프로그램 (OpenCV)

미니콘다 가상환경 `DS` 환경을 기반으로 제작된 OpenCV 전방 카메라 블랙박스 프로그램입니다.

---

## 📌 주요 사양 및 기능

1. **전방 카메라 FHD 녹화**
   - 해상도: **1920 x 1080 (FHD)**
   - 프레임레이트: **30 FPS**
2. **다양한 동영상 코덱 지원**
   - `DIVX` (`.avi` 컨테이너)
   - `XVID` (`.avi` 컨테이너)
   - `MP4` (`mp4v` / `.mp4` 컨테이너)
   - `AVC1` (`H.264` / `.mp4` 컨테이너)
3. **1분 단위 상시 분할 녹화**
   - 설정된 시간(기본 60초)마다 파일을 자동으로 분할 저장
   - 파일명 포맷: `YYYYMMDD_HHMMSS_1920x1080_CODEC.ext`
4. **가속도 센서(G-Sensor) 물리 모델링 및 충격 감지**
   - 3축 가속도($A_x, A_y, A_z$) 및 합성 $G_{total}$ 물리 시뮬레이션
   - 평상시: 도로 주행 진동 노이즈 모델링 (약 1.0G 유지)
   - 충격 발생 시: 감쇠 조화 진동(Damped Harmonic Oscillation) 파형 모델링
   - 충격 감지 임계값(기본 2.5G) 초과 시 **이벤트 녹화 자동 전환**
5. **마우스 클릭 버튼 UI 컨트롤 패널 창 제공**
   - [sensor.py](file:///Users/minwoo/Projects/ai-native-agent/learn_opencv/blackbox/sensor.py): OpenCV 기반 센서 제어창 (`Blackbox Sensor Controller`)
   - 마우스 클릭 가능한 버튼:
     - `🚨 충격 발생 (Heavy Crash: 4.5G)`
     - `⚠️ 둔턱/경미 충격 (Bump: 2.8G)`
     - `⚡ 측면 충돌 (Side Impact: 3.8G)`
     - `🔄 센서 초기화 (Reset)`
   - 실시간 G-Force 오실로스코프 파형 그래프 및 임계값 라인 표시
6. **상시 / 충격 이벤트 녹화 디렉토리 자동 분리**
   - 일반 주행: `recordings/normal/` (`..._NORMAL_...`)
   - 충격 이벤트: `recordings/event/` (`..._EVENT_...`) 영구 보존
7. **검증 도구 및 뷰어 제공**
   - [viewer.py](file:///Users/minwoo/Projects/ai-native-agent/learn_opencv/blackbox/viewer.py): 녹화된 영상을 확인하는 전용 플레이어 (트랙바 탐색, 배속, 파일 전환, EVENT/NORMAL 뱃지 구분)
   - [codec_checker.py](file:///Users/minwoo/Projects/ai-native-agent/learn_opencv/blackbox/codec_checker.py): 환경 내 코덱 설치 및 생성 가능 여부 확인
   - [verify_recording.py](file:///Users/minwoo/Projects/ai-native-agent/learn_opencv/blackbox/verify_recording.py): 녹화된 비디오의 실제 해상도, FPS, 재생 시간, FourCC 메타데이터 검증

---

## 🛠️ 실행 환경 활성화

```bash
conda activate DS
```

---

## 🚀 실행 방법

### 1. 코덱 지원 여부 사전 검증
```bash
python codec_checker.py
```

### 2. 블랙박스 기본 녹화 (1분 단위 상시 녹화)
```bash
# 기본 실행 (카메라 0번, FHD 1080p 30fps, MP4 코덱, 60초 분할)
python blackbox.py

# XVID 코덱으로 녹화
python blackbox.py --codec XVID

# DIVX 코덱으로 녹화
python blackbox.py --codec DIVX

# 특정 카메라 지정 (예: 외장 웹캠 카메라 1번)
python blackbox.py --camera 1 --codec MP4
```

### 3. 카메라 연결 없이 테스트 (가상 시뮬레이션 모드)
웹캠이 없거나 가상으로 주행 영상 녹화 동작을 확인하려면 `--test-mode` 옵션을 사용합니다.
```bash
python blackbox.py --test-mode --split-sec 60
```

### 4. 녹화 파일 전용 뷰어로 재생 및 검증 (NEW)
```bash
# recordings 폴더 내 영상 목록 자동 로드 및 재생
python viewer.py

# 특정 파일 직접 지정 재생
python viewer.py recordings/20261001_093228_1920x1080_DIVX.avi

# 창 크기 배율 조절 (기본 0.55 -> FHD 1080p를 모니터에 적절히 표시)
python viewer.py --scale 0.7
```

#### 🎮 뷰어 단축키 가이드
| 단축키 | 기능 설명 |
| :--- | :--- |
| **`Space`** | 재생 / 일시정지 (Play / Pause) |
| **`←` / `→` (또는 `A` / `D`)** | 5초 전/후 탐색 (일시정지 중에는 1프레임씩 이동) |
| **`[` / `]` (또는 `P` / `N`)** | 이전 파일 / 다음 파일 전환 |
| **`↑` / `↓` (또는 `+` / `-`)** | 재생 배속 변경 (`0.5x`, `1.0x`, `1.5x`, `2.0x`, `4.0x`) |
| **`R`** | 현재 영상 처음부터 다시 재생 |
| **`H`** | 화면 도움말(HUD) 켜기 / 끄기 토글 |
| **`트랙바 슬라이더`** | 마우스로 원하는 재생 위치(프레임)로 직접 탐색 |
| **`Q` / `ESC`** | 뷰어 종료 |

### 5. 녹화 파일 메타데이터 무결성 검증
```bash
python verify_recording.py
```

---

## ⚙️ 주요 옵션 (CLI Arguments)

| 옵션 | 기본값 | 설명 |
| :--- | :---: | :--- |
| `--camera` | `0` | 연결된 웹캠/카메라 장치 번호 |
| `--width` | `1920` | 녹화 가로 해상도 (FHD: 1920) |
| `--height` | `1080` | 녹화 세로 해상도 (FHD: 1080) |
| `--fps` | `30.0` | 녹화 프레임레이트 |
| `--codec` | `MP4` | `DIVX`, `XVID`, `MP4`, `AVC1` 중 선택 |
| `--split-sec` | `60` | 파일 분할 주기 (초 단위, 기본 60초 = 1분) |
| `--save-dir` | `recordings` | 동영상 저장 폴더 |
| `--no-preview` | `False` | 화면 프리뷰 창 없이 백그라운드 녹화 |
| `--test-mode` | `False` | 가상 주행 영상 시뮬레이션 모드 |
