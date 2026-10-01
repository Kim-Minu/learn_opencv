import cv2
import os
import numpy as np

def test_codec(fourcc_str, extension, width=640, height=480, fps=30):
    """
    지정된 코덱과 확장자로 더미 비디오를 생성하여 정상 인코딩/저장되는지 테스트합니다.
    """
    test_filename = f"temp_test_{fourcc_str}.{extension}"
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    out = cv2.VideoWriter(test_filename, fourcc, fps, (width, height))
    
    if not out.isOpened():
        if os.path.exists(test_filename):
            try:
                os.remove(test_filename)
            except Exception:
                pass
        return False, "VideoWriter 열기 실패 (FourCC 미지원 또는 라이브러리 부재)"
    
    # 더미 프레임 10개 작성 테스트
    dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(10):
        out.write(dummy_frame)
    out.release()
    
    # 파일 생성 여부 및 크기 확인
    if os.path.exists(test_filename) and os.path.getsize(test_filename) > 0:
        file_size = os.path.getsize(test_filename)
        try:
            os.remove(test_filename)
        except Exception:
            pass
        return True, f"정상 지원 (테스트 생성 크기: {file_size} bytes)"
    else:
        if os.path.exists(test_filename):
            try:
                os.remove(test_filename)
            except Exception:
                pass
        return False, "파일이 생성되지 않았거나 크기가 0 bytes"

def run_codec_check():
    print("=" * 60)
    print("       [블랙박스 코덱 지원 여부 검증 도구]")
    print(f"       OpenCV 버전: {cv2.__version__}")
    print("=" * 60)
    
    codecs_to_test = [
        ("DIVX", "avi", "DivX 코덱 (AVI 컨테이너)"),
        ("XVID", "avi", "Xvid 코덱 (AVI 컨테이너)"),
        ("mp4v", "mp4", "MPEG-4 코덱 (MP4 컨테이너)"),
        ("avc1", "mp4", "H.264/AVC 코덱 (MP4 컨테이너)"),
        ("MJPG", "avi", "Motion JPEG (호환성 높은 표준)")
    ]
    
    results = {}
    for fourcc, ext, desc in codecs_to_test:
        success, message = test_codec(fourcc, ext)
        status = "✅ [지원]" if success else "❌ [미지원]"
        print(f"{status} FourCC: '{fourcc}' ({ext.upper()}) - {desc}")
        print(f"       -> 상태: {message}")
        results[fourcc] = success
    print("=" * 60)
    return results

if __name__ == "__main__":
    run_codec_check()
