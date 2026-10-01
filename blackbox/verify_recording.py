import cv2
import os
import sys
import glob

def decode_fourcc(fourcc_int):
    """정수형 FourCC를 4자리 문자열로 디코딩"""
    try:
        return "".join([chr((int(fourcc_int) >> 8 * i) & 0xFF) for i in range(4)])
    except Exception:
        return "UNKNOWN"

def inspect_video_file(file_path):
    """비디오 파일 메타데이터 및 프레임 무결성 검증"""
    if not os.path.exists(file_path):
        print(f"[에러] 파일을 찾을 수 없습니다: {file_path}")
        return None
        
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"[에러] 비디오 파일을 열 수 없습니다: {file_path}")
        return None
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = decode_fourcc(fourcc_int)
    
    # 실제 프레임 읽기 검증
    actual_read_frames = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        actual_read_frames += 1
        
    cap.release()
    
    calc_duration = (actual_read_frames / fps) if fps > 0 else 0
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    result = {
        "file": os.path.basename(file_path),
        "size_mb": file_size_mb,
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": actual_read_frames,
        "duration_sec": calc_duration,
        "fourcc": fourcc_str
    }
    return result

def verify_all_recordings(directory="recordings"):
    print("=" * 70)
    print("            [블랙박스 녹화 파일 무결성 및 사양 검증]")
    print("=" * 70)
    
    video_files = []
    valid_exts = {'.avi', '.mp4', '.mkv', '.mov'}
    for root, _, filenames in os.walk(directory):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in valid_exts:
                video_files.append(os.path.join(root, fn))
    video_files.sort()
    
    if not video_files:
        print(f"디렉토리 '{directory}'에 녹화된 비디오 파일이 없습니다.")
        return
        
    for vf in video_files:
        info = inspect_video_file(vf)
        if info:
            print(f"📁 파일명    : {info['file']}")
            print(f"  - 파일 크기: {info['size_mb']:.2f} MB")
            print(f"  - 해상도   : {info['width']} x {info['height']} ({'FHD 1080p 일치' if info['width']==1920 and info['height']==1080 else '설정 해상도'})")
            print(f"  - FPS      : {info['fps']:.2f} fps")
            print(f"  - 총 프레임: {info['total_frames']} frames")
            print(f"  - 재생 시간: {info['duration_sec']:.2f}초 ({info['duration_sec']/60:.2f}분)")
            print(f"  - 코덱     : {info['fourcc']}")
            print("-" * 70)

if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "recordings"
    verify_all_recordings(target_dir)
