#!/usr/bin/env python3
"""
WebP to JPG 변환 스크립트

사용법:
1. 현재 폴더(또는 지정 폴더/파일) 내 webp 변환:
   python webp_to_jpg.py
   python webp_to_jpg.py 1.webp
   python webp_to_jpg.py /path/to/folder

2. 웹 URL에서 다운로드 후 바로 JPG로 저장:
   python webp_to_jpg.py --url https://example.com/sample.webp

3. 다운로드 폴더 실시간 자동 감시 (인터넷에서 다운로드 시 자동 변환):
   python webp_to_jpg.py --watch
   python webp_to_jpg.py --watch --delete-original
"""

import sys
import os
import time
import argparse
from pathlib import Path
from PIL import Image


def convert_image(img: Image.Image) -> Image.Image:
    """투명 배경(RGBA)을 고려하여 RGB로 안전하게 변환"""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgb_img = Image.new("RGB", img.size, (255, 255, 255))
        alpha = img.convert("RGBA").split()[-1]
        rgb_img.paste(img, mask=alpha)
        return rgb_img
    return img.convert("RGB")


def convert_file(webp_path: Path, delete_original: bool = False, quality: int = 95) -> Path:
    """단일 WebP 파일을 JPG로 변환"""
    webp_path = Path(webp_path).resolve()
    if not webp_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {webp_path}")

    jpg_path = webp_path.with_suffix(".jpg")

    with Image.open(webp_path) as img:
        rgb_img = convert_image(img)
        rgb_img.save(jpg_path, "JPEG", quality=quality)

    print(f"변환 완료: {webp_path.name} -> {jpg_path.name}")

    if delete_original:
        os.remove(webp_path)
        print(f"원본 삭제: {webp_path.name}")

    return jpg_path


def convert_folder(folder_path: Path, delete_original: bool = False, quality: int = 95):
    """폴더 내 모든 WebP 파일을 변환"""
    folder = Path(folder_path).resolve()
    webp_files = list(folder.glob("*.webp")) + list(folder.glob("*.WEBP"))

    if not webp_files:
        print(f"변환할 WebP 파일이 없습니다: {folder}")
        return

    print(f"총 {len(webp_files)}개 WebP 파일 변환 시작...")
    for file_path in webp_files:
        try:
            convert_file(file_path, delete_original=delete_original, quality=quality)
        except Exception as e:
            print(f"[오류] {file_path.name} 변환 실패: {e}")


def download_and_convert(url: str, output_path: str = None, quality: int = 95):
    """인터넷 URL에서 WebP를 다운로드하여 JPG로 저장"""
    try:
        import requests
    except ImportError:
        print("requests 라이브러리가 필요합니다: pip install requests")
        return

    import io

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    print(f"다운로드 중: {url}")
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    if not output_path:
        # URL에서 파일명 추출
        url_name = url.split("?")[0].rstrip("/").split("/")[-1]
        base_name = Path(url_name).stem or "downloaded_image"
        output_path = f"{base_name}.jpg"

    out_file = Path(output_path).resolve()

    with Image.open(io.BytesIO(response.content)) as img:
        rgb_img = convert_image(img)
        rgb_img.save(out_file, "JPEG", quality=quality)

    print(f"저장 완료: {out_file}")


def start_watcher(watch_dir: Path, delete_original: bool = False, quality: int = 95):
    """지정한 폴더를 감시하여 WebP 다운로드 시 자동 변환"""
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        print("watchdog 라이브러리가 필요합니다: pip install watchdog")
        return

    class Handler(FileSystemEventHandler):
        def on_created(self, event):
            if not event.is_directory and event.src_path.lower().endswith(".webp"):
                time.sleep(0.5)  # 브라우저 파일 쓰기 완료 대기
                try:
                    convert_file(Path(event.src_path), delete_original=delete_original, quality=quality)
                except Exception as e:
                    print(f"[오류] 변환 실패: {e}")

    observer = Observer()
    observer.schedule(Handler(), path=str(watch_dir), recursive=False)
    observer.start()
    print(f"폴더 감시 시작: {watch_dir}")
    print("종료하려면 Ctrl+C를 누르세요.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n감시를 종료합니다.")
    observer.join()


def main():
    parser = argparse.ArgumentParser(description="WebP to JPG 변환 유틸리티")
    parser.add_argument("target", nargs="?", default=".", help="변환할 파일 또는 폴더 경로 (기본값: 현재 폴더)")
    parser.add_argument("--url", "-u", type=str, help="다운로드할 WebP 이미지 URL")
    parser.add_argument("--output", "-o", type=str, help="출력 JPG 파일 경로 (--url 사용 시)")
    parser.add_argument("--watch", "-w", action="store_true", help="폴더 실시간 감시 모드 활성화")
    parser.add_argument("--watch-dir", type=str, default=str(Path.home() / "Downloads"), help="감시할 폴더 (기본값: ~/Downloads)")
    parser.add_argument("--delete-original", "-d", action="store_true", help="변환 완료 후 원본 WebP 파일 삭제")
    parser.add_argument("--quality", "-q", type=int, default=95, help="JPG 이미지 품질 (1~100, 기본값: 95)")

    args = parser.parse_args()

    # 1. URL 다운로드 모드
    if args.url:
        download_and_convert(args.url, args.output, quality=args.quality)
        return

    # 2. 감시 모드
    if args.watch:
        target_dir = Path(args.watch_dir).expanduser().resolve()
        start_watcher(target_dir, delete_original=args.delete_original, quality=args.quality)
        return

    # 3. 파일 또는 폴더 일괄 변환 모드
    target_path = Path(args.target).expanduser().resolve()
    if target_path.is_file():
        convert_file(target_path, delete_original=args.delete_original, quality=args.quality)
    elif target_path.is_dir():
        convert_folder(target_path, delete_original=args.delete_original, quality=args.quality)
    else:
        print(f"존재하지 않는 경로입니다: {target_path}")


if __name__ == "__main__":
    main()
