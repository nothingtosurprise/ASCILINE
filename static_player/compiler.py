import argparse
import os
import struct
import subprocess
import numpy as np

import sys

# Add parent directory to sys.path so we can import the core engine from the root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the existing engine components
from ascii_video_player2 import VideoDecoder, AsciiMapper
from codec import encode_frame, DEFAULT_LEVEL, ProfileEncoder

def extract_audio(video_path: str, output_path: str):
    print(f"[Audio] Attempting to extract audio to {output_path}...")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path, 
                "-vn", "-acodec", "libmp3lame", "-ab", "128k", "-ar", "44100", 
                output_path
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        print("[Audio] Audio extracted successfully.")
    except FileNotFoundError:
        print("[Audio] WARNING: FFmpeg not found on this system.")
        print("[Audio] The video will be compiled silently. Please install FFmpeg for audio support.")
    except subprocess.CalledProcessError:
        print("[Audio] WARNING: FFmpeg failed to extract audio. The video will be compiled silently.")

def get_video_dimensions(decoder):
    # Quick utility to get dims
    return decoder.vid_w, decoder.vid_h

def compile_video(args):
    video_path = args.video
    if not os.path.exists(video_path):
        print(f"Error: File not found -> {video_path}")
        return

    out_name = args.out or os.path.splitext(os.path.basename(video_path))[0]
    out_dir = "."
    os.makedirs(out_dir, exist_ok=True)
    
    ascf_path = os.path.join(out_dir, f"{out_name}.ascf")
    audio_path = os.path.join(out_dir, f"{out_name}.mp3")
    
    pixel_mode = args.pixel
    render_mode = args.mode
    cols = args.cols
    tolerance = args.tolerance
    level = 9 if args.hard else 3
    pixel_qb = args.quantize  # bits to drop per channel in pixel mode (0 = lossless)

    # 1. Extract audio
    extract_audio(video_path, audio_path)

    # 2. Setup Decoder
    print(f"[Video] Initializing decoder for {video_path}...")
    decoder = VideoDecoder(video_path, cols, args.rows, skip_gray=pixel_mode)
    
    # Calculate rows (from stream_server logic)
    vid_w, vid_h = get_video_dimensions(decoder)
    ratio = vid_w / max(vid_h, 1)
    if args.rows == 0:
        if pixel_mode:
            rows = max(1, round(cols / ratio))
        else:
            rows = max(1, round(cols / ratio / 2))
    else:
        rows = args.rows

    # Update decoder with actual rows if it was auto-calculated
    # Actually, VideoDecoder doesn't allow changing rows after init, so we must recreate if rows was 0
    if args.rows == 0:
        decoder.release()
        decoder = VideoDecoder(video_path, cols, rows, skip_gray=pixel_mode)

    mapper = AsciiMapper()
    source_fps = decoder.fps
    
    # Decimation logic
    MAX_FPS = 30
    if source_fps > MAX_FPS:
        skip_n = round(source_fps / MAX_FPS)
        effective_fps = source_fps / skip_n
    else:
        skip_n = 1
        effective_fps = source_fps

    print(f"[Compiler] Dimensions: {cols}x{rows} | Mode: {render_mode} | Pixel: {pixel_mode} | FPS: {effective_fps:.1f}")

    # Opt-in lossy DCT profile (tag 4): a separate profile, not a tag-race competitor.
    profile_enc = None
    if args.profile:
        if not pixel_mode:
            print("Error: --profile requires --pixel (the lossy DCT profile is pixel mode only).")
            decoder.release()
            return
        # 8x8 blocks over 4:2:0 planes require cols/rows to be multiples of 16.
        pc = ((cols + 15) // 16) * 16
        pr = ((rows + 15) // 16) * 16
        if pc != cols or pr != rows:
            cols, rows = pc, pr
            decoder.release()
            decoder = VideoDecoder(video_path, cols, rows, skip_gray=pixel_mode)
        profile_enc = ProfileEncoder(cols, rows, args.qf)
        print(f"[Compiler] Lossy DCT profile (tag 4) ON | QF={args.qf} | grid padded to {cols}x{rows}")

    char_byte_lut = np.array([ord(c) for c in mapper._lut], dtype=np.uint8)
    qb = {6: 0, 5: 2, 4: 3, 3: 5, 2: 6}.get(render_mode, 0)
    
    frame_buf = np.empty((rows, cols, 4), dtype=np.uint8) if render_mode > 1 else None

    with open(ascf_path, "wb") as f_out:
        # Write Header (14 bytes)
        # Magic: 'ASCF' (4)
        # FPS: float32 (4)
        # Mode: uint8 (1)
        # Pixel: uint8 (1)
        # Cols: uint16 (2)
        # Rows: uint16 (2)
        header = struct.pack(">4sfBBHH", b"ASCF", effective_fps, render_mode, int(pixel_mode), cols, rows)
        f_out.write(header)
        
        frame_index = 0
        prev_frame = None
        bytes_written = 14
        
        try:
            while True:
                for _ in range(skip_n - 1):
                    if not decoder.grab():
                        break
                
                try:
                    gray_frame, bgr_frame = next(decoder)
                except StopIteration:
                    break

                if pixel_mode:
                    frame_px = np.ascontiguousarray(bgr_frame)
                    if pixel_qb > 0:
                        frame_px = (frame_px >> pixel_qb) << pixel_qb
                    if profile_enc is not None:
                        msg, prev_frame = profile_enc.encode(frame_px)
                    else:
                        msg, prev_frame = encode_frame(
                            frame_px,
                            prev_frame, frame_index, level=level, tolerance=tolerance
                        )
                else:
                    indices = np.floor_divide(gray_frame, max(1, 256 // mapper._n))
                    np.clip(indices, 0, mapper._n - 1, out=indices)
                    
                    if render_mode == 1:
                        char_matrix = mapper._lut[indices]
                        lines = [''.join(r) for r in char_matrix]
                        payload = (f"{frame_index}\n" + '\n'.join(lines)).encode('utf-8')
                        msg = payload # For mode 1, we just pack the string as bytes
                    else:
                        char_codes = char_byte_lut[indices]
                        rgb = bgr_frame[:, :, ::-1]
                        if qb > 0:
                            rgb = (rgb >> qb) << qb
                        frame_buf[:, :, 0] = char_codes
                        frame_buf[:, :, 1:] = rgb
                        
                        msg, prev_frame = encode_frame(
                            frame_buf, prev_frame, frame_index, level=level, tolerance=tolerance
                        )
                
                # Write length prefix (uint32) + payload
                f_out.write(struct.pack(">I", len(msg)))
                f_out.write(msg)
                
                bytes_written += 4 + len(msg)
                frame_index += 1
                
                if frame_index % 50 == 0:
                    print(f"\r[Compiler] Compiled {frame_index} frames ({(bytes_written / 1024 / 1024):.2f} MB)...", end="")
        
        finally:
            decoder.release()

    print(f"\n[Compiler] Done! Total frames: {frame_index}. Output saved to {ascf_path} ({(bytes_written / 1024 / 1024):.2f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASCILINE Static Compiler")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--cols", type=int, default=300, help="Grid columns (default 300)")
    parser.add_argument("--rows", type=int, default=0, help="Grid rows (0 = auto)")
    parser.add_argument("--mode", type=int, default=6, choices=[1, 2, 3, 4, 5, 6], help="Render mode: 1=B&W  2=64c  3=512c  4=32Kc  5=262Kc  6=16M Ultra")
    parser.add_argument("--pixel", action="store_true", help="Pixel mode (no characters)")
    parser.add_argument("--tolerance", type=int, default=0, help="Color drift tolerance (0=lossless)")
    parser.add_argument("--hard", action="store_true", help="Use maximum zlib compression (level 9) instead of default (level 3). Slower but smaller file.")
    parser.add_argument("--quantize", type=int, default=0, choices=[0, 1, 2, 3], metavar="0-3", help="Pixel mode color quantization: bits to drop per channel (0=lossless, 1=slight, 2=medium, 3=aggressive). Reduces file size.")
    parser.add_argument("--profile", action="store_true", help="Opt-in lossy DCT compression profile (tag 4, pixel mode). ~4-5x smaller at matched quality. Implies --pixel.")
    parser.add_argument("--qf", type=int, default=70, help="Profile quality factor 1-100 (higher = better quality and larger file). Default 70.")
    parser.add_argument("--out", type=str, default="", help="Output base name")
    
    args = parser.parse_args()
    compile_video(args)
