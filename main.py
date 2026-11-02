"""
VocaVision — Terminal & Live Video AI Vision Assistive System
==============================================================
Identifies clothing articles, detects dominant colors, recognizes food,
and describes them via real-time bounding boxes and voice speech (pyttsx3).

Supports:
  - Local/USB webcams & virtual cams (DroidCam, Iriun)
  - Mobile phone cameras via IP Webcam (e.g. http://192.168.X.X:8080/video)
  - Pure terminal CLI mode & real-time visual live feed window
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np

from audio_output import initialize_tts, speak, speak_async
from camera_handler import capture_frame, parse_camera_source
from vision_engine import (
    load_models, analyze_image,
    FOOD_CLASSES, CLOTHING_CLASSES,
)

BANNER = r"""
 __      __                 ____   ____.__       .__               
 \ \    / /___   ____ _____ \   \ /   /|__| _____|__| ____   ____  
  \ \  / //  _ \_/ ___\\__  \ \   Y   / |  |/  ___/  |/  _ \ /    \ 
   \ \/ /(  <_> )  \___ / __ \_\     /  |  |\___ \|  (  <_> )   |  \
    \__/  \____/ \___  >____  / \___/   |__|____  >__|\____/|___|  /
                     \/     \/                  \/               \/ 
           AI-Powered Assistive Vision [Live Feed & Terminal]
"""

class VocaVisionApp:
    def __init__(self, camera_source=0, enable_tts=True):
        self.camera_source = parse_camera_source(camera_source)
        self.tts_enabled = enable_tts
        self.tts_engine = None

        if self.tts_enabled:
            try:
                self.tts_engine = initialize_tts()
            except Exception as e:
                print(f"[!] Warning: TTS initialization failed ({e}). Running in muted mode.")
                self.tts_enabled = False

        print("[*] Loading neural network models...")
        (
            self.food_model,
            self.clothing_model,
            self.using_pytorch,
            self.yolo_model,
            self.using_yolo,
        ) = load_models()

        if self.using_yolo:
            self.engine_name = "YOLO v3 (Custom Trained 20-Class Detector)"
        elif self.using_pytorch:
            self.engine_name = "Hybrid v2 (PyTorch MobileNetV2 + TF Food)"
        else:
            self.engine_name = "Legacy v1 (TensorFlow CNN)"

    def speak_text(self, text: str, async_mode: bool = True):
        """Speaks text if TTS is enabled."""
        if not self.tts_enabled or self.tts_engine is None:
            return
        try:
            if async_mode:
                speak_async(self.tts_engine, text)
            else:
                speak(self.tts_engine, text)
        except Exception as e:
            print(f"[!] Speech error: {e}")

    def format_bar(self, prob: float, width: int = 20) -> str:
        """Renders an ASCII bar for terminal confidence display."""
        filled = int(round(prob * width))
        return "█" * filled + "░" * (width - filled)

    # ── Single Frame Inference ────────────────────────────────────────────────
    def analyze_frame_data(self, frame: np.ndarray, save_path: str = None) -> str:
        h, w = frame.shape[:2]
        print(f"[*] Processing image ({w}x{h})...")

        start_t = time.perf_counter()
        description, annotated, food_probs, clothing_probs = analyze_image(
            self.food_model,
            self.clothing_model,
            frame,
            using_pytorch=self.using_pytorch,
            yolo_model=self.yolo_model,
            using_yolo=self.using_yolo,
        )
        latency = (time.perf_counter() - start_t) * 1000

        print("\n" + "═" * 68)
        print("  ANALYSIS RESULT")
        print("═" * 68)
        print(f"  Description : {description}")
        print(f"  Latency     : {latency:.1f} ms  | Engine: {self.engine_name}")
        print("═" * 68)

        if not self.using_yolo:
            print("\n  [Top Food Predictions]")
            for idx in np.argsort(food_probs)[::-1][:3]:
                bar = self.format_bar(food_probs[idx], width=18)
                print(f"    {FOOD_CLASSES[idx]:<16} {food_probs[idx]:>6.1%}  {bar}")

            print("\n  [Top Clothing Predictions]")
            for idx in np.argsort(clothing_probs)[::-1][:3]:
                bar = self.format_bar(clothing_probs[idx], width=18)
                print(f"    {CLOTHING_CLASSES[idx]:<16} {clothing_probs[idx]:>6.1%}  {bar}")
            print("-" * 68)

        if save_path:
            cv2.imwrite(save_path, annotated)
            print(f"[+] Saved annotated frame to: {save_path}")

        self.speak_text(description, async_mode=True)
        return description

    def analyze_image_file(self, file_path: str, save_path: str = None):
        if not os.path.isfile(file_path):
            print(f"[-] Error: File not found: '{file_path}'")
            return

        frame = cv2.imread(file_path)
        if frame is None:
            print(f"[-] Error: Unable to decode image '{file_path}'.")
            return

        out_path = save_path or (os.path.splitext(file_path)[0] + "_result.jpg")
        self.analyze_frame_data(frame, save_path=out_path)

    def analyze_webcam_snapshot(self, save_path: str = None):
        print(f"[*] Accessing camera source: {self.camera_source}...")
        try:
            frame = capture_frame(self.camera_source)
        except Exception as e:
            print(f"[-] Camera Error: {e}")
            self.speak_text("Could not open camera.")
            return

        out_path = save_path or "snapshot_latest_result.jpg"
        self.analyze_frame_data(frame, save_path=out_path)

    # ── Live Visual Video Feed Window ─────────────────────────────────────────
    def run_live_view(self, camera_source=None):
        """
        Opens a live video feed window showing real-time bounding boxes,
        color labels, FPS, and smart audio descriptions.
        """
        src = parse_camera_source(camera_source if camera_source is not None else self.camera_source)
        print(f"\n[*] Starting Live Video Stream from: {src}")
        print("[*] Controls in video window:")
        print("      [Q] or [ESC] : Close video feed & return to menu")
        print("      [SPACE]      : Speak current detection immediately")
        print("      [S]          : Save annotated screenshot")
        print("      [M]          : Toggle voice mute\n")

        cap = cv2.VideoCapture(src)
        # Minimize buffer latency to eliminate stream lag
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print(f"[-] Error: Failed to open camera source '{src}'.")
            print("    -> If using mobile IP Webcam, check phone Wi-Fi IP and start the server.")
            print("    -> If using laptop/USB webcam, try index 0 or 1.")
            return

        window_title = "VocaVision — Live Video Feed"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, 960, 600)

        self.speak_text("Live video feed started.")

        last_spoken_desc = ""
        last_spoken_time = 0.0
        speech_cooldown = 3.5  # Min seconds between auto-announcements

        prev_time = time.time()
        fps = 0.0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    # Brief wait for network stream buffering
                    time.sleep(0.01)
                    continue

                curr_time = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, (curr_time - prev_time)))
                prev_time = curr_time

                # Run fast real-time inference (disables heavy GrabCut/rembg on live feed)
                description, annotated, _, _ = analyze_image(
                    self.food_model,
                    self.clothing_model,
                    frame,
                    using_pytorch=self.using_pytorch,
                    yolo_model=self.yolo_model,
                    using_yolo=self.using_yolo,
                    allow_fallback=False,
                )

                # Audio throttling: announce when new items appear with cooldown
                now = time.time()
                is_meaningful = description and not description.startswith("I could not detect")
                if is_meaningful and (description != last_spoken_desc) and (now - last_spoken_time > speech_cooldown):
                    self.speak_text(description, async_mode=True)
                    last_spoken_desc = description
                    last_spoken_time = now

                # ── Draw HUD Banner ───────────────────────────────────────────
                h, w = annotated.shape[:2]
                hud = annotated.copy()
                cv2.rectangle(hud, (0, 0), (w, 42), (20, 20, 20), -1)
                cv2.addWeighted(hud, 0.75, annotated, 0.25, 0, annotated)

                engine_tag = "YOLO v3 (GPU)" if self.using_yolo else "Hybrid v2"
                tts_tag = "TTS: ON" if self.tts_enabled else "TTS: MUTED"
                hud_text = f"FPS: {fps:.1f}  |  {engine_tag}  |  {tts_tag}  |  Q: Exit  SPACE: Speak  S: Save"
                cv2.putText(annotated, hud_text, (12, 27),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 200), 2, cv2.LINE_AA)

                cv2.imshow(window_title, annotated)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord('q'), ord('Q'), 27):  # Q or ESC
                    print("[*] Closing live video feed.")
                    break
                elif key == ord(' '):  # SPACE -> Force speak
                    if description:
                        print(f"[Manual Trigger] {description}")
                        self.speak_text(description, async_mode=True)
                        last_spoken_time = time.time()
                elif key in (ord('s'), ord('S')):  # S -> Save snapshot
                    snap_name = f"live_snapshot_{int(time.time())}.jpg"
                    cv2.imwrite(snap_name, annotated)
                    print(f"[+] Snapshot saved to: {snap_name}")
                elif key in (ord('m'), ord('M')):  # M -> Toggle mute
                    self.tts_enabled = not self.tts_enabled
                    state = "ENABLED" if self.tts_enabled else "MUTED"
                    print(f"[*] Voice output is now {state}.")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.speak_text("Live video feed closed.")

    # ── Continuous Headless Monitor ───────────────────────────────────────────
    def run_continuous_monitor(self, interval: float = 2.0):
        src = self.camera_source
        print(f"\n[*] Starting Headless Camera Monitor from: {src} (Interval: {interval}s)")
        print("[*] Press Ctrl+C in terminal to stop monitoring.\n")
        self.speak_text("Continuous camera monitoring started.")

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"[-] Failed to open camera source '{src}'.")
            return

        last_description = ""
        scan_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(interval)
                    continue

                scan_count += 1
                description, annotated, _, _ = analyze_image(
                    self.food_model,
                    self.clothing_model,
                    frame,
                    using_pytorch=self.using_pytorch,
                    yolo_model=self.yolo_model,
                    using_yolo=self.using_yolo,
                    allow_fallback=False,
                )

                timestamp = time.strftime("%H:%M:%S")
                if description != last_description:
                    print(f"[{timestamp}] [Scan #{scan_count}] -> {description}")
                    self.speak_text(description, async_mode=True)
                    last_description = description
                    cv2.imwrite("monitor_latest.jpg", annotated)
                else:
                    print(f"[{timestamp}] [Scan #{scan_count}] (Unchanged) {description}", end="\r")

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n[*] Monitoring stopped by user.")
            self.speak_text("Camera monitoring stopped.")
        finally:
            cap.release()

    # ── Interactive Terminal Menu ─────────────────────────────────────────────
    def interactive_menu(self):
        print(BANNER)
        print(f"  Active Engine : {self.engine_name}")
        print(f"  Camera Source : {self.camera_source}")
        print(f"  Voice Output  : {'ENABLED' if self.tts_enabled else 'MUTED'}")
        print("═" * 68)

        self.speak_text("VocaVision is ready.")

        while True:
            tts_status = "ENABLED" if self.tts_enabled else "MUTED"
            print("\nSelect an action:")
            print("  [1] Live Video Stream (Real-Time Bounding Boxes & Color Marking Window)")
            print("  [2] Snap & Analyze Camera (Single Shot in Terminal)")
            print("  [3] Continuous Headless Monitor (Voice Alerts only, No Window)")
            print("  [4] Analyze Image File")
            print(f"  [5] Configure Camera Source (Current: {self.camera_source})")
            print(f"  [6] Toggle Voice Output (Currently: {tts_status})")
            print("  [7] Exit")

            try:
                choice = input("\nEnter choice [1-7]: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting VocaVision.")
                break

            if choice == "1":
                self.run_live_view()
            elif choice == "2":
                self.analyze_webcam_snapshot()
            elif choice == "3":
                try:
                    int_str = input("Scan interval in seconds [default: 2.0]: ").strip()
                    interval = float(int_str) if int_str else 2.0
                except ValueError:
                    interval = 2.0
                self.run_continuous_monitor(interval=interval)
            elif choice == "4":
                path = input("Enter path to image file: ").strip().strip('"').strip("'")
                if path:
                    self.analyze_image_file(path)
            elif choice == "5":
                print("\nEnter camera index (0, 1) or Mobile IP Webcam URL")
                print("Examples:")
                print("  - Laptop / Built-in camera : 0")
                print("  - External USB / Virtual cam : 1")
                print("  - Android IP Webcam app    : http://192.168.1.5:8080/video (or 192.168.1.5:8080)")
                print("  - DroidCam app             : http://192.168.1.5:4747/video")
                new_src = input("New Camera Source: ").strip()
                if new_src:
                    self.camera_source = parse_camera_source(new_src)
                    print(f"[+] Camera source set to: {self.camera_source}")
            elif choice == "6":
                self.tts_enabled = not self.tts_enabled
                if self.tts_enabled and self.tts_engine is None:
                    try:
                        self.tts_engine = initialize_tts()
                    except Exception as e:
                        print(f"[-] TTS error: {e}")
                        self.tts_enabled = False
                print(f"[+] Voice output is now {'ENABLED' if self.tts_enabled else 'MUTED'}.")
            elif choice == "7":
                print("\nThank you for using VocaVision. Goodbye!")
                self.speak_text("Goodbye.")
                break
            else:
                print("[-] Invalid choice. Please enter a number from 1 to 7.")


def main():
    parser = argparse.ArgumentParser(
        description="VocaVision: AI Vision & Speech Assistive System (Live Feed & Terminal)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--live-view", action="store_true", help="Launch live video stream with real-time bounding boxes and color labels")
    parser.add_argument("--image", type=str, help="Path to an image file to analyze")
    parser.add_argument("--webcam", action="store_true", help="Snap a single photo from camera and analyze in terminal")
    parser.add_argument("--monitor", action="store_true", help="Run continuous headless camera monitoring loop")
    parser.add_argument("--interval", type=float, default=2.0, help="Interval (seconds) between scans in monitor mode")
    parser.add_argument("--camera", type=str, default="0", help="Camera index (0, 1) or IP Webcam URL (e.g. http://192.168.1.X:8080/video)")
    parser.add_argument("--save", type=str, default=None, help="File path to save the annotated detection image")
    parser.add_argument("--no-tts", action="store_true", help="Disable text-to-speech audio announcements")

    args = parser.parse_args()

    app = VocaVisionApp(camera_source=args.camera, enable_tts=not args.no_tts)

    if args.live_view:
        app.run_live_view(camera_source=args.camera)
    elif args.image:
        app.analyze_image_file(args.image, save_path=args.save)
    elif args.webcam:
        app.analyze_webcam_snapshot(save_path=args.save)
    elif args.monitor:
        app.run_continuous_monitor(interval=args.interval)
    else:
        # No CLI flags -> interactive menu
        app.interactive_menu()


if __name__ == "__main__":
    main()
