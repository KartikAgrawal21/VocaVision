"""
Quick terminal test script for VocaVision models.
Tests a single image file (or webcam snapshot) without any GUI.

Usage:
    # Test with an image file:
    python test_image.py --image path/to/your/image.jpg

    # Test with a live webcam snapshot:
    python test_image.py --webcam
"""
import os
import sys
import argparse
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from vision_engine import load_models, analyze_image, FOOD_CLASSES, CLOTHING_CLASSES
from camera_handler import capture_frame

def test_image_file(path, food_model, clothing_model, using_pytorch=True, yolo_model=None, using_yolo=False):
    """Tests a single image file and prints formatted terminal results."""
    frame = cv2.imread(path)
    if frame is None:
        print(f"[-] ERROR: Could not read image at '{path}'")
        return

    print(f"\n[*] Analyzing image: {path} (Dimensions: {frame.shape[1]}x{frame.shape[0]})")

    description, annotated, food_probs, clothing_probs = analyze_image(
        food_model, clothing_model, frame,
        using_pytorch=using_pytorch,
        yolo_model=yolo_model,
        using_yolo=using_yolo
    )

    print("\n" + "=" * 60)
    print(f"DESCRIPTION: {description}")
    print("=" * 60)

    if using_yolo:
        print("\n[Detection Model: Custom YOLOv8s]")
    else:
        print("\nTop Food Predictions:")
        top_food = np.argsort(food_probs)[::-1][:5]
        for idx in top_food:
            bar = "█" * int(food_probs[idx] * 25)
            print(f"  {FOOD_CLASSES[idx]:<16} {food_probs[idx]:>6.1%}  {bar}")

        print("\nTop Clothing Predictions:")
        top_cloth = np.argsort(clothing_probs)[::-1][:5]
        for idx in top_cloth:
            bar = "█" * int(clothing_probs[idx] * 25)
            print(f"  {CLOTHING_CLASSES[idx]:<16} {clothing_probs[idx]:>6.1%}  {bar}")

    # Save annotated result image
    out_path = os.path.splitext(path)[0] + "_result.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"\n[+] Annotated result saved to: {out_path}\n")

def test_webcam(food_model, clothing_model, using_pytorch=True, yolo_model=None, using_yolo=False):
    """Captures a frame from the webcam and analyzes it in terminal."""
    print("\n[*] Capturing snapshot from webcam...")
    try:
        frame = capture_frame(0)
    except Exception as e:
        print(f"[-] Webcam Error: {e}")
        return

    print(f"[*] Snapshot captured ({frame.shape[1]}x{frame.shape[0]}). Running inference...")
    description, annotated, food_probs, clothing_probs = analyze_image(
        food_model, clothing_model, frame,
        using_pytorch=using_pytorch,
        yolo_model=yolo_model,
        using_yolo=using_yolo
    )

    print("\n" + "=" * 60)
    print(f"DESCRIPTION: {description}")
    print("=" * 60)

    out_path = "webcam_snapshot_result.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"\n[+] Annotated snapshot saved to: {out_path}\n")

def main():
    parser = argparse.ArgumentParser(description="VocaVision Headless Terminal Model Tester")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to an image file to test")
    group.add_argument("--webcam", action="store_true", help="Capture snapshot from webcam")
    args = parser.parse_args()

    print("[*] Loading VocaVision models...")
    food_model, clothing_model, using_pytorch, yolo_model, using_yolo = load_models()

    if args.image:
        test_image_file(args.image, food_model, clothing_model, using_pytorch, yolo_model, using_yolo)
    else:
        test_webcam(food_model, clothing_model, using_pytorch, yolo_model, using_yolo)

if __name__ == "__main__":
    main()
