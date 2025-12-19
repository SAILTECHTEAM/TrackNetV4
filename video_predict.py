"""Video Prediction with Sliding Window

Usage:
    python predict/video_predict.py --config config.yaml
"""

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.tracknet_v2 import TrackNet

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get('predict', {
        'model': 'best_model.pth',
        'input': 'input.mp4',
        'output': 'output.mp4',
        'threshold': 0.5,
        'dot_size': 5
    })


class SlidingWindowPredictor:
    def __init__(self, model_path, threshold=0.5, dot_size=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TrackNet().to(self.device)
        self.threshold = threshold
        self.dot_size = dot_size

        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def preprocess_frames(self, frames):
        processed = []
        for frame in frames:
            resized = cv2.resize(frame, (512, 288))
            tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1)
            processed.append(tensor)
        return torch.cat(processed, dim=0).unsqueeze(0).to(self.device)

    def predict_center_frame(self, frames):
        input_tensor = self.preprocess_frames(frames)
        with torch.no_grad():
            output = self.model(input_tensor)
        heatmap = output[0, 1].cpu().numpy()
        return heatmap

    def detect_ball(self, heatmap, original_size):
        if heatmap.max() < self.threshold:
            return None
        max_pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        model_x, model_y = max_pos[1], max_pos[0]
        scale_x = original_size[0] / 512
        scale_y = original_size[1] / 288
        return (int(model_x * scale_x), int(model_y * scale_y))

    def draw_ball(self, frame, ball_pos):
        if ball_pos is not None:
            cv2.circle(frame, ball_pos, self.dot_size, (0, 0, 255), -1)
        return frame

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {total_frames} frames, {width}x{height}, {fps:.1f}FPS")
        print(f"Device: {self.device}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_buffer = []
        processed_count = 0
        detected_count = 0

        ret, frame1 = cap.read()
        ret, frame2 = cap.read()
        if not ret:
            print("Error: Cannot read video frames")
            return

        frame_buffer = [frame1, frame2]

        with tqdm(total=total_frames - 2, desc="Processing", unit="frame") as pbar:
            while True:
                ret, frame3 = cap.read()
                if not ret:
                    break

                frame_buffer.append(frame3)

                heatmap = self.predict_center_frame(frame_buffer)
                ball_pos = self.detect_ball(heatmap, (width, height))

                center_frame = frame_buffer[1].copy()
                processed_frame = self.draw_ball(center_frame, ball_pos)
                out.write(processed_frame)

                if ball_pos is not None:
                    detected_count += 1
                processed_count += 1

                frame_buffer.pop(0)
                pbar.update(1)

        out.write(self.draw_ball(frame_buffer[0].copy(), None))
        out.write(self.draw_ball(frame_buffer[1].copy(), None))
        processed_count += 2

        cap.release()
        out.release()

        detection_rate = (detected_count / processed_count) * 100 if processed_count > 0 else 0
        print(f"Completed: {detected_count}/{processed_count} frames detected ({detection_rate:.1f}%)")
        print(f"Output: {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default=str(CONFIG_PATH))
    parser.add_argument('--model', type=str)
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--dot_size', type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_path = args.model or cfg.get('model', 'best_model.pth')
    input_video = args.input or cfg.get('input', 'input.mp4')
    output_video = args.output or cfg.get('output', 'output.mp4')
    threshold = args.threshold or cfg.get('threshold', 0.5)
    dot_size = args.dot_size or cfg.get('dot_size', 5)

    print("=" * 60)
    print("Shuttlecock Detection - Sliding Window")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Input: {input_video}")
    print(f"Output: {output_video}")
    print(f"Threshold: {threshold}")
    print(f"Dot size: {dot_size}")
    print("-" * 60)

    if not os.path.exists(input_video):
        print(f"Error: Input video not found: {input_video}")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)

    predictor = SlidingWindowPredictor(model_path, threshold, dot_size)
    predictor.process_video(input_video, output_video)

    print("=" * 60)
    print("Done!")
    print("=" * 60)
