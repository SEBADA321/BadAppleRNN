# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple, Dict
from torchvision.io import read_image, ImageReadMode
import matplotlib.pyplot as plt
from torchinfo import summary
from torch.amp import autocast, GradScaler
import os
import cv2
import time
import torch
import numpy as np
from tqdm.auto import tqdm

def get_horizon_schedule(epoch: int) -> int:
    phase = epoch // 50
    K = 2 * (2 ** phase)
    return min(K, T - 1, 512)
    
def get_split_params(models, debug: bool = False):
    split_params = {'adam': [], 'muon': []}

    for model in models:
        for name, param in model.named_parameters():
            if name.endswith(".weight") and 'W_h' in name:
                split_params['muon'].append((name, param))
                if debug: print(name, param.shape, 'muon')
            elif 'bias' in name:
                split_params['adam'].append((name, param))
                if debug: print(name, param.shape, 'adam')
            elif name.endswith(".weight") and 'W_h' not in name:
                split_params['adam'].append((name, param))
                if debug: print(name, param.shape, 'adam')
            elif 'param' in name:
                split_params['adam'].append((name, param))
                if debug: print(name, param.shape, 'adam')
            else:
                    if debug: print(name, param.shape, 'NONE_MISSING')
        
    return split_params

print(torch.cuda.is_available())

# %%
class CustomTransitionFunction(nn.Module):
    def __init__(self,hidden_size, init_weights: bool = True):
        super(CustomTransitionFunction, self).__init__()
        self.hidden_size=hidden_size
        self.W_hf=nn.Linear(hidden_size,hidden_size)
        self.W_hi=nn.Linear(hidden_size,hidden_size)
        self.W_ho=nn.Linear(hidden_size,hidden_size)
        self.W_hg=nn.Linear(hidden_size,hidden_size)
        if init_weights:
            self.init_weights()

    def init_weights(self) -> None:
        for name, param in self.named_parameters():
            if name.endswith(".weight") and 'W_h' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        self.W_hf.bias.data.fill_(1.0)
        return

    def forward(self,h,c):
        # Calculate the gates
        f=F.sigmoid(self.W_hf(h))
        i=F.sigmoid(self.W_hi(h))
        o=F.sigmoid(self.W_ho(h))

        # Calculate the cell state
        c_t=torch.tanh(self.W_hg(h))
        c=f*c+i*c_t

        # Calculate the hidden state
        h=o*torch.tanh(c)
        return h,c

class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=True,
        )

        self.norm = nn.GroupNorm(
            num_groups=1,
            num_channels=out_channels,
        )

        self.act = nn.SiLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class FrameDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int = 64,
        output_height: int = 384,
        output_width: int = 512,
    ):
        super().__init__()

        self.output_height = output_height
        self.output_width = output_width

        self.base_channels = 8
        self.base_height = output_height // 16
        self.base_width = output_width // 16

        if output_height % 16 != 0:
            raise ValueError(
                "output_height must be divisible by 16"
            )

        if output_width % 16 != 0:
            raise ValueError(
                "output_width must be divisible by 16"
            )

        self.projection = nn.Linear(
            hidden_size,
            self.base_channels
            * self.base_height
            * self.base_width,
        )

        self.block1 = DepthwiseSeparableConv(8, 16)
        self.block2 = DepthwiseSeparableConv(16, 16)
        self.block3 = DepthwiseSeparableConv(16, 8)

        self.output = nn.Conv2d(
            8,
            1,
            kernel_size=1,
        )

    def forward(self, x):
        """
        Accepts:

            [B, hidden]
            [T, B, hidden]

        Returns:

            [B, 1, H, W]
            [T, B, 1, H, W]

        Output is logits. Do NOT apply sigmoid here.
        """

        original_shape = x.shape[:-1]

        x = x.reshape(-1, x.shape[-1])

        x = self.projection(x)

        x = x.reshape(
            -1,
            self.base_channels,
            self.base_height,
            self.base_width,
        )

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        x = self.block1(x)

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        x = self.block2(x)

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        x = self.block3(x)

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        x = self.output(x)

        x = x.reshape(
            *original_shape,
            1,
            self.output_height,
            self.output_width,
        )

        return x

class TrainingTableEmbeddings(nn.Module):
    def __init__(self, h_dim: int, c_dim: int, entries: int, init_weights: bool = True):
        super(TrainingTableEmbeddings, self).__init__()
        self.h_dim: int = h_dim
        self.c_dim: int = c_dim
        self.entries: int = entries

        self.h_table = nn.Embedding(num_embeddings=self.entries, embedding_dim=self.h_dim, max_norm=None)
        self.c_table = nn.Embedding(num_embeddings=self.entries, embedding_dim=self.c_dim, max_norm=None)

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.xavier_normal_(self.h_table.weight)
        nn.init.xavier_normal_(self.c_table.weight)

    def forward(self, indices):
        return self.h_table(indices), self.c_table(indices)


# %%
class BadApple(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str | Path):
        self.dataset_path = Path(dataset_path).resolve()

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset path {self.dataset_path} does not exist"
            )

        self.frame_paths = sorted(
            p for p in self.dataset_path.iterdir() if p.is_file()
        )

        if not self.frame_paths:
            raise ValueError(f"No files found in {self.dataset_path}")

        self.frames = torch.stack([
            read_image(str(path), mode=ImageReadMode.GRAY)
            for path in self.frame_paths
        ])

    def __len__(self):
        return self.frames.shape[0]

    def __getitem__(self, index):
        return self.frames[index].float().div_(255.0)

dataset = BadApple(dataset_path='/home/tumi/Desktop/BadApple/frames')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIDDEN_SIZE = 64
H_DIM, C_DIM = HIDDEN_SIZE, HIDDEN_SIZE
ENTRIES = len(dataset) # or T

tte = TrainingTableEmbeddings(h_dim=H_DIM, c_dim=C_DIM, entries=ENTRIES).to(device)
ctf = CustomTransitionFunction(hidden_size=HIDDEN_SIZE).to(device)
fd = FrameDecoder(hidden_size=HIDDEN_SIZE, output_height=384, output_width=512).to(device)

# 2. Load the checkpoint file
checkpoint_path = "runs/last_model.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)

# 3. Load state dicts into each model
tte.load_state_dict(checkpoint["tte"])
ctf.load_state_dict(checkpoint["ctf"])
fd.load_state_dict(checkpoint["fd"])

# 4. Set to evaluation mode
tte.eval()
ctf.eval()
fd.eval()

print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")


# %%
def visualize_autonomous_rollout(
    tte, 
    ctf, 
    fd, 
    dataset, 
    start_frame: int = 0, 
    num_frames: int | None = None, 
    fps: int = 30, 
    save_video_path: str | None = None,
    device: str = "cuda"
):
    """
    Real-time OpenCV visualizer comparing Ground Truth vs Autonomous Rollout.
    """
    tte.eval()
    ctf.eval()
    fd.eval()

    total_dataset_frames = len(dataset)
    if num_frames is None:
        num_frames = total_dataset_frames - start_frame
    else:
        num_frames = min(num_frames, total_dataset_frames - start_frame)

    # Window setup
    window_name = "Bad Apple!! Autonomous Rollout Visualizer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 460)

    # Optional MP4 Video Writer
    video_writer = None
    if save_video_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # 512 + 512 = 1024 width, 384 + 60 (header) = 444 height
        video_writer = cv2.VideoWriter(save_video_path, fourcc, fps, (1024, 444))
        print(f"Recording video to: {save_video_path}")

    # Fetch initial state from the table at start_frame
    with torch.no_grad():
        t0 = torch.tensor([start_frame], device=device)
        h_curr = tte.h_table(t0)
        c_curr = tte.c_table(t0)

    paused = False
    step_forward = False
    frame_idx = 0
    prev_time = time.time()

    print("\nVisualizer running! Controls:")
    print(" [SPACE] : Pause / Resume")
    print(" [D]     : Step 1 frame forward (while paused)")
    print(" [R]     : Restart rollout from start")
    print(" [S]     : Save comparison snapshot")
    print(" [Q/ESC] : Quit\n")

    try:
        while frame_idx < num_frames:
            current_t = start_frame + frame_idx

            # 1. Ground Truth Frame (from dataset)
            gt_tensor = dataset[current_t].squeeze() # [384, 512]
            gt_img = (gt_tensor.numpy() * 255).astype(np.uint8)
            gt_bgr = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2BGR)

            # 2. Autonomous Rollout Frame (from fd(h_curr))
            with torch.no_grad():
                pred_logit = fd(h_curr)
                pred_tensor = torch.sigmoid(pred_logit).squeeze().cpu()
                pred_img = (pred_tensor.numpy() * 255).astype(np.uint8)
                pred_bgr = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)

                # Compute frame MSE
                mse_error = torch.mean((pred_tensor - gt_tensor) ** 2).item()

                # Step latent state if not paused or stepping
                if not paused or step_forward:
                    h_curr, c_curr = ctf(h_curr, c_curr)

            # 3. Build UI Layout
            # Side-by-side canvas
            canvas = np.hstack([gt_bgr, pred_bgr])

            # Header Dashboard (60px high)
            header = np.zeros((60, canvas.shape[1], 3), dtype=np.uint8)
            header[:] = (25, 25, 25) # Dark background

            # Metrics
            fps_actual = 1.0 / max(1e-5, (time.time() - prev_time))
            prev_time = time.time()
            time_str = f"{current_t // (fps * 60):02d}:{(current_t // fps) % 60:02d}.{(current_t % fps):02d}"
            
            # Left Header info
            info_left = f"Frame: {current_t:04d}/{total_dataset_frames} | Time: {time_str} | FPS: {fps_actual:.1f}"
            cv2.putText(header, info_left, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            # Right Header info
            status_str = "[PAUSED]" if paused else "[RUNNING]"
            status_color = (0, 165, 255) if paused else (0, 255, 0)
            info_right = f"MSE: {mse_error:.5f}  {status_str}"
            cv2.putText(header, info_right, (canvas.shape[1] - 320, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)

            # Panel labels on image
            cv2.putText(canvas, "Ground Truth", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "Autonomous Rollout (Recursive ctf)", (512 + 15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            # Divider line between frames
            cv2.line(canvas, (512, 0), (512, 384), (50, 50, 50), 2)

            # Combine Header + Canvas
            full_frame = np.vstack([header, canvas])

            # Write to video if enabled
            if video_writer:
                video_writer.write(full_frame)

            # Display
            cv2.imshow(window_name, full_frame)

            # Reset single-step trigger
            step_forward = False

            # Waitkey handling & interaction
            delay = 1 if paused else max(1, int(1000 / fps))
            key = cv2.waitKey(delay) & 0xFF

            if key == ord('q') or key == 27:  # Q or ESC
                print("Visualizer stopped by user.")
                break
            elif key == ord(' '):  # SPACE toggles pause
                paused = not paused
            elif key == ord('d') or key == 83:  # D or Right Arrow to step
                if paused:
                    step_forward = True
                    frame_idx += 1
            elif key == ord('r'):  # R restarts
                frame_idx = 0
                with torch.no_grad():
                    t0 = torch.tensor([start_frame], device=device)
                    h_curr = tte.h_table(t0)
                    c_curr = tte.c_table(t0)
                print("Restarted rollout from t=0")
                continue
            elif key == ord('s'):  # S saves snapshot
                snap_name = f"snapshot_frame_{current_t}.png"
                cv2.imwrite(snap_name, full_frame)
                print(f"Saved snapshot: {snap_name}")

            if not paused:
                frame_idx += 1

    finally:
        if video_writer:
            video_writer.release()
            print(f"Video saved to {save_video_path}")
        cv2.destroyAllWindows()
        cv2.waitKey(1)


# %%
visualize_autonomous_rollout(
    tte=tte,
    ctf=ctf,
    fd=fd,
    dataset=dataset,
    start_frame=0,
    num_frames=len(dataset),     # Unroll entire Bad Apple video
    fps=30,                      # 30 FPS playback
    save_video_path="bad_apple_rollout.mp4", # Set to None if you don't want to record
    device=device
)


# %%
def visualize_table_reconstruction(
    tte, 
    fd, 
    dataset, 
    start_frame: int = 0, 
    num_frames: int = None, 
    fps: int = 30, 
    save_video_path: str = None,
    device: str = "cuda"
):
    """
    Real-time visualizer for Direct Table Reconstruction (Embedding Table -> FrameDecoder).
    No RNN/transition function is used.
    """
    tte.eval()
    fd.eval()

    total_dataset_frames = len(dataset)
    if num_frames is None:
        num_frames = total_dataset_frames - start_frame
    else:
        num_frames = min(num_frames, total_dataset_frames - start_frame)

    window_name = "Bad Apple!! Direct Table Reconstruction (Decoder Only)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 460)

    # Trackbar state
    trackbar_state = {"frame_idx": start_frame, "updated_by_user": False}

    def on_trackbar(val):
        trackbar_state["frame_idx"] = val
        trackbar_state["updated_by_user"] = True

    cv2.createTrackbar("Timeline", window_name, start_frame, total_dataset_frames - 1, on_trackbar)

    # Optional MP4 Video Writer
    video_writer = None
    if save_video_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(save_video_path, fourcc, fps, (1024, 444))
        print(f"Recording video to: {save_video_path}")

    paused = False
    current_t = start_frame
    prev_time = time.time()

    print("\nVisualizer running! Controls:")
    print(" [Trackbar] : Scrub to any frame")
    print(" [SPACE]    : Pause / Play")
    print(" [A] / [D]  : Step 1 frame backward / forward")
    print(" [R]        : Jump to start (t=0)")
    print(" [S]        : Save comparison screenshot")
    print(" [Q/ESC]    : Quit\n")

    try:
        while current_t < total_dataset_frames:
            # Handle user scrubbing with trackbar
            if trackbar_state["updated_by_user"]:
                current_t = trackbar_state["frame_idx"]
                trackbar_state["updated_by_user"] = False

            # 1. Ground Truth Frame
            gt_tensor = dataset[current_t].squeeze()  # [384, 512]
            gt_img = (gt_tensor.numpy() * 255).astype(np.uint8)
            gt_bgr = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2BGR)

            # 2. Table Lookup + Decoder Reconstruction: fd(tte.h_table(t))
            with torch.no_grad():
                t_tensor = torch.tensor([current_t], device=device)
                h_table_val = tte.h_table(t_tensor)
                
                pred_logit = fd(h_table_val)
                pred_tensor = torch.sigmoid(pred_logit).squeeze().cpu()
                pred_img = (pred_tensor.numpy() * 255).astype(np.uint8)
                pred_bgr = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)

                mse_error = torch.mean((pred_tensor - gt_tensor) ** 2).item()

            # 3. Canvas Composition
            canvas = np.hstack([gt_bgr, pred_bgr])

            # Header Dashboard (60px)
            header = np.zeros((60, canvas.shape[1], 3), dtype=np.uint8)
            header[:] = (25, 25, 25)

            fps_actual = 1.0 / max(1e-5, (time.time() - prev_time))
            prev_time = time.time()
            time_str = f"{current_t // (fps * 60):02d}:{(current_t // fps) % 60:02d}.{(current_t % fps):02d}"

            info_left = f"Frame: {current_t:04d}/{total_dataset_frames} | Time: {time_str} | FPS: {fps_actual:.1f}"
            cv2.putText(header, info_left, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            status_str = "[PAUSED]" if paused else "[PLAYING]"
            status_color = (0, 165, 255) if paused else (0, 255, 0)
            info_right = f"MSE: {mse_error:.5f}  {status_str}"
            cv2.putText(header, info_right, (canvas.shape[1] - 330, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)

            # Labels on images
            cv2.putText(canvas, "Ground Truth", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "Direct Table Reconstruction (fd(h_t))", (512 + 15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.line(canvas, (512, 0), (512, 384), (50, 50, 50), 2)

            full_frame = np.vstack([header, canvas])

            if video_writer:
                video_writer.write(full_frame)

            cv2.imshow(window_name, full_frame)
            cv2.setTrackbarPos("Timeline", window_name, current_t)

            # WaitKey Handling
            delay = 1 if paused else max(1, int(1000 / fps))
            key = cv2.waitKey(delay) & 0xFF

            if key == ord('q') or key == 27:  # Q or ESC
                print("Visualizer stopped.")
                break
            elif key == ord(' '):  # SPACE
                paused = not paused
            elif key == ord('d') or key == 83:  # D or Right Arrow
                if paused and current_t < total_dataset_frames - 1:
                    current_t += 1
            elif key == ord('a') or key == 81:  # A or Left Arrow
                if paused and current_t > 0:
                    current_t -= 1
            elif key == ord('r'):  # R restart
                current_t = 0
            elif key == ord('s'):  # S snapshot
                snap_name = f"table_recon_frame_{current_t}.png"
                cv2.imwrite(snap_name, full_frame)
                print(f"Saved snapshot: {snap_name}")

            if not paused:
                current_t += 1

    finally:
        if video_writer:
            video_writer.release()
            print(f"Video saved to {save_video_path}")
        cv2.destroyAllWindows()
        cv2.waitKey(1)


# %%
visualize_table_reconstruction(
    tte=tte,
    fd=fd,
    dataset=dataset,
    start_frame=0,
    num_frames=len(dataset),
    fps=30,
    save_video_path="bad_apple_table_reconstruction.mp4", # Or None
    device=device.type
)



