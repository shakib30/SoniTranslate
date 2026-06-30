"""
Lip Sync processing using Wav2Lip with GPU support.

This module provides a robust interface for syncing video lips with audio using the
Wav2Lip model. It includes automatic GPU detection, configurable timeouts, retry
logic for model downloads, and proper cleanup of temporary files.

Usage:
    from soni_translate.lip_sync import Wav2LipProcessor
    
    processor = Wav2LipProcessor(device="cuda")  # or "cpu"/"auto"
    result = processor.process(video_path="input.mp4", audio_path="audio.wav")
"""

import os
import shutil
import subprocess
import logging
import tempfile
import time
from pathlib import Path
from typing import Optional, Union
from urllib.request import Request, urlopen
import glob

logger = logging.getLogger(__name__)

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False


class Wav2LipProcessor:
    """
    A robust Wav2Lip processor with GPU support, retry logic, and proper cleanup.

    Attributes:
        device (str): Device to run inference on ('cuda' or 'cpu').
        wav2lip_dir (str): Path to the Wav2Lip repository.
        model_path (str): Path to the model weights.
        temp_dir (str): Temporary directory for processing intermediate files.
    """

    def __init__(
        self,
        device: str = "auto",
        wav2lip_dir: str = "Wav2Lip",
        timeout_clone: int = 300,
        timeout_pip: int = 120,
        timeout_inference: int = 3600
    ):
        """
        Initialize the Wav2Lip processor.

        Args:
            device: 'cuda', 'cpu', or 'auto' (auto-detects GPU).
            wav2lip_dir: Path to the Wav2Lip repository directory.
            timeout_clone: Max seconds for git clone (default 5 min).
            timeout_pip: Max seconds for pip install (default 2 min).
            timeout_inference: Max seconds for inference (default 1 hour).
        """
        self.wav2lip_dir = wav2lip_dir
        self.model_path = os.path.join(wav2lip_dir, "checkpoints")
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if CUDA_AVAILABLE else "cpu"
        else:
            self.device = device if device in ("cuda", "cpu") else "cpu"
        
        self.timeout_clone = timeout_clone
        self.timeout_pip = timeout_pip
        self.timeout_inference = timeout_inference
        
        # Create a unique temp directory under 'temp/'
        os.makedirs("temp", exist_ok=True)
        self.temp_dir = tempfile.mkdtemp(prefix="wav2lip_", dir="temp")
        
        logger.info(f"Wav2LipProcessor initialized (device={self.device}, temp={self.temp_dir})")

    def _install_wav2lip(self) -> None:
        """Clone Wav2Lip repository with timeout and retry."""
        if os.path.exists(self.wav2lip_dir):
            logger.info("Wav2Lip already exists, skipping clone.")
            return

        logger.info("Cloning Wav2Lip repository...")
        for attempt in range(1, 4):
            try:
                result = subprocess.run(
                    ["git", "clone", "https://github.com/Rudrabha/Wav2Lip.git", self.wav2lip_dir],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_clone
                )
                result.check_returncode()
                logger.info("Wav2Lip cloned successfully.")
                return
            except subprocess.TimeoutExpired:
                logger.warning(f"Git clone timed out (attempt {attempt}/3).")
                self._clean_clone_dir()
            except subprocess.CalledProcessError as e:
                logger.error(f"Git clone failed: {e.stderr}")
                self._clean_clone_dir()
                if attempt == 3:
                    raise RuntimeError("Failed to clone Wav2Lip after 3 attempts.")
            except FileNotFoundError:
                raise RuntimeError("Git is not installed or not in PATH.")

    def _clean_clone_dir(self) -> None:
        """Remove incomplete clone directory."""
        if os.path.exists(self.wav2lip_dir):
            shutil.rmtree(self.wav2lip_dir)
            logger.debug(f"Removed incomplete clone: {self.wav2lip_dir}")

    def _download_model(self) -> str:
        """
        Download wav2lip_gan.pth with retry logic and proper cleanup on failure.

        Returns:
            Path to the downloaded model file.
        """
        os.makedirs(self.model_path, exist_ok=True)
        dest_path = os.path.join(self.model_path, "wav2lip_gan.pth")

        if os.path.exists(dest_path):
            logger.info("Wav2Lip model already exists.")
            return dest_path

        urls = [
            "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth",
            "https://huggingface.co/rippertnt/wav2lip/resolve/main/wav2lip_gan.pth"
        ]

        last_error = None
        for attempt, url in enumerate(urls, 1):
            logger.info(f"Downloading Wav2Lip model (attempt {attempt}/{len(urls)}) from: {url}")
            try:
                req = Request(url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
                })
                with urlopen(req) as response:
                    total_size = int(response.headers.get('Content-Length', 0))
                    logger.info(f"Model size: {total_size / (1024*1024):.2f} MB")

                    with open(dest_path, 'wb') as out_file:
                        block_size = 8192
                        downloaded = 0
                        while True:
                            buffer = response.read(block_size)
                            if not buffer:
                                break
                            out_file.write(buffer)
                            downloaded += len(buffer)
                            # Simple progress logging every 10%
                            if total_size > 0 and downloaded % (total_size // 10 + 1) < block_size:
                                pct = downloaded / total_size * 100
                                logger.debug(f"Downloaded {pct:.1f}%")

                logger.info("Model download completed successfully.")
                return dest_path
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to download from {url}: {e}")
                if os.path.exists(dest_path):
                    os.remove(dest_path)

        raise RuntimeError(f"All download links failed for Wav2Lip weights. Last error: {last_error}")

    def _install_requirements(self) -> None:
        """Check and install Wav2Lip-specific requirements with timeout."""
        # Don't blindly downgrade numpy/librosa if already installed by main project.
        # Instead, try importing. If fails, install needed packages.
        missing = []
        try:
            import face_alignment  # noqa: F401
        except ImportError:
            missing.append("face-alignment==1.3.5")
        try:
            import librosa  # noqa: F401
        except ImportError:
            missing.append("librosa")

        if not missing:
            logger.info("Wav2Lip dependencies already satisfied.")
            return

        logger.info(f"Installing Wav2Lip requirements: {missing}")
        try:
            result = subprocess.run(
                ["pip", "install"] + missing,
                capture_output=True,
                text=True,
                timeout=self.timeout_pip
            )
            result.check_returncode()
            logger.info("Wav2Lip requirements installed.")
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            logger.error(f"Failed to install requirements: {e}")
            raise RuntimeError("Could not install Wav2Lip requirements.")

    def _get_resize_factor(self, video_path: str) -> str:
        """
        Auto-determine resize factor based on video resolution to prevent OOM.

        Args:
            video_path: Path to the input video.

        Returns:
            Resize factor as a string ('1', '2', or '4').
        """
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                if max(width, height) > 2000:
                    return "4"
                elif max(width, height) > 1000:
                    return "2"
                logger.info(f"Video resolution: {width}x{height}. Using resize_factor=1")
            else:
                logger.warning("Could not open video for resolution check.")
        except ImportError:
            logger.warning("OpenCV not available, using default resize_factor=1")
        except Exception as e:
            logger.warning(f"Could not check video resolution: {e}")

        return "1"

    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds using ffprobe."""
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    video_path
                ],
                capture_output=True, text=True, timeout=30
            )
            return float(result.stdout.strip())
        except Exception as e:
            logger.warning(f"Could not get video duration: {e}")
            return 0.0

    def _split_video_into_chunks(
        self, video_path: str, audio_path: str, chunk_duration: int = 90
    ) -> list:
        """
        Split video into chunks for GPU-safe processing.
        Returns list of dicts: [{"video": path, "audio": path, "start": float}, ...]
        """
        duration = self._get_video_duration(video_path)
        if duration <= 0:
            logger.warning("Could not determine duration, processing as single chunk.")
            return [{"video": video_path, "audio": audio_path, "start": 0.0}]

        if duration <= chunk_duration:
            logger.info(f"Video {duration:.1f}s <= {chunk_duration}s, processing as single chunk.")
            return [{"video": video_path, "audio": audio_path, "start": 0.0}]

        chunks = []
        num_chunks = int(duration // chunk_duration) + 1
        logger.info(f"Splitting {duration:.1f}s video into {num_chunks} chunks of ~{chunk_duration}s each.")

        for i in range(num_chunks):
            start_time = i * chunk_duration
            if start_time >= duration:
                break

            chunk_video = os.path.join(self.temp_dir, f"chunk_{i:03d}.mp4")
            chunk_audio = os.path.join(self.temp_dir, f"chunk_{i:03d}.wav")

            # Split video chunk
            cmd_video = (
                f'ffmpeg -y -ss {start_time} -i "{video_path}" '
                f'-t {chunk_duration} -c copy "{chunk_video}"'
            )
            # Split audio chunk
            cmd_audio = (
                f'ffmpeg -y -ss {start_time} -i "{audio_path}" '
                f'-t {chunk_duration} -c copy "{chunk_audio}"'
            )

            try:
                run_command(cmd_video)
                run_command(cmd_audio)
                if os.path.exists(chunk_video) and os.path.exists(chunk_audio):
                    chunks.append({
                        "video": chunk_video,
                        "audio": chunk_audio,
                        "start": start_time,
                    })
                    logger.info(f"Chunk {i}: {start_time:.1f}s - {min(start_time + chunk_duration, duration):.1f}s")
                else:
                    logger.warning(f"Chunk {i} files not created, skipping.")
            except Exception as e:
                logger.error(f"Failed to create chunk {i}: {e}")

        return chunks if chunks else [{"video": video_path, "audio": audio_path, "start": 0.0}]

    def _concat_chunks(self, chunk_outputs: list, output_path: str) -> str:
        """Concatenate lip-synced chunks into final output."""
        if len(chunk_outputs) == 1:
            shutil.copy2(chunk_outputs[0], output_path)
            return output_path

        concat_file = os.path.join(self.temp_dir, "concat_list.txt")
        with open(concat_file, "w") as f:
            for chunk_path in chunk_outputs:
                f.write(f"file '{os.path.abspath(chunk_path)}'\n")

        cmd = (
            f'ffmpeg -y -f concat -safe 0 -i "{concat_file}" '
            f'-c:v libx264 -c:a aac -movflags +faststart "{output_path}"'
        )
        run_command(cmd)

        if os.path.exists(output_path):
            logger.info(f"Concatenated {len(chunk_outputs)} chunks into {output_path}")
            return output_path
        else:
            raise RuntimeError("Failed to concatenate chunks into final output.")

    def _build_command(self, video_path: str, audio_path: str, output_path: str) -> list:
        """Build the Wav2Lip inference command with GPU/CPU flags."""
        resize_factor = self._get_resize_factor(video_path)

        command = [
            "python", os.path.join(self.wav2lip_dir, "inference.py"),
            "--checkpoint_path", os.path.join(self.model_path, "wav2lip_gan.pth"),
            "--face", video_path,
            "--audio", audio_path,
            "--outfile", output_path,
            "--resize_factor", resize_factor,
            "--wav2lip_batch_size", "16",
            "--face_det_batch_size", "8",
            "--pads", "0", "10", "0", "0",
        ]

        # Enable GPU acceleration if available and requested
        if self.device == "cuda":
            logger.info("Using GPU acceleration for Wav2Lip.")
            # Wav2Lip inference.py typically picks CUDA automatically if torch.cuda is available,
            # but we can explicitly set env vars or pass flags if needed.
            # For now, we log it; the underlying torch.cuda logic in Wav2Lip handles the rest.
        else:
            logger.info("Using CPU for Wav2Lip inference (slower).")

        return command

    def process(self, video_path: str, audio_path: str, output_path: Optional[str] = None) -> str:
        """
        Process a video with Wav2Lip to sync lips with the given audio.
        Uses chunked processing for long videos to prevent GPU OOM.

        Args:
            video_path: Path to the input video file.
            audio_path: Path to the input audio file.
            output_path: Optional explicit output path. If None, generates one in temp_dir.

        Returns:
            Path to the output video file.

        Raises:
            RuntimeError: If setup or inference fails after retries.
            FileNotFoundError: If input files do not exist.
        """
        # Validate inputs
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Setup
        self._install_wav2lip()
        self._download_model()
        self._install_requirements()

        # Determine output path
        if not output_path:
            output_path = os.path.join(self.temp_dir, "lip_sync_output.mp4")

        # Split into chunks for GPU-safe processing (90s per chunk)
        chunk_duration = 90
        chunks = self._split_video_into_chunks(video_path, audio_path, chunk_duration)

        if len(chunks) == 1:
            # Single chunk - process directly
            logger.info("Processing as single chunk...")
            command = self._build_command(video_path, audio_path, output_path)
            logger.info(f"Executing Wav2Lip: {' '.join(command)}")

            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_inference,
                    check=False
                )

                if result.returncode != 0:
                    logger.error(f"Wav2Lip stderr: {result.stderr}")
                    raise subprocess.CalledProcessError(
                        result.returncode, command, output=result.stdout, stderr=result.stderr
                    )

                if os.path.exists(output_path):
                    logger.info(f"Wav2Lip completed successfully: {output_path}")
                    return output_path
                else:
                    raise RuntimeError("Wav2Lip finished but output file was not created.")

            except subprocess.TimeoutExpired:
                logger.error(f"Wav2Lip timed out after {self.timeout_inference} seconds.")
                raise RuntimeError("Lip sync processing timed out.")
        else:
            # Multi-chunk processing
            logger.info(f"Processing {len(chunks)} chunks to prevent GPU overload...")
            chunk_outputs = []

            for i, chunk in enumerate(chunks):
                chunk_output = os.path.join(self.temp_dir, f"lip_sync_chunk_{i:03d}.mp4")
                logger.info(f"Processing chunk {i+1}/{len(chunks)}: {chunk['start']:.1f}s")

                command = self._build_command(chunk["video"], chunk["audio"], chunk_output)

                try:
                    result = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout_inference,
                        check=False
                    )

                    if result.returncode != 0:
                        logger.error(f"Wav2Lip chunk {i} failed: {result.stderr}")
                        continue

                    if os.path.exists(chunk_output):
                        chunk_outputs.append(chunk_output)
                        logger.info(f"Chunk {i+1}/{len(chunks)} completed successfully.")
                    else:
                        logger.warning(f"Chunk {i+1} output not created, skipping.")

                except subprocess.TimeoutExpired:
                    logger.error(f"Wav2Lip chunk {i} timed out, skipping.")
                    continue

                # Free GPU memory between chunks
                import gc
                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass

            if not chunk_outputs:
                raise RuntimeError("All chunks failed. No output produced.")

            # Concatenate all chunks
            return self._concat_chunks(chunk_outputs, output_path)

    def cleanup(self) -> None:
        """Clean up the temporary directory and any residual files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")

    def __del__(self):
        """Destructor to ensure cleanup happens."""
        try:
            self.cleanup()
        except Exception as e:
            logger.debug(f"Cleanup in destructor failed: {e}")


# Backward-compatible wrapper function
def run_wav2lip(video_path: str, audio_path: str, output_path: Optional[str] = None) -> str:
    """
    Backward-compatible wrapper for Wav2Lip processing.

    Args:
        video_path: Path to the input video.
        audio_path: Path to the input audio.
        output_path: Optional explicit output path.

    Returns:
        Path to the output video.
    """
    processor = Wav2LipProcessor()
    try:
        result = processor.process(video_path, audio_path, output_path)
        return result
    finally:
        processor.cleanup()
