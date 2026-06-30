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
            "--face_det_batch_size", "2",
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

        # Setup Wجرا
        self._install_wav2lip()
        self._download_model()
        self._install_requirements()

        # Determine output path
        if not output_path:
            output_path = os.path.join(self.temp_dir, "lip_sync_output.mp4")

        # Build and run
        command = self._build_command(video_path, audio_path, output_path)
        logger.info(f"Executing Wav2Lip: {' '.join(command)}")

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_inference,
                check=False  # We check manually for better error logs
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
            raise RuntimeError(f"Lip sync processing timed out. Try reducing video length or resolution.")

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
