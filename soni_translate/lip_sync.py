import os
import subprocess
import logging

logger = logging.getLogger(__name__)

def run_wav2lip(video_path, audio_path, output_path):
    logger.info("Starting Wav2Lip Lip-Sync processing...")
    wav2lip_dir = "Wav2Lip"
    
    # Check if Wav2Lip is installed, if not, download it
    if not os.path.exists(wav2lip_dir):
        logger.info("Wav2Lip not found. Cloning repository...")
        subprocess.run(["git", "clone", "https://github.com/Rudrabha/Wav2Lip.git"])
        
        # Download pre-trained model
        model_path = os.path.join(wav2lip_dir, "checkpoints")
        os.makedirs(model_path, exist_ok=True)
        
        dest_path = os.path.join(model_path, "wav2lip_gan.pth")
        urls = [
            "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth",
            "https://huggingface.co/rippertnt/wav2lip/resolve/main/wav2lip_gan.pth"
        ]
        
        download_success = False
        import urllib.request
        for url in urls:
            logger.info(f"Downloading Wav2Lip model weights from: {url}")
            try:
                # Set user-agent header to avoid potential blocks
                req = urllib.request.Request(
                    url, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                )
                with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
                    meta = response.info()
                    file_size = int(meta.get("Content-Length", 0))
                    logger.info(f"Downloading {file_size / (1024*1024):.2f} MB...")
                    
                    block_size = 8192
                    downloaded = 0
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        downloaded += len(buffer)
                        out_file.write(buffer)
                logger.info("Download completed successfully.")
                download_success = True
                break
            except Exception as e:
                logger.warning(f"Failed to download from {url}: {str(e)}")
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                    
        if not download_success:
            logger.error("All download links failed for Wav2Lip weights.")
            return video_path

        # Install basic requirements if not present
        try:
            import face_alignment
            import librosa
            logger.info("Wav2Lip requirements already satisfied.")
        except ImportError:
            logger.info("Installing Wav2Lip requirements...")
            subprocess.run(["pip", "install", "librosa==0.8.0", "face-alignment==1.3.5", "numpy==1.23.5"])

    # Ensure the model exists before running
    if not os.path.exists(os.path.join(wav2lip_dir, "checkpoints", "wav2lip_gan.pth")):
        logger.error("Wav2Lip model weights not found! Lip sync failed.")
        return video_path # Fallback to original video
    
    # Determine resize factor automatically based on resolution to prevent OOM
    resize_factor = "1"
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if max(width, height) > 2000:
                resize_factor = "4"
            elif max(width, height) > 1000:
                resize_factor = "2"
            logger.info(f"Video resolution: {width}x{height}. Auto-selected resize_factor: {resize_factor}")
    except Exception as e:
        logger.warning(f"Could not check video resolution for auto-resizing: {e}")

    command = [
        "python", os.path.join(wav2lip_dir, "inference.py"),
        "--checkpoint_path", os.path.join(wav2lip_dir, "checkpoints", "wav2lip_gan.pth"),
        "--face", video_path,
        "--audio", audio_path,
        "--outfile", output_path,
        "--resize_factor", resize_factor,
        "--wav2lip_batch_size", "16",
        "--face_det_batch_size", "2"
    ]
    
    logger.info(f"Executing Wav2Lip: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
        if os.path.exists(output_path):
            logger.info("Wav2Lip completed successfully.")
            return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during Wav2Lip execution: {str(e)}")
    
    logger.error("Lip sync failed, returning original video.")
    return video_path
