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
        # Using a reliable mirror for the model weights
        subprocess.run(["wget", "-O", os.path.join(model_path, "wav2lip_gan.pth"), 
                        "https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApMqsIGmza6nGIBPb822T8lXq65R37M_N_M7A"])
        
        # Install basic requirements
        logger.info("Installing Wav2Lip requirements...")
        subprocess.run(["pip", "install", "librosa==0.8.0", "face-alignment==1.3.5", "numpy==1.23.5"])

    # Ensure the model exists before running
    if not os.path.exists(os.path.join(wav2lip_dir, "checkpoints", "wav2lip_gan.pth")):
        logger.error("Wav2Lip model weights not found! Lip sync failed.")
        return video_path # Fallback to original video
    
    command = [
        "python", os.path.join(wav2lip_dir, "inference.py"),
        "--checkpoint_path", os.path.join(wav2lip_dir, "checkpoints", "wav2lip_gan.pth"),
        "--face", video_path,
        "--audio", audio_path,
        "--outfile", output_path,
        "--resize_factor", "1"
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
