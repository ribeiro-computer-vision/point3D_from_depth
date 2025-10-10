## my_mast3r_setup.py

# Setup my mast3r detector/descriptor

import os
import subprocess
import urllib.request
import shutil
from pathlib import Path


def setup_mast3r(checkpoint_gdrive_dir: str = None, ckpt_name: str = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"):    
    """Clone mast3r repo, install dependencies, and download checkpoint."""
    def run_cmd(cmd: str):
        print(f"→ {cmd}")
        subprocess.run(cmd, shell=True, check=True)

    # --- Clone repo if missing ---
    if not os.path.exists("mast3r"):
        run_cmd("git clone --recursive https://github.com/naver/mast3r")

    # --- Change into repo ---
    os.chdir("mast3r")

    # --- Install requirements ---
    run_cmd("pip install -r requirements.txt")
    run_cmd("pip install -r dust3r/requirements.txt")

    # Optional requirements
    try:
        run_cmd("pip install -r dust3r/requirements_optional.txt")
    except subprocess.CalledProcessError:
        print("⚠️ Optional requirements failed (skipped).")






    # # First, try to install the checkpoint from Google drive (faster)
    # # If this option does not work then the setup_mast3r() function will download it
    # # from the source (i.e., Mast3r project page).
    
    # ---- Config ----
    ckpt_name = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    gdrive_src = Path(checkpoint_gdrive_dir) / ckpt_name

    # Choose ONE target dir used everywhere (match what your code later expects)
    ckpt_dir = Path("checkpoints")                     # or Path("/content/mast3r/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / ckpt_name

    # Source URL (fallback)
    url = "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

    print(f"CWD: {os.getcwd()}")
    print(f"Target checkpoint: {ckpt_path}")

    # ---- Try Google Drive copy first ----
    if ckpt_path.exists():
        print(f"✅ Checkpoint already exists: {ckpt_path}")
    else:
        if gdrive_src.exists():
            try:
                print(f"📄 Copying from Google Drive:\n  {gdrive_src}\n→ {ckpt_path}")
                shutil.copy2(gdrive_src, ckpt_path)
                print("✅ Copied from Google Drive.")
            except Exception as e:
                print(f"⚠️ Copy failed: {e}")
        else:
            print(f"ℹ️ Google Drive source not found: {gdrive_src}")

    # ---- Fallback: download only if still missing ----
    if not ckpt_path.exists():
        print(f"⬇️ Downloading to {ckpt_path} ...")
        urllib.request.urlretrieve(url, ckpt_path)
        print("✅ Checkpoint downloaded.")
    else:
        print(f"✅ Ready: {ckpt_path}")


        