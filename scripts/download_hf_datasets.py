import os
from datasets import load_dataset, DownloadMode
from pathlib import Path
from tqdm import tqdm
import shutil
import logging
from datetime import datetime

# ตั้งค่า logging
logging.basicConfig(
    filename='dataset_download.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def check_disk_space(path, required_gb=10):
    """ตรวจสอบพื้นที่ดิสก์"""
    total, used, free = shutil.disk_usage(path)
    free_gb = free // (2**30)
    if free_gb < required_gb:
        raise Exception(f"Not enough disk space. Required: {required_gb}GB, Available: {free_gb}GB")
    return free_gb

def get_downloaded_files(path):
    """ตรวจสอบไฟล์ที่ดาวน์โหลดแล้ว"""
    if not os.path.exists(path):
        return set()
    return {f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))}

def download_dataset(dataset_name, save_dir, required_space_gb=10):
    start_time = datetime.now()
    save_path = Path(save_dir) / dataset_name.replace("/", "_")
    try:
        # ตรวจสอบพื้นที่ดิสก์
        free_space = check_disk_space(save_dir, required_space_gb)
        logging.info(f"Available disk space: {free_space}GB")
        # สร้างโฟลเดอร์
        os.makedirs(save_path, exist_ok=True)
        # ตรวจสอบไฟล์ที่มีอยู่และลบไฟล์ที่ไม่เกี่ยวข้องออก เช่น .gitattributes
        existing_files = get_downloaded_files(save_path)
        if ".gitattributes" in existing_files:
            existing_files.remove(".gitattributes")
        if existing_files:
            logging.info(f"Found {len(existing_files)} existing files in dataset folder")
        # แสดง progress bar
        with tqdm(desc=f"Downloading {dataset_name}", unit="files") as pbar:
            dataset = load_dataset(
                dataset_name,
                cache_dir=str(save_path),
                download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
                ignore_verifications=True
            )
            pbar.update(1)
        # บันทึก dataset
        dataset.save_to_disk(str(save_path))
        # สรุปผลการดาวน์โหลด
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        new_files = get_downloaded_files(save_path) - existing_files
        logging.info(f"""
        Download Summary:
        - Dataset: {dataset_name}
        - Save location: {save_path}
        - Duration: {duration:.2f} seconds
        - New files: {len(new_files)}
        - Total files: {len(get_downloaded_files(save_path))}
        """)
        print(f"Successfully downloaded {dataset_name} to {save_path}")
    except Exception as e:
        error_msg = f"Error downloading {dataset_name}: {str(e)}"
        logging.error(error_msg)
        print(error_msg)
        print(f"Previously downloaded files: {existing_files}")
        raise

if __name__ == "__main__":
    # กำหนด path ที่ต้องการเก็บ dataset
    base_dir = "D:/Github/MedicalMT_/DatasetDownload"
    # รายชื่อ dataset ที่ต้องการโหลด
    datasets = [
        "future-technologies/Universal-Transformers-Dataset"
    ]
    # โหลดทีละ dataset
    for dataset_name in datasets:
        try:
            download_dataset(dataset_name, base_dir)
        except Exception as e:
            logging.error(f"Failed to download {dataset_name}: {str(e)}")
            continue
