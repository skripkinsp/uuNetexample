from pathlib import Path

def check_files():
    images = list(Path("nnUNet_raw/Dataset001_MyTask/imagesTr").glob("case_*.nii.gz"))
    labels = list(Path("nnUNet_raw/Dataset001_MyTask/labelsTr").glob("case_*.nii.gz"))
    print(f"Найдено {len(images)} изображений и {len(labels)} масок")

check_files()