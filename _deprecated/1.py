import os
import re

# 1. Проверка, что все файлы имеют формат tif/tiff
def check_tif_files(root_dir):
    for year_folder in os.listdir(root_dir):
        year_path = os.path.join(root_dir, year_folder)
        if os.path.isdir(year_path):
            for subfolder in os.listdir(year_path):
                subfolder_path = os.path.join(year_path, subfolder)
                if os.path.isdir(subfolder_path):
                    for filename in os.listdir(subfolder_path):
                        if "B03" in filename:
                            if not filename.lower().endswith((".tiff")):
                                print(f"Файл не tif/tiff: {os.path.join(subfolder_path, filename)}")

# 2. Список уникальных дат из имен файлов (tif/tiff)
def collect_sentinel_dates(root_dir):
    sentinel_dates = set()
    # Берём первую дату в формате YYYY-MM-DD из имени файла
    date_pattern = re.compile(r'(\d{4}-\d{2}-\d{2})')
    for year_folder in os.listdir(root_dir):
        year_path = os.path.join(root_dir, year_folder)
        if os.path.isdir(year_path):
            for subfolder in os.listdir(year_path):
                subfolder_path = os.path.join(year_path, subfolder)
                if os.path.isdir(subfolder_path):
                    for filename in os.listdir(subfolder_path):
                        if "B03" in filename and filename.lower().endswith((".tif", ".tiff")):
                            match = date_pattern.search(filename)
                            if match:
                                sentinel_dates.add(match.group(1))
    print("sentinel_dates = [")
    for date in sorted(sentinel_dates):
        print(f"    '{date}',")
    print("]")

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    print("Проверка формата файлов...")
    check_tif_files(root_dir)
    print("\nСбор дат...")
    collect_sentinel_dates(root_dir)
