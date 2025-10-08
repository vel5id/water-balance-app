import pysftp
import os
from datetime import datetime

# Даты для загрузки
target_dates = [
    '2020-03-27',
    '2020-04-21',
    '2020-04-26',
    '2020-05-01'
]

def download_gleam():
    print("Начинаем загрузку данных GLEAM...")
    
    cnopts = pysftp.CnOpts()
    cnopts.hostkeys = None  # не проверять ключ хоста
    
    downloaded_files = []
    
    try:
        with pysftp.Connection(
            host="hydras.ugent.be",
            username="gleamuser",
            password="GLEAM4!#h+cel_924",
            port=2225,
            cnopts=cnopts
        ) as sftp:
            print("Подключение успешно установлено")
            # Скачиваем README_GLEAM4.2.pdf из директории data
            try:
                sftp.cwd('data')
                files_in_data = sftp.listdir()
                print("\nСодержимое папки data:", files_in_data)
                if 'README_GLEAM4.2.pdf' in files_in_data:
                    print("Скачиваем README_GLEAM4.2.pdf...")
                    sftp.get('README_GLEAM4.2.pdf')
                    print("README_GLEAM4.2.pdf успешно скачан!")
                else:
                    print("README_GLEAM4.2.pdf не найден в папке data.")
                sftp.cwd('..')
            except Exception as e:
                print(f"Ошибка при скачивании README_GLEAM4.2.pdf: {e}")

            # Далее стандартная логика загрузки суточных файлов
            sftp.cwd('daily')  # папка с суточными файлами
            all_files = sftp.listdir()
            for date_str in target_dates:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                date_pattern = date_obj.strftime('%Y%m%d')
                matching_files = [f for f in all_files if date_pattern in f and f.endswith('.nc')]
                for fname in matching_files:
                    print(f"Загружаем файл: {fname}")
                    if not os.path.exists(fname):
                        sftp.get(fname)
                        downloaded_files.append(fname)
                    else:
                        print(f"Файл {fname} уже существует")
                        downloaded_files.append(fname)
        print("Загрузка завершена")
        return downloaded_files
        
    except Exception as e:
        print(f"Ошибка при загрузке: {str(e)}")
        return []

if __name__ == "__main__":
    downloaded_files = download_gleam()
    
    # Проверка загруженных файлов
    print("\nПроверка загруженных файлов:")
    for file in downloaded_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024 * 1024)  # размер в МБ
            print(f"Файл {file} успешно загружен, размер: {size:.2f} MB")
        else:
            print(f"Ошибка: файл {file} не найден")
