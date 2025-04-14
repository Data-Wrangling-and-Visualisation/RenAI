import os
import shutil

# Пути к папкам
source_dir = 'images'
target_dir = 'target_images'

# Создаем целевую папку, если она не существует
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Получаем список файлов в исходной папке
files = os.listdir(source_dir)

# Сортируем файлы по имени (чтобы брать первые 500)
files.sort()

# Копируем первые 500 файлов
for file in files[:100]:
    file_path = os.path.join(source_dir, file)
    target_path = os.path.join(target_dir, file)
    
    # Проверяем, что это файл (а не папка)
    if os.path.isfile(file_path):
        try:
            shutil.copy2(file_path, target_path)
            print(f"Скопирован файл: {file}")
        except Exception as e:
            print(f"Ошибка копирования файла {file}: {e}")
