import torch
import sys
import json

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Недостаточно аргументов"}))
        return
    category = sys.argv[1]
    file_name = sys.argv[2]
    # Предполагаем, что файлы лежат в папке processed/<category>/<file_name>
    file_path = f"processed/{category}/{file_name}"
    try:
        data = torch.load(file_path, map_location='cpu')
        # Если data является словарем, обрабатываем каждое поле,
        # а если это Tensor – конвертируем его сразу
        if isinstance(data, dict):
            json_data = {}
            for key, value in data.items():
                if torch.is_tensor(value):
                    json_data[key] = value.tolist()
                else:
                    json_data[key] = value
        elif torch.is_tensor(data):
            json_data = data.tolist()
        else:
            json_data = data
        print(json.dumps(json_data))
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == '__main__':
    main()
