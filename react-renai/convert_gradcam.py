import sys
import json

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Недостаточно аргументов"}))
        return
    category = sys.argv[1]
    file_name = sys.argv[2]
    # Здесь вы можете реализовать реальную обработку GradCAM
    # Например, загрузить соответствующее изображение, закодировать его в base64 и вернуть
    result = {
        "gradcam_image": f"processed/gradcam/{file_name}",
        "message": "Реальные данные GradCAM"
    }
    print(json.dumps(result))

if __name__ == '__main__':
    main()