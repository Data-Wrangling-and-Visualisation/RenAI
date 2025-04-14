import sys
import json

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Недостаточно аргументов"}))
        return
    category = sys.argv[1]
    file_name = sys.argv[2]
    # Реализуйте обработку Attention – например, получите карту внимания
    result = {
        "attention_map": f"processed/attention/{file_name}",
        "message": "Реальные данные Attention"
    }
    print(json.dumps(result))

if __name__ == '__main__':
    main()