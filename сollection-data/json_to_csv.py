import json
import csv
import os
from pathlib import Path

CONFIG = {
    "directories": {
        "json": "json_data",
        "csv": "processed"
    }
}

BASE_DIR = Path(__file__).resolve().parent
for key, rel_path in CONFIG["directories"].items():
    full_path = BASE_DIR / rel_path
    CONFIG["directories"][key] = str(full_path)
    full_path.mkdir(parents=True, exist_ok=True)

def merge_json_to_csv():
    json_dir = Path(CONFIG["directories"]["json"])
    csv_dir = Path(CONFIG["directories"]["csv"])
    
    json_files = list(json_dir.glob("*.json"))
    
    if not json_files:
        print(f"JSON files not found in directory {json_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to merge")
    
    csv_file_path = csv_dir / "processed_data.csv"
    
    fieldnames = set()
    
    all_data = []
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                    for item in data:
                        fieldnames.update(item.keys())
                else:
                    all_data.append(data)
                    fieldnames.update(data.keys())
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=sorted(fieldnames))
        writer.writeheader()
        writer.writerows(all_data)
    
    print(f"Data successfully written to {csv_file_path}")

if __name__ == "__main__":
    merge_json_to_csv()
