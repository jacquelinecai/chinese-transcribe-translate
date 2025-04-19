import json

file_path = "./data/translation2019zh/translation2019zh_train.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

selected_data = data[::5]

with open(file_path, "w", encoding="utf-8") as out:
    json.dump(selected_data, out, ensure_ascii=False, indent=4)

print(f"Filtered data saved to {file_path}")