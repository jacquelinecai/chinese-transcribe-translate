import json

file_path = "./data/translation2019zh/translation2019zh_train.json"

selected_data = []
with open(file_path, "r", encoding="utf-8") as file:
    for index, line in enumerate(file):
        if index % 15 == 0:
            try:
                selected_data.append(json.loads(line))
            except json.JSONDecodeError:
                pass

with open(file_path, "w", encoding="utf-8") as output_file:
    json.dump(selected_data, output_file, ensure_ascii=False, indent=4)

print(f"Filtered data saved to {file_path}")