import os

train_dir = "data/chinese_characters/train" 

for char_class in os.listdir(train_dir):
    char_path = os.path.join(train_dir, char_class)

    if os.path.isdir(char_path):
        images = os.listdir(char_path)
        
        for img in images:
            filename, file_ex = os.path.splitext(img)
            img_number = int(filename)
            if img_number < 1 or img_number > 10:
                img_path = os.path.join(char_path, img)
                os.remove(img_path)

