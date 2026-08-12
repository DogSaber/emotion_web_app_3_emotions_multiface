import os

# Change this to the folder you want to rename
FOLDER = r"dataset\train\Angry"

# Change this depending on the emotion folder
PREFIX = "angry"

valid_extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

files = [
    f for f in os.listdir(FOLDER)
    if f.lower().endswith(valid_extensions)
]

files.sort()

for i, filename in enumerate(files, start=1):
    old_path = os.path.join(FOLDER, filename)

    ext = os.path.splitext(filename)[1].lower()
    new_name = f"{PREFIX}_{i:04d}{ext}"
    new_path = os.path.join(FOLDER, new_name)

    os.rename(old_path, new_path)

print(f"Renamed {len(files)} files in {FOLDER}")