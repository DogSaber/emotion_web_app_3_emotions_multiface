import json, shutil, h5py

src = "emotion_recognition_model.h5"
bak = src + ".bak2"
shutil.copyfile(src, bak)
print("Backup written:", bak)

def walk(obj):
    if isinstance(obj, dict):
        # Replace Keras 3 dtype policy dict with float32
        if "dtype" in obj and isinstance(obj["dtype"], dict):
            d = obj["dtype"]
            if d.get("class_name") == "DTypePolicy":
                obj["dtype"] = "float32"
        for v in obj.values():
            walk(v)
    elif isinstance(obj, list):
        for v in obj:
            walk(v)

with h5py.File(src, "r+") as f:
    raw = f.attrs["model_config"]
    s = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
    cfg = json.loads(s)
    walk(cfg)
    f.attrs.modify("model_config", json.dumps(cfg).encode("utf-8"))

print("Patched dtype policy in:", src)