import json
import shutil
import h5py

src = "emotion_recognition_model.h5"
bak = src + ".bak"

# backup first
shutil.copyfile(src, bak)
print("Backup written:", bak)

with h5py.File(src, "r+") as f:
    if "model_config" not in f.attrs:
        raise RuntimeError("No model_config attribute found in this .h5 file.")

    raw = f.attrs["model_config"]
    if isinstance(raw, bytes):
        s = raw.decode("utf-8")
    else:
        s = raw

    cfg = json.loads(s)

    # Walk the config and rename InputLayer key if present
    def walk(obj):
        if isinstance(obj, dict):
            # Replace batch_shape -> batch_input_shape where found
            if "batch_shape" in obj and "batch_input_shape" not in obj:
                obj["batch_input_shape"] = obj.pop("batch_shape")
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(cfg)

    new_s = json.dumps(cfg).encode("utf-8")
    f.attrs.modify("model_config", new_s)

print("Patched model_config in:", src)