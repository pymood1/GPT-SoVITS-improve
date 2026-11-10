def get_models():
    model_size_list = [
        "medium",
        "medium.en",
        "large-v2",
        "large-v3",
        "large-v3-turbo",
    ]
    return model_size_list


asr_dict = {
    "Fun-ASR MLT-Nano (多语种)": {
        "lang": ["ko", "zh", "en", "ja", "yue"],
        "size": ["large"],
        "path": "funasr_asr.py",
        "precision": ["float32"],
        "backend": "fun-asr-mlt-nano",
    },
    "SenseVoice (极速, 5语种)": {
        "lang": ["ko", "zh", "en", "ja", "yue"],
        "size": ["large"],
        "path": "funasr_asr.py",
        "precision": ["float32"],
        "backend": "sensevoice",
    },
    "达摩 ASR (中文经典)": {
        "lang": ["zh", "yue"],
        "size": ["large"],
        "path": "funasr_asr.py",
        "precision": ["float32"],
        "backend": "classic",
    },
    "Faster Whisper (多语种)": {
        "lang": ["auto", "ko", "en", "ja", "zh", "yue"],
        "size": get_models(),
        "path": "fasterwhisper_asr.py",
        "precision": ["float32", "float16", "int8"],
    },
}
