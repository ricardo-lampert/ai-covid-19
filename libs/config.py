import yaml
import os


class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, obj(b) if isinstance(b, dict) else b)


def load_config(file_path, key="", as_dict=False):
    if not os.path.exists(file_path):
        raise RuntimeError(f"Configuration file '{file_path}' not found")
    with open(file_path, "r") as config_file:
        config = yaml.load(config_file.read(), Loader=yaml.FullLoader)
    if key == "":
        if(as_dict):
            return config
        return obj(config)
    if(as_dict):
        return config[key]
    return obj(config[key])
