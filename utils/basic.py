import json


class Config_Manager:
    def __init__(self, config_path):
        self.cfg_path = config_path

    def set_cfg_path(self, cfg_path):
        self.cfg_path = cfg_path

    def get_cfg(self):
        with open(self.cfg_path) as f:
            self.cfg = json.load(f)
        return self.cfg

    def print(self):
        print(self.cfg)