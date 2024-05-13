import yaml
import os

class TTSConfigLoader():
    def __init__(self):
        self.configurations()
        self.load_configs()

    def configurations(self):
        self.ROOT_DIR = 'config_voices'
        self.CONFIGS = 'config_lang.yaml'
        self.data = {}
        self.vkey = 'vc'
        self.VOICE_LIST = {}

    def load_configs(self):
        # todo: load main config
        with open(self.CONFIGS, "r") as yamlfile:
            self.data = yaml.load(yamlfile, Loader=yaml.FullLoader)

        self.langs, self.voices_path = self.get_subconfigs()

    def get_voice_list(self, lang: str):
        assert lang in self.langs
        path_to_lang = os.path.join(self.ROOT_DIR, self.voices_path[lang])
        if not os.path.exists(path_to_lang):
            raise FileNotFoundError()

        with open(path_to_lang, "r") as yamlfile:
            self.VOICE_LIST[lang] = yaml.load(yamlfile, Loader=yaml.FullLoader)

    def get_subconfigs(self):
        languages = self.data['languages']['codes']
        langs = [i.lower() for i in languages]
        return langs, self.data['voices']

config_loader = TTSConfigLoader()
"""
arabic / chinese / english / french
german / japanese / korean / russian / spanish
"""
config_loader.get_voice_list('french')
