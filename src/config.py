import json
import os
from typing import List, Dict, Any


class Config:
    """Container for a loaded configuration.

    Assumes that the main configuration contains a key "path_hypotheses".
    Hypotheses are loaded at initialization time from this path and made accessible
    in the configuration under the key "hypotheses".
    """

    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
        with open('paths.json') as fin:
            main_paths = json.load(fin)
        self.config['dataset']['path'] = os.path.join(main_paths['data_dir'], self.config['dataset']['path'])
        self.config['path_out'] = os.path.join(main_paths['output_dir'], self.config['path_out'])
        self.config['path_hypotheses'] = os.path.join(main_paths['configs_dir'], self.config['path_hypotheses'])
        if 'checkpoint' in self.config['predictor']:
            self.config['predictor']['checkpoint'] = os.path.join(main_paths['output_dir'], self.config['predictor']['checkpoint'])
        with open(self.config['path_hypotheses']) as fin:
            self.config['hypotheses'] = json.load(fin)

    def __getitem__(self, key):
        return self.config[key]

    def get_from_key_list(self, key_list: List[str]) -> Any:
        """Get a sub-config or config item from a key-list."""
        key_list = [key for key in key_list]
        key_list.insert(0, 'hypotheses')
        sub_config = self.config
        for key in key_list:
            sub_config = sub_config[key]
        return sub_config

    def get_hypos_for_section(self, sec_group: str, sec_name: str):
        """Get the hypothesis or multiple hypotheses for a pipeline section."""
        hypotheses_keys = self.config['prediction_pipeline'][sec_group][sec_name]['hypotheses_keys']
        results = self.get_from_key_list(hypotheses_keys)
        return results
