# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Set path to all the Waymo data and the parsed Waymo files."""
import os
from pathlib import Path

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from pyvirtualdisplay import Display

CONFIG_PATH = '/home/ctrl-sim/cfgs'

def get_scenario_dict(hydra_cfg):
    """Convert the `scenario` key in the hydra config to a true dict."""
    if isinstance(hydra_cfg['nocturne']['scenario'], dict):
        return hydra_cfg['nocturne']['scenario']
    else:
        return OmegaConf.to_container(hydra_cfg['nocturne']['scenario'], resolve=True)


def get_default_scenario_dict():
    """Construct the `scenario` dict without w/o hydra decorator."""
    GlobalHydra.instance().clear()
    initialize(config_path="./")
    cfg = compose(config_name="config")
    return get_scenario_dict(cfg)


def set_display_window():
    """Set a virtual display for headless machines."""
    if "DISPLAY" not in os.environ:
        disp = Display()
        disp.start()
