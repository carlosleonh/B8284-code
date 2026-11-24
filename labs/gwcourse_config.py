"""
gwcourse_config.py
==================
Reads the configuration of the laboratories, ``labs/config.toml``.

The file says where the data are and which events the course loads. The
functions of ``gwcourse_signal`` do not read it: they take a path or a time as
an argument. A script that needs a file of the course reads the configuration
here and passes the path on:

    from gwcourse_config import read_config, strain_file
    from gwcourse_signal import load_strain

    config = read_config()
    strain, t, fs = load_strain(strain_file(config, "GW150914", "H1"))
    t_merger = config["events"]["GW150914"]["gps"]

The notebooks do the same with ``tomllib`` in their first code cell, so that
the configuration is visible there.
"""

from __future__ import annotations

import os
import tomllib

__all__ = ["read_config", "strain_file"]


def read_config(path=None):
    """Read the configuration file of the laboratories.

    Args:
        path: the configuration file (default None: ``config.toml`` in the
            folder of this module).

    Returns:
        config: the dictionary of the file. ``config['data']['path']`` is added:
            the absolute path of the folder with the data.
    """
    # the default file is the one beside this module
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.toml")
    with open(path, "rb") as handle:
        config = tomllib.load(handle)
    # the folder of the data is given relative to the folder of the file
    folder = os.path.dirname(os.path.abspath(path))
    config["data"]["path"] = os.path.join(folder, config["data"]["dir"])
    return config


def strain_file(config, event, detector):
    """Give the file of the strain of one detector for one event.

    Args:
        config: the dictionary returned by ``read_config``.
        event: the name of the event, for example 'GW150914'.
        detector: 'H1', 'L1' or 'V1'.

    Returns:
        path: the absolute path of the hdf5 file.
    """
    name = config["events"][event][detector]
    path = os.path.join(config["data"]["path"], name)
    return path
