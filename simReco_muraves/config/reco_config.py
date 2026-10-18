from __future__ import annotations


_RECO_CONFIG = {
    "cluster_lists": {
        "max_strip_energy": 3000.0,
        "first_strip_pos": -0.528,
        "adjacent_strips_distance": 0.0165,
        "n_strips": 64,
        "min_energy_trigger_mask": 20.0,
        "split_board_index": 32,
        "single_strip_smearing_scale": 1000,
    },
    "tracking": {
        "first_strip_pos": -0.528,
        "adjacent_strips_distance": 0.0165,
        "scattering_pi": 3.141592653589793,
    },
}


def get_reco_config():
    return _RECO_CONFIG
