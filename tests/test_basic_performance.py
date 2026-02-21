from src.sim.episode import EpisodeConfig, run_episode
from src.controllers.pn import PNController

def test_pn_vortex_basic():
    cfg = EpisodeConfig(
        field="vortex",
        controller="pn",
        current_strength=0.3,
        current_comp=0.7,
        timeout_s=20.0,
        dt=0.05,
    )
    ctrl = PNController()
    res = run_episode(cfg, ctrl, seed=123)
    assert isinstance(res.success, bool)