import research_template as rt
import hydra


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def test(cfg):
    print(rt.get_path(cfg, "gtopdb.processed.ligands"))


test()
