import yaml


def test_runtime_versions_are_yaml_serializable():
    import torch

    manifest = {"torch": str(torch.__version__)}
    assert yaml.safe_load(yaml.safe_dump(manifest)) == manifest
