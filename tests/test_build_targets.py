def test_target_builder_module_imports_and_exposes_cli():
    from stable_finance.dataset import build_targets

    assert callable(build_targets.build_month)
    assert callable(build_targets.main)
