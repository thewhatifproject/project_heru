from app.config import RuntimeConfig, merge_runtime_config


def test_config_merge_nested() -> None:
    cfg = RuntimeConfig()

    updated = merge_runtime_config(
        cfg,
        {
            "prompt": "test prompt",
            "outputs": {
                "virtual_camera": True,
                "rtmp_enabled": True,
            },
        },
    )

    assert updated.prompt == "test prompt"
    assert updated.outputs.virtual_camera is True
    assert updated.outputs.rtmp_enabled is True
    assert updated.outputs.preview is True
