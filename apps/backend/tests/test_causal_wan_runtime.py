from app.pipeline.causal_wan_runtime import CausalWanRealtimeRunner


def test_steps_schedule_mapping() -> None:
    assert CausalWanRealtimeRunner._steps_from_count(1) == [700, 0]
    assert CausalWanRealtimeRunner._steps_from_count(2) == [700, 500, 0]
    assert CausalWanRealtimeRunner._steps_from_count(3) == [700, 600, 400, 0]
    assert CausalWanRealtimeRunner._steps_from_count(4) == [700, 600, 500, 400, 0]


def test_snap_size_to_multiple_of_16() -> None:
    assert CausalWanRealtimeRunner._snap_size(767) == 752
    assert CausalWanRealtimeRunner._snap_size(768) == 768
    assert CausalWanRealtimeRunner._snap_size(255) == 256
