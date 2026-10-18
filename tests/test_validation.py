import pandas as pd

from vehicle_status_animation import (
    make_demo_event_log,
    validate_event_log,
    validate_legacy_orders,
)


def test_demo_event_log_passes_validation():
    assert validate_event_log(make_demo_event_log()) == []


def test_negative_state_is_reported():
    df = make_demo_event_log()
    df.loc[df.index[0], "on_hand"] = -1
    assert any("不可為負數" in issue for issue in validate_event_log(df))


def test_bad_transition_is_reported():
    df = make_demo_event_log()
    idx = df.index[df["event_type"].eq("租賃出車")][0]
    df.loc[idx, "on_hand"] = 5  # Should be 4 after dispatching one vehicle.
    assert any("狀態不符" in issue for issue in validate_event_log(df))


def test_legacy_order_validation():
    good = pd.DataFrame({
        "pickup_date": ["2026-10-01"], "duration_days": [2],
        "return_station": ["台北"], "pickup_station": ["台北"], "on_hand": [3],
    })
    assert validate_legacy_orders(good) == []
    good.loc[0, "duration_days"] = -1
    assert any("duration_days" in issue for issue in validate_legacy_orders(good))


def test_demo_has_cross_station_return_out_and_arrival_events():
    df = make_demo_event_log()
    assert ((df["event_type"] == "異站還車結束") & (df["station"] == "台北") & (df["to_station"] == "台中")).any()
    assert ((df["event_type"] == "異站還車入站") & (df["station"] == "台中") & (df["from_station"] == "台北")).any()
