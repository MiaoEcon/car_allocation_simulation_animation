from vehicle_status_animation import (
    animate_station,
    interactive_station,
    make_demo_orders,
    save_html_player,
)


def test_primary_animation_api_is_exported():
    assert callable(animate_station)
    assert callable(interactive_station)
    assert callable(save_html_player)


def test_demo_orders_has_required_columns():
    df = make_demo_orders()
    assert {"pickup_date", "duration_days", "pickup_station", "return_station", "on_hand"}.issubset(df.columns)
    assert not df.empty


def test_demo_includes_cross_station_returns_into_station():
    df = make_demo_orders(days=14, n_in=3)
    incoming = df[(df["pickup_station"] != "台北") & (df["return_station"] == "台北")]
    assert len(incoming) == 3


def test_build_draws_incoming_and_cross_return_labels():
    from vehicle_status_animation.single_station import _build, _cols

    df = make_demo_orders(days=14, n_in=3, seed=3)
    c = _cols("pickup_date", "duration_days", "return_station", "pickup_station",
              "on_hand", "not_on_hand", "preparing", "renting", None)
    fig, draw, total_frames, n = _build(df, "台北", c, fps=2, frames_per_order=3,
                                         max_orders=None, dpi=35, incoming_prep_days=1)
    draw(total_frames - 1)
    labels = [text.get_text() for text in fig.axes[0].texts]
    assert any(label.startswith("來自 ") for label in labels)
    assert any(label.startswith("還到 ") for label in labels)
    assert n > 0
    import matplotlib.pyplot as plt
    plt.close(fig)
