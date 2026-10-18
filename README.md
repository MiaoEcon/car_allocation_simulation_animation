# Vehicle Status Animation

以單一站點為單位，將車輛配置與訂單流動製作成深色儀表板風格動畫。套件名稱為 `vehicle-status-animation`，Python 匯入名稱為 `vehicle_status_animation`。

## 安裝

```bash
python -m pip install .
```

Colab / Jupyter 互動播放器：

```bash
python -m pip install '.[notebook]'
```

## 主要 API

```python
import pandas as pd
from vehicle_status_animation import (
    make_demo_orders, animate_station, interactive_station, save_html_player,
)

orders = make_demo_orders()
# GIF 或 MP4
animate_station(orders, station="台北", out_path="taipei.gif")
# Colab / Jupyter 可拖動時間軸
interactive_station(orders, station="台北")
# 單一 HTML 播放器
save_html_player(orders, station="台北", out_path="taipei.html")
```

`animate_station` 會依 `out_path` 副檔名輸出 GIF 或 MP4；MP4 需要系統已安裝 FFmpeg。中文字型會優先尋找系統字型，必要時嘗試安裝或下載 Noto Sans TC，因此在無網路或無系統安裝權限的環境，請預先安裝可用的中文字型。

## 訂單資料格式

必需欄位：

| 欄位 | 說明 | 範例 |
|---|---|---|
| `pickup_date` | 出車日期 | `2026-10-01` |
| `duration_days` | 用車天數，正整數 | `3` |
| `pickup_station` | 出車站點 | `台北` |
| `return_station` | 預計還車站點 | `台中` |
| `on_hand` | 該筆訂單發生前可租車數 | `5` |

建議同時提供 `preparing`（整備中）與 `renting`（租賃中），並提供 `not_on_hand`（配置數中不在站的車數）。若沒有提供 `not_on_hand`，可用 `preparing + renting` 推算。若配置數可由 `on_hand + not_on_hand` 得出，建議同一月份維持一致；也可以透過 `allocation` 明確指定配置數。

### 參數欄位對應

若資料表欄位名稱不同，可在呼叫時指定 `col_date`、`col_days`、`col_return`、`col_pickup`、`col_onhand`、`col_other`、`col_prep`、`col_rent`、`col_uncovered`。

### 異站還車與他站還入

原始 `orders` 可以同時包含本站出車與他站出車的訂單：

- **本站異站還車**：`pickup_station == station` 且 `return_station != station`。車輛出車時會標示目的站；到預計還車日後，對應車格改為藍色虛線，顯示「還到 XX站」及日期，且不再視為回到本站。
- **他站還入本站**：`pickup_station != station` 且 `return_station == station`。到達日後會在配置車格之外新增車格，不計入本站配置數；初始為琥珀色「整備中」，經 `incoming_prep_days` 天後轉為青色「在站可租」，並持續標示來源站及到站日期。
- `incoming_prep_days` 可傳給 `animate_station`、`interactive_station`、`save_html_player`，預設為 `1`。

範例：

```python
orders = make_demo_orders(n_in=3)
animate_station(orders, "台北", "taipei.gif", incoming_prep_days=2)
```

請確保原始資料包含其他站的訂單，並提供可辨識的出車站、還車站、出車日期與用車天數。`max_orders` 只限制本站出車訂單的播放筆數，不會把其他站訂單當成本站訂單播放；他站還入仍會依原始訂單的到站日期顯示。

## 輸出選項

- `animate_station(...)`：輸出 GIF / MP4。
- `interactive_station(...)`：在 Colab / Jupyter 顯示播放器，提供播放、時間軸、逐單跳轉與速度選擇；需要 `ipywidgets`。
- `save_html_player(...)`：輸出可在瀏覽器開啟的 HTML 播放器。

常用參數：

- `fps`：每秒影格數。
- `frames_per_order`：每筆訂單的動畫影格數；越大，每筆訂單播放越慢。
- `max_orders`：僅播放排序後的前 N 筆本站訂單，方便快速測試。
- `allocation`：配置數，可為整數或月份對應字典，例如 `{"2026-10": 8}`。
- `incoming_prep_days`：他站還入車輛的整備天數，預設 1 天。

## 事件紀錄驗證（相容舊 API）

此版本仍保留舊版事件紀錄驗證工具：`validate_event_log`、`validate_legacy_orders`、`make_demo_event_log`。事件紀錄格式與規則請參閱 [DESIGN_AND_DATA_RULES.md](DESIGN_AND_DATA_RULES.md)。

## 開發與測試

```bash
python -m pip install -e '.[notebook]'
PYTHONPATH=src pytest -q
```

## 專案結構

```text
vehicle-status-animation/
├── README.md
├── DESIGN_AND_DATA_RULES.md
├── pyproject.toml
├── src/vehicle_status_animation/
│   ├── __init__.py          # 公開 API 與事件紀錄驗證
│   └── single_station.py    # 單一站點動畫引擎
└── tests/
```
