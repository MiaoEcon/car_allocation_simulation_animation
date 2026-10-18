"""單一站點車輛配置動畫（深色儀表板風格）。

車輛三種狀態（加總 = 該月配置數，整月固定）：
    在站可租 on_hand　　整備中 preparing（等上架）　　租賃中 renting（已出車、尚未回站）

三種輸出：
    animate_station(...)     → 存成 mp4 / gif
    interactive_station(...) → 在 Colab / Jupyter 內做出「可拖動時間軸」的播放器（報告用）
    save_html_player(...)    → 存成單一 .html，瀏覽器直接開，也有時間軸

資料模式：
    1. 舊版租賃資料：pickup_date、duration_days、pickup_station、return_station、on_hand 等欄位。
    2. 車輛狀態異動紀錄：event_date、event_type、station、from_station、to_station、
       vehicle_count、on_hand、preparing、renting。每筆紀錄只代表一個站點的異動；
       調度必須拆成「調度調出」與「調度調入」兩筆站點紀錄。狀態欄位記錄異動完成後的台數。
"""
import glob
import io
import math
import os
import subprocess
import urllib.request
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

# ───────────────────────── 配色與狀態 ─────────────────────────
BG, PANEL, PANEL_EDGE = "#0B1220", "#111B2E", "#22314D"
TEXT, MUTED = "#E6EDF7", "#8EA0BC"
TEAL, AMBER, VIOLET, ROSE, SKY = "#2DD4BF", "#F5B84B", "#A78BFA", "#FB7185", "#60A5FA"
ON, PR, RE, EM = 0, 1, 2, 3                       # 在站 / 整備中 / 租賃中 / 空車位
S_NAME = {ON: "在站", PR: "整備中", RE: "租賃中"}
S_COL = {ON: TEAL, PR: AMBER, RE: VIOLET}
S_BG = {ON: "#0E2A2E", PR: "#2B2415", RE: "#1F1A3D"}
BAY_EMPTY = "#1B2A44"

# 每一單的時間軸（0~1）
T_RET = 0.16                 # 狀態更新（還站、整備、上架）完成
T_HL0, T_HL1 = 0.16, 0.32    # 選定車位（脈動光圈）
T_GO0, T_GO1 = 0.32, 0.64    # 車輛開出 / 未覆蓋印章
T_SET = 0.66                 # 結算

# ───────────────────────── 中文字型 ─────────────────────────
_PREFERRED = ["Noto Sans CJK TC", "Noto Sans TC", "Noto Sans CJK JP", "Noto Sans CJK SC",
              "Noto Sans CJK HK", "Microsoft JhengHei", "PingFang TC", "Heiti TC",
              "WenQuanYi Zen Hei", "SimHei", "Droid Sans Fallback"]
_FONT_URLS = [
    "https://github.com/notofonts/noto-cjk/raw/main/Sans/Variable/TTF/Subset/NotoSansTC-VF.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/notosanstc/NotoSansTC%5Bwght%5D.ttf",
]


def _register(paths):
    for p in paths:
        try:
            fm.fontManager.addfont(p)
        except Exception:
            pass


def _pick():
    names = {f.name for f in fm.fontManager.ttflist}
    for n in _PREFERRED:
        if n in names:
            return n
    return None


def setup_cjk_font(verbose=True):
    """依序：現有字型 → 掃描系統 CJK 字型 → apt 安裝 → 下載 Noto Sans TC。"""
    name = _pick()
    if not name:
        _register(glob.glob("/usr/share/fonts/**/*CJK*", recursive=True))
        name = _pick()
    if not name:
        try:
            subprocess.run(["apt-get", "-qq", "install", "-y", "fonts-noto-cjk"],
                           check=True, capture_output=True, timeout=300)
        except Exception:
            pass
        _register(glob.glob("/usr/share/fonts/**/*CJK*", recursive=True))
        name = _pick()
    if not name:
        for url in _FONT_URLS:
            try:
                path = os.path.join(os.getcwd(), "NotoSansTC.ttf")
                urllib.request.urlretrieve(url, path)
                _register([path])
                name = _pick()
                if name:
                    break
            except Exception:
                continue
    if name:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        if verbose:
            print(f"[字型] 使用 {name}")
    elif verbose:
        print("[字型] 找不到中文字型：Colab 請先執行 !apt-get -qq install -y fonts-noto-cjk")
    return name


# ───────────────────────── 小工具 ─────────────────────────
def _clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def _ease_out(t):
    t = _clamp(t)
    return 1 - (1 - t) ** 3


def _ease_inout(t):
    t = _clamp(t)
    return t * t * (3 - 2 * t)


def _rbox(x, y, w, h, r, **kw):
    return FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", **kw)


def _car(ax, x, y, w, h, color, alpha=1.0, z=4):
    """原創的簡易車輛圖示（車身＋車艙＋車窗＋輪子）。"""
    ax.add_patch(_rbox(x + w * .06, y + h * .22, w * .88, h * .36, h * .12, fc=color, ec="none", alpha=alpha, zorder=z))
    ax.add_patch(_rbox(x + w * .24, y + h * .50, w * .52, h * .32, h * .10, fc=color, ec="none", alpha=alpha, zorder=z))
    ax.add_patch(_rbox(x + w * .30, y + h * .56, w * .40, h * .19, h * .05, fc=BG, ec="none", alpha=alpha * .55, zorder=z + .1))
    for fx in (.27, .73):
        ax.add_patch(Circle((x + w * fx, y + h * .22), h * .11, fc=BG, ec="#C9D4E6", lw=.8, alpha=alpha, zorder=z + .2))


# ───────────────────────── 資料處理 ─────────────────────────
def _normalize_allocation(df, col_date, col_onhand, col_other, allocation, verbose=True):
    """讓「同月配置數」一致，並令 not_on_hand = 配置數 - on_hand。

    allocation:
        None   → 以該月第一單（月初）的 on_hand + not_on_hand 當作該月配置數，其餘訂單一律改成一致
        int    → 所有月份都用這個配置數
        dict   → {"2026-10": 8, "2026-11": 10}
    """
    df = df.copy()
    month = df[col_date].dt.to_period("M")
    on = df[col_onhand].astype(int)
    if allocation is None:
        if col_other not in df.columns:
            raise ValueError(f"需要 {col_other} 欄位（或 preparing + renting 兩欄），或改傳 allocation=配置數")
        total = on + df[col_other].astype(int)
        alloc = total.groupby(month).transform("first")
        n_bad = int((total != alloc).sum())
        if n_bad and verbose:
            print(f"[配置] 有 {n_bad} 單的配置數與該月月初不一致，已統一為月初配置數")
    elif isinstance(allocation, dict):
        alloc = month.astype(str).map(allocation)
        if alloc.isna().any():
            raise ValueError("allocation 字典缺少部分月份：" + ", ".join(sorted(set(month.astype(str)[alloc.isna()]))))
        alloc = alloc.astype(int)
    else:
        alloc = pd.Series(int(allocation), index=df.index)
    over = int((on > alloc).sum())
    if over and verbose:
        print(f"[配置] 有 {over} 單 on_hand 大於配置數，not_on_hand 以 0 處理")
    df[col_other] = (alloc - on).clip(lower=0)
    return df


def _derive_renting(df, covered, station, c):
    """由訂單推算租賃中：同月內、已出車且尚未回到本站者（異站還車的視為不回本站）。"""
    dates = df[c["date"]].to_numpy("datetime64[D]")
    days = df[c["days"]].astype(int).to_numpy().astype("timedelta64[D]")
    rets = df[c["ret"]].to_numpy()
    month = df[c["date"]].dt.to_period("M").to_numpy()
    rent = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        k = 0
        for j in range(i):
            if covered[j] and month[j] == month[i]:
                back = rets[j] == station and dates[j] + days[j] <= dates[i]
                k += 0 if back else 1
        rent[i] = k
    return rent


def _resolve_status(df, covered, station, c, verbose=True):
    """回傳 (on, preparing, renting) 三個陣列，加總 = 配置數。"""
    on = df[c["on"]].astype(int).to_numpy()
    alloc = on + df[c["other"]].astype(int).to_numpy()
    room = alloc - on                                  # 不在站的總數 = 整備中 + 租賃中
    has_p, has_r = c["prep"] in df.columns, c["rent"] in df.columns
    if has_p:
        prep = np.clip(df[c["prep"]].astype(int).to_numpy(), 0, room)
        rent = room - prep
        if has_r and verbose:
            given = df[c["rent"]].astype(int).to_numpy()
            n_bad = int((given != rent).sum())
            if n_bad:
                print(f"[狀態] 有 {n_bad} 單 在站+整備中+租賃中 ≠ 配置數，已以配置數為準")
    elif has_r:
        rent = np.clip(df[c["rent"]].astype(int).to_numpy(), 0, room)
        prep = room - rent
    else:
        rent = np.minimum(_derive_renting(df, covered, station, c), room)
        prep = room - rent
        if verbose:
            print("[狀態] 資料沒有 preparing / renting 欄位：租賃中由訂單推算，整備中 = 配置 - 在站 - 租賃中")
    return on, prep, rent


def make_demo_orders(station="台北", others=("桃園", "台中"), on0=5, prep0=3, days=14,
                     prep_days=1, seed=3, month="2026-10"):
    """示範資料（在站／整備中／租賃中都照實際流動計算，不是隨機）：
    - 月初：在站 on0、整備中 prep0、租賃中 0（配置數 = on0 + prep0，整月固定）
    - 月初整備中的車，在第 3、7 天上架（整備中 -1、在站 +1）
    - 每一單進來：有車就開走（在站 -1、租賃中 +1）；沒車就是未覆蓋
    - 同站還車的訂單，到 出車日 + 用車天數 那天回站，先進整備 prep_days 天（租賃中 -1、整備中 +1），再上架（整備中 -1、在站 +1）
    - 異站還車的訂單，這台車不再回本站（本月仍計在租賃中）
    - 月初配置數比前一個有資料的月份增加時，動畫會標示「月初調車進站 +N 台」；減少時標示調車離站
    """
    rng = np.random.default_rng(seed)
    on, prep, rent = on0, prep0, 0
    returns, ready, rows = {}, {3: 1, 7: 1}, []
    for d in range(1, days + 1):
        k = ready.pop(d, 0); on += k; prep -= k                     # 上架
        r = returns.pop(d, 0); rent -= r; prep += r                 # 回站進整備
        if r:
            ready[d + prep_days] = ready.get(d + prep_days, 0) + r
        for _ in range(int(rng.choice([0, 1, 1, 2, 2, 3]))):
            dur = int(rng.integers(1, 5))
            ret = str(rng.choice([station, *others], p=[.6, .2, .2]))
            rows.append(dict(pickup_date=f"{month}-{d:02d}", duration_days=dur, pickup_station=station,
                             return_station=ret, on_hand=on, preparing=prep, renting=rent,
                             not_on_hand=prep + rent))
            if on > 0:
                on -= 1; rent += 1
                if ret == station:
                    returns[d + dur] = returns.get(d + dur, 0) + 1
    return pd.DataFrame(rows)


def make_demo_event_log(month="2026-10"):
    """產生「車輛狀態異動紀錄」示範資料。

    每列是一個站點的異動，on_hand/preparing/renting 是異動完成後的狀態快照。
    調度會產生兩列：調出站一列「調度調出」、調入站一列「調度調入」。
    同樣地，異站還車會分別記錄租賃結束站與實際還車站的異動。
    """
    states = {
        "台北": {"on_hand": 5, "preparing": 2, "renting": 0},
        "桃園": {"on_hand": 4, "preparing": 1, "renting": 0},
        "台中": {"on_hand": 3, "preparing": 1, "renting": 0},
    }
    rows = []
    seq = 0

    def add(ts, event_type, station, count=0, from_station=None, to_station=None, note=""):
        nonlocal seq
        seq += 1
        st = states[station]
        rows.append({
            "event_id": f"EV{seq:03d}", "event_date": f"{month}-{ts}",
            "event_type": event_type, "station": station,
            "from_station": from_station, "to_station": to_station,
            "vehicle_count": int(count), "on_hand": st["on_hand"],
            "preparing": st["preparing"], "renting": st["renting"], "note": note,
        })

    # 期初狀態（每站一筆）
    for st in states:
        add("01 08:00:00", "期初狀態", st, note="月初盤點")

    # 租賃出車：台北在站 -1、租賃中 +1
    states["台北"]["on_hand"] -= 1; states["台北"]["renting"] += 1
    add("01 09:00:00", "租賃出車", "台北", 1, "台北", "台北", "同站租還示例")

    # 同站還車：租賃中 -1、整備中 +1
    states["台北"]["renting"] -= 1; states["台北"]["preparing"] += 1
    add("02 10:00:00", "同站還車", "台北", 1, "台北", "台北")

    # 調度台北 → 桃園，拆成兩筆站點異動
    states["台北"]["on_hand"] -= 1
    add("03 11:00:00", "調度調出", "台北", 1, "台北", "桃園")
    states["桃園"]["on_hand"] += 1
    add("03 11:00:00", "調度調入", "桃園", 1, "台北", "桃園")

    # 台北租賃出車、異站還到台中
    states["台北"]["on_hand"] -= 1; states["台北"]["renting"] += 1
    add("04 12:00:00", "租賃出車", "台北", 1, "台北", "台中", "異站還車示例")
    # 租賃結束時，台北只減少租賃中；車輛不回台北
    states["台北"]["renting"] -= 1
    add("06 12:00:00", "異站還車結束", "台北", 1, "台北", "台中")
    # 實際還車站台中收到車輛，先進整備
    states["台中"]["preparing"] += 1
    add("06 12:00:00", "異站還車入站", "台中", 1, "台北", "台中")

    # 台北整備完成上架
    states["台北"]["preparing"] -= 1; states["台北"]["on_hand"] += 1
    add("07 09:00:00", "整備完成", "台北", 1, "台北", "台北")

    df = pd.DataFrame(rows)
    df["event_date"] = pd.to_datetime(df["event_date"], format="%Y-%m-%d %H:%M:%S")
    return df.sort_values(["event_date", "event_id"], kind="stable").reset_index(drop=True)




# ───────────────────────── 輸入資料檢查 ─────────────────────────
EVENT_REQUIRED_COLUMNS = (
    "event_date", "event_type", "station", "on_hand", "preparing", "renting"
)
EVENT_COUNT_COLUMNS = ("on_hand", "preparing", "renting")
_EVENT_DELTAS = {
    "租賃出車": {"on_hand": -1, "preparing": 0, "renting": 1},
    "同站還車": {"on_hand": 0, "preparing": 1, "renting": -1},
    "整備完成": {"on_hand": 1, "preparing": -1, "renting": 0},
    "調度調出": {"on_hand": -1, "preparing": 0, "renting": 0},
    "調度調入": {"on_hand": 1, "preparing": 0, "renting": 0},
    "異站還車結束": {"on_hand": 0, "preparing": 0, "renting": -1},
    "異站還車入站": {"on_hand": 0, "preparing": 1, "renting": 0},
}

def validate_event_log(event_log: pd.DataFrame, *, strict_transitions: bool = True) -> list[str]:
    """檢查車輛狀態異動紀錄。通過回傳空清單；有問題時回傳可讀的錯誤清單。

    檢查必填欄位、空值、日期、狀態台數、重複 event_id，
    並（預設）依事件類型驗證相鄰快照的狀態增減。站點配置可因調度／異站還車改變。
    此函數不會修改輸入資料。
    """
    errors: list[str] = []
    if not isinstance(event_log, pd.DataFrame):
        return ["輸入必須是 pandas.DataFrame。"]
    missing = [c for c in EVENT_REQUIRED_COLUMNS if c not in event_log.columns]
    if missing:
        return [f"缺少必要欄位：{', '.join(missing)}"]
    if event_log.empty:
        return ["資料沒有任何紀錄。"]
    df = event_log.copy()
    for col in ("event_type", "station"):
        if df[col].isna().any() or df[col].astype(str).str.strip().eq("").any():
            errors.append(f"{col} 不可為空。")
    dates = pd.to_datetime(df["event_date"], errors="coerce")
    if dates.isna().any():
        errors.append(f"event_date 有 {int(dates.isna().sum())} 筆無法解析的日期。")
    for col in EVENT_COUNT_COLUMNS:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any():
            errors.append(f"{col} 有空值或非數字。")
            continue
        if (values < 0).any():
            errors.append(f"{col} 不可為負數。")
        if ((values % 1) != 0).any():
            errors.append(f"{col} 必須是整數台數。")
    if "vehicle_count" in df.columns:
        counts = pd.to_numeric(df["vehicle_count"], errors="coerce")
        if counts.isna().any() or (counts < 0).any() or ((counts % 1) != 0).any():
            errors.append("vehicle_count 必須是非負整數。")
    if "event_id" in df.columns:
        ids = df["event_id"].dropna().astype(str)
        if ids.duplicated().any():
            errors.append("event_id 有重複值。")
    if dates.isna().any() or any(pd.to_numeric(df[c], errors="coerce").isna().any() for c in EVENT_COUNT_COLUMNS):
        return errors
    df["_validated_date"] = dates
    df = df.sort_values(["_validated_date"] + (["event_id"] if "event_id" in df.columns else []), kind="stable")
    df["_total"] = df[list(EVENT_COUNT_COLUMNS)].sum(axis=1)
    for station, group in df.groupby("station", sort=False):
        # 站點配置總數可因調度調入／調出及異站還車而改變，因此不要求整段期間固定；
        # 由下方事件轉換規則驗證每次改變是否有對應事件與正確快照。
        if not strict_transitions:
            continue
        prev = None
        for _, row in group.iterrows():
            typ = str(row["event_type"])
            current = {c: int(row[c]) for c in EVENT_COUNT_COLUMNS}
            count = int(row.get("vehicle_count", 1)) if pd.notna(row.get("vehicle_count", 1)) else 1
            if typ == "期初狀態":
                prev = current
                continue
            if typ not in _EVENT_DELTAS:
                errors.append(f"站點「{station}」事件 {typ!r} 不在已知事件規則中，無法驗證狀態變化。")
                prev = current
                continue
            if prev is None:
                errors.append(f"站點「{station}」第一筆事件是 {typ!r}，缺少「期初狀態」作為基準。")
                prev = current
                continue
            expected = {c: prev[c] + _EVENT_DELTAS[typ][c] * count for c in EVENT_COUNT_COLUMNS}
            if current != expected:
                errors.append(
                    f"站點「{station}」事件 {typ!r}（{row['_validated_date']}）狀態不符："
                    f"預期 {expected}，實際 {current}。請檢查事件數量、事件順序與狀態快照。"
                )
            prev = current
    return errors

def validate_legacy_orders(orders: pd.DataFrame, *, col_date="pickup_date", col_days="duration_days",
                           col_return="return_station", col_pickup="pickup_station", col_onhand="on_hand") -> list[str]:
    """檢查舊版租賃訂單格式與基本數值；回傳錯誤清單，不修改輸入資料。"""
    if not isinstance(orders, pd.DataFrame):
        return ["輸入必須是 pandas.DataFrame。"]
    required = [col_date, col_days, col_return, col_onhand]
    missing = [c for c in required if c not in orders.columns]
    if missing:
        return [f"舊版訂單缺少必要欄位：{', '.join(missing)}"]
    errors = []
    if orders.empty:
        errors.append("資料沒有任何訂單。")
    if pd.to_datetime(orders[col_date], errors="coerce").isna().any():
        errors.append(f"{col_date} 有空值或無法解析的日期。")
    days = pd.to_numeric(orders[col_days], errors="coerce")
    if days.isna().any() or (days < 0).any() or ((days % 1) != 0).any():
        errors.append(f"{col_days} 必須是非負整數。")
    on = pd.to_numeric(orders[col_onhand], errors="coerce")
    if on.isna().any() or (on < 0).any() or ((on % 1) != 0).any():
        errors.append(f"{col_onhand} 必須是非負整數。")
    if orders[col_return].isna().any():
        errors.append(f"{col_return} 不可為空。")
    if col_pickup in orders.columns and orders[col_pickup].isna().any():
        errors.append(f"{col_pickup} 不可為空。")
    for col in ("preparing", "renting", "not_on_hand"):
        if col in orders.columns:
            vals = pd.to_numeric(orders[col], errors="coerce")
            if vals.isna().any() or (vals < 0).any() or ((vals % 1) != 0).any():
                errors.append(f"{col} 必須是非負整數。")
    return errors

def _validate_or_raise(df: pd.DataFrame, event_mode: bool, c: dict) -> None:
    problems = (validate_event_log(df) if event_mode else validate_legacy_orders(
        df, col_date=c["date"], col_days=c["days"], col_return=c["ret"],
        col_pickup=c["pick"], col_onhand=c["on"]))
    if problems:
        preview = "\n - ".join(problems[:20])
        more = f"\n另有 {len(problems) - 20} 個問題未列出。" if len(problems) > 20 else ""
        raise ValueError("輸入資料檢查未通過：\n - " + preview + more)


# ───────────────────────── 逐台車輛狀態追蹤 ─────────────────────────
def _sorted_state(max_total, on, pr, re_):
    s = np.full(max_total, EM)
    s[:on] = ON; s[on:on + pr] = PR; s[on + pr:on + pr + re_] = RE
    return s


def _counts(s):
    return int((s == ON).sum()), int((s == PR).sum()), int((s == RE).sum())


def _apply_flows(s, target):
    """把「狀態計數」的變化拆成逐台車的轉換（例：租賃中→整備中、整備中→在站）。"""
    s = s.copy()
    cur = _counts(s)
    delta = [target[k] - cur[k] for k in range(3)]
    sources = Counter({k: -delta[k] for k in range(3) if delta[k] < 0})
    pref = {ON: [PR, RE], PR: [RE, ON], RE: [ON, PR]}
    flows = Counter()
    for sink in (ON, PR, RE):                                # 先填「在站」，再整備中，最後租賃中
        for _ in range(max(0, delta[sink])):
            src = next(x for x in pref[sink] if sources[x] > 0)
            sources[src] -= 1
            bay = int(np.where(s == src)[0][0])
            s[bay] = sink
            flows[(src, sink)] += 1
    return s, flows


# ───────────────────────── 核心：建立畫面 ─────────────────────────
def _build(orders, station, c, fps, frames_per_order, max_orders, dpi, allocation=None):
    setup_cjk_font()
    event_mode = {"event_type", "station", "event_date", "on_hand", "preparing", "renting"}.issubset(orders.columns)
    df = orders.copy()
    _validate_or_raise(df, event_mode, c)
    if event_mode:
        # 異動紀錄的狀態欄位代表「異動完成後」的站點快照；不套用舊版固定月配置數規則。
        c = c.copy()
        c.update(date="event_date", pick="station", on="on_hand", prep="preparing", rent="renting")
        if "duration_days" not in df.columns:
            df["duration_days"] = 0
        c["days"] = "duration_days"
        if "to_station" not in df.columns:
            df["to_station"] = df["station"]
        c["ret"] = "to_station"
        c["other"] = "_event_not_on_hand"
        df[c["other"]] = df[c["prep"]].astype(int) + df[c["rent"]].astype(int)
    else:
        if c["pick"] not in df.columns:
            df[c["pick"]] = df[c["ret"]]
    df[c["date"]] = pd.to_datetime(df[c["date"]])
    df = df[df[c["pick"]] == station].sort_values(c["date"], kind="stable").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"沒有「{station}」的車輛狀態異動紀錄／訂單")
    if c["prep"] in df.columns and c["rent"] in df.columns:
        # 整備中與租賃中分開保留；兩者合計作為不在站台數，避免沿用過期的 not_on_hand。
        df[c["other"]] = df[c["prep"]].astype(int) + df[c["rent"]].astype(int)
    if not event_mode:
        df = _normalize_allocation(df, c["date"], c["on"], c["other"], allocation)
    if max_orders:
        df = df.head(max_orders)
    n = len(df)
    on_arr = df[c["on"]].astype(int).to_numpy()
    if c["unc"] and c["unc"] in df.columns:
        unc = df[c["unc"]].astype(bool).to_numpy()
    else:
        unc = on_arr <= 0
    covered = np.ones(len(df), dtype=bool) if event_mode else ((~unc) & (on_arr > 0))
    cum_unc = np.cumsum(~covered)
    on_arr, pr_arr, re_arr = _resolve_status(df, covered, station, c)
    tot_arr = on_arr + pr_arr + re_arr
    max_total = int(tot_arr.max())

    # 舊版資料以月份配置數差異推估月初調度；異動紀錄模式已逐筆記錄調入／調出，不重複推估。
    month_keys = df[c["date"]].dt.to_period("M").to_numpy()
    month_first_total = {}
    month_first_idx = {}
    for idx, month_key in enumerate(month_keys):
        if month_key not in month_first_total:
            month_first_total[month_key] = int(tot_arr[idx])
            month_first_idx[month_key] = idx
    month_change = np.zeros(n, dtype=int)
    ordered_months = list(month_first_total)
    if not event_mode:
        for prev_month, curr_month in zip(ordered_months, ordered_months[1:]):
            month_change[month_first_idx[curr_month]] = month_first_total[curr_month] - month_first_total[prev_month]

    # 逐台車輛狀態：S0 單前 → S1 狀態更新後 → S2 出車後
    S0, S1, S2, USE, FLOW = [], [], [], [], []
    prev_end, prev_tot = None, None
    for i in range(n):
        tgt = (int(on_arr[i]), int(pr_arr[i]), int(re_arr[i]))
        if event_mode:
            s1 = _sorted_state(max_total, *tgt)
            s0 = s1.copy() if prev_end is None else prev_end.copy()
            fl = Counter()
            if prev_end is not None:
                before = _counts(prev_end)
                after = _counts(s1)
                for st, delta in zip((ON, PR, RE), (after[k] - before[k] for k in range(3))):
                    if delta:
                        fl[(EM if delta > 0 else st, st if delta > 0 else EM)] += abs(delta)
            s2, u = s1.copy(), None  # 異動紀錄的狀態已是事件完成後快照，不再模擬一次出車
        else:
            if prev_end is None or tot_arr[i] != prev_tot:     # 月初／配置數改變：重新排列
                s0 = _sorted_state(max_total, *tgt); s1, fl = s0.copy(), Counter()
            else:
                s0 = prev_end.copy(); s1, fl = _apply_flows(s0, tgt)
            s2, u = s1.copy(), None
            if covered[i]:
                u = int(np.where(s1 == ON)[0][0]); s2[u] = RE
        S0.append(s0); S1.append(s1); S2.append(s2); USE.append(u); FLOW.append(fl)
        prev_end, prev_tot = s2, tot_arr[i]

    # 停車格幾何
    cols = min(5 if max_total <= 15 else 6 if max_total <= 24 else 8, max(max_total, 1))
    rows = math.ceil(max_total / cols)
    gx0, gx1, gy0, gy1, gap = 0.95, 9.85, 2.6, 6.55, 0.22
    sw = min((gx1 - gx0 - (cols - 1) * gap) / cols, 2.0)
    sh = min((gy1 - gy0 - (rows - 1) * gap) / rows, sw * 0.62)
    sw = min(sw, sh / 0.62)
    grid_w, grid_h = cols * sw + (cols - 1) * gap, rows * sh + (rows - 1) * gap
    ox = gx0 + ((gx1 - gx0) - grid_w) / 2
    oy = gy1 - ((gy1 - gy0) - grid_h) / 2

    def bay_xy(k):
        r, cc = divmod(k, cols)
        return ox + cc * (sw + gap), oy - (r + 1) * sh - r * gap

    fig = plt.figure(figsize=(12, 6.75), dpi=dpi, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    total_frames = n * frames_per_order + int(fps * 2.5)   # 結尾停留

    def draw_bay(k, state, a=1.0):
        x, y = bay_xy(k)
        if state == EM:
            ax.add_patch(_rbox(x, y, sw, sh, .12, fc="none", ec=BAY_EMPTY, lw=1.2, ls=(0, (4, 3)), alpha=a, zorder=2))
            return
        col = S_COL[state]
        ax.add_patch(_rbox(x, y, sw, sh, .12, fc=S_BG[state], ec=col, lw=1.2, alpha=a * .95, zorder=2))
        _car(ax, x + sw * .1, y + sh * .12, sw * .8, sh * .76, col, a * (1.0 if state == ON else .88))

    def draw(f):
        if f >= n * frames_per_order:
            i, t = n - 1, 0.999
        else:
            i, t = divmod(int(f), frames_per_order)
            t = t / frames_per_order
        ax.clear()
        ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")
        r = df.iloc[i]
        s0, s1, s2, u, fl = S0[i], S1[i], S2[i], USE[i], FLOW[i]
        cov = bool(covered[i])
        total = int(tot_arr[i])
        ret = r[c["ret"]]
        # 比對站名時先去除前後空白，避免「台中」與「台中 」被誤判為同站。
        station_key = str(station).strip()
        ret_key = "" if pd.isna(ret) else str(ret).strip()
        same = ret_key == station_key
        settled = t >= T_SET
        event_type_now = str(r["event_type"]) if event_mode else ""
        cross_return_out = event_mode and event_type_now == "異站還車結束"
        cross_return_in = event_mode and event_type_now == "異站還車入站"
        legacy_cross_return = (not event_mode) and bool(ret_key) and not same
        a_ret = _ease_out(t / T_RET)
        has_fl = sum(fl.values()) > 0

        # ── 標題、階段、覆蓋率
        done = i + (1 if settled else 0)
        unc_shown = int(cum_unc[i] if settled else (cum_unc[i - 1] if i > 0 else 0))
        ax.text(0.6, 8.3, station, fontsize=28, weight="bold", color=TEXT, va="center")
        ax.text(0.6, 7.74, f"{r[c['date']]:%Y 年 %m 月 %d 日}", fontsize=12, color=MUTED, va="center")
        if event_mode:
            phase = str(r["event_type"])
            event_colors = {"租賃出車": SKY, "同站還車": TEAL, "異站還車結束": SKY,
                            "異站還車入站": SKY, "調度調出": AMBER, "調度調入": SKY,
                            "整備完成": AMBER, "期初狀態": MUTED}
            pcol = event_colors.get(phase, TEAL)
        elif has_fl and t < T_RET:
            phase, pcol = "車輛狀態更新", TEAL
        elif t < T_GO0:
            phase, pcol = "訂單進來", SKY
        elif t < T_SET:
            phase, pcol = ("車輛開出", SKY) if cov else ("無車可租", ROSE)
        else:
            phase, pcol = "結算", MUTED
        ax.text(8.0, 8.3, phase, fontsize=12, color=pcol, ha="center", va="center", weight="bold",
                bbox=dict(boxstyle="round,pad=0.5", fc=PANEL, ec=pcol, lw=1.2))
        if done > 0:
            rate = (done - unc_shown) / done
            ax.text(15.4, 8.35, f"{rate:.0%}", fontsize=30, weight="bold", ha="right", va="center",
                    color=TEAL if rate >= .9 else AMBER if rate >= .7 else ROSE)
        else:
            ax.text(15.4, 8.35, "—", fontsize=30, weight="bold", ha="right", va="center", color=MUTED)
        ax.text(15.4, 7.74, f"覆蓋率　未覆蓋 {unc_shown} ／ 已處理 {done} 單", fontsize=11,
                ha="right", va="center", color=MUTED)

        # ── 停車場面板
        flash = (not cov) and 0.30 <= t < 0.82
        ax.add_patch(_rbox(0.6, 2.3, 9.6, 5.0, .25, fc=PANEL, ec=ROSE if flash else PANEL_EDGE,
                           lw=2.2 if flash else 1.4, zorder=1))
        ax.text(0.95, 7.0, f"站內車位　配置 {total} 台", fontsize=10, color=MUTED, va="center", zorder=3)
        # 月初配置數增加／減少時，明確標示調車進出，避免車輛數看起來憑空改變。
        delta_alloc = int(month_change[i])
        if not event_mode and delta_alloc != 0 and t < 0.82:
            a_move = _ease_out(t / 0.10) * (1 - _clamp((t - 0.68) / 0.14))
            if delta_alloc > 0:
                move_text = f"月初調車進站 +{delta_alloc} 台"
                move_color = SKY
            else:
                move_text = f"月初調車離站 {delta_alloc} 台"
                move_color = AMBER
            ax.text(5.35, 6.58, move_text, fontsize=15, weight="bold", color=move_color,
                    ha="center", va="center", alpha=a_move,
                    bbox=dict(boxstyle="round,pad=0.35", fc=BG, ec=move_color, lw=1.8, alpha=a_move), zorder=9)
        if has_fl and t < 0.46:                                     # 狀態轉換標籤（例：租賃中 → 整備中）
            ab = _ease_out(t / 0.08) * (1 - _clamp((t - 0.34) / 0.10))
            for row_i, ((src, dst), cnt) in enumerate(list(fl.items())[:3]):
                src_name = S_NAME.get(src, "新增車輛")
                dst_name = S_NAME.get(dst, "調出本站")
                dst_color = S_COL.get(dst, SKY if dst == EM else MUTED)
                ax.text(9.85, 7.1 - row_i * .24, f"{src_name} → {dst_name}　×{cnt}", fontsize=10.5,
                        weight="bold", color=dst_color, ha="right", va="center", alpha=ab, zorder=9)

        # ── 各停車格
        for k in range(max_total):
            if t < T_RET and s0[k] != s1[k]:                        # 狀態更新：舊 → 新（淡入淡出）
                draw_bay(k, s0[k], 1 - a_ret); draw_bay(k, s1[k], a_ret)
            elif cov and k == u:
                if t < T_GO0:
                    draw_bay(k, s1[k])
                elif t < T_SET - 0.06:
                    draw_bay(k, EM)
                else:
                    draw_bay(k, EM); draw_bay(k, RE, _ease_inout((t - (T_SET - 0.06)) / 0.15))
            else:
                draw_bay(k, s1[k])

        # 被選中的車位：脈動光圈
        if not event_mode and cov and T_HL0 <= t < T_HL1:
            x, y = bay_xy(u)
            pulse = .5 + .5 * math.sin((t - T_HL0) / (T_HL1 - T_HL0) * math.pi * 2)
            ax.add_patch(_rbox(x - .06, y - .06, sw + .12, sh + .12, .16, fc="none", ec=SKY, lw=3,
                               alpha=.35 + .65 * pulse, zorder=6))

        # 出車：車輛升起淡出，光點沿弧線飛向「去向」
        if not event_mode and cov and T_GO0 <= t < T_GO1:
            uu = _ease_inout((t - T_GO0) / (T_GO1 - T_GO0))
            x, y = bay_xy(u)
            # 同站還車用青綠色；異站還車用藍色，讓車輛去向在動畫中一眼可辨。
            _car(ax, x + sw * .1, y + sh * .12 + .5 * uu, sw * .8, sh * .76,
                 TEAL if same else SKY, 1 - uu, z=7)
            p0 = np.array([x + sw / 2, y + sh / 2])
            p2 = np.array([10.95, 4.7])
            p1 = np.array([(p0[0] + p2[0]) / 2, max(p0[1], p2[1]) + 1.2])
            col = SKY if not same else TEAL
            for j in range(7):
                v = _clamp(uu - j * .035)
                if v <= 0:
                    continue
                pt = (1 - v) ** 2 * p0 + 2 * (1 - v) * v * p1 + v ** 2 * p2
                ax.add_patch(Circle(pt, .13 - j * .013, fc=col, ec="none", alpha=(1 - j / 7) * .9, zorder=8))

        # 異站還車：不只顯示文字框，改用「粗體事件標題 + 行進路線 + 動態車輛 + 大箭頭」
        # 讓觀眾一眼看到車輛離開本站、前往另一站，或從另一站抵達本站。
        if (cross_return_out or cross_return_in or legacy_cross_return) and 0.20 <= t < 0.96:
            if cross_return_out:
                other_station = str(r.get("to_station", "") or "目的站").strip()
                headline = "異站還車｜車輛離開本站"
                route_text = f"{station_key}  ━━━━━━━━━▶  {other_station}"
                accent = "#FF9F43"
                car_start, car_end = 2.15, 7.65
                car_y = 4.02
            elif cross_return_in:
                other_station = str(r.get("from_station", "") or "原租賃站").strip()
                headline = "異站還車｜車輛抵達本站"
                route_text = f"{other_station}  ━━━━━━━━━▶  {station_key}"
                accent = "#F97316"
                car_start, car_end = 7.65, 2.15
                car_y = 4.02
            else:
                other_station = ret_key or "異站"
                headline = "異站還車｜車輛不回本站"
                route_text = f"{station_key}  ━━━━━━━━━▶  {other_station}"
                accent = "#FF9F43"
                car_start, car_end = 2.15, 7.65
                car_y = 4.02

            # 大型事件底板與標題，使用橘色而非一般資訊提示藍框。
            overlay_alpha = 0.96
            ax.add_patch(_rbox(1.35, 3.38, 7.95, 2.28, .22, fc="#24180F",
                               ec=accent, lw=3.0, alpha=overlay_alpha, zorder=10))
            ax.text(5.32, 5.22, headline, fontsize=21, weight="bold", color=accent,
                    ha="center", va="center", zorder=13)
            ax.text(5.32, 4.78, route_text, fontsize=12.5, weight="bold", color=TEXT,
                    ha="center", va="center", zorder=13)

            # 動態車輛沿路線移動；行進方向與站點方向一致。
            travel = _ease_inout((t - 0.20) / 0.70)
            car_x = car_start + (car_end - car_start) * travel
            ax.plot([2.05, 8.45], [car_y, car_y], color=accent, lw=3.2,
                    alpha=0.78, solid_capstyle="round", zorder=11)
            # 在路線末端加上清楚的方向箭頭。
            if cross_return_in:
                arrow_start, arrow_end = (7.95, car_y), (2.45, car_y)
            else:
                arrow_start, arrow_end = (2.45, car_y), (7.95, car_y)
            ax.annotate("", xy=arrow_end, xytext=arrow_start,
                        arrowprops=dict(arrowstyle="-|>", color=accent, lw=3.0,
                                        mutation_scale=24), zorder=12)
            # 車身比一般停車格車輛大，並加上短暫光暈。
            pulse = 0.70 + 0.30 * math.sin((t - 0.20) * math.pi * 8)
            ax.add_patch(Circle((car_x + 0.27, car_y + 0.10), 0.48,
                                fc=accent, ec="none", alpha=0.18 * pulse, zorder=12))
            _car(ax, car_x, car_y - 0.20, 0.72, 0.52, accent, alpha=1.0, z=14)
            ax.text(5.32, 3.63,
                    f"目的站：{other_station}" if not cross_return_in else f"來源站：{other_station}",
                    fontsize=12, weight="bold", color=accent, ha="center", va="center", zorder=13)

        # 未覆蓋：印章
        if not event_mode and not cov and 0.30 <= t < 0.92:
            uu = _ease_out((t - 0.30) / 0.2)
            fade = 1 - _clamp((t - 0.82) / 0.1)
            ax.text(5.4, 4.65, "未覆蓋", fontsize=64 - 26 * uu, weight="bold", color=ROSE, ha="center", va="center",
                    rotation=7, alpha=uu * fade, zorder=9,
                    bbox=dict(boxstyle="round,pad=0.35", fc=BG, ec=ROSE, lw=4, alpha=uu * fade))

        # ── 右側：異動卡／租賃訂單卡
        tin = _ease_out(t / 0.18)
        a_card = tin * (1 - _clamp((t - 0.93) / 0.07))
        dx = (1 - tin) * 1.2
        if event_mode:
            etype = str(r["event_type"])
            event_colors = {"租賃出車": SKY, "同站還車": TEAL, "異站還車結束": SKY,
                            "異站還車入站": SKY, "調度調出": AMBER, "調度調入": SKY,
                            "整備完成": AMBER, "期初狀態": MUTED}
            route_color = event_colors.get(etype, TEAL)
            ax.add_patch(_rbox(10.8 + dx, 5.35, 4.6, 1.95, .2, fc=PANEL, ec=route_color, lw=1.8, alpha=a_card, zorder=3))
            ax.text(11.1 + dx, 6.98, f"狀態異動　#{i + 1}", fontsize=11, weight="bold", color=route_color, alpha=a_card, va="center", zorder=4)
            count = int(r.get("vehicle_count", 0) or 0)
            from_st = str(r.get("from_station", "") or "—")
            to_st = str(r.get("to_station", "") or "—")
            detail_rows = [("異動日期", f"{r[c['date']]:%Y/%m/%d %H:%M}"), ("異動類型", etype),
                           ("台數／路徑", f"{count} 台　{from_st} → {to_st}")]
            for j, (lab, val) in enumerate(detail_rows):
                ax.text(11.1 + dx, 6.48 - j * .42, lab, fontsize=11, color=MUTED, alpha=a_card, va="center", zorder=4)
                ax.text(12.1 + dx, 6.48 - j * .42, val, fontsize=11.5, color=route_color if j else TEXT, alpha=a_card, va="center", zorder=4)
        else:
            route_color = ROSE if not cov else TEAL if same else SKY
            ax.add_patch(_rbox(10.8 + dx, 5.35, 4.6, 1.95, .2, fc=PANEL, ec=route_color, lw=1.8, alpha=a_card, zorder=3))
            ax.text(11.1 + dx, 6.98, f"新訂單　#{i + 1}", fontsize=11, weight="bold", color=route_color, alpha=a_card, va="center", zorder=4)
            for j, (lab, val) in enumerate([("出車日", f"{r[c['date']]:%Y/%m/%d}"),
                                            ("用車", f"{int(r[c['days']])} 天"),
                                            ("還車", "本站（同站租還）" if same else f"異站還車 → {ret}")]):
                ax.text(11.1 + dx, 6.48 - j * .42, lab, fontsize=11, color=MUTED, alpha=a_card, va="center", zorder=4)
                value_color = route_color if j == 2 else TEXT
                ax.text(12.1 + dx, 6.48 - j * .42, val, fontsize=12, color=value_color, alpha=a_card, va="center", zorder=4)

        # ── 右側：車輛去向／異動方向
        t_on = 0.38 if not cov else 0.54
        a_dest = _ease_out((t - t_on) / 0.12) * (1 - _clamp((t - 0.95) / 0.05))
        if event_mode:
            etype = str(r["event_type"])
            dcol = {"調度調出": AMBER, "調度調入": SKY, "異站還車結束": SKY,
                    "異站還車入站": SKY, "同站還車": TEAL, "租賃出車": SKY,
                    "整備完成": AMBER, "期初狀態": MUTED}.get(etype, TEAL)
            ax.add_patch(_rbox(10.8, 4.2, 4.6, .95, .2, fc=PANEL, ec=dcol, lw=1.8, zorder=3))
            ax.text(11.1, 4.9, "異動說明", fontsize=9.5, color=MUTED, va="center", alpha=.9, zorder=4)
            if a_dest > 0:
                fs, ts = str(r.get("from_station", "") or "—"), str(r.get("to_station", "") or "—")
                cnt = int(r.get("vehicle_count", 0) or 0)
                note = str(r.get("note", "") or "")
                main = f"{etype}　{cnt} 台"
                ax.text(11.1, 4.48, main, fontsize=14, weight="bold", color=dcol, alpha=a_dest, va="center", zorder=4)
        else:
            dcol = ROSE if not cov else TEAL if same else SKY
            ax.add_patch(_rbox(10.8, 4.2, 4.6, .95, .2, fc=PANEL, ec=dcol if a_dest > 0 else PANEL_EDGE, lw=1.8, zorder=3))
            ax.text(11.1, 4.9, "車輛去向", fontsize=9.5, color=MUTED, va="center", alpha=.9, zorder=4)
            if a_dest > 0:
                main = "未覆蓋　無可租車輛" if not cov else "同站租還" if same else f"異站還車 → {ret}（不回本站）"
                ax.text(11.1, 4.48, main, fontsize=16, weight="bold", color=dcol, alpha=a_dest, va="center", zorder=4)
                if cov and same:
                    ax.text(15.2, 4.48, f"{int(r[c['days']])} 天後回", fontsize=10, color=MUTED, alpha=a_dest, ha="right", va="center", zorder=4)

        # ── 右側：近期紀錄
        ax.add_patch(_rbox(10.8, 2.3, 4.6, 1.75, .2, fc=PANEL, ec=PANEL_EDGE, lw=1.4, zorder=3))
        ax.text(11.1, 3.8, "近期紀錄", fontsize=10, color=MUTED, va="center", zorder=4)
        idxs = list(range(max(0, i - 3), i + (1 if settled else 0)))[-4:]
        for row, idx in enumerate(idxs):
            rr = df.iloc[idx]
            c_ = not covered[idx]
            if event_mode:
                txt = str(rr["event_type"])
                col = {"租賃出車": SKY, "同站還車": TEAL, "異站還車結束": SKY,
                       "異站還車入站": SKY, "調度調出": AMBER, "調度調入": SKY,
                       "整備完成": AMBER, "期初狀態": MUTED}.get(txt, TEAL)
            else:
                txt = "未覆蓋" if c_ else "同站租還" if rr[c["ret"]] == station else f"異站還車 → {rr[c['ret']]}"
                col = ROSE if c_ else TEAL if rr[c["ret"]] == station else SKY
            al = _ease_out((t - T_SET) / 0.15) if idx == i else 1.0
            yy = 3.45 - row * .30
            ax.add_patch(Circle((11.18, yy), .06, fc=col, ec="none", alpha=al, zorder=4))
            ax.text(11.4, yy, f"{rr[c['date']]:%m/%d}", fontsize=10, color=MUTED, alpha=al, va="center", zorder=4)
            ax.text(12.3, yy, txt, fontsize=10.5, color=col, alpha=al, va="center", zorder=4)

        # ── 左下：即時車數（三種狀態分開）
        if settled:
            n_on, n_pr, n_re = _counts(s2)
        elif t >= T_RET:
            n_on, n_pr, n_re = _counts(s1)
        else:
            c0, c1 = _counts(s0), _counts(s1)
            n_on = int(round(c0[0] + (c1[0] - c0[0]) * a_ret))
            n_pr = int(round(c0[1] + (c1[1] - c0[1]) * a_ret))
            n_re = total - n_on - n_pr
        for (x0, st, label, val) in [(0.6, ON, "在站可租", n_on), (3.75, PR, "整備中（等上架）", n_pr),
                                     (7.35, RE, "租賃中", n_re), (10.8, None, "配置總數", total)]:
            if st is not None:
                _car(ax, x0, 1.28, .85, .55, S_COL[st])
            xt = x0 + (1.05 if st is not None else 0)
            ax.text(xt, 1.78, label, fontsize=10, color=MUTED, va="center")
            ax.text(xt, 1.40, f"{val}", fontsize=20, weight="bold", color=S_COL[st] if st is not None else TEXT, va="center")

        # ── 進度條（每一單一個刻度）
        ax.add_patch(_rbox(0.6, 0.5, 14.8, .08, .04, fc=PANEL_EDGE, ec="none"))
        ax.add_patch(_rbox(0.6, 0.5, max(.08, 14.8 * (i + t) / n), .08, .04, fc=TEAL, ec="none"))
        if n <= 60:
            for k in range(n + 1):
                ax.add_patch(Rectangle((0.6 + 14.8 * k / n - .01, 0.46), .02, .16, fc=MUTED, ec="none", alpha=.45))
        ax.text(0.6, 0.82, "異動進度" if event_mode else "訂單進度", fontsize=9, color=MUTED, va="center")
        ax.text(15.4, 0.82, f"{i + 1} / {n}", fontsize=9, color=MUTED, ha="right", va="center")

    return fig, draw, total_frames, n


def _cols(col_date, col_days, col_return, col_pickup, col_onhand, col_other, col_prep, col_rent, col_uncovered):
    return dict(date=col_date, days=col_days, ret=col_return, pick=col_pickup, on=col_onhand,
                other=col_other, prep=col_prep, rent=col_rent, unc=col_uncovered)


# ───────────────────────── 輸出方式 1：mp4 / gif ─────────────────────────
def animate_station(
    orders: pd.DataFrame,
    station: str,
    out_path: str = "station.mp4",
    col_date="pickup_date",
    col_days="duration_days",
    col_return="return_station",
    col_pickup="pickup_station",
    col_onhand="on_hand",
    col_other="not_on_hand",
    col_prep="preparing",    # 整備中台數
    col_rent="renting",      # 租賃中台數
    col_uncovered=None,
    fps=20,
    frames_per_order=44,     # 每單幀數；fps=20 時約 2.2 秒/單。想更慢就加大
    max_orders=None,         # 只播前 N 單（先試跑用）
    dpi=100,
    allocation=None,         # 月配置數：None=用月初那一單；或給整數／{'2026-10': 8}
):
    c = _cols(col_date, col_days, col_return, col_pickup, col_onhand, col_other, col_prep, col_rent, col_uncovered)
    fig, draw, total, _ = _build(orders, station, c, fps, frames_per_order, max_orders, dpi, allocation)
    anim = FuncAnimation(fig, draw, frames=total, interval=1000 / fps)
    cb = lambda i, n: print(f"\r渲染 {i + 1}/{n}", end="") if i % 25 == 0 else None
    if out_path.lower().endswith(".mp4"):
        anim.save(out_path, writer="ffmpeg", fps=fps, dpi=dpi, progress_callback=cb,
                  extra_args=["-pix_fmt", "yuv420p"])
    else:
        anim.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi, progress_callback=cb)
    plt.close(fig)
    print(f"\n完成：{out_path}")
    return out_path


# ───────────────────────── 輸出方式 2：可拖動時間軸（Colab / Jupyter） ─────────────────────────
def interactive_station(
    orders: pd.DataFrame,
    station: str,
    col_date="pickup_date",
    col_days="duration_days",
    col_return="return_station",
    col_pickup="pickup_station",
    col_onhand="on_hand",
    col_other="not_on_hand",
    col_prep="preparing",
    col_rent="renting",
    col_uncovered=None,
    fps=12,
    frames_per_order=30,
    max_orders=None,
    dpi=80,
    allocation=None,
    prerender=True,          # True：先把所有畫面算好，拖動／播放就完全不卡（報告前先跑一次）
):
    """顯示播放器：▶ 播放／暫停、時間軸（可任意拖動）、上一單／下一單、速度。"""
    try:
        import ipywidgets as W
        from IPython.display import display
    except ImportError as e:
        raise ImportError("需要 ipywidgets：pip install ipywidgets（Colab 內建）") from e

    c = _cols(col_date, col_days, col_return, col_pickup, col_onhand, col_other, col_prep, col_rent, col_uncovered)
    fig, draw, total, n = _build(orders, station, c, fps, frames_per_order, max_orders, dpi, allocation)
    cache = {}

    def render(f):
        f = int(f)
        if f not in cache:
            draw(f)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, facecolor=BG)
            cache[f] = buf.getvalue()
        return cache[f]

    if prerender:
        for f in range(total):
            render(f)
            if f % 25 == 0:
                print(f"\r預先渲染 {f + 1}/{total}", end="")
        print("\r預先渲染完成，可以開始拖動時間軸。      ")

    img = W.Image(value=render(0), format="png")
    slider = W.IntSlider(min=0, max=total - 1, value=0, step=1, description="時間軸",
                         continuous_update=True, readout=False, layout=W.Layout(width="95%"))
    play = W.Play(min=0, max=total - 1, step=1, value=0, interval=int(1000 / fps), description="播放")
    W.jslink((play, "value"), (slider, "value"))
    speed = W.Dropdown(options=[("0.5×", 0.5), ("1×", 1.0), ("2×", 2.0)], value=1.0, description="速度",
                       layout=W.Layout(width="130px"))
    btn_prev = W.Button(description="◀ 上一單", layout=W.Layout(width="100px"))
    btn_next = W.Button(description="下一單 ▶", layout=W.Layout(width="100px"))
    label = W.Label()
    focus = int(frames_per_order * (T_HL0 + 0.04))   # 跳到「訂單剛進來」的畫面，方便停下來講

    def on_slider(change):
        f = change["new"]
        img.value = render(f)
        label.value = f"第 {min(f // frames_per_order, n - 1) + 1} / {n} 單"

    def jump(delta):
        play.playing = False if hasattr(play, "playing") else None
        i = min(slider.value // frames_per_order, n - 1)
        if delta < 0 and slider.value % frames_per_order > focus:    # 已看過本單後半段：先回到本單開頭
            tgt = i
        else:
            tgt = min(max(i + delta, 0), n - 1)
        slider.value = tgt * frames_per_order + focus

    slider.observe(on_slider, names="value")
    speed.observe(lambda ch: setattr(play, "interval", int(1000 / (fps * ch["new"]))), names="value")
    btn_prev.on_click(lambda _: jump(-1))
    btn_next.on_click(lambda _: jump(+1))
    on_slider({"new": 0})
    display(W.VBox([img, W.HBox([play, btn_prev, btn_next, speed, label]), slider]))
    plt.close(fig)


# ───────────────────────── 輸出方式 3：單一 HTML 檔（瀏覽器開啟，有時間軸） ─────────────────────────
def save_html_player(
    orders: pd.DataFrame,
    station: str,
    out_path: str = "station.html",
    fps=12,
    frames_per_order=16,     # 為了檔案大小，預設比 mp4 少
    dpi=72,
    max_orders=None,
    allocation=None,
    col_date="pickup_date", col_days="duration_days", col_return="return_station",
    col_pickup="pickup_station", col_onhand="on_hand", col_other="not_on_hand",
    col_prep="preparing", col_rent="renting", col_uncovered=None,
):
    """存成單一 .html：用瀏覽器開啟就有播放／暫停、逐格、可拖動的時間軸。報告時不用開 Colab。"""
    from matplotlib.animation import HTMLWriter
    c = _cols(col_date, col_days, col_return, col_pickup, col_onhand, col_other, col_prep, col_rent, col_uncovered)
    fig, draw, total, _ = _build(orders, station, c, fps, frames_per_order, max_orders, dpi, allocation)
    anim = FuncAnimation(fig, draw, frames=total, interval=1000 / fps)
    anim.save(out_path, writer=HTMLWriter(fps=fps, embed_frames=True, default_mode="once"), dpi=dpi)
    plt.close(fig)
    print(f"完成：{out_path}（{os.path.getsize(out_path) / 1e6:.1f} MB）")
    return out_path


if __name__ == "__main__":
    demo = make_demo_orders()
    print(demo[["pickup_date", "duration_days", "return_station", "on_hand", "preparing", "renting"]].head(14))
    animate_station(demo, "台北", "demo_station.gif", fps=15, frames_per_order=30, max_orders=8)
# The primary animation API is the single-station order simulation. Keep the
# event-log validation helpers above available for users migrating from the
# earlier repository API.
from .single_station import (  # noqa: E402
    animate_station,
    interactive_station,
    save_html_player,
    make_demo_orders,
    setup_cjk_font,
)
