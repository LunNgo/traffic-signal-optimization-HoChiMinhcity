#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_mapp_routes.py  –  MAPP SUMO Route Generator  v6 (HCMC Realism)
==========================================================
THAY ĐỔI THEO THỰC TẾ GIAO THÔNG TPHCM:

[1] TỰ ĐỘNG TÍNH TOÁN TỶ LỆ RẼ (KHÔNG CẦN MODULE NGOÀI)
    - Tự động đo góc (angle) giữa các đoạn đường tại nút giao.
    - Đi thẳng (chi phí x1.0) -> Rẽ phải (x1.5) -> Rẽ trái (x2.5) -> Quay đầu (x6.0).
    - Kết quả: Dijkstra tự động sinh ra tỷ lệ rẽ giống hệt thực tế mà không cần ép buộc.

[2] DỒN XE VÀO TRỤC CHÍNH (KHÔNG PHÂN BỔ ĐỀU)
    - COST_MAP được điều chỉnh để phạt rất nặng đường nhỏ (residential x3.5).
    - Bắt buộc xe phải luồn lách để thoát ra đường lớn (TC, TP) chạy cho nhanh.

[3] CHUYẾN ĐI CÓ MỤC ĐÍCH, ĐẾN NƠI LÀ BIẾN MẤT
    - Xóa bỏ việc ép xe chạy qua 'via' ngẫu nhiên gây tình trạng chạy lòng vòng.
    - OD (Origin-Destination) chia làm 4 loại chuyến đi thực tế: 
      + Rìa -> Trung tâm (Đi làm)
      + Trung tâm -> Rìa (Về nhà)
      + Rìa -> Rìa (Xuyên tâm)
      + Trung tâm -> Trung tâm (Nội bộ)
    - Xe đến đích sẽ biến mất, không gây kẹt xe ảo tích tụ.

[4] HÀNH VI XE HUNG HĂNG, LUỒN LÁCH, KHÔNG NHƯỜNG
    - Bật lại lcStrategic nhưng kết hợp với lcAssertive cao để xe luồn lách liên tục.
    - jmIgnoreFoeProb=1.0: Đèn xanh hoặc đường thẳng là chạy, tuyệt đối không nhường xe khác.
    - lcCooperative=0.0: Không hợp tác nhường làn.
"""

import xml.etree.ElementTree as ET
import random, os, sys, csv, heapq, math
from collections import defaultdict

# =============================================================================
# CẤU HÌNH
# =============================================================================
os.environ['SUMO_HOME'] = r'C:\Program Files (x86)\Eclipse\Sumo'

NET_FILE    = "mapp.net.xml"
OSM_FILE    = "mapp.osm"
CSV_FILE    = "Luuluongxe.csv"
ROUTES_OUT  = "mapp.rou.xml"
SUMOCFG_OUT = "mapp.sumocfg"

SIM_BEGIN = 0
SIM_END   = 3600

RATIO_MOTO  = 0.87
RATIO_CAR   = 0.10
RATIO_TRUCK = 0.03

# Tổng xe nền (ngoài CSV)
BG_VEHICLES = 4000

# Phân bổ thời gian: tuyến tính
SURGE_RATIO  = 0.15
SURGE_WINDOW = 600

TARGET_SPEED   = 11.0
DIJKSTRA_NOISE = 1.25
MAX_TRIES      = 12

SEED = 42
random.seed(SEED)

SKIP_TYPES = {
    "highway.footway", "highway.path", "highway.service",
    "highway.steps",   "highway.pedestrian"
}

# Ép xe dồn ra đường lớn, hạn chế chui vào đường nhỏ
COST_MAP = {
    "TC":                     0.30,
    "TP":                     0.45,
    "highway.primary":        0.40,
    "highway.primary_link":   0.50,
    "highway.secondary":      0.70,
    "highway.tertiary":       1.00,
    "highway.tertiary_link":  1.10,
    "highway.residential":    3.50, # Phạt rất nặng để không đi đường hẻm liên tục
    "default":                1.50,
}

# Ma trận OD hướng cho nhóm xe xuyên tâm
OD_PAIRS = [
    ("N", "S", 10), ("N", "E",  8), ("N", "W",  8),
    ("S", "N", 10), ("S", "E",  8), ("S", "W",  8),
    ("E", "W", 10), ("E", "N",  8), ("E", "S",  8),
    ("W", "E", 10), ("W", "N",  8), ("W", "S",  8),
]
_od_total = sum(w for _,_,w in OD_PAIRS)
OD_CDF = []
_cum = 0
for sd, sk, w in OD_PAIRS:
    _cum += w / _od_total
    OD_CDF.append((sd, sk, _cum))


# =============================================================================
# PHÂN TÍCH MẠNG & TOÁN HỌC GÓC RẼ
# =============================================================================
def cost_factor(eid, etype):
    if eid.startswith("TC"): return COST_MAP["TC"]
    if eid.startswith("TP"): return COST_MAP["TP"]
    for k, v in COST_MAP.items():
        if k.startswith("highway") and k in etype:
            return v
    return COST_MAP["default"]

def modify_network_speed(net_file, spd):
    if not os.path.exists(net_file): return
    tree = ET.parse(net_file)
    cnt = 0
    for lane in tree.getroot().iter("lane"):
        if lane.get("speed") is not None:
            lane.set("speed", f"{spd:.2f}"); cnt += 1
    tree.write(net_file, encoding="utf-8", xml_declaration=True)
    print(f"   => Cập nhật {cnt} làn về {spd} m/s.")

def load_csv(csv_file):
    vol = {}
    if not os.path.exists(csv_file): return vol
    with open(csv_file, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eid = row.get("Edge_ID", "").strip()
            vs  = row.get("Volume",  "").strip()
            if eid and vs:
                try: vol[eid] = int(float(vs))
                except: pass
    print(f"   => CSV: {len(vol)} cạnh")
    return vol

def load_network(net_file):
    with open(net_file, 'r', encoding='utf-8', errors='ignore') as f:
        root = ET.parse(f).getroot()

    jpos, dead = {}, set()
    for j in root.findall("junction"):
        jid = j.get("id")
        if j.get("x"):
            jpos[jid] = (float(j.get("x")), float(j.get("y")))
        if j.get("type") == "dead_end":
            dead.add(jid)

    edges = {}
    for e in root.findall("edge"):
        if e.get("function") == "internal": continue
        eid   = e.get("id", "")
        etype = e.get("type", "")
        if etype in SKIP_TYPES: continue
        lanes = e.findall("lane")
        if not lanes: continue
        dis = lanes[0].get("disallow", "")
        if dis == "all" or "passenger" in dis: continue
        edges[eid] = {
            "from":   e.get("from"),
            "to":     e.get("to"),
            "nlanes": len(lanes),
            "speed":  float(lanes[0].get("speed", TARGET_SPEED)),
            "length": float(lanes[0].get("length", 100.0)),
            "type":   etype,
        }

    def get_angle(dx1, dy1, dx2, dy2):
        """Tính góc rẽ tại nút giao để phân loại Trái/Phải/Thẳng."""
        ang1 = math.degrees(math.atan2(dy1, dx1))
        ang2 = math.degrees(math.atan2(dy2, dx2))
        diff = ang2 - ang1
        while diff > 180: diff -= 360
        while diff <= -180: diff += 360
        return diff

    adj = defaultdict(list)
    raw = set()
    for c in root.findall("connection"):
        fr, to = c.get("from", ""), c.get("to", "")
        if (fr and to and not fr.startswith(":") and not to.startswith(":") and fr in edges and to in edges):
            
            # Tính toán hình phạt hướng rẽ (Mô phỏng thực tế TPHCM)
            turn_penalty = 1.0
            pA = jpos.get(edges[fr]["from"])
            pB = jpos.get(edges[fr]["to"])
            pC = jpos.get(edges[to]["to"])
            
            if pA and pB and pC:
                dx1, dy1 = pB[0] - pA[0], pB[1] - pA[1]
                dx2, dy2 = pC[0] - pB[0], pC[1] - pB[1]
                if (dx1**2 + dy1**2) > 0.1 and (dx2**2 + dy2**2) > 0.1:
                    theta = get_angle(dx1, dy1, dx2, dy2)
                    if abs(theta) <= 30:
                        turn_penalty = 1.0    # Đi thẳng (Rẻ nhất, tỷ lệ chọn cao nhất)
                    elif -135 <= theta < -30:
                        turn_penalty = 1.5    # Rẽ phải (Dễ hơn)
                    elif 30 < theta <= 135:
                        turn_penalty = 2.5    # Rẽ trái (Đắt hơn vì cắt mặt)
                    else:
                        turn_penalty = 6.0    # Quay đầu U-turn (Rất đắt, hạn chế tối đa)
            
            cf = cost_factor(to, edges[to]["type"])
            base_cost = edges[to]["length"] / max(edges[to]["speed"], 0.1)
            # Tích hợp sẵn tỷ lệ rẽ tự nhiên vào đồ thị thuật toán
            cost = base_cost * cf * turn_penalty
            adj[fr].append((cost, to))
            raw.add((fr, to))

    all_to   = {d["to"]   for d in edges.values()}
    all_from = {d["from"] for d in edges.values()}
    fsrc  = [eid for eid, d in edges.items() if d["from"] in dead or d["from"] not in all_to]
    fsink = [eid for eid, d in edges.items() if d["to"] in dead or d["to"] not in all_from]

    all_cx = []
    for eid in edges:
        d = edges[eid]
        p1, p2 = jpos.get(d["from"]), jpos.get(d["to"])
        if p1 and p2: all_cx.append(((p1[0]+p2[0])/2, (p1[1]+p2[1])/2))
    cx = sum(x for x,_ in all_cx) / max(len(all_cx),1)
    cy = sum(y for _,y in all_cx) / max(len(all_cx),1)

    def dir4(eid):
        d = edges[eid]
        p1, p2 = jpos.get(d["from"]), jpos.get(d["to"])
        if p1 and p2: x, y = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
        elif p1: x, y = p1
        elif p2: x, y = p2
        else: return "C"
        dx, dy = x - cx, y - cy
        ang = math.degrees(math.atan2(dy, dx))
        if   -45 <= ang <  45: return "E"
        elif  45 <= ang < 135: return "N"
        elif ang >= 135 or ang < -135: return "W"
        else: return "S"

    src_by_dir, sink_by_dir = defaultdict(list), defaultdict(list)
    for e in fsrc: src_by_dir[dir4(e)].append(e)
    for e in fsink: sink_by_dir[dir4(e)].append(e)

    for d in "NSEW":
        if not src_by_dir[d]:  src_by_dir[d]  = fsrc
        if not sink_by_dir[d]: sink_by_dir[d] = fsink

    inner = [e for e in edges if e.startswith("TC") or e.startswith("TP") or COST_MAP.get(edges[e]["type"], 1.5) <= 1.0]

    print(f"   => edges={len(edges)} | fsrc={len(fsrc)} fsink={len(fsink)} | inner={len(inner)}")
    for d in "NSEW":
        print(f"      {d}: src={len(src_by_dir[d])} sink={len(sink_by_dir[d])}")

    return edges, fsrc, fsink, src_by_dir, sink_by_dir, inner, adj, raw


# =============================================================================
# DIJKSTRA + VALIDATE
# =============================================================================
def _dijkstra(src, sink, adj, noise=1.0):
    if src == sink: return None
    INF  = float("inf")
    dist = {src: 0.0}
    prev = {src: None}
    pq   = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == sink:
            path, cur = [], u
            while cur is not None:
                path.append(cur); cur = prev[cur]
            return list(reversed(path))
        if d > dist.get(u, INF) + 1e-9: continue
        for bc, v in adj.get(u, []):
            nd = d + bc * (random.uniform(1.0, noise) if noise > 1 else 1.0)
            if nd < dist.get(v, INF):
                dist[v] = nd; prev[v] = u
                heapq.heappush(pq, (nd, v))
    return None

def fix_route(route, edges, adj, raw):
    if not route: return None
    valid = [e for e in route if e in edges]
    if len(valid) < 2: return None

    result = [valid[0]]
    for cur in valid[1:]:
        prev_e = result[-1]
        if prev_e == cur: continue
        if (prev_e, cur) in raw:
            result.append(cur)
        else:
            bridge = _dijkstra(prev_e, cur, adj, noise=1.0)
            if bridge and len(bridge) > 1:
                result.extend(bridge[1:])
    return result if len(result) >= 2 else None

def find_route(src, sink, adj, edges, raw, via=None, noise=DIJKSTRA_NOISE, tries=MAX_TRIES):
    for attempt in range(tries):
        n = noise + attempt * 0.05
        if via and via in edges and via != src and via != sink:
            p1 = _dijkstra(src, via, adj, n)
            p2 = _dijkstra(via, sink, adj, n) if p1 else None
            raw_path = (p1 + p2[1:]) if (p1 and p2) else None
        else:
            raw_path = _dijkstra(src, sink, adj, n)

        if raw_path and len(raw_path) >= 2:
            fixed = fix_route(raw_path, edges, adj, raw)
            if fixed and len(fixed) >= 2:
                return fixed

    raw_path = _dijkstra(src, sink, adj, noise=1.0)
    if raw_path:
        return fix_route(raw_path, edges, adj, raw)
    return None


# =============================================================================
# TIỆN ÍCH OD
# =============================================================================
def surge_departs(n):
    if n <= 0: return []
    peak_end = min(SIM_BEGIN + SURGE_WINDOW, SIM_END)
    np_ = int(n * SURGE_RATIO); nr = n - np_
    t = ([random.uniform(SIM_BEGIN, peak_end) for _ in range(np_)]
         + [random.uniform(peak_end, SIM_END)  for _ in range(nr)])
    return sorted(t)

def batch_types(n):
    nm = int(n * RATIO_MOTO); nc = int(n * RATIO_CAR)
    nt = max(0, n - nm - nc)
    lst = ["motorcycle"] * nm + ["passenger"] * nc + ["delivery"] * nt
    random.shuffle(lst)
    return lst

def wc(pool, edges_dict):
    if not pool: return None
    w   = [float(edges_dict.get(e, {}).get("nlanes", 1)) for e in pool]
    tot = sum(w)
    r   = random.uniform(0, tot)
    cum = 0
    for e, wi in zip(pool, w):
        cum += wi
        if r <= cum: return e
    return pool[-1]

def pick_od(src_by_dir, sink_by_dir, inner, edges_dict):
    """
    Xác định điểm đi và đến cực kỳ rõ ràng để xe đến nơi là mất, tránh chạy lòng vòng.
    """
    r = random.random()
    if r < 0.35: # 35% Đi từ Rìa vào Trung tâm (Đi làm)
        d = random.choice(["N", "S", "E", "W"])
        src = wc(src_by_dir[d], edges_dict) if src_by_dir[d] else None
        sink = wc(inner, edges_dict) if inner else None
    elif r < 0.70: # 35% Đi từ Trung tâm ra Rìa (Về nhà)
        d = random.choice(["N", "S", "E", "W"])
        src = wc(inner, edges_dict) if inner else None
        sink = wc(sink_by_dir[d], edges_dict) if sink_by_dir[d] else None
    elif r < 0.85: # 15% Đi xuyên tâm (Rìa -> Rìa)
        r2 = random.random()
        src, sink = None, None
        for sd, sk, cum in OD_CDF:
            if r2 <= cum:
                src = wc(src_by_dir[sd], edges_dict) if src_by_dir[sd] else None
                sink = wc(sink_by_dir[sk], edges_dict) if sink_by_dir[sk] else None
                break
    else: # 15% Nội bộ Trung tâm -> Trung tâm
        src = wc(inner, edges_dict) if inner else None
        sink = wc(inner, edges_dict) if inner else None

    if not src or not sink or src == sink:
        pool = list(edges_dict.keys())
        if pool:
            src, sink = random.choice(pool), random.choice(pool)
            
    return src, sink

def mk_xml(vid, vtype, t, route):
    return (f'    <vehicle id="{vid}" type="{vtype}" depart="{t:.2f}" departLane="best">\n'
            f'        <route edges="{" ".join(route)}"/>\n'
            f'    </vehicle>')


# =============================================================================
# SINH XE
# =============================================================================
def generate_routes(edges, fsrc, fsink, src_by_dir, sink_by_dir,
                    inner, adj, raw, vol_map, routes_file):

    lines   = []
    counter = {"passenger": 0, "motorcycle": 0, "delivery": 0}
    PFX     = {"passenger": "car_", "motorcycle": "moto_", "delivery": "truck_"}
    skipped = 0

    def emit(vtype, t, route, tag):
        nonlocal skipped
        r2 = fix_route(route, edges, adj, raw) if route else None
        if not r2 or len(r2) < 2:
            skipped += 1; return False
        vid = f"{PFX[vtype]}{counter[vtype]}_{tag}"
        lines.append((t, mk_xml(vid, vtype, t, r2)))
        counter[vtype] += 1
        return True

    # ── Lớp A: Xe từ CSV (Luôn ép qua via_e để đảm bảo lưu lượng đếm)
    print("\n   [A] Xe CSV ...")
    csv_ok  = {e: v for e, v in vol_map.items() if e in edges and v > 0}
    cnt_a   = 0
    for via_e, vol in csv_ok.items():
        types = batch_types(vol)
        deps  = surge_departs(len(types))
        for t, vt in zip(deps, types):
            src, sink = pick_od(src_by_dir, sink_by_dir, inner, edges)
            if not src or not sink or src == sink:
                skipped += 1; continue
            route = find_route(src, sink, adj, edges, raw, via=via_e)
            if not route:
                route = find_route(src, sink, adj, edges, raw)
            if emit(vt, t, route, f"A"): cnt_a += 1
    print(f"      => {cnt_a} xe CSV")

    # ── Lớp B: Xe nền – KHÔNG DÙNG VIA ĐỂ TRÁNH LÒNG VÒNG
    print(f"\n   [B] {BG_VEHICLES} xe nền (Điểm đến dứt khoát, không đi vòng) ...")
    types_b = batch_types(BG_VEHICLES)
    deps_b  = surge_departs(BG_VEHICLES)
    cnt_b   = 0

    for t, vt in zip(deps_b, types_b):
        src, sink = pick_od(src_by_dir, sink_by_dir, inner, edges)
        if not src or not sink or src == sink:
            skipped += 1; continue

        route = find_route(src, sink, adj, edges, raw) # Tuyệt đối không nhét via ngẫu nhiên vào đây
        if emit(vt, t, route, "B"): cnt_b += 1

    print(f"      => {cnt_b} xe nền")

    # ── Lớp C: Coverage 
    print("\n   [C] Coverage toàn mạng ...")
    cover_pool = list(edges.keys())
    random.shuffle(cover_pool)
    n_cov    = len(cover_pool)
    types_c  = batch_types(n_cov)
    deps_c   = surge_departs(n_cov)
    cnt_c    = 0

    for t, vt, via_e in zip(deps_c, types_c, cover_pool):
        src, sink = pick_od(src_by_dir, sink_by_dir, inner, edges)
        if not src or not sink or src == sink:
            skipped += 1; continue
        route = find_route(src, sink, adj, edges, raw, via=via_e)
        if not route:
            route = find_route(src, sink, adj, edges, raw)
        if emit(vt, t, route, "C"): cnt_c += 1
    print(f"      => {cnt_c} xe coverage")

    # ── Ghi file ──────────────────────────────────────────────────────────
    lines.sort(key=lambda x: x[0])
    total = sum(counter.values())

    with open(routes_file, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n\n')

        f.write(
            '    \n'
            '    \n'
            '    \n'
            '    \n\n'
        )

        f.write(
            '    <vType id="passenger" vClass="passenger"\n'
            '           width="1.8" length="4.5" color="0.4,0.4,1.0"\n'
            '           accel="2.9" decel="4.5" emergencyDecel="7.0"\n'
            '           maxSpeed="12.0" minGap="0.5" tau="0.6" sigma="0.8"\n'
            '           latAlignment="arbitrary" minGapLat="0.2"\n'
            '           lcStrategic="1.0" lcSpeedGain="3.0" lcCooperative="0.0" lcKeepRight="0.1" lcAssertive="2.0"\n'
            '           jmIgnoreFoeProb="1.0" jmIgnoreFoeSpeed="50.0"\n'
            '           jmTimegapMinor="0.0" jmStoplineGap="0.0"\n'
            '           impatience="1.0"/>\n\n'
        )
        f.write(
            '    <vType id="motorcycle" vClass="motorcycle"\n'
            '           width="0.8" length="2.0" color="0.2,0.8,0.2"\n'
            '           accel="3.5" decel="5.5" emergencyDecel="7.5"\n'
            '           maxSpeed="15.0" minGap="0.1" tau="0.4" sigma="0.8"\n'
            '           latAlignment="arbitrary" minGapLat="0.1"\n'
            '           lcStrategic="1.0" lcSpeedGain="5.0" lcCooperative="0.0" lcKeepRight="0.1" lcAssertive="5.0"\n'
            '           jmIgnoreFoeProb="1.0" jmIgnoreFoeSpeed="50.0"\n'
            '           jmTimegapMinor="0.0" jmStoplineGap="0.0"\n'
            '           impatience="1.0"/>\n\n'
        )
        f.write(
            '    <vType id="delivery" vClass="delivery"\n'
            '           width="2.5" length="8.0" color="0.8,0.5,0.1"\n'
            '           accel="2.0" decel="4.0" emergencyDecel="6.5"\n'
            '           maxSpeed="12.0" minGap="1.0" tau="0.8" sigma="0.4"\n'
            '           latAlignment="center" minGapLat="0.5"\n' # Ép xe tải đi giữa làn, không đánh võng lách ngang
            '           lcStrategic="1.0" lcSpeedGain="1.0" lcCooperative="0.1" lcKeepRight="0.8" lcAssertive="1.0"\n' # Giảm sự hung hăng, ngoan ngoãn xếp hàng
            '           jmIgnoreFoeProb="1.0" jmIgnoreFoeSpeed="50.0"\n'
            '           jmTimegapMinor="0.0" jmStoplineGap="0.0"\n'
            '           impatience="1.0"/>\n\n'
        )

        f.write("\n".join(x for _, x in lines))
        f.write("\n</routes>\n")

    print(f"\n   ✓ {total} xe  (moto={counter['motorcycle']}, car={counter['passenger']}, truck={counter['delivery']})")
    print(f"   ! Bỏ qua {skipped} xe (không tìm được route)")
    return total

# =============================================================================
# VERIFY
# =============================================================================
def verify_routes(routes_file, edges, adj, raw):
    if not os.path.exists(routes_file): return
    print("\n   [Verify] Kiểm tra connections ...")
    root   = ET.parse(routes_file).getroot()
    total  = 0; broken = 0; examples = []
    for v in root.findall("vehicle"):
        r = v.find("route")
        if r is None: continue
        el = r.get("edges", "").split()
        total += 1
        for i in range(len(el)-1):
            if (el[i], el[i+1]) not in raw:
                broken += 1
                if len(examples) < 3:
                    examples.append(f"{v.get('id')}: {el[i]}→{el[i+1]}")
                break
    if broken == 0:
        print(f"   ✅ {total} xe – tất cả connections hợp lệ!")
    else:
        print(f"   ⚠️  {broken}/{total} xe vẫn có gap:")
        for ex in examples: print(f"      {ex}")

def print_stats(routes_file, edges):
    if not os.path.exists(routes_file): return
    try:
        root  = ET.parse(routes_file).getroot()
        s     = {"passenger": 0, "motorcycle": 0, "delivery": 0}
        total = 0
        edge_hits = defaultdict(int)
        for v in root.findall("vehicle"):
            vt = v.get("type", "")
            if vt in s: s[vt] += 1; total += 1
            r = v.find("route")
            if r is not None:
                for eid in r.get("edges", "").split():
                    edge_hits[eid] += 1
        covered  = sum(1 for e in edges if edge_hits.get(e, 0) > 0)
        zero_cov = [e for e in edges if edge_hits.get(e, 0) == 0]
        print("\n" + "═" * 58)
        print("   📊 THỐNG KÊ")
        print("═" * 58)
        print(f"   🏍️  Xe máy   : {s['motorcycle']:>7,}")
        print(f"   🚗  Ô tô     : {s['passenger']:>7,}")
        print(f"   🚛  Tải      : {s['delivery']:>7,}")
        print(f"   📌  TỔNG     : {total:>7,}")
        print(f"   🗺️  Coverage : {covered}/{len(edges)} ({covered/max(len(edges),1)*100:.1f}%)")
        if zero_cov:
            print(f"   ⚠️  Không có xe: {zero_cov[:6]}")
        print("═" * 58 + "\n")
    except Exception as e:
        print(f"[!] {e}")

# =============================================================================
# SUMOCFG
# =============================================================================
def write_sumocfg(out, net, rou):
    with open(out, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<configuration>\n\n')
        f.write(f'    <input>\n'
                f'        <net-file    value="{net}"/>\n'
                f'        <route-files value="{rou}"/>\n'
                f'    </input>\n\n')
        f.write(f'    <time>\n'
                f'        <begin value="{SIM_BEGIN}"/>\n'
                f'        <end   value="{SIM_END}"/>\n'
                f'    </time>\n\n')
        f.write('    <report>\n'
                '        <no-warnings value="true"/>\n'
                '        <no-step-log value="true"/>\n'
                '    </report>\n\n')
        f.write('    \n')
        f.write('    <processing>\n'
                '        <lateral-resolution      value="0.4"/>\n'
                '        <step-length             value="0.2"/>\n'
                '        <collision.action        value="none"/>\n'
                '        <collision.mingap-factor value="0"/>\n'
                '        <time-to-teleport        value="30"/>\n'
                '        <time-to-teleport.remove value="true"/>\n'
                '        <time-to-impatience      value="3"/>\n'
                '        <ignore-junction-blocker value="3"/>\n'
                '    </processing>\n\n'
                '</configuration>\n')
    print(f"✓ Đã ghi '{out}'")

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 66)
    print("  MAPP ROUTE GENERATOR  v6")
    print("=" * 66)
    print("  ✔ Tự động tích hợp tỷ lệ rẽ (Đi thẳng > Phải > Trái) qua góc toạ độ")
    print("  ✔ Phân bổ dồn xe mạnh vào trục chính, phạt nặng hẻm nhỏ")
    print("  ✔ Xe di chuyển dứt khoát đến đích là biến mất, KHÔNG chạy lòng vòng")
    print("  ✔ Xe hung hăng luồn lách, đèn xanh KHÔNG nhường nhịn")
    print("=" * 66)

    if not os.path.exists(NET_FILE):
        sys.exit(f"[!] Thiếu {NET_FILE}")

    print("\n[1/3] Cấu hình tốc độ mạng lưới...")
    modify_network_speed(NET_FILE, TARGET_SPEED)

    print("\n[2/3] Phân tích mạng lưới...")
    (edges, fsrc, fsink, src_by_dir, sink_by_dir, inner, adj, raw) = load_network(NET_FILE)
    vol_map = load_csv(CSV_FILE)

    print("\n[3/3] Sinh xe...")
    generate_routes(edges, fsrc, fsink, src_by_dir, sink_by_dir, inner, adj, raw, vol_map, ROUTES_OUT)

    print_stats(ROUTES_OUT, edges)
    verify_routes(ROUTES_OUT, edges, adj, raw)
    write_sumocfg(SUMOCFG_OUT, NET_FILE, ROUTES_OUT)

    print(f"\n✅ Hoàn tất!")
    print(f"   Routes : {ROUTES_OUT}")
    print(f"   Config : {SUMOCFG_OUT}\n")

if __name__ == "__main__":
    main()