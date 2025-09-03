#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, math, heapq, sys, time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from itertools import count

# ====== 基本路径 ======
SAVE_PATH = "./1.json"
OUT_PATH  = "./1.txt"

# ====== 剪枝/限流参数（可调）======
BEAM_CHILD_LIMIT   = 12     # 每个节点最多保留的子节点（按潜力排序）
FAST_DEATH_PRUNE   = True   # 快死剪
SKIP_NO_MERGE_MOVE = True   # 若存在能合并的走法，则剪掉“不合并”的走法
BAN_IMMEDIATE_BACK = True   # 禁止立即反向
TT_CAPACITY        = 1_000_000  # 置换表容量
NODE_BUDGET        = 2_000_000  # 全局节点上限（防爆跑）

# ====== 启发式权重 ======
W_EMPTY    = 600
W_CORNER   = 5000
W_MERGE    = 1.0    # 本步合并得分优先级
W_STABLE   = 0.1    # 已累计新增分参与排序

# ====== 方向编码：0/1/2/3 = 上/下/左/右 ======
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
DIRS  = (DOWN, LEFT, RIGHT, UP)     # 试走顺序（经验强）
OPPO  = {UP:DOWN, DOWN:UP, LEFT:RIGHT, RIGHT:LEFT}

# ====== 进度条（tqdm 优先，失败则用 ASCII）======
PROG_UPDATE_EVERY = 2000  # 每多少节点刷新一次进度/ETA
try:
    from tqdm import tqdm
    _USE_TQDM = True
except Exception:
    _USE_TQDM = False

class Progress:
    def __init__(self, total:int):
        self.total = total
        self.start = time.time()
        self.last = 0
        if _USE_TQDM:
            self.bar = tqdm(total=total, unit="node", dynamic_ncols=True, smoothing=0.1)
        else:
            self.bar = None

    def update(self, current:int, best:int, open_size:int):
        if _USE_TQDM:
            delta = current - self.last
            if delta > 0:
                self.bar.update(delta)
                self.bar.set_postfix(best=best, open=open_size)
        else:
            # 轻量 ASCII 进度：基于 NODE_BUDGET 的粗略比例
            pct = min(100.0, 100.0 * current / max(1, self.total))
            elapsed = time.time() - self.start
            rate = current / max(1e-9, elapsed)
            remain = (self.total - current) / max(1e-9, rate)
            bar_len = 30
            filled = int(bar_len * pct / 100.0)
            bar = "█" * filled + "-" * (bar_len - filled)
            msg = f"\r[{bar}] {pct:6.2f}% | best {best} | open {open_size} | {rate:7.0f} n/s | ETA {remain:6.1f}s"
            sys.stdout.write(msg)
            sys.stdout.flush()
        self.last = current

    def close(self):
        if _USE_TQDM:
            self.bar.close()
        else:
            sys.stdout.write("\n")
            sys.stdout.flush()

# ====== JS Mulberry32 兼容 PRNG ======
def _imul(a: int, b: int) -> int:
    return ((a & 0xFFFFFFFF) * (b & 0xFFFFFFFF)) & 0xFFFFFFFF

def mulberry32(seed_init: int):
    s = seed_init & 0xFFFFFFFF
    def rnd() -> float:
        nonlocal s
        s = (s + 0x6D2B79F5) & 0xFFFFFFFF
        t = _imul(s ^ (s >> 15), 1 | s)
        t2 = (t + _imul(t ^ (t >> 7), 61 | t)) & 0xFFFFFFFF
        t  = (t2 ^ t) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0
    return rnd

def generate_tile_info(seed: int, step: int, side: int) -> Tuple[int, int]:
    rnd = mulberry32((seed + step * 12345) & 0xFFFFFFFF)
    empty_slot_index = int(math.floor(rnd() * (side * side)))
    value = 2 if rnd() < 0.8 else 4
    return empty_slot_index, value

# ====== 棋盘基础 ======
def grid_to_matrix(flat: Tuple[int, ...], side: int) -> List[List[int]]:
    return [list(flat[i*side:(i+1)*side]) for i in range(side)]

def matrix_to_flat(mat: List[List[int]]) -> Tuple[int, ...]:
    return tuple(v for row in mat for v in row)

def is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0

def merge_value(v: int, base: int) -> int:
    if v == 2:  return base * 2
    if v == 4:  return base * 4
    if v % base == 0 and is_power_of_two(v // base): return v * 2
    return v * 2

def compress_line(line: List[int], side: int, base: int) -> Tuple[List[int], int, bool, int]:
    nz = [x for x in line if x != 0]
    out, add_score, merges = [], 0, 0
    i = 0
    while i < len(nz):
        if i + 1 < len(nz) and nz[i] == nz[i + 1]:
            m = merge_value(nz[i], base)
            out.append(m); add_score += m; merges += 1
            i += 2
        else:
            out.append(nz[i]); i += 1
    out += [0] * (side - len(out))
    moved = out != line
    return out, add_score, moved, merges

def apply_move(flat: Tuple[int, ...], side: int, base: int, direction: int):
    mat = grid_to_matrix(flat, side)
    total_gain, moved_any, total_merges = 0, False, 0

    if direction == LEFT:
        for r in range(side):
            nl, g, m, mg = compress_line(mat[r], side, base)
            mat[r] = nl; total_gain += g; moved_any |= m; total_merges += mg
    elif direction == RIGHT:
        for r in range(side):
            rev = list(reversed(mat[r]))
            nl, g, m, mg = compress_line(rev, side, base)
            mat[r] = list(reversed(nl)); total_gain += g; moved_any |= m; total_merges += mg
    elif direction == UP:
        for c in range(side):
            col = [mat[r][c] for r in range(side)]
            nl, g, m, mg = compress_line(col, side, base)
            for r in range(side): mat[r][c] = nl[r]
            total_gain += g; moved_any |= m; total_merges += mg
    elif direction == DOWN:
        for c in range(side):
            col = [mat[r][c] for r in reversed(range(side))]
            nl, g, m, mg = compress_line(col, side, base)
            for idx, r in enumerate(reversed(range(side))): mat[r][c] = nl[idx]
            total_gain += g; moved_any |= m; total_merges += mg

    return matrix_to_flat(mat), total_gain, moved_any, total_merges

def spawn(flat: Tuple[int, ...], side: int, seed: int, step: int):
    mat = grid_to_matrix(flat, side)
    empties = [(r, c) for r in range(side) for c in range(side) if mat[r][c] == 0]
    if not empties: return flat, step
    idx, value = generate_tile_info(seed, step, side)
    r, c = empties[idx % len(empties)]
    mat[r][c] = value
    return matrix_to_flat(mat), step + 1

def count_empty(flat: Tuple[int, ...]) -> int:
    return sum(1 for v in flat if v == 0)

def can_move(flat: Tuple[int, ...], side: int, base: int) -> bool:
    for d in (UP, DOWN, LEFT, RIGHT):
        _, _, moved, _ = apply_move(flat, side, base, d)
        if moved: return True
    return False

# ====== 启发式（排序用）======
def heuristic(flat: Tuple[int, ...], side: int) -> int:
    mat = grid_to_matrix(flat, side)
    score = 0
    maxv = 0
    for r in range(side):
        for c in range(side):
            v = mat[r][c]
            if not v: continue
            weight = (side - r) * 10 + (side - c)
            score += v * weight
            if v > maxv: maxv = v
    if mat[side-1][0] == maxv and maxv > 0:
        score += W_CORNER
    score += W_EMPTY * count_empty(flat)
    return score

# ====== 置换表（近似 LRU）======
class TransTable:
    def __init__(self, cap=1_000_000):
        self.cap = cap
        self.map: Dict[Tuple[Tuple[int, ...], int], Tuple[int, int]] = {}
        self.tick = 0
    def better_seen(self, key, score) -> bool:
        self.tick += 1
        if key in self.map:
            old_score, _ = self.map[key]
            if old_score >= score:
                return True
        self.map[key] = (score, self.tick)
        if len(self.map) > self.cap:
            cutoff = sorted(self.map.values(), key=lambda x:x[1])[len(self.map)//10][1]
            self.map = {k:v for k,v in self.map.items() if v[1] >= cutoff}
        return False

# ====== 搜索节点 ======
@dataclass(order=True)
class Node:
    priority: float
    flat: Tuple[int, ...] = field(compare=False)
    add_score: int = field(compare=False)
    step: int = field(compare=False)
    path: str = field(compare=False, default="")
    last_dir: Optional[int] = field(compare=False, default=None)
    dead: bool = field(compare=False, default=False)

# ====== 读取存档 ======
def read_save(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    g = data["currentGame"]
    side = int(g["side"]); base = int(g["base"])
    seed = int(g["seed"]); step = int(g.get("step", len(g["tiles"])))
    current_score = int(g["currentScore"])
    grid = [[0]*side for _ in range(side)]
    for t in g["tiles"]:
        grid[int(t["row"])][int(t["col"])] = int(t["value"])
    return side, base, seed, step, current_score, matrix_to_flat(grid)

# ====== Top-K（按新增分）======
_topk_counter = count()
def push_topk(minheap, node: Node, keep:int=16):
    heapq.heappush(minheap, (node.add_score, next(_topk_counter), node))
    if len(minheap) > keep:
        heapq.heappop(minheap)

# ====== 主搜索（深度优先 + 强剪枝 + 进度条）======
def solve(side, base, seed, step0, start_flat):
    tt = TransTable(TT_CAPACITY)
    best_heap = []            # 小顶堆保存候选终局，最后取 top3
    best_score = -1
    nodes = 0

    start = Node(priority=-(heuristic(start_flat, side)),
                 flat=start_flat, add_score=0, step=step0, path="",
                 last_dir=None, dead=not can_move(start_flat, side, base))
    frontier: List[Node] = [start]

    killer = {}  # depth -> dir
    prog = Progress(NODE_BUDGET)
    last_prog_refresh = 0

    while frontier and nodes < NODE_BUDGET:
        node = heapq.heappop(frontier)
        nodes += 1

        # 进度条节流刷新
        if nodes - last_prog_refresh >= PROG_UPDATE_EVERY:
            prog.update(nodes, best_score if best_score>=0 else 0, len(frontier))
            last_prog_refresh = nodes

        if node.dead:
            push_topk(best_heap, node)
            if node.add_score > best_score:
                best_score = node.add_score
            continue

        key = (node.flat, node.step)
        if tt.better_seen(key, node.add_score):
            continue

        children: List[Tuple[float, Node, int, int, int]] = []
        merge_exists = False
        order = list(DIRS)
        depth = len(node.path)
        if depth in killer:
            try:
                order.remove(killer[depth]); order.insert(0, killer[depth])
            except ValueError:
                pass

        for d in order:
            if BAN_IMMEDIATE_BACK and node.last_dir is not None and d == OPPO[node.last_dir]:
                continue
            new_flat, gain, moved, merges = apply_move(node.flat, side, base, d)
            if not moved:
                continue
            merge_exists |= (merges > 0)
            spawned_flat, new_step = spawn(new_flat, side, seed, node.step)
            new_dead = not can_move(spawned_flat, side, base)
            empties = count_empty(spawned_flat)
            h = heuristic(spawned_flat, side)
            key_sort = -(W_MERGE*gain + 5*empties + 0.001*h + W_STABLE*node.add_score)
            child = Node(priority=key_sort, flat=spawned_flat,
                         add_score=node.add_score + gain, step=new_step,
                         path=node.path + str(d), last_dir=d, dead=new_dead)
            children.append((key_sort, child, gain, empties, d))

        if not children:
            leaf = Node(priority=node.priority, flat=node.flat, add_score=node.add_score,
                        step=node.step, path=node.path, last_dir=node.last_dir, dead=True)
            push_topk(best_heap, leaf)
            if leaf.add_score > best_score:
                best_score = leaf.add_score
            continue

        if SKIP_NO_MERGE_MOVE and merge_exists:
            merged_children = [x for x in children if x[2] > 0]
            if merged_children:
                children = merged_children

        if FAST_DEATH_PRUNE and len(children) > 2:
            children.sort(key=lambda x: (-(x[2]), -x[3]))
            survivors = children[:2]
            for item in children[2:]:
                gain, empties = item[2], item[3]
                if gain == 0 and empties <= 2:  # 可更激进
                    continue
                survivors.append(item)
            children = survivors

        children.sort(key=lambda x: x[0])
        children = children[:BEAM_CHILD_LIMIT]

        best_local = max(children, key=lambda x: x[2])
        killer[depth] = best_local[4]

        for _, ch, _, _, _ in reversed(children):
            heapq.heappush(frontier, ch)

    # 最后刷新一次并关闭进度条
    prog.update(nodes, best_score if best_score>=0 else 0, len(frontier))
    prog.close()

    top_nodes = sorted(best_heap, key=lambda t: t[0], reverse=True)
    top_nodes = [n for _, __, n in top_nodes][:3]
    best = top_nodes[0]
    return top_nodes, best

# ====== 输出辅助 ======
def board_str(flat: Tuple[int, ...], side: int) -> str:
    mat = grid_to_matrix(flat, side)
    width = max(4, max((len(str(v)) for v in flat if v != 0), default=1))
    lines = []
    for r in range(side):
        lines.append(" ".join(f"{mat[r][c]:>{width}d}" if mat[r][c] != 0 else " " * width for c in range(side)))
    return "\n".join(lines)

# ====== 入口 ======
def main():
    side, base, seed, step0, start_score, start_flat = read_save(SAVE_PATH)
    top3, best = solve(side, base, seed, step0, start_flat)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for n in top3:
            f.write(n.path + "\n")

    print(f"参数：side={side}, base={base}, seed={seed}, 起始步数={step0}")
    print(f"最优总分（含存档已有分）：{start_score + best.add_score} | 新增分：{best.add_score} | 步数：{len(best.path)}")
    print("最终死亡棋盘：")
    print(board_str(best.flat, side))
    print(f"前三最优解已写入 {OUT_PATH}")
    if len(top3) > 1:
        print("次优/三优 (长度, 新增分)：", [(len(n.path), n.add_score) for n in top3[1:]])

if __name__ == "__main__":
    sys.setrecursionlimit(1_000_000)
    main()
