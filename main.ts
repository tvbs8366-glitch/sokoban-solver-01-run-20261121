// Sokoban solver: push-optimal A* over pushes, solved room by room.
//
// Every level is a hallway with independent rooms off it (boxes never leave
// their room in an optimal solution), so each room is solved on its own with
// A* over pushes: a state is the sorted box squares plus the player square,
// each push costs 1, and the heuristic is the minimum-cost matching of boxes
// to goals on box-ignoring shortest paths (admissible, consistent). States
// are pruned by three sound deadlocks: dead squares (a square from which no
// sequence of pushes, ignoring other boxes, reaches any goal), frozen 2x2
// blocks of walls/boxes, and frozen boxes. The first goal state popped is
// therefore push-optimal for its room, and the rooms' optima sum to the
// level's optimum P*. No clock, no randomness: budgets are counted in
// expanded states, so two runs write identical bytes.
//
// Local run: mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** [dx, dy, move letter]. */
const DIRS: [number, number, string][] = [
  [0, -1, 'u'],
  [0, 1, 'd'],
  [-1, 0, 'l'],
  [1, 0, 'r'],
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  floor: Uint8Array;
  goal: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goal = new Uint8Array(w * h);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x]!;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') { goal[i] = 1; goals.push(i); }
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  const floor = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) floor[i] = wall[i] ? 0 : 1;
  boxes.sort((a, b) => a - b);
  return { w, h, wall, floor, goal, goals, boxes, player };
}

/** BFS over `allowed` cells from `start`, avoiding `blocked`; returns the visited set plus parent links for walks. */
function bfsReach(
  lv: Level,
  start: number,
  blocked: Uint8Array,
  allowed: Uint8Array,
): { seen: Uint8Array; prev: Int32Array; prevDir: Int8Array } {
  const S = lv.w * lv.h;
  const seen = new Uint8Array(S);
  const prev = new Int32Array(S).fill(-1);
  const prevDir = new Int8Array(S).fill(-1);
  const q = new Int32Array(S);
  let qh = 0;
  let qt = 0;
  seen[start] = 1;
  q[qt++] = start;
  while (qh < qt) {
    const c = q[qh++]!;
    const cx = c % lv.w;
    const cy = (c / lv.w) | 0;
    for (let d = 0; d < 4; d++) {
      const [dx, dy] = DIRS[d]!;
      const nx = cx + dx;
      const ny = cy + dy;
      if (nx < 0 || ny < 0 || nx >= lv.w || ny >= lv.h) continue;
      const n = ny * lv.w + nx;
      if (seen[n] || blocked[n] || !allowed[n]) continue;
      seen[n] = 1;
      prev[n] = c;
      prevDir[n] = d;
      q[qt++] = n;
    }
  }
  return { seen, prev, prevDir };
}

/** The walk (u/d/l/r) from `start` to `target` through free cells, or null. Reuses one BFS. */
function walkTo(
  lv: Level,
  start: number,
  target: number,
  blocked: Uint8Array,
  allowed: Uint8Array,
): string | null {
  if (start === target) return '';
  const { seen, prev, prevDir } = bfsReach(lv, start, blocked, allowed);
  if (!seen[target]) return null;
  let moves = '';
  for (let c = target; c !== start; c = prev[c]!) moves = DIRS[prevDir[c]!]![2] + moves;
  return moves;
}

/** Shortest-path distances from `src` over `allowed` cells (-1 unreachable). */
function bfsDist(lv: Level, src: number, allowed: Uint8Array): Int32Array {
  const S = lv.w * lv.h;
  const dist = new Int32Array(S).fill(-1);
  const q = new Int32Array(S);
  let qh = 0;
  let qt = 0;
  dist[src] = 0;
  q[qt++] = src;
  while (qh < qt) {
    const c = q[qh++]!;
    const cx = c % lv.w;
    const cy = (c / lv.w) | 0;
    for (let d = 0; d < 4; d++) {
      const [dx, dy] = DIRS[d]!;
      const nx = cx + dx;
      const ny = cy + dy;
      if (nx < 0 || ny < 0 || nx >= lv.w || ny >= lv.h) continue;
      const n = ny * lv.w + nx;
      if (dist[n] >= 0 || !allowed[n]) continue;
      dist[n] = dist[c]! + 1;
      q[qt++] = n;
    }
  }
  return dist;
}

function permutations(n: number): number[][] {
  const out: number[][] = [];
  const a = Array.from({ length: n }, (_, i) => i);
  const gen = (k: number): void => {
    if (k === n) { out.push(a.slice()); return; }
    for (let i = k; i < n; i++) {
      [a[k], a[i]] = [a[i]!, a[k]!];
      gen(k + 1);
      [a[k], a[i]] = [a[i]!, a[k]!];
    }
  };
  gen(0);
  return out;
}

interface Room {
  region: Uint8Array;
  goals: number[];
  boxes: number[];
  entry: number[];
  hallway: Uint8Array;
}

/** Split a level into rooms off the hallway (the player's row, else column). Null when the shape is unexpected. */
function decompose(lv: Level, axis: 'row' | 'col'): Room[] | null {
  const S = lv.w * lv.h;
  const line = axis === 'row' ? (lv.player / lv.w) | 0 : lv.player % lv.w;
  const isHall = (c: number): boolean => (axis === 'row' ? ((c / lv.w) | 0) === line : c % lv.w === line);
  const hallway = new Uint8Array(S);
  for (let c = 0; c < S; c++) if (lv.floor[c] && isHall(c)) hallway[c] = 1;
  const seen = new Uint8Array(S);
  const rooms: Room[] = [];
  let coveredBoxes = 0;
  for (let c = 0; c < S; c++) {
    if (!lv.floor[c] || isHall(c) || seen[c]) continue;
    // One connected component off the hallway.
    const cells: number[] = [];
    const q = [c];
    seen[c] = 1;
    for (let qi = 0; qi < q.length; qi++) {
      const cur = q[qi]!;
      cells.push(cur);
      const cx = cur % lv.w;
      const cy = (cur / lv.w) | 0;
      for (const [dx, dy] of DIRS) {
        const nx = cx + dx;
        const ny = cy + dy;
        if (nx < 0 || ny < 0 || nx >= lv.w || ny >= lv.h) continue;
        const n = ny * lv.w + nx;
        if (!lv.floor[n] || isHall(n) || seen[n]) continue;
        seen[n] = 1;
        q.push(n);
      }
    }
    const boxes = cells.filter((x) => lv.boxes.includes(x));
    if (boxes.length === 0) continue;
    const goals = cells.filter((x) => lv.goal[x] === 1);
    if (goals.length !== boxes.length) return null;
    const entry = cells.filter((x) => {
      const cx = x % lv.w;
      const cy = (x / lv.w) | 0;
      return DIRS.some(([dx, dy]) => {
        const nx = cx + dx;
        const ny = cy + dy;
        return nx >= 0 && ny >= 0 && nx < lv.w && ny < lv.h && hallway[ny * lv.w + nx] === 1;
      });
    });
    if (entry.length === 0) return null;
    const region = new Uint8Array(S);
    for (const x of cells) region[x] = 1;
    coveredBoxes += boxes.length;
    rooms.push({ region, goals, boxes, entry, hallway });
  }
  if (coveredBoxes !== lv.boxes.length) return null;
  return rooms;
}

interface SolveResult {
  moves: string;
  endPlayer: number;
}

/**
 * Push-optimal A* for one region (a room, or the whole level as a fallback:
 * region = all floor, entry = [], hallway = empty). The player may walk the
 * region plus the hallway cells; boxes stay inside the region. Goal: every
 * box on a goal, with the player able to reach an entry cell (so it can leave
 * the room) when entry cells exist.
 */
function solveRegion(
  lv: Level,
  region: Uint8Array,
  goals: number[],
  boxes0: number[],
  entry: number[],
  hallway: Uint8Array,
  startPlayer: number,
  cap: number,
): SolveResult | null {
  const S = lv.w * lv.h;
  const nB = boxes0.length;
  if (nB === 0) return { moves: '', endPlayer: startPlayer };
  const goalSet = new Uint8Array(S);
  for (const g of goals) goalSet[g] = 1;
  // The player may walk the region and the hallway.
  const domain = new Uint8Array(S);
  for (let i = 0; i < S; i++) domain[i] = region[i] || hallway[i] ? 1 : 0;

  // Box-to-goal shortest paths (ignoring boxes) over the region.
  const distG: Int32Array[] = goals.map((g) => bfsDist(lv, g, region));

  // Dead squares: reverse push-reachability from the goals, ignoring boxes.
  // A square is live when a box there can be pushed, step by step, onto a goal.
  const live = new Uint8Array(S);
  {
    const q: number[] = [];
    for (const g of goals) { live[g] = 1; q.push(g); }
    for (let qi = 0; qi < q.length; qi++) {
      const x = q[qi]!;
      const xx = x % lv.w;
      const xy = (x / lv.w) | 0;
      for (const [dx, dy] of DIRS) {
        const cx = xx - dx;
        const cy = xy - dy;
        const bx = xx - 2 * dx;
        const by = xy - 2 * dy;
        if (cx < 0 || cy < 0 || cx >= lv.w || cy >= lv.h) continue;
        if (bx < 0 || by < 0 || bx >= lv.w || by >= lv.h) continue;
        const c = cy * lv.w + cx;
        const behind = by * lv.w + bx;
        if (!region[c] || live[c] || !domain[behind]) continue;
        live[c] = 1;
        q.push(c);
      }
    }
  }

  const perms = nB <= 5 ? permutations(nB) : null;
  const INF = 1e9;
  const heuristic = (boxes: number[]): number => {
    if (perms) {
      let best = INF;
      for (const p of perms) {
        let sum = 0;
        for (let i = 0; i < nB; i++) {
          const d = distG[p[i]!]![boxes[i]!]!;
          if (d < 0) { sum = INF; break; }
          sum += d;
          if (sum >= best) break;
        }
        if (sum < best) best = sum;
      }
      return best;
    }
    let sum = 0;
    for (let i = 0; i < nB; i++) {
      let m = INF;
      for (let g = 0; g < goals.length; g++) {
        const d = distG[g]![boxes[i]!]!;
        if (d >= 0 && d < m) m = d;
      }
      if (m === INF) return INF;
      sum += m;
    }
    return sum;
  };

  /** Sound deadlock tests for a box set (any true means unsolvable). */
  const isDead = (boxes: number[], boxSet: Uint8Array): boolean => {
    for (const b of boxes) if (!live[b]) return true;
    // Frozen 2x2 blocks of walls and boxes holding a box off its goal.
    for (const b of boxes) {
      const bx = b % lv.w;
      const by = (b / lv.w) | 0;
      for (const [ox, oy] of [[0, 0], [-1, 0], [0, -1], [-1, -1]] as [number, number][]) {
        const sq: number[] = [];
        let full = true;
        for (const [sx, sy] of [[0, 0], [1, 0], [0, 1], [1, 1]] as [number, number][]) {
          const nx = bx + ox + sx;
          const ny = by + oy + sy;
          if (nx < 0 || ny < 0 || nx >= lv.w || ny >= lv.h) { full = false; break; }
          const n = ny * lv.w + nx;
          if (!lv.wall[n] && !boxSet[n]) { full = false; break; }
          sq.push(n);
        }
        if (full && sq.some((n) => boxSet[n] && !goalSet[n])) return true;
      }
    }
    // Frozen boxes: a box that cannot move on either axis, where a neighbour
    // counts as blocking when it is a wall or an already-frozen box.
    const frozen = new Uint8Array(nB);
    for (let round = 0; round < nB; round++) {
      let changed = false;
      for (let i = 0; i < nB; i++) {
        if (frozen[i]) continue;
        const b = boxes[i]!;
        const bx = b % lv.w;
        const by = (b / lv.w) | 0;
        const blocked = (nx: number, ny: number): boolean => {
          if (nx < 0 || ny < 0 || nx >= lv.w || ny >= lv.h) return true;
          const n = ny * lv.w + nx;
          if (lv.wall[n]) return true;
          const j = boxes.indexOf(n);
          return j >= 0 && frozen[j] === 1;
        };
        const stuckX = blocked(bx - 1, by) && blocked(bx + 1, by);
        const stuckY = blocked(bx, by - 1) && blocked(bx, by + 1);
        if (stuckX && stuckY) { frozen[i] = 1; changed = true; }
      }
      if (!changed) break;
    }
    for (let i = 0; i < nB; i++) if (frozen[i] && !goalSet[boxes[i]!]!) return true;
    return false;
  };

  const keyOf = (boxes: number[], player: number): number => {
    let k = 0;
    for (const b of boxes) k = k * S + b;
    return k * S + player;
  };

  // A* node storage.
  const nBoxes: number[][] = [];
  const nPlayer: number[] = [];
  const nG: number[] = [];
  const nF: number[] = [];
  const nParent: number[] = [];
  const nStep: string[] = [];
  const nKey: number[] = [];
  const best = new Map<number, number>();
  const heap: number[] = [];
  const heapLess = (a: number, b: number): boolean =>
    nF[a]! < nF[b]! || (nF[a]! === nF[b]! && nG[a]! > nG[b]!);
  const heapPush = (v: number): void => {
    let i = heap.length;
    heap.push(v);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): number => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && heapLess(heap[c + 1]!, heap[c]!)) c++;
        if (!heapLess(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  const startBoxes = boxes0.slice().sort((a, b) => a - b);
  const h0 = heuristic(startBoxes);
  if (h0 >= INF) return null;
  const k0 = keyOf(startBoxes, startPlayer);
  nBoxes.push(startBoxes);
  nPlayer.push(startPlayer);
  nG.push(0);
  nF.push(h0);
  nParent.push(-1);
  nStep.push('');
  nKey.push(k0);
  best.set(k0, 0);
  heapPush(0);

  let expanded = 0;
  while (heap.length > 0) {
    const idx = heapPop();
    if (best.get(nKey[idx]!) !== nG[idx]) continue; // stale entry
    const boxes = nBoxes[idx]!;
    const player = nPlayer[idx]!;
    const g = nG[idx]!;
    // Goal: every box on a goal, and the player can still reach the door.
    if (boxes.every((b) => goalSet[b] === 1)) {
      if (entry.length === 0) {
        let moves = '';
        for (let t = idx; t > 0; t = nParent[t]!) moves = nStep[t]! + moves;
        return { moves, endPlayer: player };
      }
      const boxSet = new Uint8Array(S);
      for (const b of boxes) boxSet[b] = 1;
      const reach = bfsReach(lv, player, boxSet, domain);
      if (entry.some((e) => !boxSet[e] && reach.seen[e])) {
        let moves = '';
        for (let t = idx; t > 0; t = nParent[t]!) moves = nStep[t]! + moves;
        return { moves, endPlayer: player };
      }
    }
    if (++expanded > cap) return null;
    if (nBoxes.length > 2_000_000) return null;
    const boxSet = new Uint8Array(S);
    for (const b of boxes) boxSet[b] = 1;
    const reach = bfsReach(lv, player, boxSet, domain);
    for (let i = 0; i < nB; i++) {
      const b = boxes[i]!;
      const bx = b % lv.w;
      const by = (b / lv.w) | 0;
      for (let d = 0; d < 4; d++) {
        const [dx, dy, ch] = DIRS[d]!;
        const nx = bx + dx;
        const ny = by + dy;
        if (nx < 0 || ny < 0 || nx >= lv.w || ny >= lv.h) continue;
        const dest = ny * lv.w + nx;
        if (!region[dest] || boxSet[dest]) continue;
        const behind = (by - dy) * lv.w + (bx - dx);
        if (bx - dx < 0 || by - dy < 0 || bx - dx >= lv.w || by - dy >= lv.h) continue;
        if (!reach.seen[behind]) continue;
        const nb = boxes.slice();
        nb[i] = dest;
        nb.sort((a, b2) => a - b2);
        const nbSet = new Uint8Array(S);
        for (const x of nb) nbSet[x] = 1;
        if (isDead(nb, nbSet)) continue;
        const h = heuristic(nb);
        if (h >= INF) continue;
        const key = keyOf(nb, b);
        const g2 = g + 1;
        if ((best.get(key) ?? Infinity) <= g2) continue;
        // The player's walk to the pushing square, then the push.
        let walk = '';
        for (let c = behind; c !== player; c = reach.prev[c]!) walk = DIRS[reach.prevDir[c]!]![2] + walk;
        const t = nBoxes.length;
        nBoxes.push(nb);
        nPlayer.push(b);
        nG.push(g2);
        nF.push(g2 + h);
        nParent.push(idx);
        nStep.push(walk + ch);
        nKey.push(key);
        best.set(key, g2);
        heapPush(t);
      }
    }
  }
  return null;
}

/** Replay moves on the level; true only when every step is legal and every box ends on a goal. */
function verify(lv: Level, moves: string): boolean {
  const box = new Uint8Array(lv.w * lv.h);
  for (const b of lv.boxes) box[b] = 1;
  let player = lv.player;
  for (const m of moves) {
    const d = DIRS.find(([,, ch]) => ch === m);
    if (!d) return false;
    const [dx, dy] = d;
    const next = player + dy * lv.w + dx;
    if (lv.wall[next]) return false;
    if (box[next]) {
      const beyond = next + dy * lv.w + dx;
      if (lv.wall[beyond] || box[beyond]) return false;
      box[next] = 0;
      box[beyond] = 1;
    }
    player = next;
  }
  return (() => { for (let i = 0; i < box.length; i++) if (box[i] && !lv.goal[i]) return false; return true; })();
}

function solve(text: string): string {
  const lv = parseLevel(text);
  if (lv.boxes.length === 0) return '';
  const rooms = decompose(lv, 'row') ?? decompose(lv, 'col');
  if (rooms) {
    const S = lv.w * lv.h;
    // Global box occupancy, updated as rooms are solved (their boxes end on their goals).
    const occ = new Uint8Array(S);
    for (const b of lv.boxes) occ[b] = 1;
    let moves = '';
    let cur = lv.player;
    let ok = true;
    for (const room of rooms) {
      const res = solveRegion(lv, room.region, room.goals, room.boxes, room.entry, room.hallway, cur, 600_000);
      if (!res) { ok = false; break; }
      moves += res.moves;
      cur = res.endPlayer;
      for (const b of room.boxes) occ[b] = 0;
      for (const g of room.goals) occ[g] = 1;
      // Walk the player out of the room to the hallway for the next room.
      const domain = new Uint8Array(S);
      for (let i = 0; i < S; i++) domain[i] = room.region[i] || room.hallway[i] ? 1 : 0;
      let exit: string | null = null;
      for (const e of room.entry) {
        if (occ[e]) continue;
        const w1 = walkTo(lv, cur, e, occ, domain);
        if (w1 === null) continue;
        const ex = e % lv.w;
        const ey = (e / lv.w) | 0;
        for (const [dx, dy, ch] of DIRS) {
          const nx = ex + dx;
          const ny = ey + dy;
          if (nx < 0 || ny < 0 || nx >= lv.w || ny >= lv.h) continue;
          const n = ny * lv.w + nx;
          if (room.hallway[n] && !occ[n]) { exit = w1 + ch; cur = n; break; }
        }
        if (exit) break;
      }
      if (!exit) { ok = false; break; }
      moves += exit;
    }
    if (ok && verify(lv, moves)) return moves;
  }
  // Fallback: solve the whole level at once.
  const fb = solveRegion(lv, lv.floor, lv.goals, lv.boxes, [], new Uint8Array(lv.w * lv.h), lv.player, 300_000);
  if (fb && verify(lv, fb.moves)) return fb.moves;
  return '';
}

for (const id of readdirSync(I).sort()) {
  const moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
