// Sokoban solver: split each level into its hallway and rooms (a box that
// leaves its room can never return, so rooms are independent), solve every
// room push-optimally with A* over pushes, stitch the walks together.
// Deterministic: fixed budgets, fixed tie-breaks, no clock, no randomness.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
const IN = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUT = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const D = [[0, -1, 'u'], [0, 1, 'd'], [-1, 0, 'l'], [1, 0, 'r']];
const CAP = 250000;
const RUN = 4000000;
const used = { n: 0 };

// One BFS over `open` cells from `s`: parent map (keys = reachable set) and
// depth map. Serves reachability, shortest paths and distances alike.
function bfs(w, open, s) {
  const par = new Map([[s, -1]]), dep = new Map([[s, 0]]), q = [s];
  for (let h = 0; h < q.length; h++) {
    const c = q[h];
    for (const [dx, dy] of D) {
      const x = (c % w) + dx, y = ((c / w) | 0) + dy, t = y * w + x;
      if (x < 0 || x >= w || y < 0 || !open.has(t) || par.has(t)) continue;
      par.set(t, c);
      dep.set(t, dep.get(c) + 1);
      q.push(t);
    }
  }
  return { par, dep };
}

// Shortest walk from a to b along a BFS parent map, or null.
function path(par, a, b) {
  if (!par.has(b)) return null;
  let o = '';
  for (let t = b; t !== a; t = par.get(t)) {
    const p = par.get(t);
    o = (t === p - 1 ? 'l' : t === p + 1 ? 'r' : t < p ? 'u' : 'd') + o;
  }
  return o;
}

function parse(t) {
  const rows = t.split(/\r?\n/).filter((r) => r.length);
  const h = rows.length, w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h), goals = new Set(), boxes = [];
  let pl = 0;
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const c = rows[y][x] ?? '#', i = y * w + x;
      if (c === '#') wall[i] = 1;
      if ('.*+'.includes(c)) goals.add(i);
      if ('$*'.includes(c)) boxes.push(i);
      if ('@+'.includes(c)) pl = i;
    }
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, pl };
}

// The hallway is the player's row; a door is a floor cell in the row above or
// below it; a room is the floor component a door opens into, hallway excluded.
// Null when the shape is not what the generator promises.
function rooms(lv) {
  const { w, h, wall, goals, boxes, pl } = lv;
  const hy = (pl / w) | 0;
  const fl = (c) => c >= 0 && c < w * h && !wall[c];
  const ds = [];
  for (let x = 0; x < w; x++)
    for (const dy of [-1, 1]) {
      const y = hy + dy;
      if (y >= 0 && y < h && fl(y * w + x) && fl(hy * w + x)) ds.push(y * w + x);
    }
  const out = [], seen = new Set();
  for (const d of ds) {
    if (seen.has(d)) continue;
    const cells = new Set([d]), q = [d];
    for (let k = 0; k < q.length; k++) {
      const c = q[k];
      for (const [dx, dy] of D) {
        const x = (c % w) + dx, y = ((c / w) | 0) + dy, t = y * w + x;
        if (x < 0 || x >= w || y < 0 || y >= h || y === hy || !fl(t) || cells.has(t)) continue;
        cells.add(t);
        q.push(t);
      }
    }
    for (const c of cells) seen.add(c);
    out.push({
      cells,
      doors: ds.filter((x) => cells.has(x)),
      boxes: boxes.filter((b) => cells.has(b)),
      goals: [...goals].filter((g) => cells.has(g)),
    });
  }
  if (!out.length) return null;
  for (const r of out) if (!r.boxes.length || r.boxes.length !== r.goals.length) return null;
  if (out.reduce((a, r) => a + r.boxes.length, 0) !== boxes.length) return null;
  if (out.reduce((a, r) => a + r.goals.length, 0) !== goals.size) return null;
  return out.sort((a, b) => a.doors[0] - b.doors[0]);
}

// Push search over one area: boxes only ever stand on `area`, the player may
// also stand on `walk` (area plus the entry cell). Weight 1 is A* (optimal);
// higher weights trade optimality for speed. Null when the budget runs out.
function search(lv, area, walk, goals, bx0, pl0, doors, exit, weight, budget) {
  const w = lv.w, gs = new Set(goals);
  const md = new Map();
  for (const g of goals) {
    const { dep } = bfs(w, area, g);
    for (const [c, d] of dep) if (!md.has(c) || d < md.get(c)) md.set(c, d);
  }
  const h = (bx) => {
    let s = 0;
    for (const b of bx) {
      const d = md.get(b);
      if (d === undefined) return 1e9;
      s += d;
    }
    return s;
  };
  // Dead cells: a box there can never reach a goal. Goals start live; a cell
  // is live when some push from it lands on a live cell with the player able
  // to stand behind it. This over-approximates reachability, so pruning the
  // rest is safe.
  const live = new Set(goals);
  let ch = true;
  while (ch) {
    ch = false;
    for (const c of area) {
      if (live.has(c)) continue;
      for (const [dx, dy] of D) {
        if (live.has(c + dy * w + dx) && walk.has(c - dy * w - dx)) {
          live.add(c);
          ch = true;
          break;
        }
      }
    }
  }
  const NB = [bx0.slice()], NP = [pl0], NG = [0], NPa = [-1], NM = [''], NF = [h(bx0)], NS = [0];
  if (NF[0] >= 1e9) return null;
  const region = (bx, p) => {
    const bs = new Set(bx), op = new Set();
    for (const c of walk) if (!bs.has(c)) op.add(c);
    return bfs(w, op, p).par;
  };
  const rmin = (bx, p) => {
    let m = 1e9;
    for (const c of region(bx, p).keys()) if (c < m) m = c;
    return m;
  };
  const best = new Map([[bx0.join() + '@' + rmin(bx0, pl0), 0]]);
  const hp = [0];
  const less = (a, b) =>
    NF[a] < NF[b] || (NF[a] === NF[b] && (NG[a] > NG[b] || (NG[a] === NG[b] && NS[a] < NS[b])));
  const push = (v) => {
    let i = hp.length;
    hp.push(v);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(hp[i], hp[p])) break;
      [hp[i], hp[p]] = [hp[p], hp[i]];
      i = p;
    }
  };
  const pop = () => {
    const t = hp[0], l = hp.pop();
    if (hp.length) {
      hp[0] = l;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= hp.length) break;
        if (c + 1 < hp.length && less(hp[c + 1], hp[c])) c++;
        if (!less(hp[c], hp[i])) break;
        [hp[i], hp[c]] = [hp[c], hp[i]];
        i = c;
      }
    }
    return t;
  };
  let sq = 1, ex = 0;
  while (hp.length) {
    if (ex >= budget || used.n >= RUN) return null;
    const cur = pop(), bx = NB[cur], p = NP[cur], g = NG[cur];
    const rg = region(bx, p);
    let mn = 1e9;
    for (const c of rg.keys()) if (c < mn) mn = c;
    if (best.get(bx.join() + '@' + mn) < g) continue;
    if (bx.every((b) => gs.has(b)) && (!exit || doors.some((d) => rg.has(d)))) {
      let m = '';
      for (let t = cur; t > 0; t = NPa[t]) m = NM[t] + m;
      let e = p;
      if (exit) {
        let bl = 1e9, bt = null, bd = -1;
        for (const d of doors) {
          const s = path(rg, p, d);
          if (s !== null && s.length < bl) {
            bl = s.length;
            bt = s;
            bd = d;
          }
        }
        if (bt === null) continue;
        m += bt;
        e = bd;
      }
      return { m, pushes: g, end: e };
    }
    ex++;
    used.n++;
    const bs = new Set(bx);
    for (let i = 0; i < bx.length; i++) {
      const b = bx[i];
      for (const [dx, dy, ch2] of D) {
        const tg = b + dy * w + dx, bh = b - dy * w - dx;
        if (!area.has(tg) || bs.has(tg) || !rg.has(bh) || !live.has(tg)) continue;
        const nx = bx.slice();
        nx[i] = tg;
        nx.sort((a, z) => a - z);
        const hh = h(nx);
        if (hh >= 1e9) continue;
        const nk = nx.join() + '@' + rmin(nx, b);
        const sv = best.get(nk);
        if (sv !== undefined && sv <= g + 1) continue;
        const wk = path(rg, p, bh);
        if (wk === null) continue;
        const id = NB.length;
        NB.push(nx);
        NP.push(b);
        NG.push(g + 1);
        NPa.push(cur);
        NM.push(wk + ch2.toUpperCase());
        NF.push(g + 1 + weight * hh);
        NS.push(sq++);
        best.set(nk, g + 1);
        push(id);
      }
    }
  }
  return null;
}

// One room: optimal A* first, then a greedy-leaning retry when the optimal
// search runs out of budget. A valid non-optimal answer still scores.
function solveRoom(lv, r, entry, exit) {
  const walk = new Set(r.cells);
  walk.add(entry);
  return (
    search(lv, r.cells, walk, r.goals, r.boxes, entry, r.doors, exit, 1, CAP) ||
    search(lv, r.cells, walk, r.goals, r.boxes, entry, r.doors, exit, 6, CAP)
  );
}

// Whole-level fallback for levels that do not decompose.
function solveWhole(lv) {
  const fl = new Set();
  for (let i = 0; i < lv.w * lv.h; i++) if (!lv.wall[i]) fl.add(i);
  const f = search(lv, fl, fl, [...lv.goals], lv.boxes, lv.pl, [], false, 6, CAP);
  return f ? f.m : '';
}

function solve(t) {
  const lv = parse(t), rs = rooms(lv);
  if (!rs) return solveWhole(lv);
  const hy = (lv.pl / lv.w) | 0, plans = [];
  for (const r of rs) {
    const d = r.doors[0], en = ((d / lv.w) | 0) < hy ? d + lv.w : d - lv.w;
    const fr = solveRoom(lv, r, en, false);
    if (!fr) return '';
    let hd = fr;
    const bs = new Set(r.goals), op = new Set();
    for (const c of r.cells) if (!bs.has(c)) op.add(c);
    op.add(en);
    const rg = bfs(lv.w, op, fr.end).par;
    if (!r.doors.some((x) => rg.has(x))) {
      hd = solveRoom(lv, r, en, true);
      if (!hd) return '';
    }
    plans.push({ r, en, fr, hd });
  }
  // The room kept for last is the one whose exit costs the most pushes, so
  // the total stays minimal: every other room pays its exit, the last pays 0.
  let li = 0, bs2 = -1;
  for (let i = 0; i < plans.length; i++) {
    const s = plans[i].hd.pushes - plans[i].fr.pushes;
    if (s > bs2) {
      bs2 = s;
      li = i;
    }
  }
  const order = plans.map((_, i) => i).filter((i) => i !== li);
  order.push(li);
  let m = '', cur = lv.pl;
  const placed = new Set(lv.boxes);
  for (const i of order) {
    const p = plans[i], op = new Set();
    for (let c = 0; c < lv.w * lv.h; c++) if (!lv.wall[c] && !placed.has(c)) op.add(c);
    const wk = path(bfs(lv.w, op, cur).par, cur, p.en);
    if (wk === null) return '';
    m += wk;
    const part = i === li ? p.fr : p.hd;
    m += part.m;
    cur = part.end;
    for (const b of p.r.boxes) placed.delete(b);
    for (const g of p.r.goals) placed.add(g);
  }
  return m;
}

for (const id of readdirSync(IN).sort()) {
  let m = '';
  try {
    m = solve(readFileSync(join(IN, id, 'level.txt'), 'utf8'));
  } catch {
    m = '';
  }
  mkdirSync(join(OUT, id), { recursive: true });
  writeFileSync(join(OUT, id, 'solution.txt'), m + '\n');
}
