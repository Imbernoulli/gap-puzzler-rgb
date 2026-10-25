'use strict';
const data = JSON.parse(document.getElementById('page-data').textContent);
const NS = 'http://www.w3.org/2000/svg';
const byId = id => document.getElementById(id);
const cap = s => s.charAt(0).toUpperCase() + s.slice(1);

// Real-to-sim slider: the real frame is clipped to the left of the line.
const compare = byId('compare');
if (compare) {
  const range = byId('compare-range');
  const set = () => compare.style.setProperty('--x', `${range.value}%`);
  range.addEventListener('input', set);
  set();
}

// A piece drawn by its shape: beads on the public CAD grid, necks between neighbours.
function shapeSvg(cells, scale = 1.6) {
  const pitch = 10, rad = 4.4;
  const rows = Math.max(...cells.map(c => c[0])) + 1, cols = Math.max(...cells.map(c => c[1])) + 1;
  const w = 12 + (cols - 1) * pitch, h = 12 + (rows - 1) * pitch;
  const svg = el('svg', {class: 'shape', viewBox: `0 0 ${w} ${h}`, width: w * scale, height: h * scale, 'aria-hidden': 'true'});
  const pos = cells.map(([r, c]) => [6 + c * pitch, 6 + r * pitch]);
  const necks = el('g', {class: 'neck'}), beads = el('g', {class: 'bead'});
  pos.forEach(([x1, y1], i) => pos.slice(i + 1).forEach(([x2, y2]) => {
    if (Math.abs(x1 - x2) + Math.abs(y1 - y2) === pitch) necks.append(el('line', {x1, y1, x2, y2}));
  }));
  pos.forEach(([cx, cy]) => beads.append(el('circle', {cx, cy, r: rad})));
  svg.append(necks, beads);
  return svg;
}
const pieceName = piece => (piece.name === 'finish' ? 'Finish' : `Piece ${piece.number}`);

function el(name, attrs = {}, text) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text != null) node.textContent = text;
  return node;
}

// One player per camera rig: video -> control step -> graph node, one step clock.
function initPlayer(root, run) {
  const q = key => root.querySelector(`[data-k="${key}"]`);
  const video = q('video'), svg = q('graph'), pieces = run.pieces;
  const perSecond = run.video.steps_per_second;
  const seek = step => { video.currentTime = Math.max(0, step + 1) / perSecond; update(); };
  let shown = null;

  pieces.forEach((piece, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    if (piece.cells) button.append(shapeSvg(piece.cells, 1));
    button.append(Object.assign(document.createElement('span'), {textContent: piece.name === 'finish' ? '✓' : piece.number}));
    button.title = pieceName(piece);
    button.setAttribute('aria-label', `Jump to ${button.title}`);
    button.addEventListener('click', () => seek(piece.start));
    q('track').append(button);
    piece.button = button;
  });

  // The subgraph: one box per node, one labelled edge per success exit (parallel
  // exits fan out), and a red notch for every node's error exit to stop. Long
  // subgraphs fold into two columns so the panel stays about as tall as the video.
  function drawGraph(piece) {
    const n = piece.nodes.length, cols = n > 6 ? 2 : 1, rows = Math.ceil(n / cols);
    const W = 300, boxH = 24, gap = 18, boxW = cols === 2 ? 130 : 200, xs = cols === 2 ? [18, 166] : [50];
    const H = 6 + rows * (boxH + gap) - gap + 6;
    const at = i => ({x: xs[Math.floor(i / rows)], y: 6 + (i % rows) * (boxH + gap)});
    const arrow = `arrow-${root.dataset.rig}`;
    svg.replaceChildren();
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.classList.toggle('two-col', cols === 2);
    const defs = el('defs');
    const marker = el('marker', {id: arrow, viewBox: '0 0 8 8', refX: 7, refY: 4, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse'});
    marker.append(el('path', {d: 'M0,0 L8,4 L0,8 z', class: 'arrowhead'}));
    defs.append(marker);
    svg.append(defs);
    piece.nodes.forEach((node, i) => {
      const {x, y} = at(i);
      if (i + 1 < n) {
        const next = at(i + 1), exits = node.exits, k = exits.length;
        if (next.x === x) {
          exits.forEach((exit, j) => {
            const sx = x + boxW / 2 + (j - (k - 1) / 2) * 12;
            svg.append(el('path', {d: `M${sx},${y + boxH} C${sx},${y + boxH + gap / 2} ${x + boxW / 2},${next.y - gap / 2} ${x + boxW / 2},${next.y - 1}`, class: 'edge', 'marker-end': `url(#${arrow})`}));
          });
          const label = el('text', {x: x + boxW / 2 + 8 + (k - 1) * 6, y: y + boxH + gap / 2 + 3, class: 'edge-label'}, node.exit);
          label.append(el('title', {}, exits.join(' · ')));
          svg.append(label);
        } else {
          // Column break: from the bottom of this column to the top of the next.
          svg.append(el('path', {d: `M${x + boxW},${y + boxH / 2} C${x + boxW + 14},${y + boxH / 2} ${next.x - 14},${next.y + boxH / 2} ${next.x - 1},${next.y + boxH / 2}`, class: 'edge', 'marker-end': `url(#${arrow})`}));
        }
      }
      const group = el('g', {class: 'node', tabindex: 0, role: 'button', 'aria-label': `${node.label}: jump to this step`});
      group.append(el('rect', {x, y, width: boxW, height: boxH, rx: 5}));
      group.append(el('rect', {x: x + boxW - 4, y: y + 5, width: 2, height: boxH - 10, class: 'abort-notch'}));
      group.append(el('text', {x: x + 8, y: y + boxH / 2 + 4, class: 'node-label'}, node.label));
      if (node.relook) group.append(el('text', {x: x - 14, y: y + boxH / 2 + 5, class: 'relook'}, '↻'));
      if (cols === 1) group.append(el('text', {x: x + boxW - 10, y: y + boxH / 2 + 4, class: 'node-time', 'text-anchor': 'end'},
        node.end != null ? `${(node.end / run.control_hz).toFixed(0)} s` : ''));
      group.addEventListener('click', () => seek(node.start));
      group.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); seek(node.start); } });
      svg.append(group);
      node.group = group;
    });
  }

  function showPiece(piece) {
    if (shown === piece) return;
    shown = piece;
    q('swatch').replaceChildren(...(piece.cells ? [shapeSvg(piece.cells, 2)] : []));
    q('piece').textContent = pieceName(piece);
    q('index').textContent = piece.name === 'finish' ? 'after piece 12' : `${piece.nodes.length} steps`;
    drawGraph(piece);
  }

  function update() {
    const step = video.currentTime * perSecond;
    const current = pieces.find(p => p.end == null || step < p.end) || pieces[pieces.length - 1];
    showPiece(current);
    pieces.forEach(p => {
      p.button.classList.toggle('done', p.end != null && step >= p.end);
      p.button.classList.toggle('now', p === current);
    });
    let active = null;
    current.nodes.forEach(node => {
      const done = node.end != null && step >= node.end && node.passed;
      const now = !active && !done && step >= node.start;
      if (now) active = node;
      node.group.classList.toggle('done', done);
      node.group.classList.toggle('now', now);
    });
    const detail = active || current.nodes[current.nodes.length - 1];
    const metric = detail.metrics.placement_error_mm != null ? ` Placed within ${detail.metrics.placement_error_mm.toFixed(1)} mm.` : '';
    q('detail').textContent = detail.detail + metric;
    q('time').textContent = `t = ${(step / run.control_hz).toFixed(1)} s`;
  }

  q('speed').textContent = `${Math.round(perSecond / run.control_hz)}× real time`;
  ['timeupdate', 'seeked', 'loadedmetadata', 'ended'].forEach(name => video.addEventListener(name, update));
  let frame = null;
  const loop = () => { update(); frame = video.paused ? null : requestAnimationFrame(loop); };
  video.addEventListener('play', () => { if (!frame) frame = requestAnimationFrame(loop); });
  // ?t=<sim seconds> opens the player at that moment of the run.
  const start = Number(new URLSearchParams(location.search).get('t'));
  if (start > 0) { const go = () => seek(start * run.control_hz); video.readyState >= 1 ? go() : video.addEventListener('loadedmetadata', go, {once: true}); }
  update();
}

document.querySelectorAll('.player[data-rig]').forEach(root => {
  const run = data.rigs[root.dataset.rig];
  if (run) initPlayer(root, run);
});

// Autoplay: every clip is muted and inline, so Safari and iOS allow it. Play only
// what is on screen and in the shown tab; a browser that still blocks autoplay keeps the controls.
const clips = [...document.querySelectorAll('video[data-autoplay]')];
clips.forEach(v => { v.muted = true; v.playsInline = true; });
const visible = v => v.offsetParent !== null;
const play = v => { if (!visible(v)) return; const p = v.play(); if (p) p.catch(() => {}); };
const onScreen = new Set();
if ('IntersectionObserver' in window) {
  const watch = new IntersectionObserver(entries => entries.forEach(e => {
    if (e.isIntersecting) { onScreen.add(e.target); play(e.target); } else { onScreen.delete(e.target); e.target.pause(); }
  }), {threshold: 0.25});
  clips.forEach(v => watch.observe(v));
} else {
  clips.forEach(v => onScreen.add(v));
  clips.forEach(play);
}

// Camera-rig tabs: show one rig's views, player, clips and grid at a time.
const tabs = [...document.querySelectorAll('[data-rig-tab]')];
function selectRig(rig) {
  document.body.dataset.rig = rig;
  tabs.forEach(t => t.setAttribute('aria-selected', String(t.dataset.rigTab === rig)));
  clips.forEach(v => (visible(v) ? onScreen.has(v) && play(v) : v.pause()));
  try { localStorage.setItem('rig', rig); } catch (e) { /* storage unavailable */ }
}
tabs.forEach(t => t.addEventListener('click', () => selectRig(t.dataset.rigTab)));
let saved = null;
try { saved = localStorage.getItem('rig'); } catch (e) { /* storage unavailable */ }
const fromHash = location.hash === '#custom' ? 'custom' : location.hash === '#official' ? 'official' : null;
selectRig(fromHash || (data.rigs[saved] ? saved : 'official'));
