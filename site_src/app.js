'use strict';
const data = JSON.parse(document.getElementById('page-data').textContent);
const NS = 'http://www.w3.org/2000/svg';
const byId = id => document.getElementById(id);

// Real-to-sim slider: the real frame is clipped to the left of the line.
const compare = byId('compare');
if (compare) {
  const range = byId('compare-range');
  const set = () => compare.style.setProperty('--x', `${range.value}%`);
  range.addEventListener('input', set);
  set();
}

// Video -> control step -> graph node. One step clock drives everything.
const video = byId('run-video');
const svg = byId('graph-svg');
const pieces = data.pieces;
const perSecond = data.video.steps_per_second;
const cap = s => s.charAt(0).toUpperCase() + s.slice(1);
const seek = step => { video.currentTime = Math.max(0, step + 1) / perSecond; update(); };
let shown = null;

pieces.forEach((piece, index) => {
  const button = document.createElement('button');
  button.type = 'button';
  button.style.setProperty('--c', piece.color);
  button.style.setProperty('--w', Math.max(1, (piece.end ?? piece.start) - piece.start));
  button.title = piece.name === 'finish' ? 'Finish' : `${cap(piece.name)} · piece ${index + 1}`;
  button.setAttribute('aria-label', `Jump to ${button.title}`);
  button.addEventListener('click', () => seek(piece.start));
  byId('track').append(button);
  piece.button = button;
});

function el(name, attrs = {}, text) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text != null) node.textContent = text;
  return node;
}

// Draw the subgraph: one box per node, one labelled edge per success exit
// (parallel exits fan out), and every node's error exit to a shared abort.
function drawGraph(piece) {
  const W = 300, boxW = 176, boxH = 26, gap = 30, x0 = 34, rail = 262;
  const H = 12 + piece.nodes.length * (boxH + gap) + 18;
  svg.replaceChildren();
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const defs = el('defs');
  const marker = el('marker', {id: 'arrow', viewBox: '0 0 8 8', refX: 7, refY: 4, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse'});
  marker.append(el('path', {d: 'M0,0 L8,4 L0,8 z', class: 'arrowhead'}));
  defs.append(marker);
  svg.append(defs);
  const top = i => 12 + i * (boxH + gap);
  svg.append(el('line', {x1: rail, y1: top(0) + boxH / 2, x2: rail, y2: top(piece.nodes.length - 1) + boxH / 2 + 18, class: 'abort-rail'}));
  svg.append(el('text', {x: rail, y: top(piece.nodes.length - 1) + boxH / 2 + 30, class: 'abort-label', 'text-anchor': 'middle'}, 'abort'));
  piece.nodes.forEach((node, i) => {
    const y = top(i);
    svg.append(el('line', {x1: x0 + boxW, y1: y + boxH / 2, x2: rail, y2: y + boxH / 2, class: 'abort-edge'}));
    if (i + 1 < piece.nodes.length) {
      const exits = node.exits, k = exits.length;
      exits.forEach((exit, j) => {
        const dx = (j - (k - 1) / 2) * 14;
        const sx = x0 + boxW / 2 + dx, sy = y + boxH, ey = top(i + 1);
        svg.append(el('path', {d: `M${sx},${sy} C${sx},${sy + gap / 2} ${x0 + boxW / 2},${ey - gap / 2} ${x0 + boxW / 2},${ey - 1}`, class: 'edge', 'marker-end': 'url(#arrow)'}));
      });
      const label = el('text', {x: x0 + boxW / 2 + 10 + (k - 1) * 7, y: y + boxH + gap / 2 + 3, class: 'edge-label'}, node.exit);
      label.append(el('title', {}, exits.join(' · ')));
      svg.append(label);
    }
    const group = el('g', {class: 'node', tabindex: 0, role: 'button', 'aria-label': `${node.label}: jump to this node`});
    group.append(el('rect', {x: x0, y, width: boxW, height: boxH, rx: 5}));
    group.append(el('text', {x: x0 + 10, y: y + boxH / 2 + 4, class: 'node-label'}, node.label));
    group.append(el('text', {x: x0 + boxW - 8, y: y + boxH / 2 + 4, class: 'node-time', 'text-anchor': 'end'},
      node.end != null ? `${(node.end / data.control_hz).toFixed(0)} s` : ''));
    if (node.relook) group.append(el('text', {x: x0 - 16, y: y + boxH / 2 + 5, class: 'relook'}, '↻'));
    group.addEventListener('click', () => seek(node.start));
    group.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); seek(node.start); } });
    svg.append(group);
    node.group = group;
  });
}

function showPiece(piece) {
  if (shown === piece) return;
  shown = piece;
  const index = pieces.indexOf(piece);
  byId('now-swatch').style.setProperty('--c', piece.color);
  byId('now-piece').textContent = piece.name === 'finish' ? 'Finish' : cap(piece.name);
  byId('now-index').textContent = piece.name === 'finish' ? 'after piece 12' : `piece ${index + 1}/12 · ${piece.nodes.length} nodes`;
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
  const metric = detail.metrics.placement_error_mm != null ? ` Checkpoint: placed within ${detail.metrics.placement_error_mm.toFixed(2)} mm.` : '';
  byId('node-detail').textContent = detail.detail + metric;
  byId('clock-time').textContent = `t = ${(step / data.control_hz).toFixed(1)} s`;
}

byId('clock-speed').textContent = `${Math.round(perSecond / data.control_hz)}× real time`;
['timeupdate', 'seeked', 'loadedmetadata', 'ended'].forEach(name => video.addEventListener(name, update));
let frame = null;
const loop = () => { update(); frame = video.paused ? null : requestAnimationFrame(loop); };
video.addEventListener('play', () => { if (!frame) frame = requestAnimationFrame(loop); });
update();
