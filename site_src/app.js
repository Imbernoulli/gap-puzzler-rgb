'use strict';
// Follows the recorded run: video time -> control step -> graph node.
const data = JSON.parse(document.getElementById('page-data').textContent);
const video = document.getElementById('run-video');
const track = document.getElementById('track');
const nodesEl = document.getElementById('nodes');
const pieces = data.pieces;
const perSecond = data.video.steps_per_second;
const cap = s => s.charAt(0).toUpperCase() + s.slice(1);
let shownPiece = null;

pieces.forEach((piece, index) => {
  const button = document.createElement('button');
  button.type = 'button';
  button.style.setProperty('--c', piece.color);
  button.style.setProperty('--w', Math.max(1, (piece.end ?? piece.start) - piece.start));
  button.title = piece.name === 'finish' ? 'Finish' : `${cap(piece.name)} · piece ${index + 1}`;
  button.setAttribute('aria-label', `Jump to ${button.title}`);
  button.addEventListener('click', () => { video.currentTime = (piece.start + 1) / perSecond; update(); });
  track.append(button);
  piece.button = button;
});

function nodeMeta(node) {
  const parts = [];
  if (node.end != null) parts.push(`t ${(node.end / data.control_hz).toFixed(1)} s`);
  if (node.metrics.placement_error_mm != null) parts.push(`placed within ${node.metrics.placement_error_mm.toFixed(2)} mm`);
  else if (node.metrics.earlier_pieces_max_mm != null) parts.push(`earlier pieces ≤ ${node.metrics.earlier_pieces_max_mm.toFixed(2)} mm`);
  return parts.join(' · ');
}

function showPiece(piece) {
  if (shownPiece === piece) return;
  shownPiece = piece;
  const index = pieces.indexOf(piece);
  document.getElementById('now-swatch').style.setProperty('--c', piece.color);
  document.getElementById('now-piece').textContent = piece.name === 'finish' ? 'Finish' : piece.name;
  document.getElementById('now-index').textContent = piece.name === 'finish' ? 'After the twelfth piece' : `Piece ${index + 1} of 12`;
  nodesEl.replaceChildren(...piece.nodes.map(node => {
    const li = document.createElement('li');
    const label = document.createElement('span');
    label.textContent = node.label;
    const exit = document.createElement('span');
    exit.className = 'exit';
    exit.textContent = node.exit;
    exit.title = node.exits.join(' | ');
    const meta = document.createElement('span');
    meta.className = 'meta';
    meta.textContent = nodeMeta(node);
    li.append(label, exit, meta);
    node.li = li;
    return li;
  }));
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
    const done = node.end != null && step >= node.end;
    const now = !done && !active && step >= node.start;
    if (now) active = node;
    node.li.classList.toggle('done', done && node.passed);
    node.li.classList.toggle('refused', node.end == null && step >= node.start && video.ended);
    node.li.classList.toggle('now', now);
  });
  const detail = active || current.nodes[current.nodes.length - 1];
  document.getElementById('node-detail').textContent = detail.detail;
  document.getElementById('clock-time').textContent = `t = ${(step / data.control_hz).toFixed(1)} s`;
}

document.getElementById('clock-speed').textContent = `${(perSecond / data.control_hz).toFixed(0)}× real time`;
['timeupdate', 'seeked', 'loadedmetadata', 'ended'].forEach(name => video.addEventListener(name, update));
let frame = null;
const loop = () => { update(); frame = video.paused ? null : requestAnimationFrame(loop); };
video.addEventListener('play', () => { if (!frame) frame = requestAnimationFrame(loop); });
update();
