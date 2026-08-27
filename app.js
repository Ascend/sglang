'use strict';

const $ = (sel, root) => (root || document).querySelector(sel);

function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function short(sha, n) {
  return sha && sha.length > (n || 8) ? sha.slice(0, n || 8) : sha;
}

// Normalize a git remote (https or ssh) into an https repo base URL.
function repoUrl(remote) {
  if (!remote) return null;
  let r = String(remote).trim().replace(/\.git$/, '');
  const ssh = r.match(/^git@([^:]+):(.+)$/);
  if (ssh) r = 'https://' + ssh[1] + '/' + ssh[2];
  return r;
}

function commitUrl(remote, sha) {
  const base = repoUrl(remote);
  return base && sha ? base + '/commit/' + sha : null;
}

function fmtDate(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  if (isNaN(d)) return iso;
  return d.toLocaleString('zh-CN', { hour12: false });
}

function toast(msg) {
  let el = $('#toast');
  if (!el) {
    el = document.createElement('div');
    el.id = 'toast';
    el.className = 'toast';
    document.body.appendChild(el);
  }
  el.textContent = msg;
  el.classList.add('show');
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove('show'), 1800);
}

/* ---------------- Data ---------------- */

let allManifests = [];
let currentFilter = 'all';

async function loadAll() {
  const listEl = $('#manifest-list');
  const countEl = $('#manifest-count');
  try {
    const res = await fetch('manifests/manifests.json', { cache: 'no-store' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    allManifests = data.manifests || [];
    renderList();
    if (allManifests.length) selectManifest(allManifests[0]);
  } catch (err) {
    countEl.textContent = '0';
    listEl.innerHTML = '<div class="empty">加载失败：' + esc(err.message) + '</div>';
  }
}

function filtered() {
  if (currentFilter === 'all') return allManifests;
  const days = parseInt(currentFilter, 10);
  if (isNaN(days)) return allManifests;
  const cutoff = Date.now() - days * 86400000;
  return allManifests.filter((m) => {
    const t = new Date(m.generated_at || '').getTime();
    return isNaN(t) || t >= cutoff; // keep entries without a parseable date
  });
}

function renderList() {
  const listEl = $('#manifest-list');
  const countEl = $('#manifest-count');
  const entries = filtered();
  countEl.textContent = entries.length;
  if (!entries.length) {
    listEl.innerHTML = '<div class="empty">暂无匹配记录</div>';
    return;
  }
  listEl.innerHTML = entries.map((e) => {
    const gi = allManifests.indexOf(e);
    const tag = (e.image || '').split(':').pop() || ('#' + gi);
    const sglang = e.sglang || {};
    const commit = sglang.commit ? short(sglang.commit) : '';
    return (
      '<div class="manifest-item" data-idx="' + gi + '">' +
        '<div class="img">' + esc(tag) + '</div>' +
        '<div class="meta">' +
          (commit ? '<span class="commit">' + esc(commit) + '</span>' : '') +
          (sglang.branch ? '<span class="tag">' + esc(sglang.branch) + '</span>' : '') +
          (e.generated_at ? '<span class="date">' + esc(fmtDate(e.generated_at)) + '</span>' : '') +
        '</div>' +
      '</div>'
    );
  }).join('');

  listEl.querySelectorAll('.manifest-item').forEach((item) => {
    item.addEventListener('click', () => {
      listEl.querySelectorAll('.manifest-item').forEach((x) => x.classList.remove('active'));
      item.classList.add('active');
      const m = allManifests[parseInt(item.dataset.idx, 10)];
      if (m) selectManifest(m);
    });
  });
}

/* ---------------- Detail ---------------- */

function selectManifest(m) {
  renderDetail(m, $('#detail'));
}

function renderDetail(d, el) {
  const sglang = d.sglang || {};
  const os = d.os || {};
  const cann = d.cann || {};
  const args = d.build_args || {};
  const pins = d.declared_pip_pins || [];
  const clones = d.git_clones || [];

  const commitHref = commitUrl(sglang.remote, sglang.commit);

  el.innerHTML =
    headerCard(d) +
    sglangCard(sglang, commitHref) +
    buildArgsCard(args) +
    pinsCard(pins, clones) +
    envCard(os, cann);

  const copyBtn = el.querySelector('[data-copy]');
  if (copyBtn) {
    copyBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(copyBtn.dataset.copy).then(
        () => toast('已复制 commit'),
        () => toast('复制失败')
      );
    });
  }
}

function headerCard(d) {
  return (
    '<div class="card header-card">' +
      '<div class="img-full">' + esc(d.image || '') + '</div>' +
      '<div class="row">' +
        '<span><span class="lbl">Dockerfile</span> <code>' + esc(d.dockerfile || '—') + '</code></span>' +
        '<span><span class="lbl">生成时间</span> ' + esc(fmtDate(d.generated_at)) + '</span>' +
      '</div>' +
    '</div>'
  );
}

function sglangCard(s, href) {
  const rows = [
    ['branch', s.branch],
    ['describe', s.describe],
    ['commit_date', s.commit_date ? fmtDate(s.commit_date) : ''],
    ['remote', s.remote],
    ['dir', s.dir],
  ];
  const kv = rows.filter(([, v]) => v).map(([k, v]) =>
    '<div class="k">' + esc(k) + '</div><div class="v">' + esc(v) + '</div>'
  ).join('');

  const commitInner = href
    ? '<a class="commit-main" href="' + esc(href) + '" target="_blank" rel="noopener" title="在 GitHub 查看">' + esc(s.commit || '—') + '</a>'
    : '<span class="commit-main">' + esc(s.commit || '—') + '</span>';

  return (
    '<div class="card">' +
      '<h3><span class="n">1</span>sglang 版本（当次 commit）</h3>' +
      '<div class="commit-block">' +
        commitInner +
        '<button class="copy-btn" data-copy="' + esc(s.commit || '') + '">复制</button>' +
      '</div>' +
      '<div class="kv" style="margin-top:16px">' + kv + '</div>' +
    '</div>'
  );
}

function buildArgsCard(args) {
  const keys = Object.keys(args);
  if (!keys.length) return '';
  const rows = keys.map((k) =>
    '<tr><td class="mono">' + esc(k) + '</td><td class="mono">' + esc(args[k]) + '</td></tr>'
  ).join('');
  return (
    '<div class="card">' +
      '<h3><span class="n">2</span>构建参数（ARG）</h3>' +
      '<table><thead><tr><th>参数</th><th>值</th></tr></thead><tbody>' + rows + '</tbody></table>' +
    '</div>'
  );
}

function pinsCard(pins, clones) {
  const pinChips = pins.map((p) => '<span class="chip pin">' + esc(p) + '</span>').join('');
  const cloneChips = clones.map((c) =>
    '<span class="chip">' + esc(c.remote + ' @ ' + c.branch) + '</span>'
  ).join('');
  return (
    '<div class="card">' +
      '<h3><span class="n">3</span>声明依赖（Dockerfile 固定版本）</h3>' +
      (pinChips ? '<div class="chips">' + pinChips + '</div>' : '<p class="pip-count">无固定版本</p>') +
      (cloneChips ? '<h3 style="margin-top:20px"><span class="n">4</span>git clone 来源</h3><div class="chips">' + cloneChips + '</div>' : '') +
    '</div>'
  );
}

function envCard(os, cann) {
  const stats = [
    ['OS', os.distro],
    ['内核', os.kernel],
    ['架构', os.arch],
    ['CANN 工具链', cann.toolkit],
    ['版本文件', cann.version_files],
  ].filter(([, v]) => v);
  const items = stats.map(([k, v]) =>
    '<div class="stat"><div class="k">' + esc(k) + '</div><div class="v">' + esc(v) + '</div></div>'
  ).join('');
  return (
    '<div class="card">' +
      '<h3><span class="n">5</span>运行环境</h3>' +
      '<div class="stat-grid">' + items + '</div>' +
    '</div>'
  );
}

// Filter dropdown
$('#filter').addEventListener('change', (ev) => {
  currentFilter = ev.target.value;
  renderList();
});

loadAll();
