let FEATURES = [];

async function init() {
  FEATURES = await (await fetch('/features')).json();
  const box = document.getElementById('fields');
  box.innerHTML = FEATURES.map(f =>
    `<div><label>${f}</label><input type="number" step="any" min="0" data-f="${f}"></div>`).join('');
  loadMetrics(); loadHistory();
}

function fill(values) {
  document.querySelectorAll('[data-f]').forEach(i => i.value = values[i.dataset.f]);
}
async function loadSample(kind) {
  fill(await (await fetch('/sample/' + kind)).json());
  document.getElementById('label').value = kind === 'benign' ? 'Sample (benign)' : 'Sample (malignant)';
}
function clearForm() {
  document.querySelectorAll('[data-f]').forEach(i => i.value = '');
  document.getElementById('step2').hidden = true;
}

async function predict() {
  const features = {};
  let empty = false;
  document.querySelectorAll('[data-f]').forEach(i => {
    if (i.value === '') empty = true; features[i.dataset.f] = i.value;
  });
  const out = document.getElementById('result');
  document.getElementById('step2').hidden = false;
  if (empty) { out.innerHTML = '<p>Please fill all 30 fields or load a sample.</p>'; return; }
  const res = await fetch('/predict', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({patient_label: document.getElementById('label').value, features})});
  const d = await res.json();
  if (d.error) { out.innerHTML = `<p>${d.error}</p>`; return; }
  const mal = d.prediction === 'Malignant';
  out.innerHTML = `<div class="result ${mal ? 'malignant' : 'benign'}">
    <h3>${mal ? 'Malignant Tumour Detected' : 'Benign (Non-cancerous)'}</h3>
    <p>Confidence: <b>${d.confidence}%</b></p>
    Malignant ${d.malignant_prob}%<div class="bar"><div style="width:${d.malignant_prob}%;background:#dc2626"></div></div>
    Benign ${d.benign_prob}%<div class="bar"><div style="width:${d.benign_prob}%;background:#16a34a"></div></div>
  </div>`;
  loadHistory();
}

async function loadMetrics() {
  const m = await (await fetch('/metrics')).json();
  if (!m.test) return;
  const t = m.test;
  document.getElementById('metrics').innerHTML =
    [['Accuracy', t.accuracy + '%'], ['Precision', t.precision + '%'], ['Recall', t.recall + '%'],
     ['F1-Score', t.f1 + '%'], ['ROC-AUC', t.auc]].map(([k, v]) =>
      `<div class="stat"><b>${v}</b>${k}</div>`).join('');
}

async function loadHistory() {
  const rows = await (await fetch('/history')).json();
  document.querySelector('#hist tbody').innerHTML = rows.map(r =>
    `<tr><td>${r.id}</td><td>${r.patient_label}</td><td>${r.prediction}</td><td>${r.confidence}%</td><td>${r.created_at}</td></tr>`).join('')
    || '<tr><td colspan="5">No predictions yet.</td></tr>';
}
async function clearHistory() { await fetch('/history', {method: 'DELETE'}); loadHistory(); }

init();
