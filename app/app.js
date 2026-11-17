const form = document.querySelector('#analysis-form');
const beforeInput = form.elements.before;
const afterInput = form.elements.after;
const runButton = document.querySelector('#analyze');
const preview = document.querySelector('#preview');
const errorBox = document.querySelector('#error');
const resultSection = document.querySelector('#results');
const emptyState = document.querySelector('#empty');
const questionInput = document.querySelector('#question-input');
const interpretation = document.querySelector('#interpretation');
let interpretationTimer;
let interpretationController;

async function interpretQuestion() {
  interpretationController?.abort();
  interpretationController = new AbortController();
  interpretation.setAttribute('aria-busy', 'true');
  document.querySelector('#interpret-status').textContent = 'Reading the question…';
  try {
    const response = await fetch('/api/interpret', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: questionInput.value }),
      signal: interpretationController.signal,
    });
    if (!response.ok || !response.headers.get('content-type')?.includes('application/json')) {
      throw new Error('Question interpretation is unavailable. Restart the dashboard with run_dashboard.bat.');
    }
    const parsed = await response.json();
    if (!response.ok) throw new Error(parsed.error || 'Could not interpret this question.');
    document.querySelector('#parsed-intent').textContent = parsed.intent_label;
    document.querySelector('#parsed-operation').textContent = parsed.operation_label;
    document.querySelector('#parsed-target').textContent = parsed.target === 'change'
      ? 'Pixel-level change' : parsed.target.replaceAll('_', ' ');
    document.querySelector('#parsed-region').textContent = parsed.regions
      .map(region => region.replaceAll('_', ' ')).join(' · ');
    document.querySelector('#interpret-status').textContent = parsed.supported
      ? 'Request understood · analysis has not run yet'
      : 'Request recognized · target is not supported by the current pipeline';
    document.querySelector('#interpretation-note').textContent = parsed.explanation;
    const actions = document.querySelector('#planned-actions');
    actions.replaceChildren();
    for (const step of parsed.steps) {
      const item = document.createElement('li');
      item.textContent = step.label;
      actions.append(item);
    }
    if (!parsed.steps.length) {
      const item = document.createElement('li');
      item.textContent = 'No analysis tools will run for this request';
      actions.append(item);
    }
  } catch (error) {
    if (error.name !== 'AbortError') {
      document.querySelector('#interpret-status').textContent = error.message;
      document.querySelector('#planned-actions').replaceChildren();
    }
  } finally {
    interpretation.setAttribute('aria-busy', 'false');
  }
}

questionInput.addEventListener('input', () => {
  window.clearTimeout(interpretationTimer);
  interpretationTimer = window.setTimeout(interpretQuestion, 260);
});
interpretQuestion();

function imagePreview(input, label) {
  const file = input.files[0];
  if (!file) return;
  const zone = input.closest('.dropzone');
  zone.querySelector('.file-name')?.remove();
  const name = document.createElement('span');
  name.className = 'file-name';
  name.textContent = file.name;
  zone.append(name);
  if (/\.tiff?$/i.test(file.name) || !file.type.startsWith('image/')) return;
  preview.querySelector(`[data-label="${label}"]`)?.remove();
  const card = document.createElement('div');
  card.className = 'preview-card';
  card.dataset.label = label;
  const img = document.createElement('img');
  img.src = URL.createObjectURL(file);
  img.alt = `${label.toLowerCase()} satellite capture`;
  const caption = document.createElement('div');
  caption.textContent = `${label} · ${file.name}`;
  card.append(img, caption);
  preview.append(card);
  preview.classList.remove('hidden');
}

function updateReady() {
  runButton.disabled = !(beforeInput.files.length && afterInput.files.length);
}

beforeInput.addEventListener('change', () => { imagePreview(beforeInput, 'BEFORE'); updateReady(); });
afterInput.addEventListener('change', () => { imagePreview(afterInput, 'AFTER'); updateReady(); });

document.querySelector('#sample-pair').addEventListener('click', async event => {
  const button = event.currentTarget;
  button.disabled = true;
  button.textContent = 'Loading sample…';
  try {
    const [before, after] = await Promise.all([
      fetch('/api/sample/before'), fetch('/api/sample/after'),
    ]);
    if (!before.ok || !after.ok) throw new Error('The local OSCD sample could not be loaded.');
    const [beforeBlob, afterBlob] = await Promise.all([before.blob(), after.blob()]);
    for (const [input, blob, name] of [
      [beforeInput, beforeBlob, 'oscd100-train_000-before.png'],
      [afterInput, afterBlob, 'oscd100-train_000-after.png'],
    ]) {
      const transfer = new DataTransfer();
      transfer.items.add(new File([blob], name, { type: 'image/png' }));
      input.files = transfer.files;
      input.dispatchEvent(new Event('change', { bubbles: true }));
    }
    form.elements.pixel_size.value = '10';
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.classList.remove('hidden');
  } finally {
    button.disabled = false;
    button.textContent = 'Load OSCD sample';
  }
});

document.querySelectorAll('.dropzone').forEach(zone => {
  zone.addEventListener('dragover', event => { event.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', event => {
    event.preventDefault();
    zone.classList.remove('dragover');
    const input = zone.querySelector('input');
    if (!event.dataTransfer.files.length) return;
    input.files = event.dataTransfer.files;
    input.dispatchEvent(new Event('change', { bubbles: true }));
  });
});

function metric(label, value) {
  const box = document.createElement('div');
  box.className = 'metric';
  const title = document.createElement('span');
  title.textContent = label;
  const result = document.createElement('strong');
  result.textContent = value;
  box.append(title, result);
  return box;
}

function recordLine(container, label, value) {
  const line = document.createElement('p');
  const name = document.createElement('strong');
  name.textContent = `${label}: `;
  line.append(name, document.createTextNode(value));
  container.append(line);
}

form.addEventListener('submit', async event => {
  event.preventDefault();
  errorBox.classList.add('hidden');
  resultSection.classList.add('hidden');
  emptyState.classList.add('hidden');
  document.querySelector('#loading').classList.remove('hidden');
  runButton.disabled = true;
  try {
    const response = await fetch('/api/analyze', { method: 'POST', body: new FormData(form) });
    const data = await response.json();
    if (!response.ok && !data.tool_runs?.length) throw new Error(data.error || 'Analysis could not be completed.');

    document.querySelector('#answer').textContent = data.answer || data.error || 'No result returned.';
    document.querySelector('.answer-tag').textContent = !data.success
      ? 'PARTIAL RUN · REVIEW RECORD'
      : data.intent === 'UNSUPPORTED' ? 'OUT OF SCOPE' : 'PIXEL CHANGE ESTIMATE';
    document.querySelector('.result-caveat').classList.toggle('hidden', !data.stats);

    const metrics = document.querySelector('#metrics');
    metrics.replaceChildren();
    if (data.stats) {
      metrics.append(
        metric('Share flagged as changed', `${data.stats.percentage.toFixed(2)}%`),
        metric('Pixels flagged', data.stats.changed_pixels.toLocaleString()),
        metric('Image pixels', data.stats.total_pixels.toLocaleString()),
      );
      if (data.stats.area_m2 != null) metrics.append(metric('Estimated area', `${data.stats.area_m2.toLocaleString()} m²`));
    }

    const regionPanel = document.querySelector('#region-panel');
    const regionMetrics = document.querySelector('#region-metrics');
    regionMetrics.replaceChildren();
    for (const [name, stats] of Object.entries(data.regions || {})) {
      regionMetrics.append(metric(name.replaceAll('_', ' '), `${stats.percentage.toFixed(2)}% · ${stats.pixels.toLocaleString()} px`));
    }
    regionPanel.classList.toggle('hidden', !Object.keys(data.regions || {}).length);

    const visualPanel = document.querySelector('#visual-panel');
    if (data.visualization) {
      document.querySelector('#visualization').src = `data:image/png;base64,${data.visualization}`;
      visualPanel.classList.remove('hidden');
    } else visualPanel.classList.add('hidden');
    document.querySelector('#threshold').textContent = data.threshold == null ? '' : `CVA · Otsu ${data.threshold.toFixed(4)}`;

    const record = document.querySelector('#record');
    record.replaceChildren();
    recordLine(record, 'Status', data.success ? 'Complete' : 'Stopped on error');
    recordLine(record, 'Question route', `${data.intent} · ${data.operation}`);
    recordLine(record, 'Requested target', data.target || 'change');
    recordLine(record, 'Selected region', (data.parameters?.comparison_regions || [data.region]).join(', '));
    recordLine(record, 'Image dimensions', `${data.dimensions.width} × ${data.dimensions.height} px · ${data.dimensions.bands} bands`);
    recordLine(record, 'Before source', `${data.inputs.before.filename} · ${data.inputs.before.format} · CRS ${data.inputs.before.crs || 'not supplied'}`);
    recordLine(record, 'After source', `${data.inputs.after.filename} · ${data.inputs.after.format} · CRS ${data.inputs.after.crs || 'not supplied'}`);
    const resolution = data.inputs.pixel_resolution_m;
    recordLine(record, 'Pixel resolution', resolution ? `${resolution[0]} × ${resolution[1]} m` : 'unknown; physical area omitted');
    recordLine(record, 'Planned tools', (data.plan || []).join(' → ') || 'No CV tools run');
    for (const run of data.tool_runs || []) {
      recordLine(record, run.success ? 'Completed' : 'Failed', `${run.tool}${Object.keys(run.parameters || {}).length ? ` · ${JSON.stringify(run.parameters)}` : ''}${run.error ? ` · ${run.error}` : ''}`);
    }

    resultSection.classList.remove('hidden');
    resultSection.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
    if (!data.success) {
      errorBox.textContent = data.error || 'Analysis stopped. Review the analysis record for details.';
      errorBox.classList.remove('hidden');
    }
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.classList.remove('hidden');
    emptyState.classList.remove('hidden');
  } finally {
    document.querySelector('#loading').classList.add('hidden');
    updateReady();
  }
});

document.querySelector('#new-analysis').addEventListener('click', () => {
  resultSection.classList.add('hidden');
  errorBox.classList.add('hidden');
  emptyState.classList.remove('hidden');
  window.scrollTo({ top: 0, behavior: 'smooth' });
});
