const state = { cases: [], selected: null, session: null };

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const decisionLabels = {
  accept: "Accept match",
  retrieve_more: "Retrieve more evidence",
  human_review: "Request human review",
  reject: "Reject match",
};

async function request(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Request failed: ${response.status}`);
  }
  return response.json();
}

function percent(value) {
  return `${Math.round(Number(value || 0) * 100)}%`;
}

function score(value) {
  return Number(value || 0).toFixed(3);
}

function titleCase(value) {
  return String(value || "").replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.hidden = false;
  window.clearTimeout(toast.timer);
  toast.timer = window.setTimeout(() => { node.hidden = true; }, 3600);
}

function renderCases() {
  $("#case-count").textContent = state.cases.length;
  $("#case-list").innerHTML = state.cases.map((item) => `
    <button class="case-button" data-case="${item.case_id}" aria-label="Open ${item.title}">
      <img src="${item.query_image_url}" alt="" />
      <span><strong>${item.title}</strong><small>${titleCase(item.expected_decision)}</small></span>
    </button>
  `).join("");
  $$(".case-button").forEach((button) => {
    button.addEventListener("click", () => selectCase(button.dataset.case));
  });
}

function renderFailures(items) {
  $("#failure-cards").innerHTML = items.map((item) => `
    <article class="failure-card">
      <img src="${item.query_image_url}" alt="${item.title} fixture" />
      <div>
        <small>${titleCase(item.expected_decision)}</small>
        <h3>${item.title}</h3>
        <p>${item.risk}</p>
      </div>
    </article>
  `).join("");
}

function selectCase(caseId) {
  state.selected = state.cases.find((item) => item.case_id === caseId);
  state.session = null;
  $$(".case-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.case === caseId);
  });
  $("#empty-state").hidden = true;
  $("#case-detail").hidden = false;
  $("#result-panel").hidden = true;
  $("#case-label").textContent = `${state.selected.label} · EXPECTED ${titleCase(state.selected.expected_decision)}`;
  $("#case-title").textContent = state.selected.title;
  $("#case-summary").textContent = state.selected.summary;
  $("#query-image").src = state.selected.query_image_url;
  $("#candidate-images").innerHTML = state.selected.candidate_image_urls.map((url, index) => `
    <figure><img src="${url}" alt="Candidate evidence ${index + 1}" /><figcaption>E${index + 1}</figcaption></figure>
  `).join("");
  $("#review-status").textContent = "";
}

async function runSelected() {
  if (!state.selected) return;
  const button = $("#run-case");
  button.disabled = true;
  button.textContent = "Running OpenCV loop…";
  try {
    const record = await request(`/api/v1/cases/${state.selected.case_id}/run`, { method: "POST" });
    state.session = record;
    renderResult(record);
    await refreshMetrics();
  } catch (error) {
    toast(error.message);
  } finally {
    button.disabled = false;
    button.textContent = "Run evidence loop";
  }
}

async function runSuite() {
  const button = $("#run-suite");
  button.disabled = true;
  button.textContent = "Running six audited cases…";
  try {
    const payload = await request("/api/v1/run-all", { method: "POST" });
    renderMetrics(payload.summary);
    const last = payload.runs[payload.runs.length - 1];
    selectCase(last.case.case_id);
    state.session = last;
    renderResult(last);
    toast(`Suite complete: ${payload.runs.filter((run) => run.expectation_met).length}/${payload.runs.length} expected safe actions.`);
  } catch (error) {
    toast(error.message);
  } finally {
    button.disabled = false;
    button.textContent = "Run all six cases";
  }
}

function renderResult(record) {
  const trace = record.session.final_trace;
  const decision = trace.decision;
  $("#result-panel").hidden = false;
  $("#decision-title").textContent = decisionLabels[decision] || titleCase(decision);
  $("#decision-reasons").textContent = trace.reasons.map(titleCase).join(" · ");
  $("#result-confidence").textContent = percent(trace.confidence);
  $("#result-steps").textContent = `${record.session.acquisitions_used} acquisition${record.session.acquisitions_used === 1 ? "" : "s"}`;
  $("#result-trace").textContent = record.session.trace_valid ? "Verified" : "Invalid";
  $("#result-latency").textContent = `${record.runtime.latency_ms.toFixed(1)} ms`;
  const badge = $("#decision-badge");
  badge.className = `decision-badge ${decision === "human_review" ? "review" : decision === "reject" ? "reject" : ""}`;

  $("#measurement-rows").innerHTML = record.comparisons.map((item, index) => `
    <tr>
      <td>Evidence ${index + 1}</td>
      <td>${score(item.retrieval_similarity)}</td>
      <td>${score(item.geometry_score)}</td>
      <td>${score(item.color_score)}</td>
      <td>${score(item.quality_score)}</td>
      <td><strong>${score(item.fused_similarity)}</strong></td>
    </tr>
  `).join("") || `<tr><td colspan="6">No acquisition was required.</td></tr>`;

  $("#audit-events").innerHTML = record.session.events.map((event) => `
    <li><b>${event.sequence + 1}</b><strong>${event.event_type}</strong><code title="${event.event_hash}">${event.event_hash}</code></li>
  `).join("");
  $("#review-status").textContent = record.human_review
    ? `Recorded: ${titleCase(record.human_review.decision)}`
    : "No human decision recorded yet.";
  $("#result-panel").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function submitReview(decision) {
  if (!state.session) {
    toast("Run a case before recording a human decision.");
    return;
  }
  try {
    const record = await request(`/api/v1/sessions/${state.session.session_id}/review`, {
      method: "POST",
      body: JSON.stringify({
        decision,
        reviewer: "judge",
        note: $("#review-note").value,
      }),
    });
    state.session = record;
    $("#review-status").textContent = `Recorded: ${titleCase(record.human_review.decision)}. Model trace preserved.`;
    toast("Human review recorded without changing the model trace.");
  } catch (error) {
    toast(error.message);
  }
}

function renderMetrics(metrics) {
  $("#metric-runs").textContent = metrics.run_count;
  $("#metric-success").textContent = metrics.run_count ? percent(metrics.expectation_success_rate) : "—";
  $("#metric-traces").textContent = metrics.run_count ? percent(metrics.trace_valid_rate) : "—";
  $("#metric-latency").textContent = metrics.run_count ? `${metrics.latency_ms.p95.toFixed(1)} ms` : "—";
}

async function refreshMetrics() {
  renderMetrics(await request("/api/v1/metrics"));
}

function setupTabs() {
  $$(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      $$(".tab").forEach((item) => item.classList.toggle("active", item === tab));
      $$(".tab-panel").forEach((panel) => {
        panel.hidden = panel.id !== `tab-${tab.dataset.tab}`;
      });
    });
  });
}

async function init() {
  try {
    const [casePayload, failurePayload, metrics] = await Promise.all([
      request("/api/v1/cases"),
      request("/api/v1/failure-gallery"),
      request("/api/v1/metrics"),
    ]);
    state.cases = casePayload.cases;
    renderCases();
    renderFailures(failurePayload.cases);
    renderMetrics(metrics);
    selectCase(state.cases[0].case_id);
  } catch (error) {
    toast(error.message);
  }
}

$("#run-case").addEventListener("click", runSelected);
$("#run-suite").addEventListener("click", runSuite);
$$('[data-review]').forEach((button) => button.addEventListener("click", () => submitReview(button.dataset.review)));
setupTabs();
init();
