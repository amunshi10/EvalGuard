"use strict";

const form = document.getElementById("form");
const fileRow = document.getElementById("file-row");
const fileInput = document.getElementById("file");
const statusEl = document.getElementById("status");
const errorEl = document.getElementById("error");
const resultsEl = document.getElementById("results");
const runBtn = document.getElementById("run");

const STATUS_WORD = { pass: "PASS", warn: "WARN", fail: "FAIL", skip: "SKIP" };

/* Selecting a sample prefills the fields it was built with; choosing "upload"
   reveals the file picker and clears them. */
form.addEventListener("change", (event) => {
  if (event.target.name !== "sample") return;
  const input = event.target;
  const isUpload = input.value === "";

  fileRow.hidden = !isUpload;
  const fields = {
    group_col: isUpload ? "" : input.dataset.group || "",
    ignore_cols: isUpload ? "" : input.dataset.ignore || "",
    seed_scores: isUpload ? "" : input.dataset.seed || "",
    split_scores: isUpload ? "" : input.dataset.split || "",
  };
  for (const [id, value] of Object.entries(fields)) {
    document.getElementById(id).value = value;
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = new FormData(form);

  if (!data.get("sample") && !(fileInput.files && fileInput.files.length)) {
    return showError("Pick a sample dataset or choose a CSV file to upload.");
  }

  errorEl.hidden = true;
  resultsEl.hidden = true;
  runBtn.disabled = true;
  statusEl.textContent = "Running checks...";

  try {
    const response = await fetch("/api/analyze", { method: "POST", body: data });
    const payload = await response.json();
    if (!response.ok) return showError(payload.error || "Something went wrong.");
    render(payload);
    statusEl.textContent = "";
  } catch (err) {
    showError("Could not reach the server: " + err.message);
  } finally {
    runBtn.disabled = false;
  }
});

function showError(message) {
  statusEl.textContent = "";
  errorEl.textContent = message;
  errorEl.hidden = false;
  resultsEl.hidden = true;
}

/* Text from the server is inserted with textContent, never innerHTML, so a CSV
   value cannot inject markup into the page. */
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function render(report) {
  resultsEl.replaceChildren();

  const verdict = el("div", `verdict verdict-${report.verdict}`);
  verdict.append(el("div", "verdict-word", STATUS_WORD[report.verdict]));

  const body = el("div");
  body.append(el("p", null, report.verdict_text));
  const c = report.counts;
  body.append(
    el("div", "tally", `${c.fail} failed  ·  ${c.warn} warnings  ·  ${c.pass} passed  ·  ${c.skip} skipped`)
  );
  verdict.append(body);
  resultsEl.append(verdict);

  const d = report.dataset;
  const splits = Object.entries(d.splits).map(([k, v]) => `${k} ${v.toLocaleString()}`).join("  ·  ");
  const bits = [
    `${d.rows.toLocaleString()} rows`,
    splits,
    `${d.classes} classes`,
    `${d.feature_columns} feature columns`,
  ];
  if (d.group_column) bits.push(`grouped by ${d.group_column}`);
  resultsEl.append(el("div", "dsline", bits.join("  ·  ")));

  for (const warning of report.warnings) {
    resultsEl.append(el("div", "notice", warning));
  }

  for (const check of report.checks) {
    resultsEl.append(renderCheck(check));
  }

  resultsEl.hidden = false;
  resultsEl.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderCheck(check) {
  const card = el("section", `check check-${check.status}`);

  const head = el("div", "check-head");
  const title = el("div", "check-title");
  title.append(el("span", `badge badge-${check.status}`, STATUS_WORD[check.status]));
  title.append(el("span", "check-name", check.title));
  head.append(title);
  head.append(el("p", "headline", check.headline));

  if (check.why_it_matters) {
    const why = el("div", "why");
    why.append(el("h4", null, "Why this matters"));
    why.append(el("div", null, check.why_it_matters));
    head.append(why);
  }
  if (check.what_to_do) {
    const what = el("div", "why");
    what.append(el("h4", null, "What to do"));
    what.append(el("div", null, check.what_to_do));
    head.append(what);
  }
  for (const note of check.notes) {
    head.append(el("div", "note", note));
  }
  card.append(head);

  const entries = Object.entries(check.metrics);
  if (entries.length) {
    const strip = el("dl", "metrics");
    for (const [key, value] of entries) {
      const cell = el("div", "metric");
      cell.append(el("dt", null, key));
      cell.append(el("dd", null, String(value)));
      strip.append(cell);
    }
    card.append(strip);
  }

  if (check.tables.length) {
    const holder = el("div", "tables");
    for (const table of check.tables) {
      const details = el("details");
      details.append(el("summary", null, `${table.title} (${table.rows.length} rows)`));
      details.append(buildTable(table));
      holder.append(details);
    }
    card.append(holder);
  }

  return card;
}

function buildTable(spec) {
  const scroll = el("div", "tscroll");
  const table = el("table");

  const thead = el("thead");
  const headRow = el("tr");
  for (const column of spec.columns) {
    headRow.append(el("th", null, column.replace(/_/g, " ")));
  }
  thead.append(headRow);
  table.append(thead);

  const tbody = el("tbody");
  for (const row of spec.rows) {
    const tr = el("tr");
    for (const column of spec.columns) {
      const value = row[column];
      // A single free-text column is prose, so let it wrap instead of scrolling.
      const cls = spec.columns.length === 1 ? "wrap-cell" : null;
      tr.append(el("td", cls, value === undefined || value === null ? "" : String(value)));
    }
    tbody.append(tr);
  }
  table.append(tbody);

  scroll.append(table);
  return scroll;
}
