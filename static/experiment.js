const el = (id) => document.getElementById(id);
const actionLabels = {
  fall: [["front", "전방"], ["left", "좌측"], ["right", "우측"], ["back", "후방"]],
  nonfall: [["sit", "의자에 앉기"], ["lie", "의도적으로 눕기"], ["pick_up", "물건 줍기"], ["squat", "쪼그려 앉기"]],
};
const statusLabels = { ready: "준비", running: "기록 중", completed: "종료", interrupted: "중단" };
let state = null;
let busy = false;
let createId = null;
let reviewKey = null;
let previewTimer = 0;
let displayedRunId = null;
let displayedTrialId = null;
let lastCompletedId = null;

function message(text, error = false) {
  el("message").textContent = text;
  el("message").classList.toggle("error", error);
}

async function request(path, data) {
  const response = await fetch("/experiment/api/" + path, data === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(typeof payload.detail === "string" ? payload.detail : "입력값과 서버 상태를 확인하세요.");
  }
  return payload;
}

function currentTrial() {
  return state?.current?.trials[state.current.index];
}

async function command(name, data) {
  if (busy) return;
  busy = true;
  try {
    state = await request(name, data);
    message("저장했습니다.");
    render();
  } catch (error) {
    message(error.message, true);
  } finally {
    busy = false;
  }
}

function renderActions() {
  el("actions").replaceChildren();
  for (const [value, label] of actionLabels[el("kind").value]) {
    const item = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = value;
    input.checked = true;
    input.addEventListener("change", renderPlanned);
    item.append(input, document.createTextNode(label));
    el("actions").append(item);
  }
  createId = null;
  renderPlanned();
}

function renderPlanned() {
  const count = el("actions").querySelectorAll("input:checked").length * Number(el("sets").value);
  el("planned").textContent = " 총 " + count + "시행";
}

function options(select, values, selected) {
  select.replaceChildren();
  for (const [value, label] of values) {
    const item = document.createElement("option");
    item.value = value;
    item.textContent = label;
    select.append(item);
  }
  select.value = selected || "";
}

function addRow(body, values) {
  const row = document.createElement("tr");
  for (const value of values) {
    const cell = document.createElement("td");
    cell.textContent = value;
    row.append(cell);
  }
  body.append(row);
}

function loadReview() {
  const run = state?.current;
  const trial = run?.trials.find((item) => item.trial_id === el("review-trial").value);
  reviewKey = trial ? run.id + "/" + trial.trial_id : null;
  el("actual-action").value = trial?.actual_action || trial?.action_label || "";
  el("actual-fall").value = trial?.actual_fall == null ? "" : String(trial.actual_fall);
  el("valid").value = trial?.valid == null ? "" : String(trial.valid);
  el("notes").value = trial?.notes || "";
  el("save-review").disabled = !trial;
}

function render() {
  const run = state.current;
  const trial = currentTrial();
  if (run && run.id !== displayedRunId) {
    displayedRunId = run.id;
    el("participant").value = run.participant_id;
    el("kind").value = run.experiment_type;
    el("sets").value = run.config.sets;
    el("duration").value = run.config.observation_seconds;
    renderActions();
    for (const input of el("actions").querySelectorAll("input")) {
      input.checked = run.config.actions.includes(input.value);
    }
    renderPlanned();
  }
  for (const input of el("setup").querySelectorAll("input, select")) {
    input.disabled = run?.status === "active";
  }
  if (trial?.trial_id !== displayedTrialId) {
    displayedTrialId = trial?.trial_id;
    previewTimer = 0;
    el("preview").hidden = true;
  }
  options(el("history"), state.runs.map((item) => [
    item.id, item.participant_id + " · " + (item.experiment_type === "fall" ? "낙상" : "비낙상")
      + " · " + new Date(item.created_at).toLocaleString(),
  ]), run?.id);
  el("create").disabled = run?.status === "active";
  el("history").disabled = trial?.status === "running";
  el("start").disabled = !trial || trial.status !== "ready" || run.status !== "active";
  el("end").disabled = trial?.status !== "running";
  el("next").disabled = !trial || !["completed", "interrupted"].includes(trial.status) || run.status !== "active";
  el("finish").disabled = !run || run.status !== "active";
  el("next").textContent = run && run.index === run.trials.length - 1 ? "실험 완료" : "다음 시행";
  if (trial) {
    el("current").textContent = run.participant_id + " · " + trial.trial_number + "/" + run.trials.length
      + "시행 · " + trial.set_number + "세트 · " + trial.action_label;
    el("phase").textContent = (statusLabels[trial.status] || trial.status)
      + (run.status !== "active" ? " · 실험 종료됨" : "");
    const done = run.trials.filter((item) => ["completed", "interrupted"].includes(item.status)).length;
    el("progress").value = done / run.trials.length * 100;
    el("progress-text").textContent = done + "/" + run.trials.length + "시행 종료";
    el("confirmed").textContent = trial.fall_detected ? "확정됨" : "확정 없음";
    el("elapsed").textContent = Number(state.elapsed_seconds ?? trial.elapsed_seconds).toFixed(1)
      + " / " + trial.observation_seconds + "초";
    el("details").textContent = "수신 " + trial.frame_count + "프레임 · 관절 유효 "
      + trial.valid_pose_frames + " · UNKNOWN " + trial.unknown_frames
      + (trial.first_confirmed_at ? " · 최초 확정 " + new Date(trial.first_confirmed_at).toLocaleTimeString()
        + " (시행 시작 후 " + trial.first_confirmation_seconds + "초)" : "");
    const warnings = [];
    if (trial.connection_gaps) warnings.push("2초 초과 수신 공백 " + trial.connection_gaps + "회");
    if (trial.session_mismatch_frames) warnings.push("다른 앱 세션의 프레임 " + trial.session_mismatch_frames + "개 제외");
    if (trial.debug_dropped || trial.debug_errors) warnings.push("진단 누락 " + trial.debug_dropped + " · 저장 오류 " + trial.debug_errors);
    if (trial.processing_errors) warnings.push("프레임 분석 오류 " + trial.processing_errors + "회");
    if (trial.status === "completed" && !trial.valid_pose_frames) warnings.push("유효 관절 프레임이 없어 성능 지표에서 제외됩니다.");
    if (!state.save_frames) warnings.push("현재 CSV 모드입니다. 사진 저장은 FALL_DEBUG=frames로 재시작하세요.");
    if (state.storage_error) warnings.push("저장 확인 필요 · " + state.storage_error);
    el("warnings").textContent = warnings.join(" / ");
    if (state.save_frames && trial.debug_directory && Date.now() - previewTimer > 1000) {
      previewTimer = Date.now();
      el("preview").src = "/experiment/api/preview/" + run.id + "/" + trial.trial_id + "?t=" + previewTimer;
      el("preview-note").textContent = "현재 시행 · 왼쪽 수신 원본, 오른쪽 관절 표시. 저장된 최신 사진을 표시합니다.";
    }
    if (!trial.debug_directory || !state.save_frames) el("preview").hidden = true;
  }
  const age = state.frame_age_seconds;
  el("connection").textContent = age == null ? "수신 대기" : age > 2 ? "수신 공백 " + age.toFixed(1) + "초" : "수신 중";
  el("fsm").textContent = state.latest.fall_status || "—";
  const finished = run?.trials.filter((item) => ["completed", "interrupted"].includes(item.status)) || [];
  const newestId = finished.length ? run.id + "/" + finished.at(-1).trial_id : null;
  const selected = newestId !== lastCompletedId ? finished.at(-1)?.trial_id
    : finished.some((item) => run.id + "/" + item.trial_id === reviewKey)
      ? reviewKey.split("/")[1] : finished.at(-1)?.trial_id;
  lastCompletedId = newestId;
  options(el("review-trial"), finished.map((item) => [
    item.trial_id, item.trial_number + " · " + item.action_label,
  ]), selected);
  if ((selected ? run.id + "/" + selected : null) !== reviewKey) loadReview();
  el("trials").replaceChildren();
  for (const item of run?.trials || []) {
    addRow(el("trials"), [item.trial_number, item.set_number, item.action_label,
      statusLabels[item.status], item.status === "ready" ? "—" : item.fall_detected ? "확정" : "확정 없음",
      item.valid_pose_frames + " / " + item.frame_count,
      item.valid == null ? "미검토" : item.valid ? "유효" : "무효"]);
  }
  el("metrics").replaceChildren();
  for (const [key, label] of [["TP", "TP"], ["FN", "FN"], ["FP", "FP"], ["TN", "TN"],
    ["precision", "Precision"], ["recall", "Recall"], ["specificity", "Specificity"], ["f1", "F1"]]) {
    const item = document.createElement("span");
    const value = state.summary[key];
    item.textContent = label + " ";
    const strong = document.createElement("strong");
    strong.textContent = value == null ? "—" : key === key.toUpperCase() ? value : (value * 100).toFixed(1) + "%";
    item.append(strong);
    el("metrics").append(item);
  }
  el("excluded").textContent = "미검토·무효·중단·관절 인식 불가로 지표에서 제외된 종료 시행 " + state.summary.excluded + "건";
  el("groups").replaceChildren();
  for (const group of Object.values(state.summary.by_action)) {
    addRow(el("groups"), [group.action_label, group.trials, group.confirmed, group.not_confirmed, group.reviewed_valid]);
  }
}

el("setup").addEventListener("submit", async (event) => {
  event.preventDefault();
  const actions = [...el("actions").querySelectorAll("input:checked")].map((item) => item.value);
  if (!actions.length) return message("동작을 하나 이상 선택하세요.", true);
  createId ||= [...crypto.getRandomValues(new Uint8Array(16))].map((byte) => byte.toString(16).padStart(2, "0")).join("");
  await command("create", { experiment_id: createId, participant_id: el("participant").value,
    experiment_type: el("kind").value, actions, sets: Number(el("sets").value),
    observation_seconds: Number(el("duration").value) });
});
el("setup").addEventListener("input", () => { createId = null; renderPlanned(); });
el("kind").addEventListener("change", renderActions);
for (const name of ["start", "end", "next", "finish"]) {
  el(name).addEventListener("click", () => {
    const trial = currentTrial();
    if (trial) command(name, { experiment_id: state.current.id, trial_id: trial.trial_id });
  });
}
el("history").addEventListener("change", () => command("select", {
  experiment_id: el("history").value, trial_id: "",
}));
el("review-trial").addEventListener("change", loadReview);
el("review").addEventListener("submit", async (event) => {
  event.preventDefault();
  const booleanValue = (id) => el(id).value === "" ? null : el(id).value === "true";
  await command("review", { experiment_id: state.current.id, trial_id: el("review-trial").value,
    actual_action: el("actual-action").value, actual_fall: booleanValue("actual-fall"),
    valid: booleanValue("valid"), notes: el("notes").value });
});
el("preview").addEventListener("load", () => { el("preview").hidden = false; });
el("preview").addEventListener("error", () => { el("preview").hidden = true; });
renderActions();
async function poll() {
  if (!busy) {
    try {
      const initial = state === null;
      state = await request("state");
      if (initial) message("서버에 연결됐습니다. 실험을 설정해 주세요.");
      render();
    } catch (error) {
      message("서버 연결 확인 · " + error.message, true);
    }
  }
  setTimeout(poll, 500);
}
poll();
