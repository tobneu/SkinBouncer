const imageEl = document.getElementById("skin-image");
const filenameEl = document.getElementById("filename");
const progressTextEl = document.getElementById("progress-text");
const progressFillEl = document.getElementById("progress-fill");
const doneMessageEl = document.getElementById("done-message");
const metaRowEl = document.getElementById("meta-row");
const metaRecordedEl = document.getElementById("meta-recorded");
const metaReasonEl = document.getElementById("meta-reason");
const metaProbEl = document.getElementById("meta-prob");
const btnGoodEl = document.getElementById("btn-good");
const btnBadEl = document.getElementById("btn-bad");
const btnSkipEl = document.getElementById("btn-skip");
const btnRetrainEl = document.getElementById("btn-retrain");
const trainingProgressEl = document.getElementById("training-progress");
const trainingEpochLabelEl = document.getElementById("training-epoch-label");
const trainingChartEl = document.getElementById("training-chart");
const runComparisonEl = document.getElementById("run-comparison");

let currentState = null;
let busy = false;

function setControlsDisabled(disabled) {
  [btnGoodEl, btnBadEl, btnSkipEl, btnRetrainEl].forEach((btn) => {
    btn.disabled = disabled;
  });
}

function render(state) {
  currentState = state;
  progressTextEl.textContent = state.done
    ? `Done — reviewed ${state.total} / ${state.total}`
    : `Reviewing ${state.index + 1} / ${state.total}`;
  progressFillEl.style.width = `${state.total ? (state.index / state.total) * 100 : 0}%`;

  // can_retrain is only set by ActiveLearningAPI (not the plain labeling tool's
  // LabelingAPI), and unlike recorded_class/reason/predicted_prob it's present in
  // both the done and not-done states, so the button stays visible on the done screen.
  btnRetrainEl.classList.toggle("hidden", !state.can_retrain);

  if (state.done) {
    imageEl.classList.add("hidden");
    doneMessageEl.classList.remove("hidden");
    filenameEl.textContent = "";
    metaRowEl.classList.add("hidden");
    btnGoodEl.classList.remove("btn-current");
    btnBadEl.classList.remove("btn-current");
  } else {
    imageEl.classList.remove("hidden");
    doneMessageEl.classList.add("hidden");
    imageEl.src = state.image_data_uri;
    filenameEl.textContent = state.filename;

    // Only the active-learning queue's ActiveLearningAPI includes these fields -
    // the plain labeling tool's LabelingAPI doesn't, so the row stays hidden there.
    if ("recorded_class" in state) {
      metaRowEl.classList.remove("hidden");
      metaRecordedEl.textContent = `Currently: ${state.recorded_class}`;
      metaReasonEl.textContent = state.reason;
      metaProbEl.textContent = `p=${state.predicted_prob.toFixed(3)}`;
      btnGoodEl.classList.toggle("btn-current", state.recorded_class === "good");
      btnBadEl.classList.toggle("btn-current", state.recorded_class === "bad");
    } else {
      metaRowEl.classList.add("hidden");
    }
  }

  // Only ActiveLearningAPI sets this key at all (null until a retrain has completed
  // at least once this session).
  if ("run_comparison" in state) {
    renderRunComparison(state.run_comparison);
  }
}

function renderRunComparison(comparison) {
  if (!comparison) {
    runComparisonEl.classList.add("hidden");
    runComparisonEl.innerHTML = "";
    return;
  }
  const lines = [`New val AUC: ${comparison.current.val_auc.toFixed(3)}`];
  if (comparison.previous.length === 0) {
    lines.push("(first recorded run for this project)");
  } else {
    comparison.previous.forEach((run, i) => {
      const pct = run.pct_change == null
        ? "n/a"
        : `${run.pct_change >= 0 ? "+" : ""}${run.pct_change.toFixed(1)}%`;
      lines.push(`vs ${i + 1} round${i === 0 ? "" : "s"} ago (${run.val_auc.toFixed(3)}): ${pct}`);
    });
  }
  runComparisonEl.innerHTML = lines.map((line) => `<div>${line}</div>`).join("");
  runComparisonEl.classList.remove("hidden");
}

function renderTrainingProgress(progress) {
  trainingEpochLabelEl.textContent = `Epoch ${progress.epoch || 0} / ${progress.epochs_total || "?"}`;
  drawTrainingChart(progress.history || {});
}

function drawTrainingChart(history) {
  const ctx = trainingChartEl.getContext("2d");
  const w = trainingChartEl.width;
  const h = trainingChartEl.height;
  ctx.clearRect(0, 0, w, h);

  const series = [
    { data: history.auc || [], color: "#f59e0b" },
    { data: history.val_auc || [], color: "#14b8a6" },
  ];
  const allValues = series.flatMap((s) => s.data);
  if (allValues.length === 0) {
    return;
  }
  const pad = 8;
  const minV = Math.min(...allValues, 0.5);
  const maxV = Math.max(...allValues, 1.0);
  const range = maxV - minV || 1;

  series.forEach(({ data, color }) => {
    if (data.length === 0) {
      return;
    }
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    data.forEach((value, i) => {
      const x = pad + (i / Math.max(data.length - 1, 1)) * (w - pad * 2);
      const y = h - pad - ((value - minV) / range) * (h - pad * 2);
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
  });
}

function decide(action) {
  // Guard against keyboard auto-repeat / double-clicks firing a second
  // decide() before the first one's response has updated currentState.done.
  if (busy || (currentState && currentState.done)) {
    return;
  }
  busy = true;
  setControlsDisabled(true);
  window.pywebview.api
    .decide(action)
    .then((state) => {
      busy = false;
      setControlsDisabled(false);
      render(state);
    })
    .catch((error) => {
      // e.g. a relabel refused by the filename-collision guard in relabel_image().
      // The session's index didn't advance server-side, so the currently shown item
      // is still valid - just unstick input instead of leaving the app frozen.
      busy = false;
      setControlsDisabled(false);
      alert(`Couldn't apply that decision:\n\n${error}\n\nYou can Skip this image instead.`);
    });
}

function retrain() {
  if (busy) {
    return;
  }
  busy = true;
  setControlsDisabled(true);
  btnRetrainEl.textContent = "Training…";
  runComparisonEl.classList.add("hidden");
  trainingProgressEl.classList.remove("hidden");
  window.pywebview.api
    .retrain()
    .then(pollTrainingProgress)
    .catch((error) => {
      onRetrainSettled();
      alert(`Retrain failed:\n\n${error}`);
    });
}

function pollTrainingProgress() {
  window.pywebview.api.get_training_progress().then((progress) => {
    renderTrainingProgress(progress);
    if (progress.status === "running") {
      setTimeout(pollTrainingProgress, 750);
    } else if (progress.status === "error") {
      onRetrainSettled();
      alert(`Retrain failed:\n\n${progress.error}`);
    } else {
      window.pywebview.api.get_state().then((state) => {
        onRetrainSettled();
        render(state);
      });
    }
  });
}

function onRetrainSettled() {
  busy = false;
  setControlsDisabled(false);
  btnRetrainEl.textContent = "🔄 Retrain";
  trainingProgressEl.classList.add("hidden");
}

const KEY_TO_ACTION = {
  g: "good",
  ArrowRight: "good",
  b: "bad",
  ArrowLeft: "bad",
  " ": "skip",
  s: "skip",
};

document.addEventListener("keydown", (event) => {
  const action = KEY_TO_ACTION[event.key];
  if (action) {
    event.preventDefault();
    decide(action);
  }
});

window.addEventListener("pywebviewready", () => {
  window.pywebview.api.get_state().then(render);
});
