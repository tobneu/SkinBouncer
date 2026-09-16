const themeToggleEl = document.getElementById("theme-toggle");
const imageEl = document.getElementById("skin-image");
const skinViewsEl = document.getElementById("skin-views");
const skinViewsHintEl = document.getElementById("skin-views-hint");
const skinModelEl = document.getElementById("skin-model");
const skinModelLayersEl = document.getElementById("skin-model-layers");
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
// Captured before any reparenting happens (see beginColdStartTraining), so
// showScreen() can always move trainingProgressEl back home when returning to
// "review" - otherwise a later Retrain would render into a now-hidden subtree.
const trainingProgressHomeEl = trainingProgressEl.parentElement;
const trainingEpochLabelEl = document.getElementById("training-epoch-label");
const trainingChartEl = document.getElementById("training-chart");
const runComparisonEl = document.getElementById("run-comparison");
const confusionMatrixEl = document.getElementById("confusion-matrix");
const confusionMatrixHeadlineEl = document.getElementById("confusion-matrix-headline");
const cmTnEl = document.getElementById("cm-tn");
const cmFpEl = document.getElementById("cm-fp");
const cmFnEl = document.getElementById("cm-fn");
const cmTpEl = document.getElementById("cm-tp");
const testRatesEl = document.getElementById("test-rates");
const curationNoteEl = document.getElementById("curation-note");
const btnExportEl = document.getElementById("btn-export");
const exportResultEl = document.getElementById("export-result");

const backToOverviewEl = document.getElementById("back-to-overview");
const screenOverviewEl = document.getElementById("screen-overview");
const screenWizardEl = document.getElementById("screen-wizard");
const screenReviewEl = document.getElementById("screen-review");
const projectListEl = document.getElementById("project-list");
const projectListEmptyEl = document.getElementById("project-list-empty");
const wizardStepNameEl = document.getElementById("wizard-step-name");
const wizardStepGoodEl = document.getElementById("wizard-step-good");
const wizardStepBadEl = document.getElementById("wizard-step-bad");
const wizardStepTrainingEl = document.getElementById("wizard-step-training");
const wizardNameInputEl = document.getElementById("wizard-name-input");
const wizardGoodPathEl = document.getElementById("wizard-good-path");
const wizardBadPathEl = document.getElementById("wizard-bad-path");
const wizardGoodNextEl = document.getElementById("wizard-good-next");
const wizardCreateBtnEl = document.getElementById("wizard-create-btn");
const wizardErrorEl = document.getElementById("wizard-error");
const wizardTrainingSlotEl = document.getElementById("wizard-training-slot");

let currentState = null;
let busy = false;

// A gentle three-quarter view: enough turn to read the side of a skin, enough tilt to
// show the top of the head, without starting so far round that the face is hard to find.
const DEFAULT_CAMERA = { yaw: Math.PI / 5, pitch: Math.PI / 9 };
// Stops a drag from tipping past straight down/up, where the model reads as a puzzle.
const PITCH_LIMIT = 1.2;
const DRAG_RADIANS_PER_PIXEL = 0.011;

let camera = { ...DEFAULT_CAMERA };
let skinModel = null;
// Advanced on every skin change. An in-flight decode compares against it before
// painting, so holding down a key can't let an earlier skin land after a later one.
let skinToken = 0;

function showSkin(dataUri) {
  const token = ++skinToken;
  const image = new Image();

  const settle = (model) => {
    if (token !== skinToken) {
      return;
    }
    skinModel = model;
    // The flat texture is swapped only once the same bytes have decoded for the 3D
    // views, so all three panels change together instead of the texture jumping ahead.
    imageEl.src = dataUri;
    camera = { ...DEFAULT_CAMERA };
    drawSkinViews();
  };

  image.onload = () => settle(SkinRenderer.parse(image));
  // A texture the renderer can't read still gets shown flat - losing the 3D views is
  // better than losing the image under review entirely.
  image.onerror = () => settle(null);
  image.src = dataUri;
}

function drawSkinViews() {
  fitCanvas(skinModelEl);
  fitCanvas(skinModelLayersEl);
  SkinRenderer.draw(skinModelEl, skinModel, camera, false);
  SkinRenderer.draw(skinModelLayersEl, skinModel, camera, true);
}

/* Matches the canvas backing store to the size CSS actually gave it. The panels are
   flexible, so a fixed width/height in the markup would render at the wrong aspect
   ratio and get stretched to fit. */
function fitCanvas(canvas) {
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.round(canvas.clientWidth * ratio);
  const height = Math.round(canvas.clientHeight * ratio);
  if (width > 0 && height > 0 && (canvas.width !== width || canvas.height !== height)) {
    canvas.width = width;
    canvas.height = height;
  }
}

function attachRotation(canvas) {
  let activePointer = null;
  let lastX = 0;
  let lastY = 0;

  canvas.addEventListener("pointerdown", (event) => {
    activePointer = event.pointerId;
    lastX = event.clientX;
    lastY = event.clientY;
    // Without this the drag starts a text selection over the card instead.
    event.preventDefault();
    capture(canvas, "setPointerCapture", activePointer);
  });

  canvas.addEventListener("pointermove", (event) => {
    if (event.pointerId !== activePointer) {
      return;
    }
    camera.yaw += (event.clientX - lastX) * DRAG_RADIANS_PER_PIXEL;
    camera.pitch = clamp(
      camera.pitch + (event.clientY - lastY) * DRAG_RADIANS_PER_PIXEL,
      -PITCH_LIMIT,
      PITCH_LIMIT,
    );
    lastX = event.clientX;
    lastY = event.clientY;
    // Both canvases are redrawn from the one shared camera, so the with/without-layer
    // pair can never drift into two different angles and stop being comparable.
    drawSkinViews();
  });

  const release = (event) => {
    if (event.pointerId !== activePointer) {
      return;
    }
    const released = activePointer;
    // Cleared before the capture call, not after: if releasing were to throw, an
    // uncleared activePointer would leave the model rotating with no button held.
    activePointer = null;
    capture(canvas, "releasePointerCapture", released);
  };
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);
}

/* Pointer capture only keeps a drag alive when it wanders off the canvas - useful, but
   never essential. It throws if the browser doesn't consider the pointer active, so a
   failure here must not take the surrounding handler down with it. */
function capture(canvas, method, pointerId) {
  try {
    canvas[method](pointerId);
  } catch (error) {
    /* drag still works, it just stops tracking outside the canvas */
  }
}

function clamp(value, low, high) {
  return Math.min(Math.max(value, low), high);
}

function setControlsDisabled(disabled) {
  [btnGoodEl, btnBadEl, btnSkipEl, btnRetrainEl, btnExportEl].forEach((btn) => {
    btn.disabled = disabled;
  });
}

// Screens that don't come from a get_state() response (overview, wizard) are pure
// client-side navigation - see startWizard()/showScreen() below. "review" is the
// fallback so the three plain entrypoints (LabelingAPI/ActiveLearningAPI/
// BlindTestReviewAPI), whose state never carries a "screen" key at all, keep working
// exactly as before this was added.
function render(state) {
  const screen = state.screen || "review";
  showScreen(screen);
  if (screen === "overview") {
    renderOverview(state.projects);
  } else {
    renderReview(state);
  }
}

function showScreen(name) {
  if (name === "review" && trainingProgressEl.parentElement !== trainingProgressHomeEl) {
    trainingProgressHomeEl.appendChild(trainingProgressEl);
    // Landing on review fresh (cold-start just finished) shouldn't show a stale
    // training panel - only an in-progress Retrain should ever reveal it again.
    trainingProgressEl.classList.add("hidden");
  }
  screenOverviewEl.classList.toggle("hidden", name !== "overview");
  screenWizardEl.classList.toggle("hidden", name !== "wizard");
  screenReviewEl.classList.toggle("hidden", name !== "review");
  backToOverviewEl.classList.toggle("hidden", name === "overview");
}

function renderReview(state) {
  currentState = state;
  progressTextEl.textContent = state.done
    ? `Done — reviewed ${state.total} / ${state.total}`
    : `Reviewing ${state.index + 1} / ${state.total}`;
  progressFillEl.style.width = `${state.total ? (state.index / state.total) * 100 : 0}%`;

  // can_retrain is only set by ActiveLearningAPI (not the plain labeling tool's
  // LabelingAPI), and unlike recorded_class/reason/predicted_prob it's present in
  // both the done and not-done states, so the button stays visible on the done screen.
  btnRetrainEl.classList.toggle("hidden", !state.can_retrain);
  btnExportEl.classList.toggle("hidden", !state.can_export);

  // can_skip is only set (to false) by BlindTestReviewAPI - both other APIs leave it
  // undefined, so the button stays visible everywhere else.
  btnSkipEl.classList.toggle("hidden", state.can_skip === false);

  if (state.done) {
    skinViewsEl.classList.add("hidden");
    skinViewsHintEl.classList.add("hidden");
    doneMessageEl.classList.remove("hidden");
    // Abandons any decode still in flight, so it can't paint over the done screen.
    skinToken += 1;
    skinModel = null;
    filenameEl.textContent = "";
    metaRowEl.classList.add("hidden");
    btnGoodEl.classList.remove("btn-current");
    btnBadEl.classList.remove("btn-current");
  } else {
    skinViewsEl.classList.remove("hidden");
    skinViewsHintEl.classList.remove("hidden");
    doneMessageEl.classList.add("hidden");
    showSkin(state.image_data_uri);
    filenameEl.textContent = state.filename;

    // recorded_class is set by both ActiveLearningAPI and BlindTestReviewAPI (not the
    // plain labeling tool's LabelingAPI, so the row stays hidden there). reason and
    // predicted_prob are reveal model output, so only ActiveLearningAPI sets them -
    // BlindTestReviewAPI deliberately never does, per #10's "no model info" requirement.
    if ("recorded_class" in state) {
      metaRowEl.classList.remove("hidden");
      metaRecordedEl.textContent = `Currently: ${state.recorded_class}`;
      metaReasonEl.classList.toggle("hidden", !("reason" in state));
      metaReasonEl.textContent = state.reason ?? "";
      metaProbEl.classList.toggle("hidden", !("predicted_prob" in state));
      metaProbEl.textContent = "predicted_prob" in state ? `p=${state.predicted_prob.toFixed(3)}` : "";
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

  // Only ActiveLearningAPI sets this key - populated from the very first launch
  // (computed once a checkpoint exists), unlike run_comparison.
  if ("confusion_matrix" in state) {
    renderConfusionMatrix(state.confusion_matrix);
  }

  if ("test_curation" in state) {
    renderCurationNote(state.test_curation);
  }
}

// Shared with the run-comparison bar chart, so both visuals stretch/compress AUC
// values onto the same 0.5-1.0-or-wider scale rather than each picking their own
// range - a jump that looks big in one chart looks equally big in the other.
function aucRange(values) {
  const minV = Math.min(...values, 0.5);
  const maxV = Math.max(...values, 1.0);
  return { minV, maxV, range: maxV - minV || 1 };
}

function renderRunComparison(comparison) {
  if (!comparison) {
    runComparisonEl.classList.add("hidden");
    runComparisonEl.innerHTML = "";
    return;
  }

  const rounds = [
    { label: "Now", val_auc: comparison.current.val_auc, current: true, delta: null },
    ...comparison.previous.map((run, i) => ({
      label: `${i + 1} round${i === 0 ? "" : "s"} ago`,
      val_auc: run.val_auc,
      current: false,
      delta: run.pct_change,
    })),
  ];
  const { minV, range } = aucRange(rounds.map((r) => r.val_auc));

  const bars = rounds.map((round) => {
    const heightPct = ((round.val_auc - minV) / range) * 100;
    const deltaHtml = round.delta == null
      ? ""
      : `<div class="run-comparison-delta ${round.delta >= 0 ? "up" : "down"}">${round.delta >= 0 ? "+" : ""}${round.delta.toFixed(1)}%</div>`;
    return `
      <div class="run-comparison-bar">
        <div class="run-comparison-value">${round.val_auc.toFixed(3)}</div>
        <div class="run-comparison-fill ${round.current ? "current" : ""}" style="height: ${heightPct}%"></div>
        <div class="run-comparison-label">${round.label}</div>
        ${deltaHtml}
      </div>`;
  }).join("");

  const note = comparison.previous.length === 0
    ? `<div class="run-comparison-note">First recorded run for this project - nothing to compare against yet.</div>`
    : "";

  runComparisonEl.innerHTML = `
    <div class="run-comparison-headline">Validation accuracy: ${comparison.current.val_auc.toFixed(3)}</div>
    <div class="run-comparison-track">${bars}</div>
    ${note}`;
  runComparisonEl.classList.remove("hidden");
}

function renderConfusionMatrix(cm) {
  if (!cm) {
    confusionMatrixEl.classList.add("hidden");
    return;
  }
  const fill = (el, count) => {
    const pct = Math.round((count / cm.n) * 100);
    el.innerHTML = `<div class="count">${count}</div><div class="pct">${pct}%</div>`;
  };
  fill(cmTnEl, cm.tn);
  fill(cmFpEl, cm.fp);
  fill(cmFnEl, cm.fn);
  fill(cmTpEl, cm.tp);

  const correct = cm.tp + cm.tn;
  const correctPct = Math.round((correct / cm.n) * 100);
  confusionMatrixHeadlineEl.textContent =
    `Correctly classified ${correctPct}% of test images (${correct} of ${cm.n})`;

  // Recall and precision, phrased as what they mean for the operator's actual decision
  // rather than by name. A rate is null when its denominator is empty (e.g. no bad
  // images in the test split at all) - "n/a" is the honest answer there, not 0%.
  testRatesEl.innerHTML = [
    { label: "Catches bad skins", rate: cm.recall, hit: cm.tp, of: cm.tp + cm.fn },
    { label: "When it flags, it's right", rate: cm.precision, hit: cm.tp, of: cm.tp + cm.fp },
  ].map(({ label, rate, hit, of }) => {
    const value = rate === null ? "n/a" : `${Math.round(rate * 100)}%`;
    const detail = rate === null ? "" : `<span class="test-rate-detail">${hit} of ${of}</span>`;
    return `<div class="test-rate"><span>${label}</span><span class="test-rate-value">${value}${detail}</span></div>`;
  }).join("");

  confusionMatrixEl.classList.remove("hidden");
}

function renderCurationNote(curation) {
  // Export is never blocked on curation - this only tells the operator how much the
  // numbers above are worth, since an unreviewed test split is still scored against
  // whatever labels the original bulk sort happened to produce.
  if (!curation || curation.complete || curation.total === 0) {
    curationNoteEl.classList.add("hidden");
    return;
  }
  const left = curation.total - curation.reviewed;
  curationNoteEl.textContent =
    `⚠ ${left} of ${curation.total} test images haven't been double-checked yet. ` +
    `Run the blind test review to confirm their labels - until then the numbers above ` +
    `are only as accurate as the original sorting was.`;
  curationNoteEl.classList.remove("hidden");
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

  const style = getComputedStyle(document.documentElement);
  const series = [
    { data: history.auc || [], color: style.getPropertyValue("--series-train").trim() },
    { data: history.val_auc || [], color: style.getPropertyValue("--series-val").trim() },
  ];
  const allValues = series.flatMap((s) => s.data);
  if (allValues.length === 0) {
    return;
  }

  const margin = { top: 8, right: 10, bottom: 20, left: 30 };
  const plotW = w - margin.left - margin.right;
  const plotH = h - margin.top - margin.bottom;
  const { minV, maxV, range } = aucRange(allValues);
  const xOf = (i, len) => margin.left + (i / Math.max(len - 1, 1)) * plotW;
  const yOf = (value) => margin.top + plotH - ((value - minV) / range) * plotH;

  // Y-axis: gridlines + tick labels at min/mid/max, so the plotted lines' vertical
  // position is readable without knowing what AUC is - just "higher is better".
  ctx.strokeStyle = style.getPropertyValue("--panel-border").trim();
  ctx.fillStyle = style.getPropertyValue("--muted").trim();
  ctx.font = "10px sans-serif";
  ctx.textBaseline = "middle";
  [minV, (minV + maxV) / 2, maxV].forEach((tick) => {
    const y = yOf(tick);
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(w - margin.right, y);
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.textAlign = "right";
    ctx.fillText(tick.toFixed(2), margin.left - 6, y);
  });

  // X-axis: epoch numbers, thinned to roughly 5 ticks regardless of how many
  // epochs training ran for.
  const epochCount = Math.max(...series.map((s) => s.data.length));
  if (epochCount > 0) {
    const step = [1, 2, 5, 10, 20, 50, 100].find((s) => epochCount / s <= 5) || 100;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let epoch = 1; epoch <= epochCount; epoch += step) {
      const x = xOf(epoch - 1, epochCount);
      ctx.fillText(String(epoch), x, h - margin.bottom + 4);
    }
  }

  series.forEach(({ data, color }) => {
    if (data.length === 0) {
      return;
    }
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    data.forEach((value, i) => {
      const x = xOf(i, data.length);
      const y = yOf(value);
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
  // Stale the moment training starts - it names a checkpoint that's being replaced.
  exportResultEl.classList.add("hidden");
  // Hidden for the duration so it never shows numbers from a checkpoint that's
  // mid-replacement - render(state) brings it back once the new one is scored.
  confusionMatrixEl.classList.add("hidden");
  trainingProgressEl.classList.remove("hidden");
  backToOverviewEl.classList.add("hidden");
  window.pywebview.api
    .retrain()
    .then(() => pollTrainingProgress((progress) => {
      if (progress.status === "error") {
        onRetrainSettled();
        alert(`Retrain failed:\n\n${progress.error}`);
      } else {
        window.pywebview.api.get_state().then((state) => {
          onRetrainSettled();
          render(state);
        });
      }
    }))
    .catch((error) => {
      onRetrainSettled();
      alert(`Retrain failed:\n\n${error}`);
    });
}

// Shared by the review screen's Retrain button and the overview/wizard's cold-start
// training (see beginColdStartTraining below) - both just need the epoch/AUC chart
// kept live until training leaves the "running" state, then a caller-specific
// reaction to how it settled.
function pollTrainingProgress(onSettled) {
  window.pywebview.api.get_training_progress().then((progress) => {
    renderTrainingProgress(progress);
    if (progress.status === "running") {
      setTimeout(() => pollTrainingProgress(onSettled), 750);
    } else {
      onSettled(progress);
    }
  });
}

function onRetrainSettled() {
  busy = false;
  setControlsDisabled(false);
  btnRetrainEl.textContent = "🔄 Retrain";
  trainingProgressEl.classList.add("hidden");
  backToOverviewEl.classList.remove("hidden");
}

function exportDetector() {
  if (busy) {
    return;
  }
  busy = true;
  setControlsDisabled(true);
  btnExportEl.textContent = "Exporting…";
  exportResultEl.classList.add("hidden");
  window.pywebview.api
    .export_detector()
    .then((result) => {
      exportResultEl.classList.remove("failed");
      exportResultEl.textContent =
        `✓ Exported "${result.category}" (threshold ${result.threshold.toFixed(3)}) to ${result.dest_dir}. ` +
        `Rebuild the API image to ship it.`;
      exportResultEl.classList.remove("hidden");
    })
    .catch((error) => {
      // Shown inline rather than as an alert() - the failure is worth reading next to
      // the metrics it belongs to, and an alert would have to be dismissed first.
      exportResultEl.classList.add("failed");
      exportResultEl.textContent = `✕ Export failed: ${error}`;
      exportResultEl.classList.remove("hidden");
    })
    .finally(() => {
      busy = false;
      setControlsDisabled(false);
      btnExportEl.textContent = "📦 Export detector";
    });
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
  if (!action) {
    return;
  }
  if (action === "skip" && currentState && currentState.can_skip === false) {
    return;
  }
  event.preventDefault();
  decide(action);
});

attachRotation(skinModelEl);
attachRotation(skinModelLayersEl);

// The panels are flexible, so a resized window changes how many backing-store pixels
// each canvas needs - without this they'd stay at the old size and look soft.
window.addEventListener("resize", drawSkinViews);

// --------------------------------------------------------------------------
// Project overview + new-project wizard (scripts/run_skinbouncer.py only - the three
// plain entrypoints never navigate here, see render()'s "screen" fallback above).
// --------------------------------------------------------------------------

function renderOverview(projects) {
  projectListEl.innerHTML = "";
  projectListEmptyEl.classList.toggle("hidden", projects.length > 0);
  for (const project of projects) {
    const card = document.createElement("div");
    card.className = "project-card";
    card.onclick = () => openProject(project.project_dir);

    const name = document.createElement("span");
    name.className = "project-card-name";
    name.textContent = project.display_name;

    const status = document.createElement("span");
    status.className = "project-card-status" + (project.trained ? "" : " untrained");
    status.textContent = project.trained ? "Ready" : "Needs training";

    card.append(name, status);
    projectListEl.appendChild(card);
  }
}

let wizardState = { name: "", goodDir: null, badDir: null };

function wizardShowStep(step) {
  wizardStepNameEl.classList.toggle("hidden", step !== "name");
  wizardStepGoodEl.classList.toggle("hidden", step !== "good");
  wizardStepBadEl.classList.toggle("hidden", step !== "bad");
  wizardStepTrainingEl.classList.toggle("hidden", step !== "training");
}

function startWizard() {
  wizardState = { name: "", goodDir: null, badDir: null };
  wizardNameInputEl.value = "";
  wizardGoodPathEl.textContent = "";
  wizardBadPathEl.textContent = "";
  wizardGoodNextEl.disabled = true;
  wizardCreateBtnEl.disabled = true;
  wizardErrorEl.classList.add("hidden");
  wizardShowStep("name");
  showScreen("wizard");
}

function wizardStep(step) {
  if (step === "good" && !wizardNameInputEl.value.trim()) {
    alert("Give the project a name first.");
    return;
  }
  wizardState.name = wizardNameInputEl.value.trim();
  wizardShowStep(step);
}

function wizardPickFolder(which) {
  window.pywebview.api.pick_folder().then((result) => {
    if (!result.path) {
      return;
    }
    if (which === "good") {
      wizardState.goodDir = result.path;
      wizardGoodPathEl.textContent = result.path;
      wizardGoodNextEl.disabled = false;
    } else {
      wizardState.badDir = result.path;
      wizardBadPathEl.textContent = result.path;
      wizardCreateBtnEl.disabled = false;
    }
  });
}

// Shared by wizardCreate() and openProject()'s auto-train-on-open path: both just
// started cold-start training server-side and need the same "show the training step,
// reuse the review screen's live chart, wait for it to settle" sequence - only what
// happens on error differs (see the two call sites' onError callbacks).
function beginColdStartTraining(onError) {
  showScreen("wizard");
  wizardShowStep("training");
  backToOverviewEl.classList.add("hidden");
  // Reparented rather than duplicated: only one of {wizard, review} is ever visible
  // at a time, so the review screen's own training-progress chart/canvas can just be
  // moved here for the duration instead of building (and keeping in sync) a second copy.
  wizardTrainingSlotEl.appendChild(trainingProgressEl);
  trainingProgressEl.classList.remove("hidden");
  pollTrainingProgress((progress) => {
    if (progress.status === "error") {
      onError(progress.error);
    } else {
      // The backend has already opened the freshly-trained project by this point
      // (SkinBouncerAPI._maybe_finish_training), so this naturally lands on "review".
      window.pywebview.api.get_state().then(render);
    }
  });
}

function wizardCreate() {
  wizardErrorEl.classList.add("hidden");
  window.pywebview.api
    .create_project(wizardState.name, wizardState.goodDir, wizardState.badDir)
    .then((result) => {
      if (result.status === "error") {
        wizardErrorEl.textContent = result.message;
        wizardErrorEl.classList.remove("hidden");
        return;
      }
      beginColdStartTraining((message) => {
        // Back to the step with the Create button (not the overview) - name/good/bad
        // are all still filled in, so retrying costs nothing.
        backToOverviewEl.classList.remove("hidden");
        wizardShowStep("bad");
        wizardErrorEl.textContent = `Training failed:\n\n${message}`;
        wizardErrorEl.classList.remove("hidden");
      });
    });
}

function openProject(projectDir) {
  window.pywebview.api.open_project(projectDir).then((result) => {
    if (result.status === "error") {
      alert(`Couldn't open project:\n\n${result.message}`);
      return;
    }
    if (result.status === "started") {
      // Existing project with no checkpoint yet - same cold-start screen as the
      // wizard, just entered from the overview instead of from wizardCreate().
      beginColdStartTraining((message) => {
        alert(`Training failed:\n\n${message}`);
        backToOverview();
      });
    } else {
      window.pywebview.api.get_state().then(render);
    }
  });
}

function backToOverview() {
  window.pywebview.api.close_project().then(render);
}

// The theme actually in effect right now: the manual override if one is set via
// data-theme, otherwise whatever the OS reports through prefers-color-scheme.
function effectiveTheme() {
  return document.documentElement.getAttribute("data-theme")
    || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
}

function applyTheme(theme) {
  if (theme) {
    document.documentElement.setAttribute("data-theme", theme);
  } else {
    document.documentElement.removeAttribute("data-theme");
  }
  // Icon shows what clicking would switch *to*, not the current theme.
  themeToggleEl.textContent = effectiveTheme() === "dark" ? "☀️" : "🌙";
}

function toggleTheme() {
  const next = effectiveTheme() === "dark" ? "light" : "dark";
  applyTheme(next);
  window.pywebview.api.set_theme(next);
}

window.addEventListener("pywebviewready", () => {
  window.pywebview.api.get_settings().then((settings) => applyTheme(settings.theme));
  window.pywebview.api.get_state().then(render);
});
