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

  // can_skip is only set (to false) by BlindTestReviewAPI - both other APIs leave it
  // undefined, so the button stays visible everywhere else.
  btnSkipEl.classList.toggle("hidden", state.can_skip === false);

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
  window.pywebview.api
    .retrain()
    .then((state) => {
      busy = false;
      setControlsDisabled(false);
      btnRetrainEl.textContent = "🔄 Retrain";
      render(state);
    })
    .catch((error) => {
      busy = false;
      setControlsDisabled(false);
      btnRetrainEl.textContent = "🔄 Retrain";
      alert(`Retrain failed:\n\n${error}`);
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

window.addEventListener("pywebviewready", () => {
  window.pywebview.api.get_state().then(render);
});
