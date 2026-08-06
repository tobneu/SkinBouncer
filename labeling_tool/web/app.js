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

let currentState = null;
let busy = false;

function render(state) {
  currentState = state;
  progressTextEl.textContent = state.done
    ? `Done — reviewed ${state.total} / ${state.total}`
    : `Reviewing ${state.index + 1} / ${state.total}`;
  progressFillEl.style.width = `${state.total ? (state.index / state.total) * 100 : 0}%`;

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
}

function decide(action) {
  // Guard against keyboard auto-repeat / double-clicks firing a second
  // decide() before the first one's response has updated currentState.done.
  if (busy || (currentState && currentState.done)) {
    return;
  }
  busy = true;
  window.pywebview.api.decide(action).then((state) => {
    busy = false;
    render(state);
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
  if (action) {
    event.preventDefault();
    decide(action);
  }
});

window.addEventListener("pywebviewready", () => {
  window.pywebview.api.get_state().then(render);
});
