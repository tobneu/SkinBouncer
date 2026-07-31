const imageEl = document.getElementById("skin-image");
const filenameEl = document.getElementById("filename");
const progressTextEl = document.getElementById("progress-text");
const progressFillEl = document.getElementById("progress-fill");
const doneMessageEl = document.getElementById("done-message");

function render(state) {
  progressTextEl.textContent = state.done
    ? `Done — reviewed ${state.total} / ${state.total}`
    : `Reviewing ${state.index + 1} / ${state.total}`;
  progressFillEl.style.width = `${state.total ? (state.index / state.total) * 100 : 0}%`;

  if (state.done) {
    imageEl.classList.add("hidden");
    doneMessageEl.classList.remove("hidden");
    filenameEl.textContent = "";
  } else {
    imageEl.classList.remove("hidden");
    doneMessageEl.classList.add("hidden");
    imageEl.src = state.image_data_uri;
    filenameEl.textContent = state.filename;
  }
}

function decide(action) {
  window.pywebview.api.decide(action).then(render);
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
