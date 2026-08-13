/* Draws a Minecraft skin texture as a rotatable player model on a plain 2D canvas.
 *
 * No WebGL and no third-party renderer: the app is loaded from local files with no
 * network access, so anything external would have to be vendored as a binary blob, and
 * WebGL availability inside the bundled QtWebEngine isn't something the app can rely on.
 *
 * A 2D canvas is enough here because the camera is orthographic. A skin is only twelve
 * boxes (six body parts, each with an outer layer), every face is a flat rectangle, and
 * an orthographic projection maps a rectangle to a parallelogram - which is exactly what
 * a canvas affine transform can express. So each face is one transformed drawImage, and
 * the result is geometrically exact rather than an approximation: there's no perspective
 * divide to get wrong, which is the usual reason 2D-canvas texture mapping looks warped.
 *
 * Coordinate system: +X right, +Y up, +Z out of the screen toward the viewer. The model
 * is centred on the origin and spans y -16..16, in skin-texture pixels.
 */

const SkinRenderer = (() => {
  // Corner offsets as multiples of each box's half-extents. p0 is the face's texture
  // top-left, p1 its top-right and p3 its bottom-left; p2 falls out of the other three
  // because an orthographic projection keeps the face a parallelogram.
  //
  // The four side faces are wound so their texture strip runs continuously around the
  // body - right, front, left, back - matching how a skin is unwrapped. Top and bottom
  // both run left-to-right in +X; top runs back-to-front, bottom front-to-back.
  const FACE_CORNERS = {
    right: { p0: [-1, 1, -1], p1: [-1, 1, 1], p3: [-1, -1, -1] },
    front: { p0: [-1, 1, 1], p1: [1, 1, 1], p3: [-1, -1, 1] },
    left: { p0: [1, 1, 1], p1: [1, 1, -1], p3: [1, -1, 1] },
    back: { p0: [1, 1, -1], p1: [-1, 1, -1], p3: [1, -1, -1] },
    top: { p0: [-1, 1, -1], p1: [1, 1, -1], p3: [-1, 1, 1] },
    bottom: { p0: [-1, -1, 1], p1: [1, -1, 1], p3: [-1, -1, -1] },
  };
  const FACES = Object.keys(FACE_CORNERS);

  // Mirroring a limb across X swaps which side is which; front/back/top/bottom keep
  // their own texture but get flipped along u.
  const MIRRORED_FACE = {
    right: "left", left: "right", front: "front", back: "back", top: "top", bottom: "bottom",
  };

  // Grows every face very slightly within its own plane. Canvas antialiases the edge of
  // each drawImage, so abutting faces would otherwise show a hairline seam between them.
  const SEAM_OVERLAP = 1.004;

  // How far each outer layer sits outside its base box, in texture pixels. The hat is
  // conventionally looser than the rest; the gap also keeps the two layers from
  // z-fighting during the depth sort.
  const OVERLAY_INFLATE = { head: 0.5, other: 0.25 };

  /* The six face rectangles of a box whose unwrapped texture starts at (x, y). Every
   * part in a skin follows this one layout, so the whole UV table is this function
   * called with different origins and dimensions. */
  function uvRects(x, y, w, h, d) {
    return {
      right: [x, y + d, d, h],
      front: [x + d, y + d, w, h],
      left: [x + d + w, y + d, d, h],
      back: [x + 2 * d + w, y + d, w, h],
      top: [x + d, y, w, d],
      bottom: [x + d + w, y, w, d],
    };
  }

  /* Reads the pixels a classic model uses but a slim one leaves blank. Two samples sit
   * in each arm, and all four must be fully transparent before a skin is treated as
   * slim - stray pixels are common, and guessing classic for an ambiguous skin only
   * costs one pixel of arm width, while guessing slim clips a real arm. */
  function detectSlim(ctx, scale) {
    const points = [[50, 16], [54, 20], [42, 48], [46, 52]];
    return points.every(([x, y]) => ctx.getImageData(x * scale, y * scale, 1, 1).data[3] === 0);
  }

  /* Inspects a loaded skin image once, so rotating it afterwards costs no pixel reads. */
  function parse(image) {
    const width = image.naturalWidth || image.width;
    const height = image.naturalHeight || image.height;
    // A skin is 64 texture pixels wide by convention; HD skins are exact multiples, so
    // every UV coordinate below scales by the same factor.
    const scale = Math.max(1, Math.round(width / 64));
    // The pre-1.8 format is half as tall: no left limbs and no outer layer but the hat.
    const legacy = height * 2 <= width;

    const probe = document.createElement("canvas");
    probe.width = width;
    probe.height = height;
    const probeCtx = probe.getContext("2d", { willReadFrequently: true });
    probeCtx.drawImage(image, 0, 0);

    const slim = legacy ? false : detectSlim(probeCtx, scale);
    return { image, scale, legacy, slim, parts: buildParts(legacy, slim) };
  }

  function buildParts(legacy, slim) {
    const armWidth = slim ? 3 : 4;
    const parts = [
      { name: "head", center: [0, 12, 0], size: [8, 8, 8], uv: [0, 0], overlayUv: [32, 0] },
      { name: "body", center: [0, 2, 0], size: [8, 12, 4], uv: [16, 16], overlayUv: [16, 32] },
      { name: "rightArm", center: [-4 - armWidth / 2, 2, 0], size: [armWidth, 12, 4], uv: [40, 16], overlayUv: [40, 32] },
      { name: "leftArm", center: [4 + armWidth / 2, 2, 0], size: [armWidth, 12, 4], uv: [32, 48], overlayUv: [48, 48] },
      { name: "rightLeg", center: [-2, -10, 0], size: [4, 12, 4], uv: [0, 16], overlayUv: [0, 32] },
      { name: "leftLeg", center: [2, -10, 0], size: [4, 12, 4], uv: [16, 48], overlayUv: [0, 48] },
    ];

    if (legacy) {
      for (const part of parts) {
        // A 64x32 skin stores only the right limbs; the left ones are the same texture
        // mirrored, which is how the game itself renders them.
        if (part.name === "leftArm") {
          part.uv = [40, 16];
          part.mirror = true;
        } else if (part.name === "leftLeg") {
          part.uv = [0, 16];
          part.mirror = true;
        }
        if (part.name !== "head") {
          part.overlayUv = null;
        }
      }
    }
    return parts;
  }

  function corner(center, half, offsets) {
    return [
      center[0] + half[0] * offsets[0],
      center[1] + half[1] * offsets[1],
      center[2] + half[2] * offsets[2],
    ];
  }

  function addBox(faces, part, uvOrigin, inflate, scale) {
    const [w, h, d] = part.size;
    const half = [w / 2 + inflate, h / 2 + inflate, d / 2 + inflate];
    const rects = uvRects(uvOrigin[0], uvOrigin[1], w, h, d);

    for (const face of FACES) {
      const offsets = FACE_CORNERS[face];
      const rect = rects[part.mirror ? MIRRORED_FACE[face] : face];

      // A mirrored limb keeps these corners exactly as they are and flips u at paint
      // time instead. Swapping the corners here would reverse the face's winding, and
      // the backface test below reads winding to decide what's visible - a mirrored
      // part would then render inside-out, showing its far side through its near one.
      faces.push({
        p0: expand(corner(part.center, half, offsets.p0), part.center),
        p1: expand(corner(part.center, half, offsets.p1), part.center),
        p3: expand(corner(part.center, half, offsets.p3), part.center),
        rect: rect.map((v) => v * scale),
        mirrorU: Boolean(part.mirror),
      });
    }
  }

  function expand(point, center) {
    return [0, 1, 2].map((i) => center[i] + (point[i] - center[i]) * SEAM_OVERLAP);
  }

  function collectFaces(model, includeOverlay) {
    const faces = [];
    for (const part of model.parts) {
      addBox(faces, part, part.uv, 0, model.scale);
      if (includeOverlay && part.overlayUv) {
        const inflate = part.name === "head" ? OVERLAY_INFLATE.head : OVERLAY_INFLATE.other;
        addBox(faces, part, part.overlayUv, inflate, model.scale);
      }
    }
    return faces;
  }

  /* Yaw about Y, then pitch about X. Positive yaw turns the model's right side toward
   * the viewer; positive pitch looks down on it from above. */
  function rotate(point, camera) {
    const [x, y, z] = point;
    const cy = Math.cos(camera.yaw);
    const sy = Math.sin(camera.yaw);
    const rx = x * cy + z * sy;
    const rz = -x * sy + z * cy;

    const cp = Math.cos(camera.pitch);
    const sp = Math.sin(camera.pitch);
    return [rx, y * cp - rz * sp, y * sp + rz * cp];
  }

  function draw(canvas, model, camera, includeOverlay) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!model) {
      return;
    }

    // The model is 32 tall and, once turned, at most ~18 across. Fitting both against a
    // little headroom keeps a rotating model from ever clipping its own canvas.
    const scale = Math.min(w / 20, h / 35);
    const originX = w / 2;
    const originY = h / 2;
    const project = (point) => {
      const [x, y, z] = rotate(point, camera);
      return [originX + x * scale, originY - y * scale, z];
    };

    const faces = collectFaces(model, includeOverlay)
      .map((face) => ({ ...face, p0: project(face.p0), p1: project(face.p1), p3: project(face.p3) }))
      .filter(isFrontFacing);

    // Painter's algorithm over every face at once, base and outer layer together: the
    // outer layer sits slightly proud of its own base face, so depth order alone puts
    // it on top without needing a separate pass - and an outer face on the far side of
    // the model still ends up correctly hidden behind nearer body parts.
    faces.sort((a, b) => depth(a) - depth(b));

    ctx.imageSmoothingEnabled = false;
    for (const face of faces) {
      paint(ctx, model.image, face);
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }

  /* Screen y grows downward, so a face pointing at the viewer winds clockwise and its
   * edge cross-product points away. Everything else is inside the model and skipped. */
  function isFrontFacing(face) {
    const ex = face.p1[0] - face.p0[0];
    const ey = face.p1[1] - face.p0[1];
    const fx = face.p3[0] - face.p0[0];
    const fy = face.p3[1] - face.p0[1];
    return ex * fy - ey * fx > 0;
  }

  function depth(face) {
    return (face.p0[2] + face.p1[2] + face.p3[2]) / 3;
  }

  function paint(ctx, image, face) {
    const [sx, sy, sw, sh] = face.rect;
    if (sw <= 0 || sh <= 0) {
      return;
    }
    // Maps the face's texture rectangle onto its projected parallelogram: the u axis
    // runs p0->p1 and the v axis p0->p3, which together are the affine transform.
    // A mirrored face starts from p1 and runs u backwards instead, which flips the
    // texture without disturbing the corner order the backface test depends on. The v
    // axis is the same either way, since p2-p1 equals p3-p0 on a parallelogram.
    const origin = face.mirrorU ? face.p1 : face.p0;
    const uEnd = face.mirrorU ? face.p0 : face.p1;
    ctx.setTransform(
      (uEnd[0] - origin[0]) / sw,
      (uEnd[1] - origin[1]) / sw,
      (face.p3[0] - face.p0[0]) / sh,
      (face.p3[1] - face.p0[1]) / sh,
      origin[0],
      origin[1],
    );
    ctx.drawImage(image, sx, sy, sw, sh, 0, 0, sw, sh);
  }

  return { parse, draw };
})();
