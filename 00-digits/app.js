"use strict";
(() => {
  // net.ts
  var Net = class {
    // b[j]    = bias of neuron j (a fixed push added to its score)
    constructor(nIn, nOut) {
      this.nIn = nIn;
      this.nOut = nOut;
      this.W = [];
      for (let i = 0; i < nIn; i++) {
        this.W[i] = [];
        for (let j = 0; j < nOut; j++) {
          this.W[i][j] = 0;
        }
      }
      this.b = [];
      for (let j = 0; j < nOut; j++) {
        this.b[j] = 0;
      }
    }
    // SCORE: from image to percentages.
    forward(x) {
      const z = [];
      for (let j = 0; j < this.nOut; j++) {
        let sum = this.b[j];
        for (let i = 0; i < this.nIn; i++) {
          sum += x[i] * this.W[i][j];
        }
        z[j] = sum;
      }
      return softmax(z);
    }
    // TRAIN with a single example. Returns the loss (only used for display).
    train(x, label, lr) {
      const p = this.forward(x);
      const loss = -Math.log(p[label] + 1e-12);
      for (let j = 0; j < this.nOut; j++) {
        let y = 0;
        if (j === label) y = 1;
        const g = p[j] - y;
        for (let i = 0; i < this.nIn; i++) {
          this.W[i][j] -= lr * g * x[i];
        }
        this.b[j] -= lr * g;
      }
      return loss;
    }
  };
  function softmax(z) {
    let max = z[0];
    for (let j = 1; j < z.length; j++) {
      if (z[j] > max) max = z[j];
    }
    const e = [];
    for (let j = 0; j < z.length; j++) {
      e[j] = Math.exp(z[j] - max);
    }
    let total = 0;
    for (let j = 0; j < z.length; j++) {
      total += e[j];
    }
    const p = [];
    for (let j = 0; j < z.length; j++) {
      p[j] = e[j] / total;
    }
    return p;
  }
  function argmax(v) {
    let best = 0;
    for (let i = 1; i < v.length; i++) {
      if (v[i] > v[best]) best = i;
    }
    return best;
  }

  // net2.ts
  var Net2 = class {
    // b2[j]    = b of output j
    constructor(nIn, nHid, nOut) {
      this.nIn = nIn;
      this.nHid = nHid;
      this.nOut = nOut;
      this.W1 = [];
      for (let i = 0; i < nIn; i++) {
        this.W1[i] = [];
        for (let h = 0; h < nHid; h++) {
          this.W1[i][h] = random(0.1);
        }
      }
      this.b1 = [];
      for (let h = 0; h < nHid; h++) {
        this.b1[h] = 0;
      }
      this.W2 = [];
      for (let h = 0; h < nHid; h++) {
        this.W2[h] = [];
        for (let j = 0; j < nOut; j++) {
          this.W2[h][j] = random(0.3);
        }
      }
      this.b2 = [];
      for (let j = 0; j < nOut; j++) {
        this.b2[j] = 0;
      }
    }
    // SCORE: from image to percentages. Also returns what each hidden neuron
    // gave, because training needs it.
    forward(x) {
      const hid = [];
      for (let h = 0; h < this.nHid; h++) {
        let sum = this.b1[h];
        for (let i = 0; i < this.nIn; i++) {
          sum += x[i] * this.W1[i][h];
        }
        if (sum < 0) sum = 0;
        hid[h] = sum;
      }
      const z = [];
      for (let j = 0; j < this.nOut; j++) {
        let sum = this.b2[j];
        for (let h = 0; h < this.nHid; h++) {
          sum += hid[h] * this.W2[h][j];
        }
        z[j] = sum;
      }
      return { hid, p: softmax(z) };
    }
    // TRAIN with a single example. Returns the loss (only used for display).
    train(x, label, lr) {
      const { hid, p } = this.forward(x);
      const errOut = [];
      for (let j = 0; j < this.nOut; j++) {
        let y = 0;
        if (j === label) y = 1;
        errOut[j] = p[j] - y;
      }
      const errHid = [];
      for (let h = 0; h < this.nHid; h++) {
        let blame = 0;
        for (let j = 0; j < this.nOut; j++) {
          blame += errOut[j] * this.W2[h][j];
        }
        if (hid[h] === 0) blame = 0;
        errHid[h] = blame;
      }
      for (let h = 0; h < this.nHid; h++) {
        for (let j = 0; j < this.nOut; j++) {
          this.W2[h][j] -= lr * errOut[j] * hid[h];
        }
      }
      for (let j = 0; j < this.nOut; j++) {
        this.b2[j] -= lr * errOut[j];
      }
      for (let i = 0; i < this.nIn; i++) {
        for (let h = 0; h < this.nHid; h++) {
          this.W1[i][h] -= lr * errHid[h] * x[i];
        }
      }
      for (let h = 0; h < this.nHid; h++) {
        this.b1[h] -= lr * errHid[h];
      }
      return -Math.log(p[label] + 1e-12);
    }
  };
  function random(max) {
    return (Math.random() * 2 - 1) * max;
  }

  // preprocess.ts
  var SIZE = 16;
  function preprocess(ink, w, h) {
    let minX = w, minY = h, maxX = -1, maxY = -1;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (ink(x, y) > 0) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }
    if (maxX < 0) return null;
    const side = (Math.max(maxX - minX, maxY - minY) + 1) * (SIZE / 12);
    const x0 = (minX + maxX + 1) / 2 - side / 2;
    const y0 = (minY + maxY + 1) / 2 - side / 2;
    const cell = side / SIZE;
    const out = [];
    for (let gy = 0; gy < SIZE; gy++) {
      for (let gx = 0; gx < SIZE; gx++) {
        const ys = Math.floor(y0 + gy * cell);
        const ye = Math.max(ys + 1, Math.floor(y0 + (gy + 1) * cell));
        const xs = Math.floor(x0 + gx * cell);
        const xe = Math.max(xs + 1, Math.floor(x0 + (gx + 1) * cell));
        let sum = 0, n = 0;
        for (let y = ys; y < ye; y++) {
          for (let x = xs; x < xe; x++) {
            n++;
            if (x >= 0 && y >= 0 && x < w && y < h) sum += ink(x, y);
          }
        }
        out[gy * SIZE + gx] = sum / n;
      }
    }
    return out;
  }

  // app.ts
  var LR = 0.1;
  var HIDDEN = 32;
  var EPOCHS = 50;
  var STORAGE_KEY = "neurona.examples";
  var net = new Net(SIZE * SIZE, 10);
  var net2 = new Net2(SIZE * SIZE, HIDDEN, 10);
  var examples = loadExamples();
  var epochsLeft = 0;
  var epochsDone = 0;
  var $ = (id) => document.getElementById(id);
  var pad = $("pad");
  var ctx = pad.getContext("2d", { willReadFrequently: true });
  var drawing = false;
  function clearPad() {
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, pad.width, pad.height);
    update();
  }
  pad.addEventListener("pointerdown", (e) => {
    drawing = true;
    pad.setPointerCapture(e.pointerId);
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 22;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    update();
  });
  pad.addEventListener("pointermove", (e) => {
    if (!drawing) return;
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    update();
  });
  pad.addEventListener("pointerup", () => drawing = false);
  function readInput() {
    const w = pad.width;
    const data = ctx.getImageData(0, 0, w, pad.height).data;
    return preprocess((x, y) => 1 - data[(y * w + x) * 4] / 255, w, pad.height);
  }
  function update() {
    const x = readInput();
    drawPreview(x);
    drawPrediction("1", x ? net.forward(x) : null);
    drawPrediction("2", x ? net2.forward(x).p : null);
  }
  function drawPreview(x) {
    const c = $("preview");
    const g = c.getContext("2d");
    const img = g.createImageData(SIZE, SIZE);
    for (let i = 0; i < SIZE * SIZE; i++) {
      const v = x ? Math.round((1 - x[i]) * 255) : 255;
      img.data.set([v, v, v, 255], i * 4);
    }
    g.putImageData(img, 0, 0);
  }
  function drawPrediction(which, p) {
    const best = p ? argmax(p) : -1;
    $("guess" + which).textContent = p ? String(best) : "\u2013";
    const bars = $("bars" + which);
    bars.innerHTML = "";
    for (let d = 0; d < 10; d++) {
      const pct = p ? p[d] * 100 : 0;
      bars.insertAdjacentHTML(
        "beforeend",
        `<div class="bar ${d === best ? "best" : ""}">
                <span>${d}</span>
                <div class="track"><div class="fill" style="width:${pct}%"></div></div>
                <em>${pct.toFixed(0)}%</em>
            </div>`
      );
    }
  }
  function saveExample(label) {
    const x = readInput();
    if (!x) return;
    if ($("testMode").checked) {
      test.total++;
      if (argmax(net.forward(x)) === label) test.ok1++;
      if (argmax(net2.forward(x).p) === label) test.ok2++;
      renderTest();
      clearPad();
      return;
    }
    const rounded = [];
    for (let i = 0; i < x.length; i++) {
      rounded[i] = Math.round(x[i] * 100) / 100;
    }
    examples.push({ x: rounded, y: label });
    persist();
    renderCounts();
    clearPad();
  }
  var test = { total: 0, ok1: 0, ok2: 0 };
  function renderTest() {
    if (test.total === 0) {
      $("testScore").textContent = "no tests yet";
      return;
    }
    const pct = (n) => `${n}/${test.total} (${Math.round(n / test.total * 100)}%)`;
    $("testScore").textContent = `1 layer: ${pct(test.ok1)} \xB7 with hidden layer: ${pct(test.ok2)}`;
  }
  function renderCounts() {
    const box = $("labels");
    box.innerHTML = "";
    for (let d = 0; d < 10; d++) {
      let n = 0;
      for (const ex of examples) {
        if (ex.y === d) n++;
      }
      const btn = document.createElement("button");
      btn.innerHTML = `${d}<small>${n}</small>`;
      btn.onclick = () => saveExample(d);
      box.appendChild(btn);
    }
    $("total").textContent = `${examples.length} examples`;
  }
  function loadExamples() {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    } catch {
      return [];
    }
  }
  function persist() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(examples));
    } catch {
    }
  }
  function trainEpoch() {
    if (epochsLeft <= 0) return;
    shuffle(examples);
    let loss1 = 0;
    let loss2 = 0;
    for (const ex of examples) {
      loss1 += net.train(ex.x, ex.y, LR);
      loss2 += net2.train(ex.x, ex.y, LR);
    }
    let correct1 = 0;
    let correct2 = 0;
    for (const ex of examples) {
      if (argmax(net.forward(ex.x)) === ex.y) correct1++;
      if (argmax(net2.forward(ex.x).p) === ex.y) correct2++;
    }
    epochsDone++;
    epochsLeft--;
    const n = examples.length;
    $("status").innerHTML = `epoch ${epochsDone}<br>1 layer: average loss ${(loss1 / n).toFixed(3)} \xB7 accuracy on the examples ${Math.round(correct1 / n * 100)}%<br>with hidden layer: average loss ${(loss2 / n).toFixed(3)} \xB7 accuracy on the examples ${Math.round(correct2 / n * 100)}%`;
    drawWeights();
    update();
    requestAnimationFrame(trainEpoch);
  }
  function shuffle(a) {
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
  }
  var selected = { net: 1, n: 0 };
  function weightOf(sel, i) {
    if (sel.net === 1) return net.W[i][sel.n];
    return net2.W1[i][sel.n];
  }
  function biasOf(sel) {
    if (sel.net === 1) return net.b[sel.n];
    return net2.b1[sel.n];
  }
  function drawWeights() {
    drawThumbs($("weights"), 1, 10, (n) => `${n}`);
    drawThumbs($("hidden"), 2, HIDDEN, (n) => `${n + 1}`);
    drawNumbers();
  }
  function drawThumbs(box, netNumber, count, label) {
    if (!box.children.length) {
      for (let n = 0; n < count; n++) {
        const fig = document.createElement("figure");
        fig.innerHTML = `<canvas width="${SIZE}" height="${SIZE}"></canvas><figcaption>${label(n)}</figcaption>`;
        fig.onclick = () => {
          selected = { net: netNumber, n };
          drawWeights();
        };
        box.appendChild(fig);
      }
    }
    const figures = box.querySelectorAll("figure");
    for (let n = 0; n < count; n++) {
      const sel = { net: netNumber, n };
      let max = 1e-9;
      for (let i = 0; i < SIZE * SIZE; i++) {
        if (Math.abs(weightOf(sel, i)) > max) max = Math.abs(weightOf(sel, i));
      }
      const g = figures[n].querySelector("canvas").getContext("2d");
      const img = g.createImageData(SIZE, SIZE);
      for (let i = 0; i < SIZE * SIZE; i++) {
        const v = weightOf(sel, i) / max;
        let pos = 0;
        let neg = 0;
        if (v > 0) pos = Math.round(v * 255);
        if (v < 0) neg = Math.round(-v * 255);
        img.data.set([255 - pos, 255 - neg, 255 - pos - neg, 255], i * 4);
      }
      g.putImageData(img, 0, 0);
      figures[n].classList.toggle("selected", selected.net === netNumber && selected.n === n);
    }
  }
  function drawNumbers() {
    let title = `1-layer network \xB7 neuron for digit ${selected.n}`;
    let table = "W";
    if (selected.net === 2) {
      title = `Hidden-layer network \xB7 detector ${selected.n + 1} of ${HIDDEN}`;
      table = "W1";
    }
    $("selTitle").textContent = title;
    $("bias").textContent = `b = ${biasOf(selected).toFixed(3)}`;
    let html = "";
    for (let row = 0; row < SIZE; row++) {
      html += "<tr>";
      for (let col = 0; col < SIZE; col++) {
        const i = row * SIZE + col;
        const w = weightOf(selected, i);
        let color = "transparent";
        if (w > 0) color = `rgba(46, 160, 67, ${Math.min(1, w * 2)})`;
        if (w < 0) color = `rgba(220, 50, 50, ${Math.min(1, -w * 2)})`;
        html += `<td style="background:${color}" title="pixel ${i} \xB7 ${table}[${i}][${selected.n}]">${w.toFixed(2)}</td>`;
      }
      html += "</tr>";
    }
    $("numbers").innerHTML = html;
  }
  $("clear").onclick = clearPad;
  $("train").onclick = () => {
    if (!examples.length) return;
    const wasIdle = epochsLeft <= 0;
    epochsLeft += EPOCHS;
    if (wasIdle) trainEpoch();
  };
  $("reset").onclick = () => {
    net = new Net(SIZE * SIZE, 10);
    net2 = new Net2(SIZE * SIZE, HIDDEN, 10);
    epochsLeft = 0;
    epochsDone = 0;
    $("status").textContent = "networks reset (network 1 at zero, network 2 random)";
    drawWeights();
    update();
  };
  var mnist = window.MNIST;
  if (!mnist) $("loadMnist").disabled = true;
  $("loadMnist").onclick = () => {
    net = new Net(SIZE * SIZE, 10);
    net.W = mnist.net1.W;
    net.b = mnist.net1.b;
    net2 = new Net2(SIZE * SIZE, HIDDEN, 10);
    net2.W1 = mnist.net2.W1;
    net2.b1 = mnist.net2.b1;
    net2.W2 = mnist.net2.W2;
    net2.b2 = mnist.net2.b2;
    epochsLeft = 0;
    epochsDone = 0;
    $("status").innerHTML = `networks trained on MNIST (60,000 digits)<br>accuracy on 10,000 unseen digits: 1 layer ${mnist.acc1.toFixed(1)}% \xB7 with hidden layer ${mnist.acc2.toFixed(1)}%`;
    drawWeights();
    update();
  };
  $("testReset").onclick = () => {
    test = { total: 0, ok1: 0, ok2: 0 };
    renderTest();
  };
  $("wipe").onclick = () => {
    if (!confirm("Delete all saved examples?")) return;
    examples = [];
    persist();
    renderCounts();
  };
  document.addEventListener("keydown", (e) => {
    if (e.key >= "0" && e.key <= "9") saveExample(Number(e.key));
    else if (e.key === " " || e.key === "Escape") {
      e.preventDefault();
      clearPad();
    }
  });
  renderCounts();
  renderTest();
  drawWeights();
  clearPad();
})();
