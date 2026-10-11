// UI: draw, save examples, train and see what the networks have learned.
// The interesting part is in net.ts and net2.ts; this is just plumbing.

import { Net, argmax } from "./net"
import { Net2 } from "./net2"
import { SIZE, preprocess } from "./preprocess"

const LR = 0.1 // learning rate: how much each weight moves on each step
const HIDDEN = 32 // hidden layer neurons (network 2)
const EPOCHS = 50 // how many full passes over all examples when pressing "Train"
const STORAGE_KEY = "neurona.examples"

type Example = { x: number[]; y: number }

let net = new Net(SIZE * SIZE, 10) // network 1: a single layer
let net2 = new Net2(SIZE * SIZE, HIDDEN, 10) // network 2: with a hidden layer
let examples: Example[] = loadExamples()
let epochsLeft = 0
let epochsDone = 0

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T

// ---------- Drawing ----------

const pad = $<HTMLCanvasElement>("pad")
const ctx = pad.getContext("2d", { willReadFrequently: true })!
let drawing = false

function clearPad() {
    ctx.fillStyle = "#fff"
    ctx.fillRect(0, 0, pad.width, pad.height)
    update()
}

pad.addEventListener("pointerdown", (e) => {
    drawing = true
    pad.setPointerCapture(e.pointerId)
    ctx.strokeStyle = "#000"
    ctx.lineWidth = 22
    ctx.lineCap = "round"
    ctx.lineJoin = "round"
    ctx.beginPath()
    ctx.moveTo(e.offsetX, e.offsetY)
    ctx.lineTo(e.offsetX, e.offsetY)
    ctx.stroke()
    update()
})
pad.addEventListener("pointermove", (e) => {
    if (!drawing) return
    ctx.lineTo(e.offsetX, e.offsetY)
    ctx.stroke()
    update()
})
pad.addEventListener("pointerup", () => (drawing = false))

// ---------- Preprocessing: from the big canvas to 16×16 ----------
// The code is in preprocess.ts (shared with the MNIST training).

function readInput(): number[] | null {
    const w = pad.width
    const data = ctx.getImageData(0, 0, w, pad.height).data
    // ink: black = 1, white = 0
    return preprocess((x, y) => 1 - data[(y * w + x) * 4] / 255, w, pad.height)
}

// ---------- Live prediction ----------

function update() {
    const x = readInput()
    drawPreview(x)
    drawPrediction("1", x ? net.forward(x) : null)
    drawPrediction("2", x ? net2.forward(x).p : null)
}

function drawPreview(x: number[] | null) {
    const c = $<HTMLCanvasElement>("preview")
    const g = c.getContext("2d")!
    const img = g.createImageData(SIZE, SIZE)
    for (let i = 0; i < SIZE * SIZE; i++) {
        const v = x ? Math.round((1 - x[i]) * 255) : 255
        img.data.set([v, v, v, 255], i * 4)
    }
    g.putImageData(img, 0, 0)
}

function drawPrediction(which: string, p: number[] | null) {
    const best = p ? argmax(p) : -1
    $("guess" + which).textContent = p ? String(best) : "–"
    const bars = $("bars" + which)
    bars.innerHTML = ""
    for (let d = 0; d < 10; d++) {
        const pct = p ? p[d] * 100 : 0
        bars.insertAdjacentHTML(
            "beforeend",
            `<div class="bar ${d === best ? "best" : ""}">
                <span>${d}</span>
                <div class="track"><div class="fill" style="width:${pct}%"></div></div>
                <em>${pct.toFixed(0)}%</em>
            </div>`,
        )
    }
}

// ---------- Examples ----------

function saveExample(label: number) {
    const x = readInput()
    if (!x) return

    // In test mode nothing is saved: we check whether each network is right and keep score.
    if ($<HTMLInputElement>("testMode").checked) {
        test.total++
        if (argmax(net.forward(x)) === label) test.ok1++
        if (argmax(net2.forward(x).p) === label) test.ok2++
        renderTest()
        clearPad()
        return
    }

    const rounded: number[] = []
    for (let i = 0; i < x.length; i++) {
        rounded[i] = Math.round(x[i] * 100) / 100
    }
    examples.push({ x: rounded, y: label })
    persist()
    renderCounts()
    clearPad()
}

// ---------- Test mode ----------
// Drawings the networks have NEVER seen: the real comparison.

let test = { total: 0, ok1: 0, ok2: 0 }

function renderTest() {
    if (test.total === 0) {
        $("testScore").textContent = "no tests yet"
        return
    }
    const pct = (n: number) => `${n}/${test.total} (${Math.round((n / test.total) * 100)}%)`
    $("testScore").textContent = `1 layer: ${pct(test.ok1)} · with hidden layer: ${pct(test.ok2)}`
}

function renderCounts() {
    const box = $("labels")
    box.innerHTML = ""
    for (let d = 0; d < 10; d++) {
        let n = 0
        for (const ex of examples) {
            if (ex.y === d) n++
        }
        const btn = document.createElement("button")
        btn.innerHTML = `${d}<small>${n}</small>`
        btn.onclick = () => saveExample(d)
        box.appendChild(btn)
    }
    $("total").textContent = `${examples.length} examples`
}

function loadExamples(): Example[] {
    try {
        return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]")
    } catch {
        return []
    }
}

function persist() {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(examples))
    } catch {}
}

// ---------- Training ----------
// An "epoch" = one pass over all examples, in random order.
// We do one epoch per frame so you can watch the weights change.

function trainEpoch() {
    if (epochsLeft <= 0) return

    shuffle(examples)
    let loss1 = 0
    let loss2 = 0
    for (const ex of examples) {
        loss1 += net.train(ex.x, ex.y, LR)
        loss2 += net2.train(ex.x, ex.y, LR)
    }
    // Count how many examples each network gets right now
    let correct1 = 0
    let correct2 = 0
    for (const ex of examples) {
        if (argmax(net.forward(ex.x)) === ex.y) correct1++
        if (argmax(net2.forward(ex.x).p) === ex.y) correct2++
    }

    epochsDone++
    epochsLeft--
    const n = examples.length
    $("status").innerHTML =
        `epoch ${epochsDone}<br>` +
        `1 layer: average loss ${(loss1 / n).toFixed(3)} · accuracy on the examples ${Math.round((correct1 / n) * 100)}%<br>` +
        `with hidden layer: average loss ${(loss2 / n).toFixed(3)} · accuracy on the examples ${Math.round((correct2 / n) * 100)}%`
    drawWeights()
    update()
    requestAnimationFrame(trainEpoch)
}

function shuffle<T>(a: T[]) {
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1))
        ;[a[i], a[j]] = [a[j], a[i]]
    }
}

// ---------- Show the weights ----------
// Every neuron that looks at the pixels has 256 weights, one per pixel: they
// can be painted as a 16×16 image.
// Green = "ink on this pixel adds in favor". Red = "subtracts". White = doesn't matter.
//   - Network 1: the 10 output neurons (the template of each digit).
//   - Network 2: the 32 hidden neurons (what each one detects).
// Click any of them = see its numbers in the table.

type Selected = { net: number; n: number } // net 1 = 1-layer network, net 2 = hidden layer
let selected: Selected = { net: 1, n: 0 }

// Weight of pixel i in the selected neuron
function weightOf(sel: Selected, i: number): number {
    if (sel.net === 1) return net.W[i][sel.n]
    return net2.W1[i][sel.n]
}
function biasOf(sel: Selected): number {
    if (sel.net === 1) return net.b[sel.n]
    return net2.b1[sel.n]
}

function drawWeights() {
    drawThumbs($("weights"), 1, 10, (n) => `${n}`)
    drawThumbs($("hidden"), 2, HIDDEN, (n) => `${n + 1}`)
    drawNumbers()
}

function drawThumbs(box: HTMLElement, netNumber: number, count: number, label: (n: number) => string) {
    // Create the thumbnails the first time
    if (!box.children.length) {
        for (let n = 0; n < count; n++) {
            const fig = document.createElement("figure")
            fig.innerHTML = `<canvas width="${SIZE}" height="${SIZE}"></canvas><figcaption>${label(n)}</figcaption>`
            fig.onclick = () => {
                selected = { net: netNumber, n: n }
                drawWeights()
            }
            box.appendChild(fig)
        }
    }

    const figures = box.querySelectorAll("figure")
    for (let n = 0; n < count; n++) {
        const sel = { net: netNumber, n: n }
        let max = 1e-9
        for (let i = 0; i < SIZE * SIZE; i++) {
            if (Math.abs(weightOf(sel, i)) > max) max = Math.abs(weightOf(sel, i))
        }
        const g = figures[n].querySelector("canvas")!.getContext("2d")!
        const img = g.createImageData(SIZE, SIZE)
        for (let i = 0; i < SIZE * SIZE; i++) {
            const v = weightOf(sel, i) / max // -1..1
            let pos = 0
            let neg = 0
            if (v > 0) pos = Math.round(v * 255)
            if (v < 0) neg = Math.round(-v * 255)
            img.data.set([255 - pos, 255 - neg, 255 - pos - neg, 255], i * 4)
        }
        g.putImageData(img, 0, 0)
        figures[n].classList.toggle("selected", selected.net === netNumber && selected.n === n)
    }
}

// ---------- The weights as numbers ----------
// The 256 weights of the selected neuron, placed at their pixel's position, and its b.

function drawNumbers() {
    let title = `1-layer network · neuron for digit ${selected.n}`
    let table = "W"
    if (selected.net === 2) {
        title = `Hidden-layer network · detector ${selected.n + 1} of ${HIDDEN}`
        table = "W1"
    }
    $("selTitle").textContent = title
    $("bias").textContent = `b = ${biasOf(selected).toFixed(3)}`

    let html = ""
    for (let row = 0; row < SIZE; row++) {
        html += "<tr>"
        for (let col = 0; col < SIZE; col++) {
            const i = row * SIZE + col // pixel number (row of the weight table)
            const w = weightOf(selected, i)
            let color = "transparent"
            if (w > 0) color = `rgba(46, 160, 67, ${Math.min(1, w * 2)})`
            if (w < 0) color = `rgba(220, 50, 50, ${Math.min(1, -w * 2)})`
            html += `<td style="background:${color}" title="pixel ${i} · ${table}[${i}][${selected.n}]">${w.toFixed(2)}</td>`
        }
        html += "</tr>"
    }
    $("numbers").innerHTML = html
}

// ---------- Buttons and keyboard ----------

$("clear").onclick = clearPad
$("train").onclick = () => {
    if (!examples.length) return
    const wasIdle = epochsLeft <= 0
    epochsLeft += EPOCHS
    if (wasIdle) trainEpoch()
}
$("reset").onclick = () => {
    net = new Net(SIZE * SIZE, 10)
    net2 = new Net2(SIZE * SIZE, HIDDEN, 10)
    epochsLeft = 0
    epochsDone = 0
    $("status").textContent = "networks reset (network 1 at zero, network 2 random)"
    drawWeights()
    update()
}
// Networks already trained on MNIST (mnist-train.ts writes them to mnist-weights.js).
// Their weights are copied into the app's two networks.
const mnist = (window as any).MNIST
if (!mnist) $<HTMLButtonElement>("loadMnist").disabled = true
$("loadMnist").onclick = () => {
    net = new Net(SIZE * SIZE, 10)
    net.W = mnist.net1.W
    net.b = mnist.net1.b
    net2 = new Net2(SIZE * SIZE, HIDDEN, 10)
    net2.W1 = mnist.net2.W1
    net2.b1 = mnist.net2.b1
    net2.W2 = mnist.net2.W2
    net2.b2 = mnist.net2.b2
    epochsLeft = 0
    epochsDone = 0
    $("status").innerHTML =
        `networks trained on MNIST (60,000 digits)<br>` +
        `accuracy on 10,000 unseen digits: 1 layer ${mnist.acc1.toFixed(1)}% · with hidden layer ${mnist.acc2.toFixed(1)}%`
    drawWeights()
    update()
}
$("testReset").onclick = () => {
    test = { total: 0, ok1: 0, ok2: 0 }
    renderTest()
}
$("wipe").onclick = () => {
    if (!confirm("Delete all saved examples?")) return
    examples = []
    persist()
    renderCounts()
}

// Keys 0-9 = save the drawing as that digit. Space/Escape = clear.
document.addEventListener("keydown", (e) => {
    if (e.key >= "0" && e.key <= "9") saveExample(Number(e.key))
    else if (e.key === " " || e.key === "Escape") {
        e.preventDefault()
        clearPad()
    }
})

renderCounts()
renderTest()
drawWeights()
clearPad()
