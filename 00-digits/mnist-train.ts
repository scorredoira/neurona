// Trains both networks with MNIST (60,000 digits from hundreds of people),
// tests them with another 10,000 they have never seen, and saves the weights to
// mnist-weights.js so the app can load them.
//
// Usage:
//   esbuild mnist-train.ts --bundle --platform=node --outfile=mnist-train.js && node mnist-train.js

import { readFileSync, writeFileSync } from "node:fs"
import { gunzipSync } from "node:zlib"
import { Net, argmax } from "./net"
import { Net2 } from "./net2"
import { SIZE, preprocess } from "./preprocess"

const HIDDEN = 32 // same as in the app
const LR = 0.01
const EPOCHS = 15

type Example = { x: number[]; y: number }

// ---------- Read MNIST ----------
// Format: a header with a few numbers (type, count, rows, columns) and then
// the raw bytes. Each image is 28×28 bytes: 0 = white, 255 = ink.

function load(imagesFile: string, labelsFile: string): Example[] {
    const img = gunzipSync(readFileSync("../data/mnist/" + imagesFile))
    const lab = gunzipSync(readFileSync("../data/mnist/" + labelsFile))
    if (img.readUInt32BE(0) !== 2051 || lab.readUInt32BE(0) !== 2049) {
        throw new Error("this does not look like an MNIST file")
    }
    const count = img.readUInt32BE(4)
    const rows = img.readUInt32BE(8)
    const cols = img.readUInt32BE(12)

    const examples: Example[] = []
    for (let k = 0; k < count; k++) {
        const start = 16 + k * rows * cols
        // Same preprocessing as your drawings: crop, center, 16×16
        const x = preprocess((px, py) => img[start + py * cols + px] / 255, cols, rows)
        if (x) examples.push({ x: x, y: lab[8 + k] })
    }
    return examples
}

// ---------- Train and test ----------

function accuracy(examples: Example[], predict: (x: number[]) => number[]): number {
    let ok = 0
    for (const ex of examples) {
        if (argmax(predict(ex.x)) === ex.y) ok++
    }
    return (ok / examples.length) * 100
}

function shuffle<T>(a: T[]) {
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1))
        const tmp = a[i]
        a[i] = a[j]
        a[j] = tmp
    }
}

console.log("reading MNIST...")
const train = load("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")
const test = load("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
console.log(`${train.length} to train, ${test.length} to test\n`)

const net = new Net(SIZE * SIZE, 10)
const net2 = new Net2(SIZE * SIZE, HIDDEN, 10)
const predict1 = (x: number[]) => net.forward(x)
const predict2 = (x: number[]) => net2.forward(x).p

let acc1 = 0
let acc2 = 0
for (let e = 1; e <= EPOCHS; e++) {
    const t0 = Date.now()
    shuffle(train)
    for (const ex of train) {
        net.train(ex.x, ex.y, LR)
        net2.train(ex.x, ex.y, LR)
    }
    acc1 = accuracy(test, predict1)
    acc2 = accuracy(test, predict2)
    const secs = ((Date.now() - t0) / 1000).toFixed(1)
    console.log(
        `epoch ${e} (${secs}s) · accuracy on unseen digits: ` +
            `1 layer ${acc1.toFixed(1)}% · with hidden layer ${acc2.toFixed(1)}%`,
    )
}

// ---------- Save the weights for the app ----------

function round(v: number): number {
    return Math.round(v * 10000) / 10000
}
function roundTable(t: number[][]): number[][] {
    const out: number[][] = []
    for (let i = 0; i < t.length; i++) {
        out[i] = []
        for (let j = 0; j < t[i].length; j++) {
            out[i][j] = round(t[i][j])
        }
    }
    return out
}
function roundList(l: number[]): number[] {
    const out: number[] = []
    for (let i = 0; i < l.length; i++) {
        out[i] = round(l[i])
    }
    return out
}

const weights = {
    acc1: round(acc1),
    acc2: round(acc2),
    net1: { W: roundTable(net.W), b: roundList(net.b) },
    net2: { W1: roundTable(net2.W1), b1: roundList(net2.b1), W2: roundTable(net2.W2), b2: roundList(net2.b2) },
}
writeFileSync("mnist-weights.js", "window.MNIST = " + JSON.stringify(weights) + "\n")
console.log("\nweights saved to mnist-weights.js")
