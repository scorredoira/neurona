// Neural network with ONE HIDDEN LAYER:
//
//   pixels (256)  ──W1, b1──►  hidden (32, with ReLU)  ──W2, b2──►  outputs (10)  ──softmax──►  percentages
//
// Hidden neurons work like the neurons in net.ts, but they look at the pixels
// and their result goes through ReLU (if negative, it becomes 0).
// Output neurons look at what the hidden neurons gave, not at the pixels.

import { softmax } from "./net"

export class Net2 {
    nIn: number // pixels
    nHid: number // hidden neurons
    nOut: number // output neurons (one per digit)

    W1: number[][] // W1[i][h] = weight of pixel i towards hidden neuron h
    b1: number[] // b1[h]    = b of hidden neuron h
    W2: number[][] // W2[h][j] = weight of hidden neuron h towards output j
    b2: number[] // b2[j]    = b of output j

    constructor(nIn: number, nHid: number, nOut: number) {
        this.nIn = nIn
        this.nHid = nHid
        this.nOut = nOut

        // Here we CANNOT start with all weights at zero: if every hidden neuron
        // starts the same, they all get the same blame and the same correction,
        // and they stay identical forever (32 copies of the same neuron).
        // That is why every weight starts with a small random number.
        this.W1 = []
        for (let i = 0; i < nIn; i++) {
            this.W1[i] = []
            for (let h = 0; h < nHid; h++) {
                this.W1[i][h] = random(0.1)
            }
        }
        this.b1 = []
        for (let h = 0; h < nHid; h++) {
            this.b1[h] = 0
        }

        this.W2 = []
        for (let h = 0; h < nHid; h++) {
            this.W2[h] = []
            for (let j = 0; j < nOut; j++) {
                this.W2[h][j] = random(0.3)
            }
        }
        this.b2 = []
        for (let j = 0; j < nOut; j++) {
            this.b2[j] = 0
        }
    }

    // SCORE: from image to percentages. Also returns what each hidden neuron
    // gave, because training needs it.
    forward(x: number[]): { hid: number[]; p: number[] } {
        // 1. Hidden layer: ink × weight, add up, add b, then ReLU.
        const hid: number[] = []
        for (let h = 0; h < this.nHid; h++) {
            let sum = this.b1[h]
            for (let i = 0; i < this.nIn; i++) {
                sum += x[i] * this.W1[i][h]
            }
            if (sum < 0) sum = 0 // ReLU
            hid[h] = sum
        }

        // 2. Output layer: what each hidden neuron gave × weight, add up, add b.
        const z: number[] = []
        for (let j = 0; j < this.nOut; j++) {
            let sum = this.b2[j]
            for (let h = 0; h < this.nHid; h++) {
                sum += hid[h] * this.W2[h][j]
            }
            z[j] = sum
        }

        // 3. Percentages, same as net.ts.
        return { hid: hid, p: softmax(z) }
    }

    // TRAIN with a single example. Returns the loss (only used for display).
    train(x: number[], label: number, lr: number): number {
        const { hid, p } = this.forward(x)

        // 1. Error of each output neuron: what it said − what it should have said.
        //    (Same as net.ts.)
        const errOut: number[] = []
        for (let j = 0; j < this.nOut; j++) {
            let y = 0
            if (j === label) y = 1
            errOut[j] = p[j] - y
        }

        // 2. BACKPROPAGATION: error of each hidden neuron = its share of the blame.
        //    For each output: its error × how much this hidden neuron influenced it (the weight).
        //    Add them all up. Uses the W2 weights from BEFORE correcting them.
        //    If the hidden neuron was switched off by ReLU (gave 0), it had no influence: blame 0.
        const errHid: number[] = []
        for (let h = 0; h < this.nHid; h++) {
            let blame = 0
            for (let j = 0; j < this.nOut; j++) {
                blame += errOut[j] * this.W2[h][j]
            }
            if (hid[h] === 0) blame = 0
            errHid[h] = blame
        }

        // 3. Correct the output layer (like net.ts, with "what the hidden neuron
        //    gave" instead of the ink).
        for (let h = 0; h < this.nHid; h++) {
            for (let j = 0; j < this.nOut; j++) {
                this.W2[h][j] -= lr * errOut[j] * hid[h]
            }
        }
        for (let j = 0; j < this.nOut; j++) {
            this.b2[j] -= lr * errOut[j]
        }

        // 4. Correct the hidden layer (like net.ts, with the hidden neuron's error
        //    we just computed).
        for (let i = 0; i < this.nIn; i++) {
            for (let h = 0; h < this.nHid; h++) {
                this.W1[i][h] -= lr * errHid[h] * x[i]
            }
        }
        for (let h = 0; h < this.nHid; h++) {
            this.b1[h] -= lr * errHid[h]
        }

        return -Math.log(p[label] + 1e-12)
    }
}

// Random number between -max and +max
function random(max: number): number {
    return (Math.random() * 2 - 1) * max
}
