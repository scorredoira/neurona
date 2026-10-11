// Neural network with ONE layer: every input pixel connects directly to each
// of the 10 output neurons (one per digit).
//
//   input x (256 pixels, 0..1)  ──W, b──►  z (10 scores)  ──softmax──►  p (10 percentages)
//
// No browser APIs: plain TypeScript, just arrays and loops.

export class Net {
    nIn: number
    nOut: number
    W: number[][] // W[i][j] = weight of pixel i towards the neuron of digit j
    b: number[] // b[j]    = bias of neuron j (a fixed push added to its score)

    constructor(nIn: number, nOut: number) {
        this.nIn = nIn
        this.nOut = nOut
        // Everything starts at zero: the network knows nothing and gives 10% to each digit.
        // (Fine with one layer. With hidden layers it is NOT: every hidden neuron
        // would learn the same thing. See net2.ts.)
        this.W = []
        for (let i = 0; i < nIn; i++) {
            this.W[i] = []
            for (let j = 0; j < nOut; j++) {
                this.W[i][j] = 0
            }
        }
        this.b = []
        for (let j = 0; j < nOut; j++) {
            this.b[j] = 0
        }
    }

    // SCORE: from image to percentages.
    forward(x: number[]): number[] {
        // 1. Each neuron j adds up ink × weight for every pixel, starting from its b.
        const z: number[] = []
        for (let j = 0; j < this.nOut; j++) {
            let sum = this.b[j]
            for (let i = 0; i < this.nIn; i++) {
                sum += x[i] * this.W[i][j]
            }
            z[j] = sum
        }

        // 2. Softmax: turns the 10 scores (which can be anything) into
        //    positive percentages that add up to 1.
        return softmax(z)
    }

    // TRAIN with a single example. Returns the loss (only used for display).
    train(x: number[], label: number, lr: number): number {
        const p = this.forward(x)

        // Loss: -log(percentage given to the right answer).
        // 100% on the right one → 0. 1% → 4.6. Lower is better.
        const loss = -Math.log(p[label] + 1e-12)

        for (let j = 0; j < this.nOut; j++) {
            // What neuron j should have said: 1 if it is the right answer, 0 if not.
            let y = 0
            if (j === label) y = 1

            // ERROR of neuron j = what it said − what it should have said.
            //   negative → it fell short, its weights must go up
            //   positive → it overshot, its weights must go down
            const g = p[j] - y

            // CORRECT: new weight = old weight − lr × error × ink.
            // A pixel without ink (x[i] = 0) did not take part, so it does not change.
            for (let i = 0; i < this.nIn; i++) {
                this.W[i][j] -= lr * g * x[i]
            }
            this.b[j] -= lr * g
        }

        return loss
    }
}

export function softmax(z: number[]): number[] {
    // Find the highest score (see the trick below).
    let max = z[0]
    for (let j = 1; j < z.length; j++) {
        if (z[j] > max) max = z[j]
    }

    // 1. Raise 2.718 to each score, so they are all positive.
    //    Trick: subtract the highest so the numbers do not explode.
    //    It does not change the final percentages.
    const e: number[] = []
    for (let j = 0; j < z.length; j++) {
        e[j] = Math.exp(z[j] - max)
    }

    // 2. Add them all up.
    let total = 0
    for (let j = 0; j < z.length; j++) {
        total += e[j]
    }

    // 3. Divide each by the total: percentages that add up to 1 (100%).
    const p: number[] = []
    for (let j = 0; j < z.length; j++) {
        p[j] = e[j] / total
    }
    return p
}

export function argmax(v: number[]): number {
    let best = 0
    for (let i = 1; i < v.length; i++) {
        if (v[i] > v[best]) best = i
    }
    return best
}
