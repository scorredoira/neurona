// Preprocessing: from an image of any size to 16×16 = 256 ink values.
// Crops the stroke, centers it and scales it so it always takes up the same space.
// Used by the app (your drawings) and by the MNIST training, so the network
// receives both the same way.

export const SIZE = 16

// ink(x, y) = ink of the pixel (0 = none, 1 = fully black)
export function preprocess(ink: (x: number, y: number) => number, w: number, h: number): number[] | null {
    // Box that contains the stroke
    let minX = w, minY = h, maxX = -1, maxY = -1
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            if (ink(x, y) > 0) {
                if (x < minX) minX = x
                if (x > maxX) maxX = x
                if (y < minY) minY = y
                if (y > maxY) maxY = y
            }
        }
    }
    if (maxX < 0) return null // empty image

    // Square centered on the stroke, with a margin: the digit takes 12 of the 16 pixels
    const side = (Math.max(maxX - minX, maxY - minY) + 1) * (SIZE / 12)
    const x0 = (minX + maxX + 1) / 2 - side / 2
    const y0 = (minY + maxY + 1) / 2 - side / 2
    const cell = side / SIZE

    // Each output pixel = average ink of its piece of the image.
    // If the image is small (MNIST is 28×28 and the stroke can take less), a
    // piece can be smaller than one pixel: then we take the pixel at its start.
    const out: number[] = []
    for (let gy = 0; gy < SIZE; gy++) {
        for (let gx = 0; gx < SIZE; gx++) {
            const ys = Math.floor(y0 + gy * cell)
            const ye = Math.max(ys + 1, Math.floor(y0 + (gy + 1) * cell))
            const xs = Math.floor(x0 + gx * cell)
            const xe = Math.max(xs + 1, Math.floor(x0 + (gx + 1) * cell))
            let sum = 0, n = 0
            for (let y = ys; y < ye; y++) {
                for (let x = xs; x < xe; x++) {
                    n++
                    if (x >= 0 && y >= 0 && x < w && y < h) sum += ink(x, y)
                }
            }
            out[gy * SIZE + gx] = sum / n
        }
    }
    return out
}
