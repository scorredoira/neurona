# Lesson 1 · Autograd

**What this lesson explains:** what a neuron is (inputs, weights, score), how
much the result changes when a weight moves (sensitivity), how that combines
through several steps (chain rule), and how the computer calculates it by
itself for every number at once (autograd). That last part is what training
needs, and it is the first piece of our library, `nn/`.

## See it work

```
go run ./01-autograd
```

```
number      value  grad
result       3.00  1.00
score of A   1.00  3.00
C's weight   3.00  1.00
A's weight   2.00  1.50
ink          0.50  6.00

check: A's weight 2.00 → 2.01 moves the result by 0.015 → sensitivity 1.50
```

The program builds a chain of two tiny neurons and calculates one number,
the `result`. Then, without trying anything, it tells you for **every**
number that took part how much the result moves if that number moves a
little: that is the `grad` column. The last line checks one of them the slow
way, by actually moving it, and gets the same 1.5.

Why this matters: training any network in this course (the digits, Pong,
the conversation) is exactly this, repeated. Calculate the `grad` of every
weight, move each weight a little in the direction that improves the answer,
and do it again. With millions of weights, trying them one by one is
impossible; this calculates all of them in one go.

## The question

How does the program know the answer is 1.5 for A's weight, and 6 for the
ink, without moving anything? To answer it we need a few ideas, from zero:
what a neuron is, what "sensitivity" means, and how it travels through
several steps. One idea per section, each with small numbers you can check
by hand.

## Pixel and input

An image is a grid of small squares called **pixels**. Each pixel is a number
that says how much ink it has: 0 = white, 1 = black.

A 2×2 image with a vertical line on the left:

```
┌───┬───┐
│ ■ │   │      ink:   1   0
├───┼───┤
│ ■ │   │             1   0
└───┴───┘
```

For the computer it is just a list of 4 numbers: `[1, 0, 1, 0]`.

An **input** is each number a neuron receives from outside. If the neuron
looks at an image, each pixel is an input. If it looks at the output of other
neurons, each of those outputs is an input. The neuron does not care where
they come from.

## Weight and neuron

A **neuron** is two things:

1. Its **weights**: numbers it keeps, one per input.
2. A **fixed rule**: multiply each input by its weight and add everything up.

The rule is the same for every neuron. What makes one neuron different from
another is its weights.

- Positive weight: "ink here is a good sign".
- Negative weight: "ink here is a bad sign".

The rule is a plain function:

```go
func neuron(inputs []float64, weights []float64) float64 {
	sum := 0.0
	for i := 0; i < len(inputs); i++ {
		sum += inputs[i] * weights[i]
	}
	return sum
}
```

The inputs are not part of the neuron: they come from outside and change with
every image. The weights belong to the neuron and only change when training.

Later we add one extra number per neuron, the *bias*, added at the end.

## Score

The **score** is what comes out of the neuron: the sum of input × weight.
With weights `[+1, -1, +1, -1]` (a "line on the left" detector):

| image | inputs | score |
|---|---|---|
| line on the left | 1, 0, 1, 0 | 1 + 0 + 1 + 0 = **2** |
| line on the right | 0, 1, 0, 1 | 0 − 1 + 0 − 1 = **−2** |

The value is arbitrary: no minimum, no maximum. It only means something when
compared with the scores of other neurons, and the highest one wins. Turning
scores into probabilities between 0 and 1 is a later step, *softmax*, which
is not part of the neuron.

## How many neurons

One neuron already looks at **all** the inputs. The number of neurons does
not depend on the number of inputs: it depends on **how many things you want
to detect**. Each neuron gives one score, so it answers one question.

| task | neurons | weights per neuron |
|---|---|---|
| line on the left? (2×2 image) | 1 | 4 |
| line on the left or on top? (2×2 image) | 2 | 4 |
| which digit, 0-9? (28×28 image = 784 pixels) | 10 | 784 |

For digits that is 10 × 784 = 7,840 weights. The answer is the digit whose
neuron gives the highest score.

A **neural network** is a set of neurons organized in **layers**. A layer is
a group of neurons side by side that all receive the same inputs, each with
its own weights. Neurons in the same layer are never connected to each other:
connections only go from one layer to the next.

With one layer (the 10 digit neurons) it is already a network: every neuron
looks at the pixels and the highest score wins. With several layers, the
scores of one layer are the inputs of the next. A layer in the middle, whose
output is not the final answer, is a **hidden layer**.

Why connect layers: to **combine** things. An 8 and a 9 both have a loop on
top; what tells them apart is what comes with it. Layer 1 finds parts, layer
2 combines the parts into digits:

```
           layer 1 (parts)                layer 2 (digits)
pixels ──▶ "loop on top?"      ──┐
       ──▶ "loop at bottom?"   ──┼──▶ "8?"  weights [+1, +1, −1]
       ──▶ "stroke on right?"  ──┘──▶ "9?"  weights [+1, −1, +1]
```

| image | layer 1 scores (top loop, bottom loop, stroke) | "8?" | "9?" |
|---|---|---|---|
| an 8 | 1, 1, 0 | 1 + 1 − 0 = **2** | 1 − 1 + 0 = **0** |
| a 9 | 1, 0, 1 | 1 + 0 − 1 = **0** | 1 − 0 + 1 = **2** |

In practice nobody designs the parts by hand: training finds them. And it
only works with one extra step between layers; without it, two layers of
multiply-and-add collapse into one. We will see that step later.

## Training

Nobody knows the right weights in advance (in the examples above I made them
up because the problem is tiny). The network starts with arbitrary weights
and adjusts them little by little until it gets the answers right. That is
**training**.

## Sensitivity (gradient)

**Sensitivity**: how much a result moves when you move one number a little.
Measure it by nudging: move the number by 0.01, see how much the result
moves, divide by 0.01.

Same neuron, weights `[+1, -1, +1, -1]`, image with a gray pixel
`[1, 0, 0.5, 0]`, score 1.5. Nudging each weight by +0.01:

| weight | new score | change | change ÷ 0.01 | ink of its pixel |
|---|---|---|---|---|
| 0 | 1.51 | +0.01 | **1** | 1 |
| 1 | 1.5 | 0 | **0** | 0 |
| 2 | 1.505 | +0.005 | **0.5** | 0.5 |
| 3 | 1.5 | 0 | **0** | 0 |

The sensitivity of the score to each weight is the ink of its pixel: the
weight only appears in ink × weight, so moving it by 0.01 moves the score by
ink × 0.01. Weights over white pixels do not matter for this image.

Training uses it to know which weights to move and how much. Its technical
name is **derivative** or **gradient**; in the code, `Grad`.

## Chain rule: one operation after another

In a network, the score of one neuron is the input of the next. So a weight
affects the final result **through** other operations. How sensitive is the
final result to that weight?

The smallest case: one pixel, neuron A with one weight, and neuron C that
takes A's score as its only input.

```
pixel ──▶ neuron A ──▶ score of A ──▶ neuron C ──▶ result
 0.5      weight 2         1          weight 3        3
```

| step | calculation | value |
|---|---|---|
| score of A | 0.5 × 2 | 1 |
| result | 1 × 3 | 3 |

Question: if A's weight moves a little, how much does the result move?

Nudge it from 2 to 2.01 and follow the change through each step:

| step | before | after | change |
|---|---|---|---|
| A's weight | 2 | 2.01 | +0.01 |
| score of A | 0.5 × 2 = 1 | 0.5 × 2.01 = 1.005 | +0.005 |
| result | 1 × 3 = 3 | 1.005 × 3 = 3.015 | +0.015 |

The result moved 0.015 for a 0.01 nudge: sensitivity **1.5**.

Where does 1.5 come from? Look at each step on its own:

- A's score moves **0.5** times what A's weight moves (0.5 is the ink: the
  rule from the previous section).
- The result moves **3** times what A's score moves (3 is C's weight: the
  same rule, now A's score is the input and 3 is the other factor).

Chained: 0.5 × 3 = **1.5**. The nudge gets multiplied by 0.5 at the first
step and by 3 at the second.

That is the **chain rule**: the sensitivity of the final result to a number
is the product of the sensitivities of each step along the way.

Why it matters: you never need to nudge anything. Each operation only needs
to know its own small rule (for a multiplication: "the other factor"), and
multiplying those along the path gives the sensitivity of the result to
every number, however long the chain is.

## Automatic: numbers that remember where they came from

To apply the chain rule, the computer needs to know, for every number, which
operation made it and from which numbers. So instead of plain numbers we use
small boxes that keep four things:

| field | what it holds |
|---|---|
| `Data` | the value |
| `Grad` | its sensitivity: how much the final result moves if this number moves |
| `Op` | the operation that made it (`*`), or nothing if it is input data |
| `Inputs` | the numbers that went into that operation |

Same example as before. Each line creates a box:

| box | Data | Op | Inputs |
|---|---|---|---|
| ink | 0.5 | | |
| A's weight | 2 | | |
| C's weight | 3 | | |
| score of A | 0.5 × 2 = **1** | `*` | ink, A's weight |
| result | 1 × 3 = **3** | `*` | score of A, C's weight |

Following `Inputs` from the result back gives the whole path:

```
result (3) ◀── * ──┬── score of A (1) ◀── * ──┬── ink (0.5)
                   │                          └── A's weight (2)
                   └── C's weight (3)
```

All `Grad` start at 0. Then **backward** walks the path from the result to
the inputs:

**Step 1.** The result moves exactly as much as itself: its `Grad` is **1**.

**Step 2.** The result was made by `*` from score of A and C's weight. Rule
of multiplication: each factor's sensitivity is "the other factor", times the
`Grad` of the result (that is the chain rule: multiply along the path).

| box | calculation | Grad |
|---|---|---|
| score of A | C's weight × result's Grad = 3 × 1 | **3** |
| C's weight | score of A × result's Grad = 1 × 1 | **1** |

**Step 3.** Score of A was made by `*` from ink and A's weight. Same rule,
now with score of A's `Grad` (3):

| box | calculation | Grad |
|---|---|---|
| A's weight | ink × score of A's Grad = 0.5 × 3 | **1.5** |
| ink | A's weight × score of A's Grad = 2 × 3 | **6** |

A's weight gets **1.5**, the same we measured by nudging. And without
nudging anything, we got the sensitivity of **every** number at once. You can
check any of them by nudging: C's weight 3 → 3.01 gives a result of
1 × 3.01 = 3.01, a change of 0.01, so sensitivity 1. ✓

Why backwards: a box can only pass its `Grad` to its inputs once its own
`Grad` is complete, and it is completed by the boxes that come after it.

### In the code

That is all `nn/value.go` does. The box:

```go
type Value struct {
	Data   float64
	Grad   float64
	Op     string
	Inputs []*Value
	backward func()
}
```

Multiplying two boxes creates a new one that remembers its inputs, and stores
the rule to pass the `Grad` back (step 2 and step 3 above):

```go
func (a *Value) Mul(b *Value) *Value {
	out := &Value{Data: a.Data * b.Data, Op: "*", Inputs: []*Value{a, b}}
	out.backward = func() {
		a.Grad += b.Data * out.Grad
		b.Grad += a.Data * out.Grad
	}
	return out
}
```

(`+=` instead of `=` because a number can be used in more than one place;
exercise 4 of "Play with it" shows why.)

`Backward()` puts the boxes in order (each one after the boxes it was made
from), sets the result's `Grad` to 1 (step 1), and calls each box's
`backward` from the last to the first.

The program `01-autograd/main.go` builds this exact example:

```go
ink := nn.New(0.5)
weightA := nn.New(2)
weightC := nn.New(3)

scoreA := ink.Mul(weightA)
result := scoreA.Mul(weightC)

result.Backward()
```

That is the output you saw at the start: `Grad` 1.5 for A's weight and 6 for
the ink, the same numbers as steps 1-3 above.

## Play with it

Change a number in `01-autograd/main.go`, guess the new `grad` before
running, then run it to check.

1. **More ink.** Set `ink := nn.New(1)`. A's weight now gets `grad` **3**
   (ink × C's weight = 1 × 3): with more ink, A's weight matters more.
2. **A negative weight.** Set `weightC := nn.New(-3)`. A's weight gets
   **−1.5**: moving it up now moves the result *down*. The sign tells
   training which way to move each weight.
3. **One more step.** After `result`, add `final := result.Mul(nn.New(2))`
   and call `final.Backward()` instead of `result.Backward()`. Every `grad`
   doubles (A's weight **3**, ink **12**): one more step on the path, one
   more factor in the chain. The check line still measures `result`, so it
   keeps saying 1.5.
4. **A number used twice.** `sq := ink.Mul(ink)` and `sq.Backward()`
   instead of `result.Backward()`; look only at the ink row. Ink is
   0.5, so `sq` is 0.25 and ink's `grad` is **1**: the multiplication passes
   0.5 to each of its two inputs, and both are the same box, so they add up.
   That is why the code uses `+=` and not `=`: with `=` the second would
   overwrite the first and you would get 0.5, which is wrong (nudge it:
   0.51 × 0.51 = 0.2601, a change of 0.0101, sensitivity ≈ 1).
