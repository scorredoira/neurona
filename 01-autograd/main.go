// Lesson 1 · Autograd
//
// The example from the lesson:
//
//	score of A = ink × A's weight
//	result     = score of A × C's weight
//
// Backward computes the sensitivity of the result to every number, and we
// check A's weight by nudging it a little.
package main

import (
	"fmt"

	"neurona/nn"
)

func main() {
	ink := nn.New(0.5)
	weightA := nn.New(2)
	weightC := nn.New(3)

	scoreA := ink.Mul(weightA)
	result := scoreA.Mul(weightC)

	result.Backward()

	fmt.Println("number      value  grad")
	show("result", result)
	show("score of A", scoreA)
	show("C's weight", weightC)
	show("A's weight", weightA)
	show("ink", ink)

	// Check: nudge A's weight by 0.01, compute again, and see how much the result changes.
	nudged := ink.Data * (weightA.Data + 0.01) * weightC.Data
	change := nudged - result.Data
	fmt.Printf("\ncheck: A's weight %.2f → %.2f moves the result by %.3f → sensitivity %.2f\n",
		weightA.Data, weightA.Data+0.01, change, change/0.01)
}

func show(name string, v *nn.Value) {
	fmt.Printf("%-10s %6.2f %5.2f\n", name, v.Data, v.Grad)
}
