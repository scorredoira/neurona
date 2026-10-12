package nn

// Value is a number that remembers where it came from, so the sensitivity of
// a final result to every number that took part can be computed automatically.
type Value struct {
	Data   float64  // its value
	Grad   float64  // sensitivity: how much the final result changes if this number moves
	Op     string   // operation that made it ("" if it is input data)
	Inputs []*Value // numbers that went into that operation

	// backward passes this number's Grad to its Inputs, using the rule of its
	// operation. Input data has no backward.
	backward func()
}

// New creates a Value from a plain number (input data or a weight).
func New(data float64) *Value {
	return &Value{Data: data}
}

// Mul returns a × b.
func (a *Value) Mul(b *Value) *Value {
	out := &Value{Data: a.Data * b.Data, Op: "*", Inputs: []*Value{a, b}}
	out.backward = func() {
		// Rule of multiplication: if you move one factor, the product moves
		// "the other factor" times as much. Chained (multiplied) with how much
		// the final result depends on the product.
		a.Grad += b.Data * out.Grad
		b.Grad += a.Data * out.Grad
	}
	return out
}

// Backward computes the Grad of every number that took part in v.
func (v *Value) Backward() {
	// Order the numbers so that each one comes after the numbers it was made
	// from. Walking that list backwards, a number's Grad is always complete
	// before it is passed on to its inputs.
	var order []*Value
	visited := map[*Value]bool{}
	var visit func(n *Value)
	visit = func(n *Value) {
		if visited[n] {
			return
		}
		visited[n] = true
		for _, in := range n.Inputs {
			visit(in)
		}
		order = append(order, n)
	}
	visit(v)

	// The final result changes exactly as much as itself: sensitivity 1.
	v.Grad = 1
	for i := len(order) - 1; i >= 0; i-- {
		if order[i].backward != nil {
			order[i].backward()
		}
	}
}
