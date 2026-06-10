# June 9 Meeting Notes
## **Possible next steps:**

1. **Same experiment but trained simultaneously on a variety of lengths:**
    - e.g., 10, 20, 50.
2. **Extend theory and empirical results to the case where forget can be anywhere.**
3. **Add standard transformer components to the model, including:**
    - Token embedding and unembedding
    - MLP at each layer
    - Layer normalization
    - Multiple layers?
    - *Note:* I'm not sure if we want these components for remaining experiments — for comparison with theoretical analysis, it's easier to remove them.
4. **Add additional target tokens (but each instance still has exactly one or zero targets present) — still binary classification (y/n):**
    - Does performance drop with a large number of targets?
    - Do we need token embedding, MLP, layer norm to help when vocab is large? What does the theory say in this case?
    - **Variant:** not binary classification. Output the target(or other unique value) if present, or ‘n’ otherwise
        - Again, what does the theory say?
5. **Add additional non-target tokens:**
    - Combine with multiple non-targets.
    - Again, does the theory imply that infinite length generalization is possible?
6. **Back to single target token, single non-target token, but now can have multiple targets present and task is to output the number of targets:**
    - e.g., (0,1,2,…,5?,…10?)
    - Can we use a different loss function that encourages ‘close’ estimate?
    - Does that improve accuracy or training time?

## **Other separate lines of research to consider in future:**

### **1. Mechanistic interpretation**

- Go back to the simplest model (single target token in loc 0, single non-target token).
- Demonstrate explicitly the calculation with $K, Q, V$ matrices.
- How are the $a$ and $b$ values produced — can we understand why $a > b$?
- Other visualizations + interpretation — we can look in the literature for this.

### **2. Longer outputs**

- e.g., if $r, s, t$ are targets and $u$ is non-target, then:
    - `"usuutun"` $\mapsto$ `":st"`
    - `"uuu"` $\mapsto$ `":n"`
    - `"uuuuturuuusun"` $\mapsto$ `":trs"`

### **3. Position-dependent task**
*(Higher priority than 1, 2?)*

- e.g., target is valid if in that `"st"` order:
    - `uuutsunn` $\mapsto$ $n$
    - `uusutunn` $\mapsto$ $n$
    - `uustununu` $\mapsto$ $y$
- This will need relative position encoding.
- Main methods to try are:
    - RoPE
    - (T5-style) bias on attention.