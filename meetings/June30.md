# June 30 Meeting Notes

We tried and failed to find a better way of adjusting the formula for total target and non-target weights.

* **for $k=1$ target:**

```math
m_{\text{target}} = \frac{e^{\alpha \Delta}}{e^{\alpha \Delta} + n - 1}, \quad m_{\text{non-target}} = \frac{n - 1}{e^{\alpha \Delta} + n - 1}
```


* **for $k$ targets:**

```math
m_{\text{target}} = \frac{k e^{\alpha \Delta}}{k e^{\alpha \Delta} + n - k}, \quad m_{\text{non-target}} = \frac{n - k}{k e^{\alpha \Delta} + n - k}
```


#### **idea to explore (AI-assisted):**

* if we allow the system to train for very large $n$, what is the resulting value of $\Delta$, as a function of $n$?

#### **another idea: instead of using**

```math
z_i = \sum_{j=0}^{n-1} a_{ij} v_j
```

* $z_i$ : attention-weighted value vector
* $n-1$ : seq length
* $a_{ij}$ : attention weights
* $v_j$ : value vector for position j

restrict to the top-$K$ attention weights. So find the set $J$ of indices so that $\{a_{ij}\}_{j \in J}$ are the top $K$ weights in $a_{i0}, a_{i1}, \dots, a_{i, n-1}$

```math
z_i = \frac{\sum_{j \in J} a_{ij} v_j}{\sum_{j \in J} \alpha_{ij}}
``` 

same weighted sum but only for top $K$ weights. Does it make sense?


* Start writing report based on binary classification results.
* If desired, do empirical experiments for other tasks
  * e.g. position-dependent,
  * longer outputs.