
# Mantissa Truncation and Rounding Residuals in Floating Point Representation

## Mantissa Truncation

* IEEE 754 single-precision floats are represented as:

  $\text{Value} = (-1)^S \times 1.F \times 2^{E - 127}$

  where:

  * $S$: sign bit
  * $E$: 8-bit exponent
  * $F$: 23-bit mantissa

* **Truncation** keeps only the most significant $n$ bits of $F$ (e.g., $n = 7$).

* The truncated value $\tilde{x}$ is a quantized version of $x$.

* The discarded bits carry **irreversible precision loss**.

---

## Residual Sign and Truncation Direction

* Define residual:

  $r = x - \tilde{x}$

* If discarded bits are all zero:

  $x = \tilde{x}, \quad r = 0$

* If discarded bits are non-zero:

  $x > \tilde{x}, \quad r > 0$

* **Truncation always rounds toward zero**:

  * For positive $x$: $r \geq 0$
  * For negative $x$: $r \leq 0$

---

## Rounding Logic in Compressed Representation

Let:

* $L$: last kept mantissa bit
* $R$: rounding bit (first discarded)
* $T$: sticky bits (all remaining discarded bits)

**Rounding rules**:

1. $L = 0, R = 0 $ → round down
2. $L = 1, R = 0 $ → round down
3. $L = 0, R = 1 $ → round up
4. $L = 1, R = 1 $ → round up

---

## Recovering Rounding Direction from Compressed Bits?

* Impossible deterministically:

  * The rounding decision depends on discarded bits.
  * Same truncated pattern can come from multiple original values.
  * Mapping is **many-to-one**.

* Only **probabilistic inference** is possible if a rounding policy and distributional assumptions are introduced.

* Statistical analysis can reveal **residual distributions**, but not exact rounding operations.

---

## Quantization in Mantissa Representation

Consider a floating-point number:

$x = m_x \cdot 2^e, \quad m_x \in [1,2), \; e \in \mathbb{Z}$

* Mantissa stored with only $m$ bits:

  $\hat{x} = \tilde{m}_x \cdot 2^e, \quad \tilde{m}_x = \text{Trunc}_m(m_x)$

* Residual:

  $\Delta = \hat{x} - x = (\tilde{m}_x - m_x) \cdot 2^e$

---

## Residual Bounds

* Max difference between mantissas:

  $|\tilde{m}_x - m_x| < 2^{-m}$

  hence:

  $|\Delta| < 2^{e - m}$

* Taking log base 10:

  $\log_{10} |\Delta| < (e - m)\log_{10}(2)$

* Approximate log of true value:

  $\log_{10}|x| \approx e \log_{10}(2) \quad \implies \quad e \approx \frac{\log_{10}|x|}{\log_{10}(2)}$

* Substitution:

  $\log_{10}|\Delta| < \log_{10}|x| - m \log_{10}(2)$

---

## Symmetric Log Residual Bounds

* General bound:

  $\log_{10}|\Delta| \leq \log_{10}|x| - m \log_{10}(2)$

* If truncation always rounds down ($\Delta < 0$):

  $\log_{10}\Delta = -\log_{10}|x| + m\log_{10}(2)$

* **Final symmetric bounds**:

  $\log_{10} |\Delta|_{\text{signed}} \in 
  \left[
  -\log_{10}|x| + m\log_{10}(2), \;\; \log_{10}|x| - m\log_{10}(2)
  \right]$

---

## Interpretation

* These bounds form **two diagonal lines** on a log-log plot of $\log_{10}|x|$ vs. $\log_{10}|\Delta|$.
* Slopes: $\pm 1$.
* Vertical intercepts: determined by $m\log_{10}(2)$.
