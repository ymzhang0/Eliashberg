---
title: Simple field integrations
---



\title{Simple field integrations}
\author{Yiming Zhang}
\date{2025-07-22}



# Introduction

Here we intriduce some useful results of simplified but heuristic field integrations.

# Hubbard-Stratonovich Transformation Derivation

**1. Standard Gaussian Identity**
We start with the auxiliary Gaussian integral for a variable $y$:
$$

    \int_{-\infty}^{\infty} \exp\left(-\frac{1}{2}y^2\right) dy = \sqrt{2\pi}

$$

**2. Shift Transformation**
Introduce a shift $y \to y - cx^2$ to couple $y$ with $x^2$. The integration measure remains unchanged ($dy$):
$$

\begin{aligned}
    \sqrt{2\pi} &= \int_{-\infty}^{\infty} \exp\left[-\frac{1}{2}(y - cx^2)^2\right] dy \\
    &= \int_{-\infty}^{\infty} \exp\left(-\frac{1}{2}y^2 + cyx^2 - \frac{1}{2}c^2 x^4\right) dy \\
    &= \exp\left(-\frac{1}{2}c^2 x^4\right) \int_{-\infty}^{\infty} \exp\left(-\frac{1}{2}y^2 + cyx^2\right) dy
\end{aligned}

$$

Rearranging to isolate the quartic term:
$$

    \exp\left(\frac{1}{2}c^2 x^4\right) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty} \exp\left(-\frac{1}{2}y^2 + cyx^2\right) dy

$$

**3. Coefficient Matching**
To match the target term $\exp(bx^4)$, we set $\frac{1}{2}c^2 = b$, which implies $c = \sqrt{2b}$. Substituting this back:
$$

    \exp(bx^4) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty} \exp\left(-\frac{1}{2}y^2 + \sqrt{2b}\,x^2 y\right) dy

$$

**4. Full Integral Transformation**
Substitute the above identity into the original integral $I = \int_{-\infty}^{\infty} \exp(-ax^2 + bx^4) dx$:
$$

\begin{aligned}
    I &= \int_{-\infty}^{\infty} e^{-ax^2} \left[ \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty} e^{-\frac{1}{2}y^2 + \sqrt{2b}\,x^2 y} dy \right] dx \\
      &= \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty} dy \, e^{-\frac{1}{2}y^2} \underbrace{\int_{-\infty}^{\infty} dx \, \exp\left[ -\left(a - y\sqrt{2b}\right)x^2 \right]}_{\text{Gaussian integral over } x}
\end{aligned}

$$

**5. Integrating out $x$**
Performing the inner Gaussian integral over $x$:
$$

    I = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty} \exp\left(-\frac{1}{2}y^2\right) \sqrt{\frac{\pi}{a - y\sqrt{2b}}} \, dy

$$

