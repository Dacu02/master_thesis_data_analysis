# Geometrical approximation
Given a recorded trajectory, or part of it, the goal is to find a Bezier curve that approximates it sufficiently well.
## Problem formalization
A Bezier curve is a parametric curve defined by a set of control points. The cubic Bezier curve is defined by four control points $\underline{P_0}, \underline{P_1}, \underline{P_2}, \underline{P_3}$:
$$\underline {B_{P_0,P_1,P_2,P_3}} (s)=(1-s)^3 \underline{P_0}+3(1-s)^2s\underline{P_1}+3(1-s)s^2\underline{P_2}+s^3\underline{P_3},\quad s\in\left[0,1\right]$$

Known parameters:
$$\underline{P_0} = \underline{(0,0,0)},\quad \underline{P_3} = \underline{(\overline{x_e}, \overline{y_e}, \overline{z_e})},\atop \underline{P_1} = \underline{P_0}+\begin{bmatrix}\cos\theta_s \cos\psi_s\\\sin\psi_s\\\sin\theta_s\cos\psi_s\end{bmatrix}\lambda_s,\quad \underline{P_2} = \underline{P_3}-\begin{bmatrix}\cos\theta_e \cos\psi_e\\\sin\psi_e\\\sin\theta_e\cos\psi_e\end{bmatrix}\lambda_e,\quad\int_0^1\|B'(s)\|ds=D$$
Where $$\theta_s=\overline \theta_s,\quad\psi_s=\overline \psi_s,\quad\theta_e=\overline \theta_e,\quad\psi_e=\overline \psi_e\quad D=\overline D$$
Thus, the bezier curve can be expressed as a function of the parameters: $$\underline B\left(s,\underline{P_0},\underline{P_3},\theta_s,\psi_s,\theta_e,\psi_e\right)=\underline B(s,\lambda_s,\lambda_e)=\atop(1-s)^3 \underline{P_0}+3(1-s)^2s\left(\underline{P_0} + \begin{bmatrix}\cos\theta_s \cos\psi_s\\\sin\psi_s\\\sin\theta_s\cos\psi_s\end{bmatrix}\lambda_s\right)+3(1-s)s^2\left(\underline{P_3} - \begin{bmatrix}\cos\theta_e \ cos\psi_e\\\sin\psi_e\\\sin\theta_e\cos\psi_e\end{bmatrix}\lambda_e\right)+s^3\underline{P_3}$$
It is an optimization problem with two unknowns and one free degree of freedom over a total of 12: $$\lambda_s,\lambda_e$$

May $\underline {x_n}(t_k)$ be the $k$-th point of the sampled recorded trajectory $\underline x(t)$ at time $t_k$, 
$t_k \in \{0, T, \ldots, T{\left(n-1\right)}\},k\in\{0,1,\ldots,n-1\}$ while $T={1}/{f}$ sampling period and $n$ the number of samples. The goal is to minime the mean square error between the recorded trajectory and the Bezier curve that approximates it:
$$\text{MSE}= \frac{1}{n}\sum_{k=0}^{n-1}\left\|\underline {x_n}(t_k) - \underline B(s_k,\lambda_s,\lambda_e) \right\|^2$$
While respecting the constraint that the length of the Bezier curve is equal to the length of the recorded trajectory: $$\int_0^1\|B'(s)\|ds = D$$

It is important to note that $s_k$ is a function that depends on the Bézier curve, its role is to map the index $k$ of the recorded trajectory to the parameter $s$ of the Bézier curve. 

May $\sigma_k$ be the cumulative normalized arc length at the $k$-th point: $$\sigma_k = \dfrac{\sum_{j=0}^{k-1}\left\|\underline {p_{j}} - \underline {p_{j+1}}\right\|}{\sum_{j=0}^{n-2}\left\|\underline {p_{j}} - \underline {p_{j+1}}\right\|},\quad \sigma_0=0,\quad\underline {p_{k}}=\underline{x_n}(t_k)$$ 

And $\Lambda_B(s)$ be the cumulative normalized arc length of the Bézier curve at parameter $s$: $$\Lambda_B(s,\lambda_s,\lambda_e) = \dfrac{\int_0^s \|B'(u,\lambda_s,\lambda_e)\| du}{\int_0^1 \|B'(u,\lambda_s,\lambda_e)\| du}$$

Then, $s_k$ can be defined as the inverse of $\Lambda_B(s)$ evaluated at $\sigma_k$: $$s_k\left(\lambda_s,\lambda_e\right) = \Lambda_B^{-1}(\sigma_k)$$

The optimization problem can ultimately be expressed as:
$$\min_{\lambda_s,\lambda_e} \sum_{k=0}^{n-1} \left\|\underline {x_n}(t_k) - \underline B\left(s_k\left(\lambda_s,\lambda_e\right),\lambda_s,\lambda_e\right) \right\|^2$$
Such that:
$$\\s_k(\lambda_s,\lambda_e)=\Lambda_B^{-1}(\sigma_k),\quad\int_0^{s_k}\left\|B'(u,\lambda_s,\lambda_e)\right\|du=\sigma_k D$$
## Implementation details
### Angles $\theta,\psi$
The directions of the initial and final segments of the recorded trajectory are noisy, thus the angles $\theta_s,\psi_s,\theta_e,\psi_e$ are computed as the direction between the start (or end) point and the first point of the recorded trajectory that is at least distant $1\ mm$ away from it.
### Minimization problem seed
Since the optimization problem is non-convex, it is important to provide a good initial guess for the parameters $\lambda_s$ and $\lambda_e$. The initial guess is the best solution that minimizes the MSE when the constraint on the length of the Bezier curve is ignored. In this case, the problem can be expressed as a linear least squares problem, since the problem is affine in the parameters $\lambda_s$ and $\lambda_e$. 

Writing $$B(s) = \underbrace{\left[(1-s)^3 + 3(1-s)^2 s\right] P_0 + \left[3(1-s)s^2 + s^3\right] P_3}_{\text{base}(s)} + \underbrace{3(1-s)^2 s}_{c_s(s)} \lambda_s \hat{d}_s + \underbrace{-3(1-s)s^2}_{c_e(s)} \lambda_e \hat{d}_e$$
Setting $$\text{col}_s = c_s(s)\,\hat{d}_s, \qquad \text{col}_e = c_e(s)\,\hat{d}_e$$

The curve is then expressed as: $$B = \text{base} + \lambda_s\,\text{col}_s + \lambda_e\,\text{col}_e$$

The LSQ problem is solved as
$$A = \begin{pmatrix} \text{col}_s(s_1) & \text{col}_e(s_1) \\ \vdots & \vdots \\ \text{col}_s(s_n) & \text{col}_e(s_n) \end{pmatrix}, \qquad
b = \begin{pmatrix} p_1 - \text{base}(s_1) \\ \vdots \\ p_n - \text{base}(s_n) \end{pmatrix}, \qquad
u = \begin{pmatrix} \lambda_s \\ \lambda_e \end{pmatrix}$$

$$\text{MSE}(u) = \frac{1}{n}\lVert A u - b \rVert^2$$
Where $\hat{d}$ represents the unit vectors in the directions of the initial and final segments of the recorded trajectory.
### Bézier curve integral
The integral of the arc length of the Bezier curve is computed using Gauss-Legendre quadrature, which is a numerical integration method that approximates the integral of a function by evaluating it at specific points (the Gauss points) and weighting the results.
### $\Lambda^{-1}$
As $\Lambda_B(s)$ contains integrals, it is not simple to compute its inverse analytically. Instead, a grid of values of $\Lambda_B(s)$ is computed for a set of $s$ values, and then the inverse is approximated by interpolating the grid. 
