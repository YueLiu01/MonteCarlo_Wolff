# Monte Carlo simulation of 2D Ising model with a long-range interacting defect line

We use the Wolff cluster Monte Carlo algorithm to simulate a 2d Ising model on a square lattice of size $(L, L)$ with a defect line at $x = L/2$. PBC along both directions is applied. The temperature is $T = 1/\beta$. The Hamiltonian is 

$$ H = -J \sum_{\braket{i,j}} s_i s_j - \sum_{i,j \in \rm{defect~line}} \frac{K}{\left[\frac{L}{\pi}\sin\left(\frac{\pi|i-j|}{L}\right)\right]^\alpha} s_i s_j, $$ 

where $s_i = \pm 1$.
In the following I fix $J=1$. The mapping between 2d classical Ising and 1d quantum TFIM $H = -ZZ - h X$ is,

$$ \beta = -\frac{1}{2} \ln{\tanh{(\beta h)}}. $$

The critical point of the bulk is $h^{(B)}_c=1$ and, correspondingly, $\beta^{(B)}_c \approx 0.44$.

In **MC_code.py**, we compute the magnetization in the bulk and along the defect line for different parameters and system sizes.

In **MC_code_parallel.py**, we implement parallelization to speed up the simulation.
