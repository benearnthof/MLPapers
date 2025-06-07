https://www-eng.lbl.gov/~shuman/NEXT/MATERIALS&COMPONENTS/Xe_damage/Crank-The-Mathematics-of-Diffusion.pdf
Reference work from 1975 
Introduces the Diffusion Equation from first principles, along with diffusion on manifolds and numerical methods to solve diffusion equations.
Connections to chemical processes

Introduces the diffusion process with water example: 
"DIFFUSION is the process by which matter is transported from one part of a
system to another as a result of random molecular motions. It is usually
illustrated by the classical experiment in which a tall cylindrical vessel has
its lower part filled with iodine solution, for example, and a column of clear
water is poured on top, carefully and slowly, so that no convection currents
are set up. At first the coloured part is separated from the clear by a sharp,
well-defined boundary. Later it is found that the upper part becomes coloured,
the colour getting fainter towards the top, while the lower part becomes cor-
respondingly less intensely coloured. After sufficient time the whole solution
appears uniformly coloured. There is evidently therefore a transfer of iodine
molecules from the lower to the upper part of the vessel taking place in the
absence of convection currents. The iodine is said to have diffused into the
water" 

This appears at first removed from thermodynamics and statistical models but one may imagine the experiment with both liquids at absolute zero -- no mixing would occur. Thus, the connection of diffusion processes to temperature, and by extension thermodynamics, is immediately apparent. How does this help us in any way to generate images of cats?

##### 1.1 Introduction 
The opening paragraph already skirts around the connection to information theory: "Simply because there are more iodine molecules in the lower element than in the upper one, there is a net transfer from the lower to the upper side of the section as a result of the molecular motions." While this is correct, the reason why this appears irreversible is an entropic one: The number of mixed states is simply orders of magnitude larger than the number of sorted, or near sorted, states. We observe this in the source distribution of diffusion probabilistic models where the Gaussian we sample from initially is the maximum entropy distribution (with specified mean and variance) among all real-valued distributions supported on the complete sampling space. [[DDPM]] [[Nonequilibrium Thermodynamics]] [[Geometric Perspective on Diffusion Models]]

##### 1.2 "The mathematical theory of diffusion in isotropic
substances is therefore based on the hypothesis that the rate of transfer of
diffusing substance through unit area of a section is proportional to the
concentration gradient measured normal to the section, i.e."

##### 1.3 Differential equation of diffusion
Fick's first and second laws:
<span id="ficks-law"></span>$$
F = -D \frac{\partial C}{\partial x}
$$
Here F is the rate of transfer per unit area of section, C the concentration of diffusion substance, x the space coordinate measured normal to the section and D is the diffusion coefficient.

We can generalize this idea by first reducing it to a small, rectangular volume. The rate of increase of diffusing substance in the element must then be equal to the differences in- and out flow when pairing parallel surfaces of this rectangular volume (for an isotropic medium). Here we assume that both the volume is constant in size and the medium incompressible, this leads us to:

	$\frac{\partial C}{\partial t}+\frac{\partial F_x}{\partial x}+\frac{\partial F_y}{\partial y}+\frac{\partial F_z}{\partial z}=0$

For a constant diffusion coefficient D, with individual rates of transfer $F_x, F_y, F_z$ as defined in [Fick's Law](#ficks-law) we can then reduce this to: 
	$\frac{\partial C}{\partial t}=D\left(\frac{\partial^2 C}{\partial x^2}+\frac{\partial^2 C}{\partial y^2}+\frac{\partial^2 C}{\partial z^2}\right)$,
	where, in the case of one-dimensional diffusion along a single axis we obtain:
	$\frac{\partial C}{\partial t}=D \frac{\partial^2 C}{\partial x^2}$.
Which is commonly known in the Chemistry literature as Fick's second law.

This also generalizes in cases where the diffusion coefficient is non constant and by simple coordinate transforms we can also apply this to diffusion on cylinders or spheres -- the differential calculus remains the same, regardless of coordinate system.

	$\frac{\partial C}{\partial t}=\operatorname{div}(D \operatorname{grad} C)$.

The direct analogue of Fick's equations from thermodynamics is: 

	$\begin{aligned} & F=-K \partial \theta / \partial x \\ & \frac{\partial \theta}{\partial t}=\left(\frac{K}{c \rho}\right) \frac{\partial^2 \theta}{\partial x^2}\end{aligned}$
Where $\theta$ is the temperature, $K$ is the heat conductivity of the medium, and $\rho$ and $c$ are the density and specific heat respectively (representing the heat capacity per unit volume in the denominator).
Here, of course, the transported medium is not saline or iodine, but heat itself.

#### 2. Methods of Solution When the Diffusion Coefficient is Constant
