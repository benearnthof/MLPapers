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

1.1 The opening paragraph already skirts around the connection to information theory: "Simply because there are more iodine molecules in the lower element than in the upper one, there is a net transfer from the lower to the upper side of the section as a result of the molecular motions." While this is correct, the reason why this appears irreversible is an entropic one: The number of mixed states is simply orders of magnitude larger than the number of sorted, or near sorted, states. We observe this in the source distribution of diffusion probabilistic models where the Gaussian we sample from initially is the maximum entropy distribution (with specified mean and variance) among all real-valued distributions supported on the complete sampling space. [[DDPM]] [[Nonequilibrium Thermodynamics]] [[Geometric Perspective on Diffusion Models]]

1.2 "The mathematical theory of diffusion in isotropic
substances is therefore based on the hypothesis that the rate of transfer of
diffusing substance through unit area of a section is proportional to the
concentration gradient measured normal to the section, i.e."
	$F=-D \partial C / \partial x$

1.3 Differential equation of diffusion
