	     MATLAB functions for BrainGrid data analysis

These files include scripts, functions, and .cpp files for BrainGrid
data analysis. We have attempted to divide these files into two
categories: those that are likely to be widely applicable, perhaps
with modest modification (in this directory) and those (in the
"examples" subdirectory) that can serve as examples of using the
general functions, but are unlikely to be usable "off the shelf".

------------------------------------------------------------


burstiness.m	Burstiness index computation and plot.

calc_aburst.m	Calculate average burstiness index (script) for
		20,000-25,000 seconds and 25,000-30,000
		seconds. Useful for plotting parameter response
		diagrams.

calc_afrate.m	Function to calculate average firing rate for
		20,000-25,000 seconds and 25,000-30,000 seconds, plus
		their values normalized relative to the target rate,
		tR.

calc_fraction.m	Script that calculates an estimate of the fraction of
		neurons with stable connectivity radii. A neuron's
		radius is considered "stable" if the difference
		between its value at the end of the simulation and
		10,000 seconds before is less than an arbitrary
		minimum (hard coded to 0.5 spikes/second). Returns
		fraction of all neurons, of non-edge neurons, and of
		edge neurons.

calc_mradii.m	Function that calculates the mean and variance of the
		radii of connectivity of inhibitory and endogenously
		active (starter) cells.

examples/	Directory containing example code that uses functions
		in this directory to do analysis and plotting specific
		to BraingGrid simulations of dissociated cortical
		tissue development.

hist_nconnect.m	Compute and plot a histogram of connections (number of
		other neurons connected to) as of the end of a
		simulation.

plot_channels.m	Raster plot

