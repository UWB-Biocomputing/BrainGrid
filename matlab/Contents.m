	     MATLAB functions for BrainGrid data analysis

These files include scripts, functions, and .cpp files for
BrainGrid data analysis. For MATLAB help compatibility, this file
contains a summary of each file's purpose.

------------------------------------------------------------

batch.m		Reads a bunch of simulation XML files and generates a
		wide selection of graphs. This was originally written
		as a function that would iterate over a 2D range of
		parameters (tR, tE), where tR is target rate and tE is
		the fraction of excitatory cells. it would then
		produce figures for each simulation result in that
		range. Uses growth2() to produce the figures.

burstiness2.m	Burstiness index computation and plot.

ca100_batch.m	like ca_batch, but computes more statistics and only
		returns values; no plotting.

ca_batch.m	Reads a set of XML files and generates comparative
		graphs of firing rate statistics ("bifurcation
		diagrams").

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

cb100_batch.m	Script that calculates burstiness statistics for
		overall network for a range of the parameters (tR,
		tE). Uses calc_aburst.

cb_batch.m	Script that uses calc_aburst to generate 2D
		parameter response diagrams.

growth2.m	Main function that creates useful single-simulation
		graphs:
		1. raster plot (using plot_channels)
		2. triple plot of connectivity radii history, change
		   in radii history (first difference), and firing
		   rate history. Lines are colorized as red
		   (inhibitory cells), blue (endogenously active
		   cells), green (edge neurons), and black (all other
		   neurons -- interior excitatory cells that don't
		   fire on their own).
		3. final radii, plotted as circles on a 2D spatial
		   grid (using plotradii)
		4. final firing rates, plotted as filled circles on a
		   2D spatial grid (using plotrates)
		5. plot of burstiness index history (using
		   burstiness2)



