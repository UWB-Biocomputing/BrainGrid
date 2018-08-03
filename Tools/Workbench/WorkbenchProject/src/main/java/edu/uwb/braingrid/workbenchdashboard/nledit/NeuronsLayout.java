package edu.uwb.braingrid.workbenchdashboard.nledit;

import java.util.ArrayList;
import java.util.Collections;

public class NeuronsLayout {
	// neuron type index
	/** neuron type index for other neurons */
	public static final int OTR = 0;
	/** neuron type index for inhibitory neurons */
	public static final int INH = 1;
	/** neuron type index for active neurons */
	public static final int ACT = 2;
	/** neuron type index for probed neurons */
	public static final int PRB = 3;
	/** neuron type index for overlapping INH and ACT neurons */
	public static final int OVP = 4;
	
	/** an array to store index of active neurons */
	public ArrayList<Integer> activeNList = new ArrayList<Integer>();
	/** an array to store index of inhibitory neurons */
	public ArrayList<Integer> inhNList = new ArrayList<Integer>();
	/** an array to store index of probed neurons */
	public ArrayList<Integer> probedNList = new ArrayList<Integer>();
	
	public NeuronsLayout() {
		
	}
	
	/**
	 * Returns the NeuronType of the index given
	 * TODO: This needs to be turned into an enums. There are parts in the code (somewhere) that uses the int values to make decisions. 
	 * Use getOrdinal() and other properties of enums to fix this.
	 * @param index
	 * @return
	 */
	public int getNeuronType(int index) {
		int cIndex = OTR;
		if (activeNList.contains(index)
				&& inhNList.contains(index)) {
			cIndex = OVP;
		} else if (activeNList.contains(index)) {
			cIndex = ACT;
		} else if (inhNList.contains(index)) {
			cIndex = INH;
		}
		return cIndex;
	}
	
	static public String getNeuronTypeName(int index) {
		switch (index) {
			case NeuronsLayout.OTR: /* neuron type index for other neurons */
				return "other neuron";
			case NeuronsLayout.INH: /* neuron type index for inhibitory neurons */
				return "inhibitory neuron";
			case NeuronsLayout.ACT: /* neuron type index for active neurons */
				return "active neuron";
			case NeuronsLayout.PRB: /* neuron type index for probed neurons */
				return "probed neuron";
			case NeuronsLayout.OVP: /* neuron type index for overlapping INH and ACT neurons */
				return "overlapping IHH and ACT neuron";
			default:
				return "unknown neuron";
		}
	}
	
	public boolean isProbed(int index) {
		return probedNList.contains(index);
	}
	

	public void changeIndex(int neuronType, int index) {
		switch (neuronType) {
		case INH: // inhibitory neurons edit mode
			if (!inhNList.contains(index)) {
				inhNList.add(index);
				Collections.sort(inhNList);
				if (activeNList.contains(index)) {
					activeNList.remove((Integer)index);
				}
			} else {
				inhNList.remove((Integer)index);
			}
			break;

		case ACT: // active neurons edit mode
			if (!activeNList.contains(index)) {
				activeNList.add(index);
				Collections.sort(activeNList);
				if (inhNList.contains(index)) {
					// The casting is necessary. Otherwise is uses .remove(<index in the array>) vs. .remove(<an object equal to this>);
					inhNList.remove((Integer)index);
				}
			} else {
				activeNList.remove((Integer)index);
			}
			break;

		case PRB: // probed neurons edit mode
			if (!probedNList.contains(index)) {
				probedNList.add(index);
				Collections.sort(probedNList);
			} else {
				probedNList.remove((Integer)index);
			}
			break;
		}
	}
}
