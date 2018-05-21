package edu.uwb.braingrid.workbenchdashboard.nledit;

import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Point;
import java.awt.geom.Point2D;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.Random;

import javax.swing.JOptionPane;

import edu.uwb.braingrid.workbenchdashboard.nledit.NLedit.RepType;

public class NL_Sim_Util {
	
	LayoutPanel layout_panel_;
	NeuronsLayout neurons_layout_;
	
	public NL_Sim_Util(LayoutPanel layout_panel, NeuronsLayout neurons_layout) {
		layout_panel_ = layout_panel;
		neurons_layout_ = neurons_layout;
	}
	
	/**
	 * The function repPattern generates a new neurons list from the old list, the
	 * new size and repeat type of which are specified by parameters, newSize and
	 * rtype respectively.
	 *
	 * @param newSize
	 *            size for the new pattern.
	 * @param list
	 *            the old neurons list.
	 * @param rtype
	 *            repeat type, CLEAR, REPEAT, or ALT.
	 * @return New neurons list of new size.
	 */
	/**
	 * @param newSize
	 * @param list
	 * @param rtype
	 * @return
	 */
	public ArrayList<Integer> repPattern(Dimension newSize, ArrayList<Integer> list, RepType rtype) {
		ArrayList<Integer> newNList = new ArrayList<Integer>();

		if (rtype != RepType.CLEAR) { // if rtype is clear, we just clear the
			// list
			int newX = newSize.width;
			int newY = newSize.height;
			Dimension size = layout_panel_.getLayoutSize();
			int sizeX = size.width;
			int sizeY = size.height;

			Iterator<Integer> iter = list.iterator();
			while (iter.hasNext()) {
				int index = iter.next();
				int x = index % sizeX;
				int y = index / sizeX;
				for (int i = 0; i <= (newY / sizeY); i++) {
					for (int j = 0; j <= (newX / sizeX); j++) {
						if (rtype == RepType.REPEAT) {
							if ((y + sizeY * i) < newY && (x + sizeX * j) < newX) {
								newNList.add((y + sizeY * i) * newX + (x + sizeX * j));
							}
						} else if (rtype == RepType.ALT) {
							int tx, ty;
							if (j % 2 == 0) {
								tx = x;
							} else {
								tx = sizeX - x - 1;
							}
							if (i % 2 == 0) {
								ty = y;
							} else {
								ty = sizeY - y - 1;
							}
							if ((ty + sizeY * i) < newY && (tx + sizeX * j) < newX) {
								newNList.add((ty + sizeY * i) * newX + (tx + sizeX * j));
							}
						}
					}
				}
			}
		}

		return newNList;
	}

	/**
	 * The 'Generate pattern' menu handler.
	 */
	public void actionGeneratePattern() {
		GPatternPanel gpatPanel = new GPatternPanel();
		// int result = JOptionPane.showConfirmDialog(this, gpatPanel,
		// "Generate pattern", JOptionPane.OK_CANCEL_OPTION);
		int result = JOptionPane.showConfirmDialog(layout_panel_, gpatPanel, "Generate pattern",
				JOptionPane.OK_CANCEL_OPTION);
		if (result == JOptionPane.OK_OPTION) { // Afirmative
			try {
				float ratioInh = Float.parseFloat(gpatPanel.tfields[GPatternPanel.idxINH].getText());
				float ratioAct = Float.parseFloat(gpatPanel.tfields[GPatternPanel.idxACT].getText());

				// validate ratios
				if ((ratioInh < 0 || ratioInh > 1.0) || (ratioAct < 0 || ratioAct > 1.0)
						|| (ratioInh + ratioAct > 1.0)) {
					throw new NumberFormatException();
				}

				if (gpatPanel.btns[GPatternPanel.idxREG].isSelected()) {
					genRegularPattern(ratioInh, ratioAct);
				} else if (gpatPanel.btns[GPatternPanel.idxRND].isSelected()) {
					genRandomPattern(ratioInh, ratioAct);
				}

				Graphics g = layout_panel_.getGraphics();
				layout_panel_.writeToGraphics(g);
			} catch (NumberFormatException ne) {
				JOptionPane.showMessageDialog(null, "Invalid ratio.");
			}
		}
	}
	
	/**
	 * The function genRandomPattern generates randomly distributed pattern of
	 * active and inhibitory neurons, each ratio of which are specified by
	 * parameters, ratioInh and ratioAct.
	 *
	 * @param ratioInh
	 *            ratio of inhibitory neurons.
	 * @param ratioAct
	 *            ratio of active neurons.
	 */
	public void genRandomPattern(float ratioInh, float ratioAct) {
		neurons_layout_.inhNList = getRandomPointsIndex(ratioInh, null);
		neurons_layout_.activeNList = getRandomPointsIndex(ratioAct, neurons_layout_.inhNList);
	}
	
	/**
	 * The function getRandomPointsIndex generates randomly distributed pattern of
	 * neurons and returns a indexes list of it.
	 *
	 * @param ratio
	 *            ratio of neurons.
	 * @param occupiedNList
	 *            a neurons indexes that already occupy the layout.
	 * @return an indexes of neurons generated.
	 */
	public ArrayList<Integer> getRandomPointsIndex(float ratio, ArrayList<Integer> occupiedNList) {
		ArrayList<Integer> newNList = new ArrayList<Integer>();
		ArrayList<Integer> freeNList = new ArrayList<Integer>();
		Dimension dim = layout_panel_.getLayoutSize();
		int height = dim.height;
		int width = dim.width;
		int size = height * width;
		int newNListSize = (int) (size * ratio);

		// create a free list
		for (int i = 0; i < size; i++) {
			if (occupiedNList == null || !occupiedNList.contains(i)) {
				freeNList.add(i);
			}
		}

		// create a new neurons list
		Random rnd = new Random();
		while (freeNList.size() > 0 && newNList.size() < newNListSize) {
			int i = rnd.nextInt(freeNList.size());
			newNList.add(freeNList.get(i));
			freeNList.remove(i);
		}
		Collections.sort(newNList);

		return newNList;
	}
	
	/**
	 * The function genRegularPattern generates regularly distributed pattern of
	 * active and inhibitory neurons, each ratio of which are specified by
	 * parameters, ratioInh and ratioAct. The function tries to locate two arrays of
	 * neurons, of which NNI is maximum.
	 *
	 * @param ratioInh
	 *            ratio of inhibitory neurons.
	 * @param ratioAct
	 *            ratio of active neurons.
	 */
	public void genRegularPattern(float ratioInh, float ratioAct) {
		neurons_layout_.inhNList.clear();
		neurons_layout_.activeNList.clear();
		float ratio = ratioInh + ratioAct;
		if (ratio == 0) {
			return;
		}

		if (ratioInh == 0) {
			neurons_layout_.activeNList = getRegularPointsIndex(ratioAct);
			return;
		} else if (ratioAct == 0) {
			neurons_layout_.inhNList = getRegularPointsIndex(ratioInh);
			return;
		}

		// ratioInh != 0 && ratioAct != 0
		neurons_layout_.activeNList = getRegularPointsIndex(ratioAct);
		neurons_layout_.inhNList = getRegularPointsIndex(ratioInh);

		findLargestNNIPointsIndexPair(ratioInh, ratioAct);
	}
	
	/**
	 * The function findLargestNNIPointsIndexPair generates regularly distributed
	 * pattern of active and inhibitory neurons, each ratio of which are specified
	 * by parameters, ratioInh and ratioAct. The function tries to locate two arrays
	 * of neurons, of which NNI is maximum.
	 *
	 * @param ratioInh
	 *            ratio of inhibitory neurons.
	 * @param ratioAct
	 *            ratio of active neurons.
	 * @return indexes of neurons lists generated.
	 */
	public void findLargestNNIPointsIndexPair(float ratioInh, float ratioAct) {
		ArrayList<Point> pts0 = new ArrayList<Point>();
		ArrayList<Point> pts1 = new ArrayList<Point>();
		Dimension dim = layout_panel_.getLayoutSize();
		int height = dim.height;
		int width = dim.width;
		int size = height * width;
		int newNListSize;
		if (ratioInh > ratioAct) {
			newNListSize = (int) (size * ratioInh);
			pts0 = cnvIndexList2Points(neurons_layout_.activeNList);
			pts1 = cnvIndexList2Points(neurons_layout_.inhNList);
		} else {
			newNListSize = (int) (size * ratioAct);
			pts0 = cnvIndexList2Points(neurons_layout_.inhNList);
			pts1 = cnvIndexList2Points(neurons_layout_.activeNList);
		}
		double len = Math.sqrt((double) size / (double) newNListSize);

		ArrayList<Point> union = new ArrayList<Point>(pts0);
		union.addAll(pts1);
		double maxNNI = calcNearestNeighborIndex(union);
		ArrayList<Point> maxPts0 = pts0;
		ArrayList<Point> maxPts1 = pts1;
		for (int xShift = (int) Math.floor(-len / 2); xShift <= Math.ceil(len / 2); xShift++) {
			for (int yShift = (int) Math.floor(-len / 2); yShift <= Math.ceil(len / 2); yShift++) {
				if (xShift == 0 && yShift == 0) {
					continue;
				}
				int xShift0 = (int) Math.ceil((double) xShift / 2);
				int xShift1 = (int) Math.ceil((double) -xShift / 2);
				int yShift0 = (int) Math.ceil((double) yShift / 2);
				int yShift1 = (int) Math.ceil((double) -yShift / 2);
				// System.out.println("xShift = " + xShift + ", xShift0 = " +
				// xShift0 + ", xShift1 = " + xShift1);
				ArrayList<Point> sftPts0 = getShiftedPoints(pts0, xShift0, yShift0);
				ArrayList<Point> sftPts1 = getShiftedPoints(pts1, xShift1, yShift1);
				union = new ArrayList<Point>(sftPts0);
				union.addAll(sftPts1);
				double nni = calcNearestNeighborIndex(union);
				if (nni > maxNNI) {
					maxNNI = nni;
					maxPts0 = sftPts0;
					maxPts1 = sftPts1;
				}
			}
		}

		if (ratioInh > ratioAct) {
			neurons_layout_.activeNList = cnvPoints2IndexList(maxPts0);
			neurons_layout_.inhNList = cnvPoints2IndexList(maxPts1);
		} else {
			neurons_layout_.inhNList = cnvPoints2IndexList(maxPts0);
			neurons_layout_.activeNList = cnvPoints2IndexList(maxPts1);
		}
	}
	
	/**
	 * The function getShiftedPoints gets the shifted points of array. The shift
	 * amount is specified by the parameter, sx, and sy.
	 *
	 * @param pts
	 *            a points of array to be shifted.
	 * @param sx
	 *            shift x value.
	 * @param sy
	 *            shift y value.
	 * @return a points of array shifted
	 */
	public ArrayList<Point> getShiftedPoints(ArrayList<Point> pts, int sx, int sy) {
		Dimension dim = layout_panel_.getLayoutSize();
		int width = dim.width;
		int height = dim.height;

		ArrayList<Point> sftPts = new ArrayList<Point>();
		Iterator<Point> iter = pts.iterator();
		while (iter.hasNext()) {
			Point pt = iter.next();
			int x = (width + (pt.x + sx)) % width;
			int y = (height + (pt.y + sy)) % height;
			sftPts.add(new Point(x, y));
		}

		return sftPts;
	}
	
	/**
	 * The function getRegularPointsIndex generates regularly distributed pattern of
	 * neurons and returns a indexes list of it.
	 *
	 * @param ratio
	 *            ratio of neurons.
	 * @return an indexes of neurons generated.
	 */
	public ArrayList<Integer> getRegularPointsIndex(float ratio) {
		Dimension dim = layout_panel_.getLayoutSize();
		int height = dim.height;
		int width = dim.width;
		int size = height * width;
		int newNListSize = (int) (size * ratio);
		double len = Math.sqrt((double) size / (double) newNListSize);

		int nx = (int) Math.round(width / len);
		int ny = (int) Math.round(height / len);
		dim.width = nx;
		dim.height = ny;

		// find evenly spaced margin
		double mx, my;
		if (nx * len > width) {
			mx = (len - (nx * len - width)) / 2.0;
		} else {
			mx = (len + (width - nx * len)) / 2.0;
		}
		if (ny * len > height) {
			my = (len - (ny * len - height)) / 2.0;
		} else {
			my = (len + (height - ny * len)) / 2.0;
		}
		mx = Math.floor(mx);
		my = Math.floor(my);

		// create points of array, which are regularly distributed
		ArrayList<Point> pts = new ArrayList<Point>();
		for (double y = my; Math.round(y) < height; y += len) {
			for (double x = mx; Math.round(x) < width; x += len) {
				pts.add(new Point((int) Math.round(x), (int) Math.round(y)));
			}
		}

		// convert points to index list
		ArrayList<Integer> newNList = cnvPoints2IndexList(pts);

		return newNList;
	}
	
	/**
	 * The function cnvIndexList2Points converts indexes array to points array.
	 *
	 * @param nList
	 *            a indexes array.
	 * @return - a points array converted from indexes array.
	 */
	public ArrayList<Point> cnvIndexList2Points(ArrayList<Integer> nList) {
		ArrayList<Point> pts = new ArrayList<Point>();

		// convert index list to points array
		int width = layout_panel_.getLayoutSize().width;
		Iterator<Integer> iter = nList.iterator();
		while (iter.hasNext()) {
			int idx = iter.next();
			pts.add(new Point(idx % width, idx / width)); // Point(x, y)
		}

		return pts;
	}
	
	/**
	 * The function cnvPoints2IndexList converts points array to indexes array.
	 *
	 * @param pts
	 *            a points array.
	 * @return an indexes array converted from points array.
	 */
	public ArrayList<Integer> cnvPoints2IndexList(ArrayList<Point> pts) {
		int width = layout_panel_.getLayoutSize().width;
		ArrayList<Integer> newNList = new ArrayList<Integer>();

		Iterator<Point> iter = pts.iterator();
		while (iter.hasNext()) {
			Point pt = iter.next();
			newNList.add(pt.y * width + pt.x);
		}
		Collections.sort(newNList);

		return newNList;
	}
	
	/**
	 * The function calcNearestNeighborIndex calculates a NNI value of points.
	 *
	 * @param pts
	 *            an array of points.
	 * @return a NNI value.
	 */
	public double calcNearestNeighborIndex(ArrayList<Point> pts) {
		int width = layout_panel_.getLayoutSize().width;
		int height = layout_panel_.getLayoutSize().height;

		// calculate average nearest neighbor
		double avgNN = 0;
		int n = pts.size();
		for (int i = 0; i < n; i++) {
			double minDist = Float.MAX_VALUE;
			for (int j = 0; j < n; j++) {
				if (i != j) {
					Point pti = pts.get(i);
					Point ptj = pts.get(j);
					double dist = Point2D.distanceSq(pti.x, pti.y, ptj.x, ptj.y);
					if (minDist > dist) {
						minDist = dist;
					}
				}
			}
			avgNN += Math.sqrt(minDist);
		}
		avgNN = avgNN / (double) n;

		// calculate estimated average neighbor
		double estANN = 1.0 / (2.0 * Math.sqrt((double) n / (double) (width * height)));

		return avgNN / estANN;
	}
	
	/**
	 * The 'Arrange probes' menu handler.
	 */
	public void actionArrangeProbes() {
		AProbesPanel aprbPanel = new AProbesPanel();
		int result = JOptionPane.showConfirmDialog(layout_panel_, aprbPanel, "Arrange probes", JOptionPane.OK_CANCEL_OPTION);
		if (result == JOptionPane.OK_OPTION) { // Afirmative
			try {
				int numProbes = Integer.parseInt(aprbPanel.tfield.getText());

				// validate number
				Dimension dim = layout_panel_.getLayoutSize();
				if (numProbes > dim.height * dim.width) {
					throw new NumberFormatException();
				}

				genProbes(numProbes);

				Graphics g = layout_panel_.getGraphics();
				layout_panel_.writeToGraphics(g);
			} catch (NumberFormatException ne) {
				JOptionPane.showMessageDialog(null, "Invalid number.");
			}
		}
	}
	
	/**
	 * The function genProbes generates a indexes list of probes of specified
	 * number, which are evenly distributed.
	 *
	 * @param numProbes
	 *            number of probes.
	 */
	public void genProbes(int numProbes) {
		if (numProbes == 0) {
			return;
		}
		Dimension dim = layout_panel_.getLayoutSize();
		int height = dim.height;
		int width = dim.width;
		int size = height * width;
		float ratio = (float) numProbes / size;

		neurons_layout_.probedNList = getRegularPointsIndex(ratio);
	}
	
	/**
	 * The function getStatisticalMsg returns a statistical message string of the
	 * layout.
	 *
	 * @param bHtml
	 *            true if reruns html format message string.
	 * @return a statistical message string of the layout.
	 */
	public String getStatisticalMsg(boolean bHtml) {
		String message;
		Dimension dsize = layout_panel_.getLayoutSize();
		int size = dsize.height * dsize.width;
		String head = "", tail = "", nl = "\n", tabs = "\t\t", redh = "", redt = "";

		if (bHtml) {
			head = "<html>";
			tail = "</html>";
			nl = "<br>";
			tabs = "&nbsp;&nbsp;";
			redh = "<font color=\"red\">";
			redt = "</font>";
		}

		float nniInhNList = (float) calcNearestNeighborIndex(cnvIndexList2Points(neurons_layout_.inhNList));
		float nniActNList = (float) calcNearestNeighborIndex(cnvIndexList2Points(neurons_layout_.activeNList));
		ArrayList<Integer> union = new ArrayList<Integer>(neurons_layout_.inhNList);
		union.addAll(neurons_layout_.activeNList);
		float nniUNList = (float) calcNearestNeighborIndex(cnvIndexList2Points(union));
		float nniPrbNList = (float) calcNearestNeighborIndex(cnvIndexList2Points(neurons_layout_.probedNList));

		message = head + "Number of Inhibitory neurons: " + neurons_layout_.inhNList.size();
		message += ", Ratio = " + ((float) neurons_layout_.inhNList.size() / size) + nl;
		message += "Number of Active neurons: " + neurons_layout_.activeNList.size();
		message += ", Ratio = " + ((float) neurons_layout_.activeNList.size() / size) + nl;
		ArrayList<Integer> itr = intersection(neurons_layout_.inhNList, neurons_layout_.activeNList);
		if (itr.size() > 0) {
			message += redh + "Number of overlapping neurons: " + itr.size() + redt + nl;
		}
		message += "Number of Probed neurons: " + neurons_layout_.probedNList.size() + nl;
		message += tabs + "Inhibitory probed neurons = "
				+ intersection(neurons_layout_.inhNList, neurons_layout_.probedNList).size() + nl;
		message += tabs + "Active probed neurons = "
				+ intersection(neurons_layout_.activeNList, neurons_layout_.probedNList).size() + nl;

		message += nl + "Nearest Neighbor Index: " + nl;
		message += tabs + "Inhibitory neurons = " + nniInhNList + nl;
		message += tabs + "Active neurons = " + nniActNList + nl;
		message += tabs + "Inhibitory + Active neurons = " + nniUNList + nl;
		message += tabs + "Probed neurons = " + nniPrbNList + tail;
		return message;
	}
	
	/**
	 * The function intersection returns intersection of two array list.
	 *
	 * @param list1
	 * @param list2
	 * @return the intersection of the two array list.
	 */
	private <T> ArrayList<T> intersection(ArrayList<T> list1, ArrayList<T> list2) {
		ArrayList<T> list = new ArrayList<T>();

		for (T t : list1) {
			if (list2.contains(t)) {
				list.add(t);
			}
		}

		return list;
	}
}
