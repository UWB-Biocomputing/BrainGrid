package edu.uwb.braingrid.tools.nledit;

import java.awt.Color;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.GraphicsConfiguration;
import java.awt.GraphicsEnvironment;
import java.awt.Insets;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.Toolkit;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.util.ArrayList;
import java.util.Collections;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.SwingUtilities;

/**
 * The LayoutPanel class handles layout window of the Growth Simulation Layout
 * Editor. The class is a sub-class of JPanel, which shows neurons layout
 * consisting of three kind of neurons, as well as probed neurons. The panel
 * provides editable function of each neuron.
 * 
 * @author Fumitaka Kawasaki
 * @version 1.2
 */
@SuppressWarnings("serial")
public class LayoutPanel extends JPanel implements MouseListener {
	static final int defaultN = 100; // the default system size
	static final int defaultCellWidth = 28; // the default size of each cell
	static final int minCellWidth = 7;
	static final int maxCellWidth = 56;
	static final Color bgColor = new Color(255, 255, 255);// white background
	private int cellWidth; // each cell's width in the window
	private Insets theInsets; // the insets of the window
	private Color nColor[]; // neuron color

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

	private int xlen; // number of cells x-axis
	private int ylen; // number of cells y-axis

	/** minimum number of cells for x-axis */
	public static final int minXCells = 5;
	/** minimum number of cells for y-axis */
	public static final int minYCells = 5;
	/** maximum number of cells for x-axis */
	public static final int maxXCells = 500;
	/** maximum number of cells for y-axis */
	public static final int maxYCells = 500;
	/** default number of cells for x-axis */
	public static final int defXCells = 10;
	/** default number of cells for y-axis */
	public static final int defYCells = 10;

	/** an array to store index of active neurons */
	public ArrayList<Integer> activeNList = new ArrayList<Integer>();
	/** an array to store index of inhibitory neurons */
	public ArrayList<Integer> inhNList = new ArrayList<Integer>();
	/** an array to store index of probed neurons */
	public ArrayList<Integer> probedNList = new ArrayList<Integer>();

	private JScrollPane scrollPane;

	private ControlFrame ctrlFrame; // reference to the control panel

	/**
	 * A class constructor, which initializes global stuff.
	 * 
	 * @param ctrlFrame
	 *            reference to the control panel
	 * @param size
	 *            size of the layout.
	 */
	public LayoutPanel(ControlFrame ctrlFrame, Dimension size) {
		this.ctrlFrame = ctrlFrame;

		xlen = size.width;
		ylen = size.height;

		cellWidth = defaultCellWidth;

		// initialize window and graphics:
		theInsets = getInsets();

		// set the windows size
		setPreferredSize(new Dimension(xlen * cellWidth + theInsets.left
				+ theInsets.right, ylen * cellWidth + theInsets.top
				+ theInsets.bottom));

		// define colors of each type of neurons
		nColor = new Color[5];
		nColor[OTR] = new Color(0x00FF00); // other neurons - green
		nColor[INH] = new Color(0xFF0000); // inhibitory neurons - red
		nColor[ACT] = new Color(0x0000FF); // starter neurons - blue
		nColor[PRB] = new Color(0x000000); // probed neurons - black
		nColor[OVP] = new Color(0xFFFF00); // overlapping neurons - yellow

		// register for mouse events on the window
		addMouseListener(this);
	}

	/**
	 * Initialize graphics objects
	 */
	public void startGraphics() {
		Graphics g = getGraphics();
		g.setColor(bgColor);
		g.fillRect(theInsets.left, theInsets.top, xlen * cellWidth, ylen
				* cellWidth);
	}

	/**
	 * Update the graphical window.
	 * 
	 * @param g
	 *            a graphic object to draw
	 */
	public void writeToGraphics(Graphics g) {
		for (int j = 0; j < ylen; j++) {
			for (int i = 0; i < xlen; i++) {
				if (true) {
					int cIndex = OTR;
					int iNeuron = j * xlen + i;
					if (activeNList.contains(iNeuron)
							&& inhNList.contains(iNeuron)) {
						cIndex = OVP;
					} else if (activeNList.contains(iNeuron)) {
						cIndex = ACT;
					} else if (inhNList.contains(iNeuron)) {
						cIndex = INH;
					}

					g.setColor(nColor[cIndex]);
					int x = theInsets.left + i * cellWidth;
					int y = theInsets.top + j * cellWidth;
					g.fillOval(x, y, cellWidth, cellWidth);

					if (probedNList.contains(iNeuron)) {
						g.setColor(nColor[PRB]);
						g.drawOval(x, y, cellWidth, cellWidth);
						if (cellWidth >= minCellWidth) { // MyPrintable may set
															// smaller cellWidth
							g.drawOval(x + 1, y + 1, cellWidth - 2,
									cellWidth - 2);
						}
					}
				}
			}
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see javax.swing.JComponent#paintComponent(java.awt.Graphics)
	 */
	public void paintComponent(Graphics g) {
		if (g != null) {
			writeToGraphics(g);
			scrollPane.repaint();
		}

	}

	/*
	 * (non-Javadoc) Toggle inhibitory, active or probed neuron type depending
	 * on the current edit mode (neuron type).
	 * 
	 * @see java.awt.event.MouseListener#mouseClicked(java.awt.event.MouseEvent)
	 */
	public void mouseClicked(MouseEvent e) {
		// find a point to click
		Point pt = e.getPoint();
		int i = (pt.x - theInsets.left) / cellWidth;
		int j = (pt.y - theInsets.top) / cellWidth;
		Integer index = j * xlen + i;

		int neuronType = ctrlFrame.getNeuronType();

		switch (neuronType) {
		case INH: // inhibitory neurons edit mode
			if (!inhNList.contains(index)) {
				inhNList.add(index);
				Collections.sort(inhNList);
				if (activeNList.contains(index)) {
					activeNList.remove(index);
				}
			} else {
				inhNList.remove(index);
			}
			break;

		case ACT: // active neurons edit mode
			if (!activeNList.contains(index)) {
				activeNList.add(index);
				Collections.sort(activeNList);
				if (inhNList.contains(index)) {
					inhNList.remove(index);
				}
			} else {
				activeNList.remove(index);
			}
			break;

		case PRB: // probed neurons edit mode
			if (!probedNList.contains(index)) {
				probedNList.add(index);
				Collections.sort(probedNList);
			} else {
				probedNList.remove(index);
			}
			break;
		}

		Graphics g = getGraphics();
		writeToGraphics(g);
	}

	public void mouseEntered(MouseEvent e) {
		// TODO Auto-generated method stub

	}

	public void mouseExited(MouseEvent e) {
		// TODO Auto-generated method stub

	}

	public void mousePressed(MouseEvent e) {
		// TODO Auto-generated method stub

	}

	public void mouseReleased(MouseEvent e) {
		// TODO Auto-generated method stub

	}

	/**
	 * Set the scroll panel.
	 * 
	 * @param scrollpane
	 */
	public void setScrollPane(JScrollPane scrollpane) {
		this.scrollPane = scrollpane;
	}

	/**
	 * Get the scroll panel.
	 * 
	 * @return reference to the scroll panel.
	 */
	public JScrollPane getScrollPane() {
		return scrollPane;
	}

	/**
	 * Change the layout size.
	 * 
	 * @param size
	 *            new layout size.
	 */
	public void changeLayoutSize(Dimension size) {
		xlen = size.width;
		ylen = size.height;

		resetLayoutPanel();
	}

	/**
	 * Change cell size.
	 * 
	 * @param inc
	 *            true when increasing cell size, false decreasing cell size.
	 */
	public void changeCellSize(boolean inc) {
		if (inc == true && cellWidth != maxCellWidth) {
			cellWidth *= 2;
		} else if (inc == false && cellWidth != minCellWidth) {
			cellWidth /= 2;
		} else {
			return;
		}

		resetLayoutPanel();
	}

	/**
	 * Adjust scroll pane and window size.
	 */
	private void resetLayoutPanel() {
		// set the windows size
		theInsets = getInsets();
		setPreferredSize(new Dimension(xlen * cellWidth + theInsets.left
				+ theInsets.right, ylen * cellWidth + theInsets.top
				+ theInsets.bottom));

		// adjust scroll pane and window size
		JFrame frame = (JFrame) SwingUtilities.getRoot(this);
		Container c = frame.getContentPane();
		c.remove(scrollPane);
		scrollPane = new JScrollPane(this,
				JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED,
				JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);
		c.add(scrollPane);
		Rectangle screen = getUsableScreenBounds(null);
		Dimension size = scrollPane.getPreferredSize();
		if (size.width > screen.width) {
			size.width = screen.width;
		}
		if (size.height > screen.height - 20) {
			size.height = screen.height - 20;
		}
		scrollPane.setPreferredSize(size);
		frame.pack();
		frame.setLocationRelativeTo(null);
	}

	/**
	 * Returns the usable area of the screen where applications can place its
	 * windows. The method subtracts from the screen the area of taskbars,
	 * system menus and the like.
	 * 
	 * @param gconf
	 *            the GraphicsConfiguration of the monitor
	 * @return the rectangle of the screen where one can place windows
	 */
	public static Rectangle getUsableScreenBounds(GraphicsConfiguration gconf) {
		if (gconf == null) {
			gconf = GraphicsEnvironment.getLocalGraphicsEnvironment()
					.getDefaultScreenDevice().getDefaultConfiguration();
		}
		Rectangle bounds = new Rectangle(gconf.getBounds());
		try {
			Toolkit toolkit = Toolkit.getDefaultToolkit();
			Insets insets = toolkit.getScreenInsets(gconf);
			bounds.y += insets.top;
			bounds.x += insets.left;
			bounds.height -= (insets.top + insets.bottom);
			bounds.width -= (insets.left + insets.right);
		} catch (Exception ex) {
			System.out
					.println("There was a problem getting screen-related information.");
		}
		return bounds;
	}

	/**
	 * Gets the layout size.
	 * 
	 * @return layout size.
	 */
	public Dimension getLayoutSize() {
		return new Dimension(xlen, ylen);
	}

	/**
	 * Gets the cell width.
	 * 
	 * @return the cell width.
	 */
	public int getCellWidth() {
		return cellWidth;
	}

	/**
	 * Sets the cell width.
	 * 
	 * @param cellWidth
	 *            - cell width to set.
	 */
	public void setCellWidth(int cellWidth) {
		this.cellWidth = cellWidth;
	}

	/**
	 * Gets the reference to the CtrlFrame class object.
	 * 
	 * @return reference to the CtrlFrame class object.
	 */
	public ControlFrame getCtrlFrame() {
		return ctrlFrame;
	}
}
