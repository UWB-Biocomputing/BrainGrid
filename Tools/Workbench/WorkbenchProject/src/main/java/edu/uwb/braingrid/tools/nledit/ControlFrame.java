package edu.uwb.braingrid.tools.nledit;

import edu.uwb.braingrid.workbench.WorkbenchManager;
import edu.uwb.braingrid.workbench.ui.WorkbenchControlFrame;
import edu.uwb.braingrid.workbench.utils.DateTime;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Point;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.event.KeyEvent;
import java.awt.geom.Point2D;
import java.awt.print.PrinterException;
import java.awt.print.PrinterJob;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.Random;

import javax.swing.ButtonGroup;
import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JRadioButtonMenuItem;
import javax.swing.JScrollPane;
import javax.swing.KeyStroke;

import org.jdom2.Document;
import org.jdom2.Element;
import org.jdom2.JDOMException;
import org.jdom2.input.SAXBuilder;
import org.jdom2.output.Format;
import org.jdom2.output.XMLOutputter;

/**
 * The Growth Simulation Layout Editor is a editor to visually edit neurons
 * layout for Growth Simulator. Three different kinds of neurons exist in the
 * layout, inhibitory neurons, starter (active) neurons, and other neurons
 * (excitatory, non starter neurons). Also any of these neurons can be probed
 * neurons. The layout editor can specify/edit locations of these neurons
 * visually, and generates neurons lists for each kind of neurons and save them
 * to xml format files. The layout editor can also read xml neurons list files
 * to show and edit neurons layout.
 *
 * The ControlPanel class controls operations for the Growth Simulation Layout
 * Editor. The class is a sub-class of JFrame, which contains a layout panel,
 * and three groups of menus, File, Edit, and Layout.
 *
 * @author Fumitaka Kawasaki (modified by Del Davis)
 * @version 1.2
 */
@SuppressWarnings("serial")
public class ControlFrame extends JFrame implements ActionListener {

    private LayoutPanel layoutPanel; // reference to the layout panel

    // menus
    private JMenuBar menuBar = new JMenuBar();

    // File menu
    private JMenu fileMenu = new JMenu("File");
    private JMenuItem importItem = new JMenuItem("Import...", KeyEvent.VK_I);
    private JMenuItem exportItem = new JMenuItem("Export...", KeyEvent.VK_E);
    private JMenuItem clearItem = new JMenuItem("Clear", KeyEvent.VK_C);
    private JMenuItem printItem = new JMenuItem("Print...", KeyEvent.VK_P);
    private JMenuItem exitItem = new JMenuItem("Exit", KeyEvent.VK_X);

    // Edit menu
    private JMenu editMenu = new JMenu("Edit");
    private ButtonGroup editGroup = new ButtonGroup();
    private JRadioButtonMenuItem inhNItem = new JRadioButtonMenuItem(
            "Inhibitory neurons");
    private JRadioButtonMenuItem activeNItem = new JRadioButtonMenuItem(
            "Active neurons");
    private JRadioButtonMenuItem probedNItem = new JRadioButtonMenuItem(
            "Probed neurons");

    // Layout menu
    private JMenu layoutMenu = new JMenu("Layout");
    private JMenuItem bcellItem = new JMenuItem("Bigger cells", KeyEvent.VK_B);
    private JMenuItem scellItem = new JMenuItem("Smaller cells", KeyEvent.VK_S);
    private JMenuItem csizeItem = new JMenuItem("Modify size...", KeyEvent.VK_M);
    private JMenuItem gpatItem = new JMenuItem("Generate pattern...",
            KeyEvent.VK_G);
    private JMenuItem aprbItem = new JMenuItem("Arrange probes...",
            KeyEvent.VK_A);
    private JMenuItem sdatItem = new JMenuItem("Statistical data...",
            KeyEvent.VK_D);

    // Reference to workbench (or other frame code launching NLEdit)
    private WorkbenchControlFrame parent;
    private WorkbenchManager workbenchMgr;

    // repeat type for modify size
    enum RepType {

        REPEAT, ALT, CLEAR
    }

    /**
     * A class constructor, which creates UI components, and registers action
     * listener.
     */
    public ControlFrame(String string) {
        super(string);
    }

    /**
     * Added as a constructor for the workbench. This constructor tracks a
     * parent work bench control frame.
     *
     * @param mgr
     * @see ControlFrame(java.lang.String)
     * @param string - title text
     * @param parent - the workbench that created this frame
     */
    public ControlFrame(String string, WorkbenchControlFrame parent, WorkbenchManager mgr) {
        super(string);
        /* Set Parent Reference for Action Handlers */
        try {
            this.parent = (WorkbenchControlFrame) parent;
        } catch (ClassCastException e) {
            this.parent = null;
        }

        workbenchMgr = mgr;

        parent.setEnabled(false);
    }

    /**
     * Enables the parent frame (which was been disabled upon construction if
     * this frame was created from a parent) and then performs a default dispose
     * operation.
     */
    @Override
    public void dispose() {
        if (parent != null) {
            parent.setEnabled(true);
        }
        super.dispose();
    }

    /**
     * The function init initializes the class. Create all menus and a layout
     * panel.
     *
     * @param sizeX width of the layout panel.
     * @param sizeY height of the layout panel.
     */
    @SuppressWarnings("deprecation")
	public void init(int sizeX, int sizeY) {
        Container c = getContentPane();

        // build menu items
        // File menu
        menuBar.add(fileMenu);
        fileMenu.add(clearItem);
        clearItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_C, Toolkit
                .getDefaultToolkit().getMenuShortcutKeyMask()));
        clearItem.addActionListener(this);
        fileMenu.addSeparator();
        fileMenu.add(importItem);
        importItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_I, Toolkit
                .getDefaultToolkit().getMenuShortcutKeyMask()));
        importItem.addActionListener(this);
        fileMenu.add(exportItem);
        exportItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_E, Toolkit
                .getDefaultToolkit().getMenuShortcutKeyMask()));
        exportItem.addActionListener(this);
        fileMenu.addSeparator();
        fileMenu.add(printItem);
        printItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_P, Toolkit
                .getDefaultToolkit().getMenuShortcutKeyMask()));
        printItem.addActionListener(this);
        fileMenu.addSeparator();
        fileMenu.add(exitItem);
        exitItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_X, Toolkit
                .getDefaultToolkit().getMenuShortcutKeyMask()));
        exitItem.addActionListener(this);

        // Edit menu
        menuBar.add(editMenu);
        editGroup.add(inhNItem);
        editMenu.add(inhNItem);
        inhNItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_I,
                KeyEvent.CTRL_DOWN_MASK));
        inhNItem.setMnemonic(KeyEvent.VK_I);
        inhNItem.addActionListener(this);
        editGroup.add(activeNItem);
        editMenu.add(activeNItem);
        activeNItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_A,
                KeyEvent.CTRL_DOWN_MASK));
        activeNItem.setMnemonic(KeyEvent.VK_A);
        activeNItem.addActionListener(this);
        editGroup.add(probedNItem);
        editMenu.add(probedNItem);
        probedNItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_P,
                KeyEvent.CTRL_DOWN_MASK));
        probedNItem.setMnemonic(KeyEvent.VK_P);
        probedNItem.addActionListener(this);
        inhNItem.setSelected(true);

        // Layout menu
        menuBar.add(layoutMenu);
        layoutMenu.add(bcellItem);
        bcellItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_B, Toolkit
                .getDefaultToolkit().getMenuShortcutKeyMask()));
        bcellItem.addActionListener(this);
        layoutMenu.add(scellItem);
        scellItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_S, Toolkit
                .getDefaultToolkit().getMenuShortcutKeyMask()));
        scellItem.addActionListener(this);
        layoutMenu.addSeparator();
        layoutMenu.add(csizeItem);
        csizeItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_M, Toolkit
                .getDefaultToolkit().getMenuShortcutKeyMask()));
        csizeItem.addActionListener(this);
        layoutMenu.addSeparator();
        layoutMenu.add(gpatItem);
        gpatItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_G, Toolkit
                .getDefaultToolkit().getMenuShortcutKeyMask()));
        gpatItem.addActionListener(this);
        layoutMenu.add(aprbItem);
        aprbItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_A, Toolkit
                .getDefaultToolkit().getMenuShortcutKeyMask()));
        aprbItem.addActionListener(this);
        layoutMenu.addSeparator();
        layoutMenu.add(sdatItem);
        sdatItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_D, Toolkit
                .getDefaultToolkit().getMenuShortcutKeyMask()));
        sdatItem.addActionListener(this);

        setJMenuBar(menuBar);

        // create a layout panel and add scroll to it
        layoutPanel = new LayoutPanel(this, new Dimension(sizeX, sizeY));
        JScrollPane scrollpane = new JScrollPane(layoutPanel,
                JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED,
                JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);
        layoutPanel.setScrollPane(scrollpane);
        c.add(scrollpane);

        // handle resize window
        getRootPane().addComponentListener(new ComponentAdapter() {
            public void componentResized(ComponentEvent e) {
                layoutPanel.getScrollPane()
                        .setPreferredSize(getPreferredSize());
            }
        });
        pack();
    }

    /**
     * The public function getNeuronType returns the current edit mode, which is
     * called from LayoutPanel.
     *
     * @return Current edit mode: LayoutPanel.INH - inhibitory neurons edit
     * mode. LayoutPanel.ACT - active neurons edit mode. LayoutPanel.PRB -
     * probed neurons edit mode.
     */
    public int getNeuronType() {
        if (inhNItem.isSelected()) {
            return LayoutPanel.INH;
        } else if (activeNItem.isSelected()) {
            return LayoutPanel.ACT;
        } else if (probedNItem.isSelected()) {
            return LayoutPanel.PRB;
        }
        return LayoutPanel.OTR;
    }

    /**
     * The function readNeuronListFromFile reads neurons list from the file
     * specified by nameNListFile and stores neurons index in list.
     *
     * @param nameNListFile file path of the neurons list (xml format).
     * @param list array list to store neurons index.
     * @param type type of neurons.
     */
    private void readNeuronListFromFile(String nameNListFile,
            ArrayList<Integer> list, int type) {
        if (nameNListFile == null || nameNListFile.length() == 0) {
            return;
        }

        try {
            // read a xml file
            Document doc = new SAXBuilder().build(new File(nameNListFile));
            Element root = doc.getRootElement();
            if ((root != null)
                    && ((root.getName() == "A" && type == LayoutPanel.ACT)
                    || (root.getName() == "I" && type == LayoutPanel.INH)
                    || (root
                    .getName() == "P" && type == LayoutPanel.PRB))) {
                list.clear();
                String[] parts = root.getValue().split("[ \n\r]");

                Dimension size = layoutPanel.getLayoutSize();
                int numNeurons = size.height * size.width;
                for (String part : parts) {
                    try {
                        int index = Integer.parseInt(part);
                        if (index < numNeurons) { // ignore indexes greater than
                            // numNeurons
                            list.add(index);
                        }
                    } catch (NumberFormatException e) {
                        System.err.println("Illegal number :" + part);
                    }
                }
            }
        } catch (JDOMException je) {
            System.err.println(je);
        } catch (IOException ie) {
            System.err.println(ie);
        }
    }

    /**
     * The function writeNeuronListToFile creates a neurons list file specified
     * by list and type.
     *
     * @param nameNListFile file path of the neurons list.
     * @param list array list of neurons index.
     * @param type type of neurons.
     */
    private void writeNeuronListToFile(String nameNListFile,
            ArrayList<Integer> list, int type) {
        if (nameNListFile == null || nameNListFile.length() == 0) {
            return;
        }

        try {
            Element root = null;
            if (type == LayoutPanel.INH) { // inhibitory neurons
                root = new Element("I");
            } else if (type == LayoutPanel.ACT) { // active neurons
                root = new Element("A");
            } else if (type == LayoutPanel.PRB) { // probed neurons
                root = new Element("P");
            }

            // create a xml file
            String sList = "";
            Iterator<Integer> iter = list.iterator();
            while (iter.hasNext()) {
                sList += " " + iter.next();
            }
            root.setText(sList);

            Document doc = new Document();
            doc.setRootElement(root);

            XMLOutputter xmlOutput = new XMLOutputter(Format.getPrettyFormat());
            xmlOutput.output(doc, new FileOutputStream(nameNListFile));
        } catch (IOException ie) {
            System.err.println(ie);
        }
    }

    /**
     * The function repPattern generates a new neurons list from the old list,
     * the new size and repeat type of which are specified by parameters,
     * newSize and rtype respectively.
     *
     * @param newSize size for the new pattern.
     * @param list the old neurons list.
     * @param rtype repeat type, CLEAR, REPEAT, or ALT.
     * @return New neurons list of new size.
     */
    /**
     * @param newSize
     * @param list
     * @param rtype
     * @return
     */
    private ArrayList<Integer> repPattern(Dimension newSize,
            ArrayList<Integer> list, RepType rtype) {
        ArrayList<Integer> newNList = new ArrayList<Integer>();

        if (rtype != RepType.CLEAR) { // if rtype is clear, we just clear the
            // list
            int newX = newSize.width;
            int newY = newSize.height;
            Dimension size = layoutPanel.getLayoutSize();
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
                            if ((y + sizeY * i) < newY
                                    && (x + sizeX * j) < newX) {
                                newNList.add((y + sizeY * i) * newX
                                        + (x + sizeX * j));
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
                            if ((ty + sizeY * i) < newY
                                    && (tx + sizeX * j) < newX) {
                                newNList.add((ty + sizeY * i) * newX
                                        + (tx + sizeX * j));
                            }
                        }
                    }
                }
            }
        }

        return newNList;
    }

    /**
     * The function changeLayoutSize generates new neurons lists of inhNList,
     * activeNList, and activeNList, and changes the size of the layout panel.
     *
     * @param newSize size for the new layout panel.
     * @param rtype repeat type, CLEAR, REPEAT, or ALT.
     */
    private void changeLayoutSize(Dimension newSize, RepType rtype) {
        layoutPanel.inhNList = repPattern(newSize, layoutPanel.inhNList, rtype);
        layoutPanel.activeNList = repPattern(newSize, layoutPanel.activeNList,
                rtype);
        layoutPanel.probedNList = repPattern(newSize, layoutPanel.activeNList,
                rtype);

        layoutPanel.changeLayoutSize(newSize);
    }

    /**
     * The function cnvPoints2IndexList converts points array to indexes array.
     *
     * @param pts a points array.
     * @return an indexes array converted from points array.
     */
    private ArrayList<Integer> cnvPoints2IndexList(ArrayList<Point> pts) {
        int width = layoutPanel.getLayoutSize().width;
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
     * The function cnvIndexList2Points converts indexes array to points array.
     *
     * @param nList a indexes array.
     * @return - a points array converted from indexes array.
     */
    private ArrayList<Point> cnvIndexList2Points(ArrayList<Integer> nList) {
        ArrayList<Point> pts = new ArrayList<Point>();

        // convert index list to points array
        int width = layoutPanel.getLayoutSize().width;
        Iterator<Integer> iter = nList.iterator();
        while (iter.hasNext()) {
            int idx = iter.next();
            pts.add(new Point(idx % width, idx / width)); // Point(x, y)
        }

        return pts;
    }

    /**
     * The function getRegularPointsIndex generates regularly distributed
     * pattern of neurons and returns a indexes list of it.
     *
     * @param ratio ratio of neurons.
     * @return an indexes of neurons generated.
     */
    private ArrayList<Integer> getRegularPointsIndex(float ratio) {
        Dimension dim = layoutPanel.getLayoutSize();
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
     * The function getShiftedPoints gets the shifted points of array. The shift
     * amount is specified by the parameter, sx, and sy.
     *
     * @param pts a points of array to be shifted.
     * @param sx shift x value.
     * @param sy shift y value.
     * @return a points of array shifted
     */
    private ArrayList<Point> getShiftedPoints(ArrayList<Point> pts, int sx,
            int sy) {
        Dimension dim = layoutPanel.getLayoutSize();
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
     * The function findLargestNNIPointsIndexPair generates regularly
     * distributed pattern of active and inhibitory neurons, each ratio of which
     * are specified by parameters, ratioInh and ratioAct. The function tries to
     * locate two arrays of neurons, of which NNI is maximum.
     *
     * @param ratioInh ratio of inhibitory neurons.
     * @param ratioAct ratio of active neurons.
     * @return indexes of neurons lists generated.
     */
    private void findLargestNNIPointsIndexPair(float ratioInh, float ratioAct) {
        ArrayList<Point> pts0 = new ArrayList<Point>();
        ArrayList<Point> pts1 = new ArrayList<Point>();
        Dimension dim = layoutPanel.getLayoutSize();
        int height = dim.height;
        int width = dim.width;
        int size = height * width;
        int newNListSize;
        if (ratioInh > ratioAct) {
            newNListSize = (int) (size * ratioInh);
            pts0 = cnvIndexList2Points(layoutPanel.activeNList);
            pts1 = cnvIndexList2Points(layoutPanel.inhNList);
        } else {
            newNListSize = (int) (size * ratioAct);
            pts0 = cnvIndexList2Points(layoutPanel.inhNList);
            pts1 = cnvIndexList2Points(layoutPanel.activeNList);
        }
        double len = Math.sqrt((double) size / (double) newNListSize);

        ArrayList<Point> union = new ArrayList<Point>(pts0);
        union.addAll(pts1);
        double maxNNI = calcNearestNeighborIndex(union);
        ArrayList<Point> maxPts0 = pts0;
        ArrayList<Point> maxPts1 = pts1;
        for (int xShift = (int) Math.floor(-len / 2); xShift <= Math
                .ceil(len / 2); xShift++) {
            for (int yShift = (int) Math.floor(-len / 2); yShift <= Math
                    .ceil(len / 2); yShift++) {
                if (xShift == 0 && yShift == 0) {
                    continue;
                }
                int xShift0 = (int) Math.ceil((double) xShift / 2);
                int xShift1 = (int) Math.ceil((double) -xShift / 2);
                int yShift0 = (int) Math.ceil((double) yShift / 2);
                int yShift1 = (int) Math.ceil((double) -yShift / 2);
                // System.out.println("xShift = " + xShift + ", xShift0 = " +
                // xShift0 + ", xShift1 = " + xShift1);
                ArrayList<Point> sftPts0 = getShiftedPoints(pts0, xShift0,
                        yShift0);
                ArrayList<Point> sftPts1 = getShiftedPoints(pts1, xShift1,
                        yShift1);
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
            layoutPanel.activeNList = cnvPoints2IndexList(maxPts0);
            layoutPanel.inhNList = cnvPoints2IndexList(maxPts1);
        } else {
            layoutPanel.inhNList = cnvPoints2IndexList(maxPts0);
            layoutPanel.activeNList = cnvPoints2IndexList(maxPts1);
        }
    }

    /**
     * The function genRegularPattern generates regularly distributed pattern of
     * active and inhibitory neurons, each ratio of which are specified by
     * parameters, ratioInh and ratioAct. The function tries to locate two
     * arrays of neurons, of which NNI is maximum.
     *
     * @param ratioInh ratio of inhibitory neurons.
     * @param ratioAct ratio of active neurons.
     */
    private void genRegularPattern(float ratioInh, float ratioAct) {
        layoutPanel.inhNList.clear();
        layoutPanel.activeNList.clear();
        float ratio = ratioInh + ratioAct;
        if (ratio == 0) {
            return;
        }

        if (ratioInh == 0) {
            layoutPanel.activeNList = getRegularPointsIndex(ratioAct);
            return;
        } else if (ratioAct == 0) {
            layoutPanel.inhNList = getRegularPointsIndex(ratioInh);
            return;
        }

        // ratioInh != 0 && ratioAct != 0
        layoutPanel.activeNList = getRegularPointsIndex(ratioAct);
        layoutPanel.inhNList = getRegularPointsIndex(ratioInh);

        findLargestNNIPointsIndexPair(ratioInh, ratioAct);
    }

    /**
     * The function getRandomPointsIndex generates randomly distributed pattern
     * of neurons and returns a indexes list of it.
     *
     * @param ratio ratio of neurons.
     * @param occupiedNList a neurons indexes that already occupy the layout.
     * @return an indexes of neurons generated.
     */
    private ArrayList<Integer> getRandomPointsIndex(float ratio,
            ArrayList<Integer> occupiedNList) {
        ArrayList<Integer> newNList = new ArrayList<Integer>();
        ArrayList<Integer> freeNList = new ArrayList<Integer>();
        Dimension dim = layoutPanel.getLayoutSize();
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
     * The function genRandomPattern generates randomly distributed pattern of
     * active and inhibitory neurons, each ratio of which are specified by
     * parameters, ratioInh and ratioAct.
     *
     * @param ratioInh ratio of inhibitory neurons.
     * @param ratioAct ratio of active neurons.
     */
    private void genRandomPattern(float ratioInh, float ratioAct) {
        layoutPanel.inhNList = getRandomPointsIndex(ratioInh, null);
        layoutPanel.activeNList = getRandomPointsIndex(ratioAct,
                layoutPanel.inhNList);
    }

    /**
     * The function genProbes generates a indexes list of probes of specified
     * number, which are evenly distributed.
     *
     * @param numProbes number of probes.
     */
    private void genProbes(int numProbes) {
        if (numProbes == 0) {
            return;
        }
        Dimension dim = layoutPanel.getLayoutSize();
        int height = dim.height;
        int width = dim.width;
        int size = height * width;
        float ratio = (float) numProbes / size;

        layoutPanel.probedNList = getRegularPointsIndex(ratio);
    }

    /**
     * The function calcNearestNeighborIndex calculates a NNI value of points.
     *
     * @param pts an array of points.
     * @return a NNI value.
     */
    private double calcNearestNeighborIndex(ArrayList<Point> pts) {
        int width = layoutPanel.getLayoutSize().width;
        int height = layoutPanel.getLayoutSize().height;

        // calculate average nearest neighbor
        double avgNN = 0;
        int n = pts.size();
        for (int i = 0; i < n; i++) {
            double minDist = Float.MAX_VALUE;
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    Point pti = pts.get(i);
                    Point ptj = pts.get(j);
                    double dist = Point2D
                            .distanceSq(pti.x, pti.y, ptj.x, ptj.y);
                    if (minDist > dist) {
                        minDist = dist;
                    }
                }
            }
            avgNN += Math.sqrt(minDist);
        }
        avgNN = avgNN / (double) n;

        // calculate estimated average neighbor
        double estANN = 1.0 / (2.0 * Math.sqrt((double) n
                / (double) (width * height)));

        return avgNN / estANN;
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

    /**
     * The function getStatisticalMsg returns a statistical message string of
     * the layout.
     *
     * @param bHtml true if reruns html format message string.
     * @return a statistical message string of the layout.
     */
    public String getStatisticalMsg(boolean bHtml) {
        String message;
        Dimension dsize = layoutPanel.getLayoutSize();
        int size = dsize.height * dsize.width;
        String head = "", tail = "", nl = "\n", tabs = "\t\t", redh = "", redt
                = "";

        if (bHtml) {
            head = "<html>";
            tail = "</html>";
            nl = "<br>";
            tabs = "&nbsp;&nbsp;";
            redh = "<font color=\"red\">";
            redt = "</font>";
        }

        float nniInhNList
                = (float) calcNearestNeighborIndex(cnvIndexList2Points(layoutPanel.inhNList));
        float nniActNList
                = (float) calcNearestNeighborIndex(cnvIndexList2Points(layoutPanel.activeNList));
        ArrayList<Integer> union = new ArrayList<Integer>(layoutPanel.inhNList);
        union.addAll(layoutPanel.activeNList);
        float nniUNList
                = (float) calcNearestNeighborIndex(cnvIndexList2Points(union));
        float nniPrbNList
                = (float) calcNearestNeighborIndex(cnvIndexList2Points(layoutPanel.probedNList));

        message = head + "Number of Inhibitory neurons: "
                + layoutPanel.inhNList.size();
        message += ", Ratio = " + ((float) layoutPanel.inhNList.size() / size)
                + nl;
        message += "Number of Active neurons: "
                + layoutPanel.activeNList.size();
        message += ", Ratio = "
                + ((float) layoutPanel.activeNList.size() / size) + nl;
        ArrayList<Integer> itr = intersection(layoutPanel.inhNList,
                layoutPanel.activeNList);
        if (itr.size() > 0) {
            message += redh + "Number of overlapping neurons: " + itr.size()
                    + redt + nl;
        }
        message += "Number of Probed neurons: "
                + layoutPanel.probedNList.size() + nl;
        message += tabs
                + "Inhibitory probed neurons = "
                + intersection(layoutPanel.inhNList, layoutPanel.probedNList)
                .size() + nl;
        message += tabs
                + "Active probed neurons = "
                + intersection(layoutPanel.activeNList, layoutPanel.probedNList)
                .size() + nl;

        message += nl + "Nearest Neighbor Index: " + nl;
        message += tabs + "Inhibitory neurons = " + nniInhNList + nl;
        message += tabs + "Active neurons = " + nniActNList + nl;
        message += tabs + "Inhibitory + Active neurons = " + nniUNList + nl;
        message += tabs + "Probed neurons = " + nniPrbNList + tail;
        return message;
    }

    /**
     * The 'Import...' menu handler.
     */
    private void actionImport() {
        ImportPanel myPanel = new ImportPanel();
        int result = JOptionPane.showConfirmDialog(this, myPanel, "Import",
                JOptionPane.OK_CANCEL_OPTION);
        if (result == JOptionPane.OK_OPTION) { // Afirmative
            readNeuronListFromFile(
                    myPanel.tfields[ImportPanel.idxInhList].getText(),
                    layoutPanel.inhNList, LayoutPanel.INH);
            readNeuronListFromFile(
                    myPanel.tfields[ImportPanel.idxActList].getText(),
                    layoutPanel.activeNList, LayoutPanel.ACT);
            readNeuronListFromFile(
                    myPanel.tfields[ImportPanel.idxPrbList].getText(),
                    layoutPanel.probedNList, LayoutPanel.PRB);

            Graphics g = layoutPanel.getGraphics();
            layoutPanel.writeToGraphics(g);
        }
    }

    /**
     * The 'Export...' menu handler.
     */
    private void actionExport() {
        ExportPanel myPanel = new ExportPanel(ImportPanel.nlistDir);
        int result = JOptionPane.showConfirmDialog(this, myPanel, "Export",
                JOptionPane.OK_CANCEL_OPTION);
        Long functionStartTime = System.currentTimeMillis();
        Long accumulatedTime = 0L;
        if (result == JOptionPane.OK_OPTION) { // Afirmative
            writeNeuronListToFile(
                    myPanel.tfields[ExportPanel.idxInhList].getText(),
                    layoutPanel.inhNList, LayoutPanel.INH);
            // add to workbench project
            if (null != workbenchMgr && workbenchMgr.isProvEnabled()) {
                Long startTime = System.currentTimeMillis();
                workbenchMgr.getProvMgr().addFileGeneration(
                        "InhibitoryNeuronListExport" + java.util.UUID.randomUUID(),
                        "neuronListExport", "NLEdit", null, false,
                        myPanel.tfields[ExportPanel.idxInhList].getText(), null,
                        null);
                accumulatedTime = DateTime.sumProvTiming(startTime, accumulatedTime);
            }

            writeNeuronListToFile(
                    myPanel.tfields[ExportPanel.idxActList].getText(),
                    layoutPanel.activeNList, LayoutPanel.ACT);
            // add to workbench project
            if (null != workbenchMgr && workbenchMgr.isProvEnabled()) {
                Long startTime = System.currentTimeMillis();
                workbenchMgr.getProvMgr().addFileGeneration(
                        "ActiveNeuronListExport" + java.util.UUID.randomUUID(),
                        "neuronListExport", "NLEdit", null, false,
                        myPanel.tfields[ExportPanel.idxActList].getText(), null,
                        null);
                accumulatedTime = DateTime.sumProvTiming(startTime, accumulatedTime);
            }

            writeNeuronListToFile(
                    myPanel.tfields[ExportPanel.idxPrbList].getText(),
                    layoutPanel.probedNList, LayoutPanel.PRB);
            // add to workbench project
            if (null != workbenchMgr && workbenchMgr.isProvEnabled()) {
                Long startTime = System.currentTimeMillis();
                workbenchMgr.getProvMgr().addFileGeneration(
                        "ProbedNeuronListExport" + java.util.UUID.randomUUID(),
                        "neuronListExport", "NLEdit", null, false,
                        myPanel.tfields[ExportPanel.idxPrbList].getText(), null,
                        null);
                accumulatedTime = DateTime.sumProvTiming(startTime, accumulatedTime);
            }
        }
        DateTime.recordFunctionExecutionTime("ControlFrame", "actionExport",
                System.currentTimeMillis() - functionStartTime,
                workbenchMgr.isProvEnabled());
        if (workbenchMgr.isProvEnabled()) {
            DateTime.recordAccumulatedProvTiming("ControlFrame", "actionExport",
                    accumulatedTime);
        }
    }

    /**
     * The 'Print...' menu handler.
     */
    private void actionPrint() {
        // get PrinterJob
        PrinterJob job = PrinterJob.getPrinterJob();
        MyPrintable printable = new MyPrintable(job.defaultPage(), layoutPanel);

        // setup Printable, Pageable
        job.setPrintable(printable);
        job.setPageable(printable);

        // display print dialog and print
        if (job.printDialog()) {
            try {
                job.print();
            } catch (PrinterException e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * The 'Clear' menu handler.
     */
    private void actionClear() {
        layoutPanel.inhNList.clear();
        layoutPanel.activeNList.clear();
        layoutPanel.probedNList.clear();

        Graphics g = layoutPanel.getGraphics();
        layoutPanel.writeToGraphics(g);
    }

    /**
     * The 'Bigger cells' menu handler.
     */
    private void actionBiggerCells() {
        layoutPanel.changeCellSize(true);
    }

    /**
     * The 'Smaller cells' menu handler.
     */
    private void actionSmallerCells() {
        layoutPanel.changeCellSize(false);
    }

    /**
     * The 'Modify size...' menu handler.
     */
    private void actionModifySize() {
        CSizePanel csizePanel = new CSizePanel(layoutPanel.getLayoutSize(),
                false);
        int result = JOptionPane.showConfirmDialog(this, csizePanel,
                "Modify layout size", JOptionPane.OK_CANCEL_OPTION);
        if (result == JOptionPane.OK_OPTION) { // Afirmative
            try {
                int sizeX = Integer
                        .parseInt(csizePanel.tfields[CSizePanel.idxX].getText());
                int sizeY = Integer
                        .parseInt(csizePanel.tfields[CSizePanel.idxY].getText());
                if (sizeX >= LayoutPanel.minXCells
                        && sizeX <= LayoutPanel.maxXCells
                        && sizeY >= LayoutPanel.minYCells
                        && sizeY <= LayoutPanel.maxYCells) {
                    RepType rtype = RepType.CLEAR;
                    if (csizePanel.newButton.isSelected()) {
                        rtype = RepType.CLEAR;
                    } else if (csizePanel.rptButton.isSelected()) {
                        rtype = RepType.REPEAT;
                    } else if (csizePanel.altButton.isSelected()) {
                        rtype = RepType.ALT;
                    }

                    changeLayoutSize(new Dimension(sizeX, sizeY), rtype);
                } else {
                    JOptionPane.showMessageDialog(null,
                            "Size x must be between " + LayoutPanel.minXCells
                            + " and " + LayoutPanel.maxXCells
                            + ", or size y must be between "
                            + LayoutPanel.minYCells + " and "
                            + LayoutPanel.maxYCells + ".");
                }
            } catch (NumberFormatException ne) {
                JOptionPane.showMessageDialog(null, "Invalid number.");
            }
        }
    }

    /**
     * The 'Generate pattern' menu handler.
     */
    private void actionGeneratePattern() {
        GPatternPanel gpatPanel = new GPatternPanel();
        int result = JOptionPane.showConfirmDialog(this, gpatPanel,
                "Generate pattern", JOptionPane.OK_CANCEL_OPTION);
        if (result == JOptionPane.OK_OPTION) { // Afirmative
            try {
                float ratioInh = Float
                        .parseFloat(gpatPanel.tfields[GPatternPanel.idxINH]
                                .getText());
                float ratioAct = Float
                        .parseFloat(gpatPanel.tfields[GPatternPanel.idxACT]
                                .getText());

                // validate ratios
                if ((ratioInh < 0 || ratioInh > 1.0)
                        || (ratioAct < 0 || ratioAct > 1.0)
                        || (ratioInh + ratioAct > 1.0)) {
                    throw new NumberFormatException();
                }

                if (gpatPanel.btns[GPatternPanel.idxREG].isSelected()) {
                    genRegularPattern(ratioInh, ratioAct);
                } else if (gpatPanel.btns[GPatternPanel.idxRND].isSelected()) {
                    genRandomPattern(ratioInh, ratioAct);
                }

                Graphics g = layoutPanel.getGraphics();
                layoutPanel.writeToGraphics(g);
            } catch (NumberFormatException ne) {
                JOptionPane.showMessageDialog(null, "Invalid ratio.");
            }
        }
    }

    /**
     * The 'Arrange probes' menu handler.
     */
    private void actionArrangeProbes() {
        AProbesPanel aprbPanel = new AProbesPanel();
        int result = JOptionPane.showConfirmDialog(this, aprbPanel,
                "Arrange probes", JOptionPane.OK_CANCEL_OPTION);
        if (result == JOptionPane.OK_OPTION) { // Afirmative
            try {
                int numProbes = Integer.parseInt(aprbPanel.tfield.getText());

                // validate number
                Dimension dim = layoutPanel.getLayoutSize();
                if (numProbes > dim.height * dim.width) {
                    throw new NumberFormatException();
                }

                genProbes(numProbes);

                Graphics g = layoutPanel.getGraphics();
                layoutPanel.writeToGraphics(g);
            } catch (NumberFormatException ne) {
                JOptionPane.showMessageDialog(null, "Invalid number.");
            }
        }
    }

    /**
     * The 'Statistical data...' menu handler.
     */
    private void actionStatisticalData() {
        String message = getStatisticalMsg(true);

        JOptionPane.showMessageDialog(null, message, "Statistical data",
                JOptionPane.PLAIN_MESSAGE);
    }

    /*
     * (non-Javadoc)
     * 
     * @see
     * java.awt.event.ActionListener#actionPerformed(java.awt.event.ActionEvent)
     */
    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == importItem) { // Import
            actionImport();
        } else if (e.getSource() == exportItem) { // Export
            actionExport();
        } else if (e.getSource() == printItem) { // Print
            actionPrint();
        } else if (e.getSource() == clearItem) { // Clear
            actionClear();
        } else if (e.getSource() == bcellItem) { // Bigger cells
            actionBiggerCells();
        } else if (e.getSource() == scellItem) { // Smaller cells
            actionSmallerCells();
        } else if (e.getSource() == csizeItem) { // Change size
            actionModifySize();
        } else if (e.getSource() == gpatItem) { // Generate pattern
            actionGeneratePattern();
        } else if (e.getSource() == aprbItem) { // Arrange probes
            actionArrangeProbes();
        } else if (e.getSource() == sdatItem) { // Statistical data
            actionStatisticalData();
        } else if (e.getSource() == exitItem) {
            dispose(); // modified for workbench
        }
    }

    // main function - start from here
    public static void main(String args[]) {
        int sizeX;
        int sizeY;

        // show change size window to set initial size
        CSizePanel csizePanel = new CSizePanel(new Dimension(
                LayoutPanel.defXCells, LayoutPanel.defYCells), true);
        while (true) {
            int result = JOptionPane.showConfirmDialog(null, csizePanel,
                    "Set initial size", JOptionPane.OK_CANCEL_OPTION);
            if (result == JOptionPane.OK_OPTION) { // Afirmative
                try {
                    sizeX = Integer
                            .parseInt(csizePanel.tfields[CSizePanel.idxX]
                                    .getText());
                    sizeY = Integer
                            .parseInt(csizePanel.tfields[CSizePanel.idxY]
                                    .getText());
                    if (sizeX >= LayoutPanel.minXCells
                            && sizeX <= LayoutPanel.maxXCells
                            && sizeY >= LayoutPanel.minYCells
                            && sizeY <= LayoutPanel.maxYCells) {
                        break;
                    } else {
                        JOptionPane.showMessageDialog(null,
                                "Size x must be between "
                                + LayoutPanel.minXCells + " and "
                                + LayoutPanel.maxXCells
                                + ", or size y must be between "
                                + LayoutPanel.minYCells + " and "
                                + LayoutPanel.maxYCells + ".");
                    }
                } catch (NumberFormatException e) {
                    JOptionPane.showMessageDialog(null, "Invalid number.");
                }
            } else { // cancel - exit the program
                System.exit(0);
            }
        }

        // create a control window
        ControlFrame frame = new ControlFrame("Growth Simulation Layout Editor");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.init(sizeX, sizeY);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }

    public static void runNLEdit(WorkbenchControlFrame parentFrame, WorkbenchManager workbenchMgr) {
        int sizeX;
        int sizeY;

        // show change size window to set initial size
        CSizePanel csizePanel = new CSizePanel(new Dimension(
                LayoutPanel.defXCells, LayoutPanel.defYCells), true);
        while (true) {
            int result = JOptionPane.showConfirmDialog(null, csizePanel,
                    "Set initial size", JOptionPane.OK_CANCEL_OPTION);
            if (result == JOptionPane.OK_OPTION) { // Afirmative
                try {
                    sizeX = Integer
                            .parseInt(csizePanel.tfields[CSizePanel.idxX]
                                    .getText());
                    sizeY = Integer
                            .parseInt(csizePanel.tfields[CSizePanel.idxY]
                                    .getText());
                    if (sizeX >= LayoutPanel.minXCells
                            && sizeX <= LayoutPanel.maxXCells
                            && sizeY >= LayoutPanel.minYCells
                            && sizeY <= LayoutPanel.maxYCells) {
                        break;
                    } else {
                        JOptionPane.showMessageDialog(null,
                                "Size x must be between "
                                + LayoutPanel.minXCells + " and "
                                + LayoutPanel.maxXCells
                                + ", or size y must be between "
                                + LayoutPanel.minYCells + " and "
                                + LayoutPanel.maxYCells + ".");
                    }
                } catch (NumberFormatException e) {
                    JOptionPane.showMessageDialog(null, "Invalid number.");
                }
            } else { // cancel - exit the program
                return; // Old Code: System.exit(0);
            }
        }

        // create a control window
        ControlFrame frame = new ControlFrame("Growth Simulation Layout Editor",
                parentFrame, workbenchMgr);
        // modified for workbench
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.init(sizeX, sizeY);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }
}
