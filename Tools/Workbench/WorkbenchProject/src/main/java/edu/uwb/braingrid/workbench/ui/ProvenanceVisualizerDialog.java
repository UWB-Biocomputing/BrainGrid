package edu.uwb.braingrid.workbench.ui;

import org.graphstream.graph.Edge;
import org.graphstream.graph.Graph;
import org.graphstream.graph.Node;
import org.graphstream.graph.implementations.MultiGraph;
import org.graphstream.graph.implementations.SingleGraph;
import org.graphstream.ui.spriteManager.Sprite;
import org.graphstream.ui.spriteManager.SpriteManager;
import org.graphstream.ui.swingViewer.ViewPanel;
import org.graphstream.ui.view.Viewer;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;

public class ProvenanceVisualizerDialog extends JFrame {
    private Graph graph;
    SpriteManager sman ;
    private JPanel contentPane;
    private JButton buttonOK;
    private JButton buttonCancel;
    private JPanel visualizationPanel;
    private JButton hideBtn;
    private JButton showBtn;

    private Viewer viewer;
    private ViewPanel view;
    protected static String styleSheet =
            "node { size: 20px; fill-color: rgb(150,150,150); }" +
                    "node.important {fill-color: blue; size: 50px; text-alignment: under;}" +
                    "edge { fill-color: rgb(255,50,50); size: 2px; }" +
                    "edge.cut { fill-color: rgba(200,200,200,128);  text-alignment: along;}" +
                    "sprite#CC { size: 0px; text-color: rgb(150,100,100); text-size: 20; }" +
                    "sprite#M  { size: 0px; text-color: rgb(100,150,100); text-size: 20; }" ;

    public ProvenanceVisualizerDialog() {
        setTitle("Provenance Visualizer");
        setResizable(true);
        setContentPane(contentPane);
        graph = new MultiGraph("embedded");
        sman = new SpriteManager(graph);
        graph.addAttribute("ui.stylesheet", styleSheet);
        viewer = new Viewer(graph, Viewer.ThreadingModel.GRAPH_IN_ANOTHER_THREAD);
        viewer.enableAutoLayout();
        view = viewer.addDefaultView(false);
        view.setPreferredSize(new Dimension(300, 500));
        visualizationPanel.add(view);
        //setModal(true);
        getRootPane().setDefaultButton(buttonOK);
        setPreferredSize(new Dimension(300, 500));
        setExtendedState(MAXIMIZED_BOTH);

        Node nodeA = graph.addNode("A");
        nodeA.addAttribute("ui.class", "important");
        //Sprite s = sman.addSprite("ASprite");
        nodeA.addAttribute("ui.label", "hahaklsjdklgjkdfahgjkhfajkdhgfdjlsaf");
        //s.attachToNode("A");
        //nodeA.addAttribute("ui.style", "fill-mode: dyn-plain; fill-color: rgb(0,100,255); size: 50;");
        //nodeA.addAttribute("ui.size", 50);
        Node nodeB = graph.addNode("B");
        nodeB.setAttribute("ui.style", "fill-mode: dyn-plain; size: 30; fill-color: red;");
        Node nodeC = graph.addNode("C");
        //graph.addEdge("AB", "A", "B");
        Edge edge = graph.addEdge("BC", "B", "C");
        edge.addAttribute("ui.class", "cut");
        edge.addAttribute("ui.label", "Edge BC");
        graph.addEdge("CA", "C", "A");

        buttonOK.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                onOK();
            }
        });

        buttonCancel.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                onCancel();
            }
        });

        hideBtn.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                nodeC.addAttribute("ui.hide");
                edge.addAttribute("ui.hide");
            }
        });

        showBtn.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                nodeC.removeAttribute("ui.hide");
                edge.removeAttribute("ui.hide");
            }
        });

        // call onCancel() when cross is clicked
        setDefaultCloseOperation(DO_NOTHING_ON_CLOSE);
        addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent e) {
                onCancel();
            }
        });

        // call onCancel() on ESCAPE
        contentPane.registerKeyboardAction(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                onCancel();
            }
        }, KeyStroke.getKeyStroke(KeyEvent.VK_ESCAPE, 0), JComponent.WHEN_ANCESTOR_OF_FOCUSED_COMPONENT);
    }

    private void onOK() {
        // add your code here
        dispose();
    }

    private void onCancel() {
        // add your code here if necessary
        dispose();
    }

    public static void main(String[] args) {
        System.setProperty("org.graphstream.ui.renderer", "org.graphstream.ui.j2dviewer.J2DGraphRenderer");
        /*
        ProvenanceVisualizerDialog dialog = new ProvenanceVisualizerDialog();
        dialog.pack();
        dialog.setVisible(true);
        System.exit(0);
        */

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            @Override
            public void run() {
                new ProvenanceVisualizerDialog().setVisible(true);
            }
        });
    }
}
