package edu.uwb.braingrid.workbenchdashboard.nledit;

import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.print.PageFormat;
import java.awt.print.Pageable;
import java.awt.print.Printable;
import java.awt.print.PrinterException;

/**
 * The MyPrintable handles printing function.
 * 
 * @author Fumitaka Kawasaki
 * @version 1.2
 */
public class MyPrintable implements Printable, Pageable {
	private PageFormat pf;
	private LayoutPanel layoutPanel;
	private NL_Sim_Util nl_sim_util_;

	public MyPrintable(LayoutPanel layoutPanel, NL_Sim_Util nl_sim_util) {
		this.layoutPanel = layoutPanel;
		nl_sim_util_ = nl_sim_util;
	}

	public MyPrintable(PageFormat pf, LayoutPanel layoutPanel, NL_Sim_Util nl_sim_util) {
		this.pf = pf;
		this.layoutPanel = layoutPanel;
		nl_sim_util_ = nl_sim_util;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.awt.print.Printable#print(java.awt.Graphics,
	 * java.awt.print.PageFormat, int)
	 */
	public int print(Graphics g, PageFormat pf, int pageIndex)
			throws PrinterException {
		if (pageIndex == 0) {
			// the first page, draw layout
			int imWidth = (int) pf.getImageableWidth();
			int imHeight = (int) pf.getImageableHeight();
			int cellWidth = layoutPanel.getCellWidth();
			Dimension dim = layoutPanel.getLayoutSize();
			int xlen = dim.width;
			int ylen = dim.height;
			int lWidth = xlen * cellWidth; // layout width
			int lHeight = ylen * cellWidth; // layout height

			int tcWidth = cellWidth;
			if (lWidth > imWidth) {
				tcWidth = imWidth / xlen;
			}
			if (lHeight > imHeight) {
				if (imHeight / ylen < tcWidth) {
					tcWidth = imHeight / ylen;
				}
			}

			// adjust cell size to fit the page
			if (tcWidth < cellWidth) {
				layoutPanel.setCellWidth(tcWidth);
			}

			// shift to allocate margin, and write layout
			g.translate((int) pf.getImageableX(), (int) pf.getImageableY());
			layoutPanel.writeToGraphics(g);

			// restore to original cell size
			if (tcWidth < cellWidth) {
				layoutPanel.setCellWidth(cellWidth);
			}

			return Printable.PAGE_EXISTS;
		}
		if (pageIndex == 1) {
			// the second page, draw statistical data
			String msg = nl_sim_util_ .getStatisticalMsg(false);
			drawStringMultiLine((Graphics2D) g, pf, msg);

			return Printable.PAGE_EXISTS;
		} else {
			return Printable.NO_SUCH_PAGE;
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.awt.print.Pageable#getNumberOfPages()
	 */
	public int getNumberOfPages() {
		return 2; // number of pages
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.awt.print.Pageable#getPageFormat(int)
	 */
	public PageFormat getPageFormat(int pageIndex)
			throws IndexOutOfBoundsException {
		return pf; // return page format
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.awt.print.Pageable#getPrintable(int)
	 */
	public Printable getPrintable(int pageIndex)
			throws IndexOutOfBoundsException {
		return this;
	}

	/**
	 * draw multi-line string
	 * 
	 * @param g
	 *            graphics object
	 * @param pf
	 *            page format
	 * @param str
	 *            string to draw
	 */
	private void drawStringMultiLine(Graphics2D g, PageFormat pf, String str) {
		int x = (int) pf.getImageableX();
		int y = (int) pf.getImageableY();

		for (String line : str.split("\n")) {
			g.drawString(line, x, y += g.getFontMetrics().getHeight());
		}
	}
}
