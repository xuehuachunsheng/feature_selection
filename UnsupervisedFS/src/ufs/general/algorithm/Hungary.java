package ufs.general.algorithm;

import java.util.Arrays;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

/**
 * The Hungary method solving allocating problem.
 * @author Yanxue
 *
 */
public class Hungary {

	Matrix graph;

	int n, m;

	int minMatchValue;

	Matrix mapMatrix;

	int[] mapIndices;

	public static final int MAX_ITE_NUM = 1000;
	
	public Hungary(Matrix pGraph) {
		graph = pGraph.plus(Ret.NEW, false, 0);
		n = (int) pGraph.getRowCount();
		m = (int) pGraph.getColumnCount();
		if (n != m) {
			graphSqureChange();
		}
	}

	private void graphSqureChange() {
		if (n < m) {
			graph = graph.appendVertically(Ret.LINK,
					Matrix.Factory.zeros(m - n, m));
		} else {
			graph = graph.appendHorizontally(Ret.LINK,
					Matrix.Factory.zeros(n, n - m));
		}
		n = (int) graph.getRowCount();
		m = n;
	}

	public void findMinMatch() {
		// Compute C'
		Matrix rowMinValue = graph.min(Ret.NEW, 1);
		Matrix tC = Matrix.Factory.emptyMatrix();

		for (int i = 0; i < n; i++) {
			tC = tC.appendVertically(Ret.LINK, graph.selectRows(Ret.LINK, i)
					.minus(rowMinValue.getAsInt(i, 0)));
		}

		Matrix columnMinValue = tC.min(Ret.NEW, 0);
		Matrix _tC = Matrix.Factory.emptyMatrix();
		for (int i = 0; i < m; i++) {
			_tC = _tC.appendHorizontally(
					Ret.LINK,
					tC.selectColumns(Ret.LINK, i).minus(
							columnMinValue.getAsInt(0, i)));
		}
		//System.out.println("C(1) computed");
		Matrix tMapMatrix = constructMapAndUpdate(_tC)[0];
		int tCount = 0;
		while (!isOptimal(tMapMatrix) && tCount++ < MAX_ITE_NUM) {
			Matrix[] tMatrix = constructMapAndUpdate(_tC);
			tMapMatrix = tMatrix[0];
			_tC = tMatrix[1];
		}
		
		mapMatrix = tMapMatrix;
		mapIndices = new int[n];
		Arrays.fill(mapIndices, -1);
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				if(mapMatrix.getAsInt(i, j) == 1) {
					mapIndices[i] = j;
					break;
				}
			}
		}
	}

	private Matrix[] constructMapAndUpdate(Matrix c) {
		Matrix tMap = Matrix.Factory.zeros(n, m);
		Matrix updateC = c.plus(Ret.NEW, false, 0);

		int[][] rowZeroIndices = getRowZeroIndices(c);

		int[] indexSequence = findMinToMaxRowZeroCountIndexSequence(rowZeroIndices);
		boolean[] rowComputed = new boolean[n];
		boolean[] columnComputed = new boolean[m];
		for (int i = 0; i < n; i++) {
			int currentRow = indexSequence[i];
			for (int j = 0; j < rowZeroIndices[currentRow].length; j++) {
				if (!columnComputed[rowZeroIndices[currentRow][j]]) {
					tMap.setAsInt(1, currentRow, rowZeroIndices[currentRow][j]);
					columnComputed[rowZeroIndices[currentRow][j]] = true;
					// 1) Flag for having bracket.
					rowComputed[currentRow] = true;
					break;
				}
			}
		}
		//System.out.println("C(1)\r\n" + tMap);
		
		if (isOptimal(tMap)) {
			return new Matrix[] { tMap, updateC };
		}
		// C' --> C''
		boolean[] rowFlag = new boolean[n];
		// 1)
		for (int i = 0; i < n; i++) {
			rowFlag[i] = !rowComputed[i];
		}
		//System.out.println("C(1): " + Arrays.toString(rowFlag));
		
		boolean[] columnFlag = new boolean[m];

		boolean[] _rowFlag = new boolean[n];
		boolean[] _columnFlag = new boolean[m];

		while (!Arrays.equals(_rowFlag, rowFlag)
				|| !Arrays.equals(_columnFlag, columnFlag)) {
			
			_rowFlag = rowFlag;
			_columnFlag = columnFlag;
			
			// 2) Flag column indices for all the zero elements in those
			// bracket-flaged row.
			for (int i = 0; i < n; i++) {
				// flaged row
				if (rowFlag[i]) {
					for (int j = 0; j < rowZeroIndices[i].length; j++) {
						columnFlag[rowZeroIndices[i][j]] = true;
					}
				}
			}
			//System.out.println("C(1)" + Arrays.toString(columnFlag));
			
			// 3) Flag row indices for those bracket-flaged elements in flaged
			// columns.
			for (int i = 0; i < m; i++) {
				if (columnFlag[i]) {
					for (int j = 0; j < n; j++) {
						if (tMap.getAsInt(j, i) == 1) {
							rowFlag[j] = true;
							break;
						}
					}
				}
			}
		}

		// 5) Find minimum element in those locations uncovered by lines.
		int tMinValue = Integer.MAX_VALUE;
		for (int i = 0; i < n; i++) {
			// skip row Lines
			if (!rowFlag[i]) {
				continue;
			}

			for (int j = 0; j < m; j++) {
				if (!columnFlag[j]) {
					if (c.getAsInt(i, j) < tMinValue) {
						tMinValue = c.getAsInt(i, j);
					}
				}
			}
		}

		// 6) Minus the minimum value for those flaged rows.
		for (int i = 0; i < n; i++) {
			if (rowFlag[i]) {
				for (int j = 0; j < m; j++) {
					updateC.setAsInt(updateC.getAsInt(i, j) - tMinValue, i, j);
				}
			}
		}
		// 6) Plus the minimum value for those flaged columns.
		for (int i = 0; i < m; i++) {
			if (columnFlag[i]) {
				for (int j = 0; j < n; j++) {
					updateC.setAsInt(updateC.getAsInt(j, i) + tMinValue, j, i);
				}
			}
		}
		
		return new Matrix[] { tMap, updateC };
	}

	private int[] findMinToMaxRowZeroCountIndexSequence(int[][] rowZeroIndices) {
		int[] tSequence = new int[n];
		int tIndex = 0;
		boolean[] rowComputed = new boolean[n];
		while (tIndex < n) {
			int minZeroCountIndex = 0;
			int minZeroCount = Integer.MAX_VALUE;

			for (int i = 0; i < n; i++) {
				if (rowComputed[i]) {
					continue;
				}

				if (rowZeroIndices[i].length < minZeroCount) {
					minZeroCount = rowZeroIndices[i].length;
					minZeroCountIndex = i;
				}

			}
			tSequence[tIndex++] = minZeroCountIndex;
			rowComputed[minZeroCountIndex] = true;
		}
		return tSequence;
	}

	private int[][] getRowZeroIndices(Matrix c) {

		int[][] tRowZeroIndices = new int[n][];
		int[] tRowZeroCounts = new int[n];

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				if (c.getAsInt(i, j) == 0) {
					tRowZeroCounts[i]++;
				}
			}
		}

		for (int i = 0; i < n; i++) {
			tRowZeroIndices[i] = new int[tRowZeroCounts[i]];
			tRowZeroCounts[i] = 0;
			for (int j = 0; j < m; j++) {
				if (c.getAsInt(i, j) == 0) {
					tRowZeroIndices[i][tRowZeroCounts[i]++] = j;
				}
			}
		}

		return tRowZeroIndices;
	}

	/**
	 * Judge if the map matrix is optimal.
	 * 
	 * @param mapC
	 * @return
	 */
	private boolean isOptimal(Matrix mapC) {
		return mapC.sum(Ret.NEW, Matrix.ALL, false).getAsInt(0, 0) == n;
	}
	
	public int[] getMapIndices() {
		return mapIndices;
	}

	public static void main(String[] args) {
		int[][] m = null;
		m = new int[][]{ 
				{ 12, 7, 9, 7, 9 }, 
				{ 8, 9, 6, 6, 6 },
				{ 7, 17, 12, 14, 9 }, 
				{ 15, 14, 6, 6, 10 }, 
				{ 4, 10, 7, 10, 9 } 
		};
		m = new int[][]{
				{2, 15, 13, 4}, 
				{10, 4, 14, 15},
				{9, 14, 16, 13},
				{7, 8, 11, 9}, 
		};
		Matrix mMatrix = Matrix.Factory.zeros(m.length, m[0].length);
		
		for (int i = 0; i < m.length; i++) {
			for (int j = 0; j < m[i].length; j++) {
				mMatrix.setAsInt(m[i][j], i, j);
			}
		}
		
		Hungary h = new Hungary(mMatrix);
		h.findMinMatch();
	}
}
