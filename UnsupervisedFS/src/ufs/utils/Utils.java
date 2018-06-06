package ufs.utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;

import org.ujmp.core.Matrix;
import org.ujmp.core.SparseMatrix;
import org.ujmp.core.calculation.Calculation.Ret;
import org.ujmp.core.doublematrix.SparseDoubleMatrix;
import org.ujmp.core.util.DistanceMeasure;
import org.ujmp.jmatio.ImportMatrixMAT;

import weka.clusterers.EM;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.DistanceFunction;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class Utils {

	public static Matrix testMatrix = null;

	static {
		/**
		 * 4, 3; 3, 4; 0, -1.
		 */
		testMatrix = Matrix.Factory.zeros(3, 2);
		testMatrix.setAsDouble(4, 0, 0);
		testMatrix.setAsDouble(3, 0, 1);

		testMatrix.setAsDouble(3, 1, 0);
		testMatrix.setAsDouble(4, 1, 1);

		testMatrix.setAsDouble(0, 2, 0);
		testMatrix.setAsDouble(-1, 2, 1);
	}

	public static Scanner scanner = new Scanner(System.in);

	public static Random random = new Random();

	// Sorting order
	public enum Order {
		ASC, DESC
	}

	/**
	 * sort an array but return the indices. E.g. by given array [3, 4, 2, 1],
	 * this method will return [3, 2, 0, 1]. If it is descending order, it will
	 * return [1, 0, 2, 3]
	 * 
	 * @param valueArray
	 *            The given value array.
	 * @param o
	 *            If the sort is ascending or descending.
	 * @param k
	 *            Obtain the pre k-th indices. k <= |valueArray| and k > 0
	 * @return the pre k-th indices of sorted array.
	 */
	public static int[] argSort(double[] valueArray, Order o, int k) {
		return Arrays.copyOf(argSort(valueArray, o), k);
	}

	/**
	 * Compute the neighbors indices by L1 norm. It uses insertion method
	 * instead of sorting method.
	 * 
	 * @param originalSamples
	 *            The original samples. Each row of it is a sample. We mark it
	 *            as X ∈ R^(n1*m), thus the number of samples is n1.
	 * @param rowSamples
	 *            The given samples that you need to compute neighbors indices
	 *            in the original samples. We mark it as Y ∈ R^(n2*m), thus the
	 *            number of samples is n2.
	 * @param k
	 *            the number of neighbors.
	 * @return A indices matrix M ∈ N^(n2*k), the i-th row of which stores the k
	 *         neighbors indices of the i-th row of Y. Note that M_ij ∈ {0, 1,
	 *         2, ..., n1-1}.
	 */
	// public static Matrix kNeighborsIndicesMatrix(Matrix originalSamples,
	// Matrix rowSamples, int k) {
	//
	// Matrix resultM = Matrix.Factory.zeros(rowSamples.getRowCount(), k);
	//
	// for (int i = 0; i < rowSamples.getRowCount(); i++) {
	// System.out.print("The neighbors of " + i + "th instance: ");
	// int[] neighborIndices = new int[k];
	// double[] neighborDistances = new double[k];
	// Arrays.fill(neighborIndices, -1);
	// Arrays.fill(neighborDistances, Double.MAX_VALUE);
	// Matrix currentRow = rowSamples.selectRows(Ret.LINK, i);
	// for (int j = 0; j < originalSamples.getRowCount(); j++) {
	// // filter the equivalent vector.
	// // We do not consider itself.
	// if (originalSamples.selectRows(Ret.LINK,
	// j).equals(rowSamples.selectRows(Ret.LINK, i))) {
	// continue;
	// }
	// double distance = currentRow.minus(originalSamples.selectRows(Ret.LINK,
	// j)).norm1();
	//
	// if (distance >= neighborDistances[k - 1]) {
	// continue;
	// }
	// int m = 0;
	// while (m < k) {
	// if (distance < neighborDistances[m]) {
	// break;
	// }
	// m++;
	// }
	// for (int l = k - 2; l >= m; l--) {
	// neighborIndices[l + 1] = neighborIndices[l];
	// neighborDistances[l + 1] = neighborDistances[l];
	// }
	// neighborIndices[m] = j;
	// neighborDistances[m] = distance;
	// }
	// for (int j = 0; j < neighborIndices.length; j++) {
	// resultM.setAsInt(neighborIndices[j], i, j);
	// }
	// System.out.println(Arrays.toString(neighborIndices));
	// }
	//
	// return resultM;
	// }

	/**
	 * Compute the farthest instance's indices by L1 norm. It uses insertion
	 * method instead of sorting method.
	 * 
	 * @param originalSamples
	 *            The original samples. Each row of it is a sample. We mark it
	 *            as X ∈ R^(n1*m), thus the number of samples is n1.
	 * @param rowSamples
	 *            The given samples that you need to compute neighbors indices
	 *            in the original samples. We mark it as Y ∈ R^(n2*m), thus the
	 *            number of samples is n2.
	 * @param k
	 *            the number of neighbors.
	 * @return A indices matrix M ∈ N^(n2*k), the i-th row of which stores the k
	 *         farthest instance indices of the i-th row of Y. Note that M_ij ∈
	 *         {0, 1, 2, ..., n1-1}.
	 */
	public static Matrix kFarthestInstanceIndices(Matrix originalSamples,
			Matrix rowSamples, int k) {

		Matrix resultM = Matrix.Factory.zeros(rowSamples.getRowCount(), k);

		for (int i = 0; i < rowSamples.getRowCount(); i++) {
			System.out
					.print("The farest neighborses of " + i + "th instance: ");
			int[] farObjIndices = new int[k];
			double[] farObjDistances = new double[k];
			Arrays.fill(farObjIndices, -1);
			Arrays.fill(farObjDistances, Double.MIN_VALUE);
			Matrix currentRow = rowSamples.selectRows(Ret.LINK, i);
			for (int j = 0; j < originalSamples.getRowCount(); j++) {
				// filter the equivalent vector.
				// We do not consider itself.
				if (originalSamples.selectRows(Ret.LINK, j).equals(
						rowSamples.selectRows(Ret.LINK, i))) {
					continue;
				}
				double distance = currentRow.minus(
						originalSamples.selectRows(Ret.LINK, j)).normF();

				if (distance <= farObjDistances[k - 1]) {
					continue;
				}
				int m = 0;
				while (m < k) {
					if (distance > farObjDistances[m]) {
						break;
					}
					m++;
				}
				for (int l = k - 2; l >= m; l--) {
					farObjIndices[l + 1] = farObjIndices[l];
					farObjDistances[l + 1] = farObjDistances[l];
				}
				farObjIndices[m] = j;
				farObjDistances[m] = distance;
			}
			for (int j = 0; j < farObjIndices.length; j++) {
				resultM.setAsInt(farObjIndices[j], i, j);
			}
			System.out.println(Arrays.toString(farObjIndices));
		}

		return resultM;
	}

	public static File writeIntArrayToTxt(int[] array, String fileName) {
		try {
			File f = new File(fileName);
			if (!f.exists()) {

				f.createNewFile();

				// } else {
				// System.out.println("The file is exists. Do you want to override it?Y/N");
				// String s = scanner.next();
				// if (!(s.equals("Y") || s.equals("y"))) {
				// System.out.println("Not override");
				// return null;
				// }
			}
			StringBuffer sbBuffer = new StringBuffer();
			for (int i = 0; i < array.length; i++) {
				sbBuffer.append(array[i] + "\r\n");
			}
			BufferedWriter bw = new BufferedWriter(new FileWriter(f));
			bw.write(sbBuffer.toString());
			bw.close();
			return f;
		} catch (IOException e) {
			e.printStackTrace();
		}
		return null;
	}

	/**
	 * sort an array but return the indices. E.g. by given array [3, 4, 2, 1],
	 * this method will return [3, 2, 0, 1]. If it is descending order, it will
	 * return [1, 0, 2, 3]
	 * 
	 * @param valueArray
	 *            The given value array.
	 * @param o
	 *            If the sort is ascending or descending.
	 * @return the indices of sorted array.
	 */
	public static int[] argSort(double[] valueArray, Order o) {
		Map.Entry<Integer, Double>[] entries = new MyEntry[valueArray.length];
		for (int i = 0; i < entries.length; i++) {
			entries[i] = new MyEntry(i, o == Order.DESC ? -valueArray[i]
					: valueArray[i]);
		}
		// lambda function
		Arrays.sort(entries, (o1, o2) -> o1.getValue().compareTo(o2.getValue()));
		int[] indices = new int[valueArray.length];
		for (int i = 0; i < valueArray.length; i++) {
			indices[i] = entries[i].getKey();
		}
		return indices;
	}

	/**
	 * Compute the neighbors indices by l2 norm. It uses insertion method
	 * instead of sorting method.
	 * 
	 * @param originalSamples
	 *            The original samples. Each row of it is a sample. We mark it
	 *            as X ∈ R^(n1*m), thus the number of samples is n1.
	 * @param rowSamples
	 *            The given samples that you need to compute neighbors indices
	 *            in the original samples. We mark it as Y ∈ R^(n2*m), thus the
	 *            number of samples is n2.
	 * @param k
	 *            the number of neighbors.
	 * @return A indices matrix M ∈ N^(n2*k), the i-th row of which stores the k
	 *         neighbors indices of the i-th row of Y. Note that M_ij ∈ {0, 1,
	 *         2, ..., n1-1}.
	 */
	public static Matrix kNeighborsIndicesMatrix(Matrix originalSamples,
			Matrix rowSamples, int k) {
		int n = (int) originalSamples.getRowCount();
		int m = (int) rowSamples.getRowCount();
		Matrix neighbors = Matrix.Factory.zeros(m, k);
		int i = 0;

		while (i < m) {
			MyEntry<Integer, Double>[] t = new MyEntry[n];

			for (int j = 0, counter = 0; j < n; j++) {
				// Compute normF
				t[counter++] = new MyEntry<>(j, rowSamples
						.selectRows(Ret.LINK, i)
						.minus(originalSamples.selectRows(Ret.LINK, j)).normF());

			}

			Arrays.sort(t, (o1, o2) -> o1.getValue().compareTo(o2.getValue()));
//			System.out.print("The neighbors of the " + i + "th index: ");
			for (int j = 0; j < k; j++) {
				neighbors.setAsInt(t[j].getKey(), i, j);
//				System.out.print(t[j].getKey() + " ");
			}
//			System.out.println();
			i++;
		}
		return neighbors;
	}

	/**
	 * Return k-nearest neighbors of each row. The neighbors of i-th row does
	 * not contain itself. We use l2 norm method to measure the distance.
	 * 
	 * @param data
	 *            The data itself.
	 * @return neighborsIndices. result[i-1] stores the row numbers of k
	 *         neighbors in matrix of the i-st row.
	 */
	public static Matrix kNeighborsIndicesMatrix(Matrix data, int numNeighbors) {
		int n = (int) data.getRowCount();
		Matrix neighbors = Matrix.Factory.zeros(n, numNeighbors);
		int i = 0;

		while (i < n) {
			MyEntry<Integer, Double>[] t = new MyEntry[n-1];

			for (int j = 0, counter = 0; j < n; j++) {
				if (j != i) {
					// Compute normF
					t[counter++] = new MyEntry<>(j, data
							.selectRows(Ret.LINK, i)
							.minus(data.selectRows(Ret.LINK, j)).normF());
				}
			}

			Arrays.sort(t, (o1, o2) -> o1.getValue().compareTo(o2.getValue()));
			// System.out.print("The neighbors of the " + i + "th index: ");
			for (int j = 0; j < numNeighbors; j++) {
				neighbors.setAsInt(t[j].getKey(), i, j);
				// System.out.print(t[j].getKey() + " ");
			}
			// System.out.println();
			// System.out.println("Instance: " + i);
			i++;
		}
		return neighbors;
	}

	/**
	 * Return k-farthest instance of each row. We use L1 norm method to measure
	 * the distance.
	 * 
	 * @param data
	 *            The data itself.
	 * @return neighborsIndices. result[i-1] stores the row numbers of k
	 *         neighbors in matrix of the i-st row.
	 */
	public static Matrix farthestInstanceIndicesMatrix(Matrix data,
			int numFarthestInstances) {
		int n = (int) data.getRowCount();
		Matrix neighbors = Matrix.Factory.zeros(n, numFarthestInstances);
		int i = 0;

		while (i < n) {
			MyEntry<Integer, Double>[] t = new MyEntry[n - 1];

			for (int j = 0, counter = 0; j < n; j++) {
				if (j != i) {
					// Compute normF
					t[counter++] = new MyEntry(j, data.selectRows(Ret.LINK, i)
							.minus(data.selectRows(Ret.LINK, j)).normF());
				}
			}

			Arrays.sort(t, (o1, o2) -> o2.getValue().compareTo(o1.getValue()));
			System.out.print("The neighbors of the " + i + "th index: ");
			for (int j = 0; j < numFarthestInstances; j++) {
				neighbors.setAsInt(t[j].getKey(), i, j);
				System.out.print(t[j].getKey() + " ");
			}
			System.out.println();
			// System.out.println("Instance: " + i);
			i++;
		}
		return neighbors;
	}

	public static class MyEntry<K, V> implements Map.Entry<K, V> {
		K key;
		V value;

		public MyEntry(K key, V value) {
			super();
			this.key = key;
			this.value = value;
		}

		@Override
		public V getValue() {
			return value;
		}

		@Override
		public V setValue(V value) {
			V v = this.value;
			this.value = value;
			return v;
		}

		@Override
		public K getKey() {
			return key;
		}

		@Override
		public String toString() {
			return "(" + key + ", " + value + ")";
		}

	}

	/**
	 * Store a 2-d matrix to a arff file. The matrix stores all the data but not
	 * stores attribute name. The file would be like this type:
	 * 
	 * <pre>
	 * 	\@relation relationName
	 * 	\@attribute b1 real
	 *  \@attribute b2 real
	 *  ...
	 *  \@attribute bm real
	 *  \@data
	 *  matrix
	 * </pre>
	 * 
	 * @param matrix
	 *            the data matrix. R^n*m n rows and m columns.
	 * 
	 * @param fileName
	 *            the file name String.
	 * @throws IOException
	 */
	public static File writeMatrixToArffFile(Matrix matrix, String fileName,
			String relationName) throws Exception {
		File f = new File(fileName);
		if (!f.exists()) {
			f.createNewFile();
		} else {
			System.out
					.println("The file is exists. Do you want to override it?Y/N");
			String s = scanner.next();
			if (!(s.equals("Y") || s.equals("y"))) {
				System.out.println("Not override");
				return null;
			}
		}
		StringBuffer sbBuffer = new StringBuffer("@relation " + relationName
				+ "\r\n");

		int m = (int) matrix.getColumnCount();
		for (int i = 0; i < m; i++) {
			sbBuffer.append("@attribute b" + i + " real\r\n");
		}
		sbBuffer.append("@data\r\n");
		for (int i = 0; i < matrix.getRowCount(); i++) {
			sbBuffer.append(matrix.getAsString(i, 0));
			for (int j = 1; j < matrix.getColumnCount(); j++) {
				sbBuffer.append("," + matrix.getAsString(i, j));
			}
			sbBuffer.append("\r\n");
		}

		BufferedWriter bw = new BufferedWriter(new FileWriter(f));
		bw.write(sbBuffer.toString());
		bw.close();
		return f;
	}

	public static File writeStringToFile(String content, String fileName,
			boolean append) {
		File f = new File(fileName);
		try {
			if (!f.exists()) {
				f.createNewFile();
			}
			FileWriter fw = new FileWriter(fileName, append);
			fw.write(content);
			fw.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return f;
	}

	public static File writeMatrixToArffFileWithClassMessage(
			Matrix informationMatrix, Matrix classMessageColumnVector,
			String fileName, String relationName) throws Exception {
		File f = new File(fileName);
		if (!f.exists()) {
			f.createNewFile();
		} else {
			System.out
					.println("The file is exists. Do you want to override it?Y/N");
			String s = scanner.next();
			if (!(s.equals("Y") || s.equals("y"))) {
				System.out.println("Not override");
				return null;
			}
		}
		StringBuffer sbBuffer = new StringBuffer("@relation " + relationName
				+ "\r\n");
		// Add normal attributes
		int m = (int) informationMatrix.getColumnCount();
		for (int i = 0; i < m; i++) {
			sbBuffer.append("@attribute b" + i + " real\r\n");
		}
		// Add the class attribute
		sbBuffer.append("@attribute c {");
		{
			Set<Integer> clazzSet = new LinkedHashSet<>();
			for (int i = 0; i < classMessageColumnVector.getRowCount(); i++) {
				clazzSet.add(classMessageColumnVector.getAsInt(i, 0));
			}
			Integer[] clazzArray = clazzSet.toArray(new Integer[0]);
			sbBuffer.append(clazzArray[0]);
			for (int i = 1; i < clazzArray.length; i++) {
				sbBuffer.append("," + clazzArray[i]);
			}
		}
		// Add data
		sbBuffer.append("}\r\n@data\r\n");
		for (int i = 0; i < informationMatrix.getRowCount(); i++) {
			sbBuffer.append(informationMatrix.getAsString(i, 0));
			for (int j = 1; j < informationMatrix.getColumnCount(); j++) {
				sbBuffer.append("," + informationMatrix.getAsString(i, j));
			}
			sbBuffer.append("," + classMessageColumnVector.getAsInt(i, 0));
			sbBuffer.append("\r\n");
		}

		BufferedWriter bw = new BufferedWriter(new FileWriter(f));
		bw.write(sbBuffer.toString());
		bw.close();
		return f;
	}

	public static File writeMatrixToTxt(Matrix m, String fileName,
			String delimiter, Class<? extends Number> class_type)
			throws Exception {
		if (class_type == Integer.class) {
			return writeMatrixToTxtIntType(m, fileName, delimiter);
		}
		if (class_type == Double.class) {
			return writeMatrixToTxtDoubleType(m, fileName, delimiter);
		}
		return null;
	}

	private static File writeMatrixToTxtIntType(Matrix m, String fileName,
			String delimiter) throws Exception {
		File f = new File(fileName);
		if (!f.exists()) {
			f.createNewFile();
		} else {
			System.out
					.println("The file is exists. Do you want to override it?Y/N");
			String s = scanner.next();
			if (!(s.equals("Y") || s.equals("y"))) {
				System.out.println("Not override");
				return null;
			}
		}
		StringBuffer sbBuffer = new StringBuffer();
		for (int i = 0; i < m.getRowCount(); i++) {
			sbBuffer.append(m.getAsInt(i, 0));
			for (int j = 1; j < m.getColumnCount(); j++) {
				sbBuffer.append(delimiter + m.getAsInt(i, j));
			}
			sbBuffer.append("\r\n");
		}
		BufferedWriter bw = new BufferedWriter(new FileWriter(f));
		bw.write(sbBuffer.toString());
		bw.close();
		return f;
	}

	private static File writeMatrixToTxtDoubleType(Matrix m, String fileName,
			String delimiter) throws Exception {
		File f = new File(fileName);
		if (!f.exists()) {
			f.createNewFile();
		} else {
			System.out
					.println("The file is exists. Do you want to override it?Y/N");
			String s = scanner.next();
			if (!(s.equals("Y") || s.equals("y"))) {
				System.out.println("Not override");
				return null;
			}
		}
		StringBuffer sbBuffer = new StringBuffer();
		for (int i = 0; i < m.getRowCount(); i++) {
			sbBuffer.append(m.getAsDouble(i, 0));
			for (int j = 1; j < m.getColumnCount(); j++) {
				sbBuffer.append(delimiter + m.getAsDouble(i, j));
			}
			sbBuffer.append("\r\n");
		}
		BufferedWriter bw = new BufferedWriter(new FileWriter(f));
		bw.write(sbBuffer.toString());
		bw.close();
		return f;
	}

	/**
	 * Generate 2-D matrix from the given file. Default the element is double
	 * type. We do not check if the data are illegal.
	 * 
	 * @param f
	 *            The data file.
	 * @param delimiter
	 *            The delimiter among elements. Generally, the delimiter is ',',
	 *            ' ' or some other common symbols.
	 * @return 2-D matrix.
	 */
	public static Matrix load2DMatrixFromTxt(File f, String delimiter,
			Class<? extends Number> class_type) {
		if (class_type == Integer.class) {
			return load2DMatrixFromTxtIntType(f, delimiter);
		}
		if (class_type == Double.class) {
			return load2DMatrixFromTxtDoubleType(f, delimiter);
		}
		return null;
	}

	private static Matrix load2DMatrixFromTxtDoubleType(File f, String delimiter) {
		try {
			BufferedReader bf = new BufferedReader(new FileReader(f));
			String s = null;
			List<String[]> list = new ArrayList<>();
			while ((s = bf.readLine()) != null && !s.trim().equals("")) {
				list.add(s.trim().split(delimiter));
			}
			Matrix result = Matrix.Factory.zeros(list.size(),
					list.get(0).length);
			for (int i = 0; i < list.size(); i++) {
				for (int j = 0; j < list.get(i).length; j++) {
					result.setAsDouble(Double.parseDouble(list.get(i)[j]), i, j);
				}
			}
			return result;
		} catch (Exception e) {
			e.printStackTrace();
		}
		// throw new RuntimeException("Unknown Error occurred in
		// Utils.loadtxt(File, String).");
		return null;
	}

	private static Matrix load2DMatrixFromTxtIntType(File f, String delimiter) {
		try {
			BufferedReader bf = new BufferedReader(new FileReader(f));
			String s = null;
			List<String[]> list = new ArrayList<>();
			while ((s = bf.readLine()) != null) {
				list.add(s.trim().split(delimiter));
			}
			Matrix result = Matrix.Factory.zeros(list.size(),
					list.get(0).length);
			for (int i = 0; i < list.size(); i++) {
				for (int j = 0; j < list.get(i).length; j++) {
					result.setAsInt(Integer.parseInt(list.get(i)[j]), i, j);
				}
			}
			return result;
		} catch (Exception e) {
			e.printStackTrace();
		}
		// throw new RuntimeException("Unknown Error occurred in
		// Utils.loadtxt(File, String).");
		return null;
	}

	/**
	 * Generate 2-D sparse matrix from the given file. Default the element is
	 * double type. We do not check if the data are illegal. The type of data
	 * should be listed as follow:
	 * 
	 * <pre>
	 * 5[delimiter]3[delimiter]10 9[delimiter]100[delimiter]23.5 ...
	 * x[delimiter]y[delimiter]v
	 * 
	 * <pre>
	 * It represents that M[x, y] = v. The first column represents the
	 * x-location of an element. The second column represents the y-location of
	 * an element. The third column represents the value.
	 * 
	 * @param f
	 *            The data file.
	 * @param delimiter
	 *            The delimiter among x-location, y-location and the value.
	 *            Generally, the delimiter is ',', ' ' or some other common
	 *            symbols.
	 * @return 2-D matrix.
	 */
	public static SparseMatrix load2DSparseMatrixFromTxt(File f,
			String delimiter, long numRow, long numColumn,
			Class<? extends Number> class_type) {
		if (class_type == Integer.class) {
			return load2DSparseMatrixFromTxtIntType(f, delimiter, numRow,
					numColumn);
		}
		if (class_type == Double.class) {
			return load2DSparseMatrixFromTxtDoubleType(f, delimiter, numRow,
					numColumn);
		}
		return null;
	}

	private static SparseDoubleMatrix load2DSparseMatrixFromTxtDoubleType(
			File f, String delimiter, long numRow, long numColumn) {
		try {
			SparseDoubleMatrix result = SparseDoubleMatrix.Factory.zeros(
					numRow, numColumn);
			BufferedReader bf = new BufferedReader(new FileReader(f));
			String s = null;
			while ((s = bf.readLine()) != null && !s.equals("")) {
				String[] parts = s.trim().split(delimiter);
				result.setAsDouble(Double.parseDouble(parts[2]),
						Long.parseLong(parts[0]), Long.parseLong(parts[1]));
			}
			bf.close();
			return result;
		} catch (Exception e) {
			e.printStackTrace();
		}
		// throw new RuntimeException("Unknown Error occurred in
		// Utils.loadtxt(File, String).");
		return null;
	}

	public static Matrix loadMatrix2DFromMat(File f, String whichMatrix)
			throws IOException {
		return ImportMatrixMAT.fromFile(f, whichMatrix);
	}

	public static void writeMatrix2DToMat(File f, Matrix matrix, String whichMatrix)
			throws IOException {
		org.ujmp.jmatio.ExportMatrixMAT.toFile(f, matrix, whichMatrix);
		
	}
	
	public static SparseDoubleMatrix convertDensityMatrixToSparse(Matrix X) {
		SparseDoubleMatrix resultM = SparseDoubleMatrix.Factory.zeros(
				X.getRowCount(), X.getColumnCount());
		for (int i = 0; i < X.getRowCount(); i++) {
			for (int j = 0; j < X.getColumnCount(); j++) {
				resultM.setAsDouble(X.getAsDouble(i, j), i, j);
			}
		}
		return resultM;
	}

	private static SparseMatrix load2DSparseMatrixFromTxtIntType(File f,
			String delimiter, long numRow, long numColumn) {
		try {
			SparseMatrix result = SparseMatrix.Factory.zeros(numRow, numColumn);
			BufferedReader bf = new BufferedReader(new FileReader(f));
			String s = null;
			while ((s = bf.readLine()) != null && !s.equals("")) {
				String[] parts = s.trim().split(delimiter);
				result.setAsInt(Integer.parseInt(parts[2]),
						Long.parseLong(parts[0]), Long.parseLong(parts[1]));
			}
			bf.close();
			return result;
		} catch (Exception e) {
			e.printStackTrace();
		}
		// throw new RuntimeException("Unknown Error occurred in
		// Utils.loadtxt(File, String).");
		return null;
	}

	/**
	 * Generate 2-D sparse matrix from the given file. Default the element is
	 * double type. We do not check if the data are illegal. The type of data
	 * should be listed as follow:
	 * 
	 * <pre>
	 * 5[delimiter]3[delimiter]10 9[delimiter]100[delimiter]23.5 ...
	 * x[delimiter]y[delimiter]v
	 * 
	 * <pre>
	 * It represents that M[x, y] = v. The first column represents the
	 * x-location of an element. The second column represents the y-location of
	 * an element. The third column represents the value.
	 * 
	 * @param f
	 *            The data file.
	 * @param delimiter
	 *            The delimiter among x-location, y-location and the value.
	 *            Generally, the delimiter is ',', ' ' or some other common
	 *            symbols.
	 * @return 2-D matrix.
	 */
	public static File write2DSparseMatrixToTxt(SparseMatrix matrix,
			String fileName, String delimiter,
			Class<? extends Number> class_type) {
		if (class_type == Double.class) {
			return write2DSparseDoubleMatrixToTxt(matrix, fileName, delimiter);
		}
		if (class_type == Integer.class) {
			return write2DSparseIntMatrixToTxt(matrix, fileName, delimiter);
		}
		return null;
	}

	private static File write2DSparseIntMatrixToTxt(SparseMatrix matrix,
			String fileName, String delimiter) {
		try {
			File f = new File(fileName);
			if (!f.exists()) {
				f.createNewFile();
			} else {
				System.out
						.println("The file is exists. Do you want to override it?Y/N");
				String s = scanner.next();
				if (!(s.equals("Y") || s.equals("y"))) {
					System.out.println("Not override");
					return null;
				}
			}

			StringBuilder sbBuffer = new StringBuilder();

			for (long[] cor : matrix.availableCoordinates()) {
				sbBuffer.append(cor[0] + delimiter + cor[1] + delimiter
						+ matrix.getAsInt(cor[0], cor[1]) + "\r\n");
			}

			BufferedWriter bf = new BufferedWriter(new FileWriter(fileName));
			bf.write(sbBuffer.toString());
			bf.close();
			return f;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	private static File write2DSparseDoubleMatrixToTxt(SparseMatrix matrix,
			String fileName, String delimiter) {
		try {
			File f = new File(fileName);
			if (!f.exists()) {
				f.createNewFile();
			} else {
				System.out
						.println("The file is exists. Do you want to override it?Y/N");
				String s = scanner.next();
				if (!(s.equals("Y") || s.equals("y"))) {
					System.out.println("Not override");
					return null;
				}
			}

			StringBuilder sbBuffer = new StringBuilder();

			for (long[] cor : matrix.availableCoordinates()) {
				sbBuffer.append(cor[0] + delimiter + cor[1] + delimiter
						+ matrix.getAsDouble(cor[0], cor[1]) + "\r\n");
			}

			BufferedWriter bf = new BufferedWriter(new FileWriter(fileName));
			bf.write(sbBuffer.toString());
			bf.close();
			return f;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	@Deprecated
	public static class I implements Comparable<I> {
		int index;
		Comparable value;

		public I(int index, Comparable value) {
			this.index = index;
			this.value = value;
		}

		@Override
		public int compareTo(I o) {
			return value.compareTo(o.value);
		}

		public int getIndex() {
			return index;
		}

		public Comparable getValue() {
			return value;
		}

		@Override
		public String toString() {
			return "I [index=" + index + ", value=" + value + "]";
		}

	}

	/**
	 * The norm2 of two instances. An instance can be treated as a vector.
	 * 
	 * <pre>
	 *  ||a1 - a2||2
	 * </pre>
	 * 
	 * @param a1
	 * @param a2
	 * @return
	 */
	public static double norm2(Instance a1, Instance a2) {
		double dis = 0;
		for (int i = 0; i < a1.numAttributes(); i++) {
			dis += (a1.value(i) - a2.value(i)) * (a1.value(i) - a2.value(i));
		}
		return Math.sqrt(dis);
	}

	/**
	 * Obtain the negative number matrix of the given matrix. The result is -X
	 * 
	 * @param X
	 *            The sample matrix
	 * @return -X
	 */
	public static Matrix negativeMatrix(Matrix X) {
		int n = (int) X.getRowCount();
		int m = (int) X.getColumnCount();
		Matrix negX = Matrix.Factory.zeros(n, m);

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				negX.setAsDouble(-X.getAsDouble(i, j), i, j);
			}
		}

		return negX;
	}

	/**
	 * <pre>
	 * Return argmin ||data.instance(i) - a||2
	 *          i
	 * </pre>
	 * 
	 * @param data
	 * @param a
	 * @return
	 */
	public static int minNorm2Index(Instances data, Instance a) {

		double tMinDistance = Double.MAX_VALUE;
		int tMinDistanceIndex = 0;
		for (int j = 0; j < data.numInstances(); j++) {
			double tDistance = norm2(data.instance(j), a);
			if (tDistance < tMinDistance) {
				tMinDistance = tDistance;
				tMinDistanceIndex = j;
			}
		}
		return tMinDistanceIndex;
	}

	/**
	 * Obtain the max value and the index of the max value in a given array.
	 * 
	 * @param array
	 *            The given array.
	 * @return new int[]{a, b}. The a stores the max value and b stores the
	 *         index of max value.
	 */
	public static int[] maxValueAndIndex(int[] array) {
		int maxValue = Integer.MIN_VALUE;
		int maxValueIndex = 0;
		for (int i = 0; i < array.length; i++) {
			if (array[i] > maxValue) {
				maxValue = array[i];
				maxValueIndex = i;
			}
		}
		return new int[] { maxValue, maxValueIndex };
	}

	public double median(Matrix a) {
		return 0.0;
	}

	/**
	 * Instances to Matrix with label. The last column of the matrix is the
	 * label message.
	 */
	public static Matrix instancesToMatrixWithLabel(Instances pInstances) {
		Matrix resMat = Matrix.Factory.zeros(pInstances.numInstances(),
				pInstances.numAttributes());
		for (int i = 0; i < pInstances.numInstances(); i++) {
			for (int j = 0; j < pInstances.numAttributes() - 1; j++) {
				resMat.setAsDouble(pInstances.instance(i).value(j), i, j);
			}
			resMat.setAsInt((int) pInstances.instance(i).classValue(), i,
					pInstances.numAttributes() - 1);
		}
		return resMat;
	}

	/**
	 * Instances to Matrix with label. The last column of the matrix is the
	 * label message.
	 */
	public static Matrix instancesToMatrixWithoutLabel(Instances pInstances) {
		Matrix resMat = Matrix.Factory.zeros(pInstances.numInstances(),
				pInstances.numAttributes());
		for (int i = 0; i < pInstances.numInstances(); i++) {
			for (int j = 0; j < pInstances.numAttributes(); j++) {
				resMat.setAsDouble(pInstances.instance(i).value(j), i, j);
			}

		}
		return resMat;
	}

	/**
	 * Matrix to instances. The last column of the matrix is the label message.
	 * The value of each feature are type of numeric. But the label message must
	 * be type of integer. It is not suitable for the regression problem.
	 */
	public static Instances matrixWithLabelToInstances(Matrix pMatrix,
			String pRelation) {
		int n = (int) pMatrix.getRowCount();
		int m = (int) pMatrix.getColumnCount();
		Set<Integer> labels = new HashSet<>();
		for (int i = 0; i < n; i++) {
			labels.add(pMatrix.getAsInt(i, m - 1));
		}
		FastVector fv = new FastVector(m);
		for (int i = 0; i < m - 1; i++) {
			fv.addElement(new Attribute("a" + i));
		}
		FastVector fvClassVal = new FastVector(labels.size());
		for (Iterator<Integer> i = labels.iterator(); i.hasNext();) {
			fvClassVal.addElement(i.next() + "");
		}
		Attribute classAttr = new Attribute("clazz", fvClassVal);
		fv.addElement(classAttr);
		// Construct a empty instances.
		Instances instances = new Instances(pRelation, fv, n);
		for (int i = 0; i < n; i++) {
			Instance ins = new Instance(m);
			for (int j = 0; j < m - 1; j++) {
				ins.setValue((Attribute) fv.elementAt(j),
						pMatrix.getAsDouble(i, j));
			}
			ins.setValue((Attribute) fv.elementAt(m - 1),
					pMatrix.getAsInt(i, m - 1) + "");
			instances.add(ins);
		}
		instances.setClassIndex(m - 1);
		return instances;
	}

	/**
	 * Matrix to instances. The value of each feature are type of numeric.
	 */
	public static Instances matrixWithoutLabelToInstances(Matrix pMatrix,
			String pRelation) {
		int n = (int) pMatrix.getRowCount();
		int m = (int) pMatrix.getColumnCount();

		FastVector fv = new FastVector(m);
		for (int i = 0; i < m; i++) {
			fv.addElement(new Attribute("a" + i));
		}

		// Construct a empty instances.
		Instances instances = new Instances(pRelation, fv, n);
		for (int i = 0; i < n; i++) {
			Instance ins = new Instance(m);
			for (int j = 0; j < m; j++) {
				ins.setValue((Attribute) fv.elementAt(j),
						pMatrix.getAsDouble(i, j));
			}

			instances.add(ins);
		}
		return instances;
	}

	/**
	 * Centralize a matrix by the given axiom.
	 * 
	 * @param X
	 *            The given original matrix
	 * @param axiom
	 *            axiom = 0, centralize X by each column. axiom = 1, centralize
	 *            X by each row.
	 * @return
	 */
	public static Matrix centralize(Matrix X, int axiom) {
		Matrix meanX = X.mean(Ret.NEW, axiom, false);
		int n = (int) X.getRowCount();
		int m = (int) X.getColumnCount();
		Matrix centerX = Matrix.Factory.zeros(n, m);
		if (axiom == 0) {
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					centerX.setAsDouble(
							X.getAsDouble(j, i) - meanX.getAsDouble(0, i), j, i);
				}
			}
		} else { // axiom == 1
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < m; j++) {
					centerX.setAsDouble(
							X.getAsDouble(i, j) - meanX.getAsDouble(i, 0), i, j);
				}
			}
		}
		return centerX;
	}

	/**
	 * The multiply of two given sparse double matrix. return a×b
	 */
	@Deprecated
	public static SparseDoubleMatrix mtimes(Matrix a, Matrix b) {
		if (a.getColumnCount() != b.getRowCount()) {
			throw new RuntimeException("The matrix scale is illagel");
		}
		int n = (int) a.getRowCount();
		int m = (int) b.getColumnCount();
		SparseDoubleMatrix X = SparseDoubleMatrix.Factory.zeros(n, m);
		ArrayList<Integer>[] a_avaliable_row_indices = new ArrayList[n];
		for (int i = 0; i < a_avaliable_row_indices.length; i++) {
			a_avaliable_row_indices[i] = new ArrayList<>();
		}
		ArrayList<Double>[] a_avaliable_row_values = new ArrayList[n];
		for (int i = 0; i < a_avaliable_row_values.length; i++) {
			a_avaliable_row_values[i] = new ArrayList<>();
		}
		ArrayList<Integer>[] b_avaliable_column_indices = new ArrayList[m];
		for (int i = 0; i < b_avaliable_column_indices.length; i++) {
			b_avaliable_column_indices[i] = new ArrayList<>();
		}
		ArrayList<Double>[] b_avaliable_column_values = new ArrayList[m];
		for (int i = 0; i < b_avaliable_column_values.length; i++) {
			b_avaliable_column_values[i] = new ArrayList<>();
		}
		Iterator<long[]> aiterator = a.availableCoordinates().iterator();
		Iterator<long[]> biterator = b.availableCoordinates().iterator();
		while (aiterator.hasNext()) {
			long[] next = aiterator.next();
			a_avaliable_row_indices[(int) next[0]].add((int) next[1]);
			a_avaliable_row_values[(int) next[0]].add(a.getAsDouble(next[0],
					next[1]));
		}
		while (biterator.hasNext()) {
			long[] next = biterator.next();
			b_avaliable_column_indices[(int) next[1]].add((int) next[0]);
			b_avaliable_column_values[(int) next[1]].add(b.getAsDouble(next[0],
					next[1]));
		}
		for (int i = 0; i < a_avaliable_row_indices.length; i++) {
			List<Integer> a_row = a_avaliable_row_indices[i];
			for (int j = 0; j < b_avaliable_column_indices.length; j++) {
				List<Integer> b_column = b_avaliable_column_indices[j];
				double sum = 0;
				int a_point = 0;
				int b_point = 0;
				while (a_point < a_row.size() && b_point < b_column.size()) {
					if (a_row.get(a_point) < b_column.get(b_point)) {
						a_point++;
					} else if (a_row.get(a_point) > b_column.get(b_point)) {
						b_point++;
					} else {
						sum += a_avaliable_row_values[i].get(a_point)
								* b_avaliable_column_values[j].get(b_point);
						a_point++;
						b_point++;
					}
				}
				X.setAsDouble(sum, i, j);
			}
		}
		return X;
	}

	/**
	 * Return cluster results.
	 * 
	 * @param pData
	 * @param pNumClusters
	 * @return
	 * @throws Exception
	 */
	public static int[] kMeansClassify(Instances pData, int pNumClusters)
			throws Exception {
		// int numClusters = instances.attribute(instances.numAttributes() -
		// 1).numValues();
		Instances removeClassInstances = new Instances(pData);
		removeClassInstances.deleteAttributeAt(removeClassInstances
				.numAttributes() - 1);

		SimpleKMeans clusterer = new SimpleKMeans();
		DistanceFunction df = new weka.core.ManhattanDistance();
		df.setInstances(removeClassInstances);
		clusterer.setDistanceFunction(df);
		clusterer.setNumClusters(pNumClusters);

		// clusterer.setPreserveInstancesOrder(false);
		clusterer.buildClusterer(removeClassInstances);
		// System.out.println(clusterer.getClusterCentroids().instance(0));
		// int[] assignment = clusterer.getAssignments();
		int[] assignment = new int[pData.numInstances()];

		for (int i = 0; i < assignment.length; i++) {
			assignment[i] = clusterer.clusterInstance(removeClassInstances
					.instance(i));
		}
		return assignment;
	}

	/**
	 * Obtain the classify accuracy (purity) by K-means cluster.
	 * 
	 * @param instances
	 *            The given instances. It must have class attribute and the
	 *            index of the class attribute must be numAttribute - 1.
	 * @param numClusters
	 *            The number of clusters provided by users. If we know the
	 *            number of classes of the given instances, which is normally
	 *            equals with the parameter.
	 * 
	 * @return the accuracy of classify, but not cluster.
	 * @throws Exception
	 */
	public static double kMeansClassifyAccuracy(Instances instances,
			int numClusters) throws Exception {
		int[] assignment = kMeansClassify(instances, numClusters);
		// System.out.println(Arrays.toString(assignment));
		List<Integer>[] heaps = new ArrayList[numClusters];
		for (int i = 0; i < numClusters; i++) {
			heaps[i] = new ArrayList<>();
		}
		for (int i = 0; i < assignment.length; i++) {
			heaps[assignment[i]].add(i);
		}
		// Purity clustering accuracy
		int correctClassNum = 0;
		for (int i = 0; i < heaps.length; i++) {
			int[] classInstancesInCluster = new int[numClusters];
			for (int j = 0; j < heaps[i].size(); j++) {
				int classIndex = (int) instances.instance(heaps[i].get(j))
						.value(instances.numAttributes() - 1);
				if (classIndex < classInstancesInCluster.length) {
					classInstancesInCluster[classIndex]++;
				}
			}
			correctClassNum += Utils.maxValueAndIndex(classInstancesInCluster)[0];
		}
		return (correctClassNum + 0.0) / instances.numInstances();

	}

	/**
	 * Obtain the classify accuracy (purity) by K-means cluster.
	 * 
	 * @param instances
	 *            The given instances. It must have class attribute and the
	 *            index of the class attribute must be numAttribute - 1.
	 * @param numClusters
	 *            The number of clusters provided by users. If we know the
	 *            number of classes of the given instances, which is normally
	 *            equals with the parameter.
	 * 
	 * @return the accuracy of classify, but not cluster.
	 * @throws Exception
	 */
	public static double eMClassifyAccuracy(Instances instances, int numClusters)
			throws Exception {
		// int numClusters = instances.attribute(instances.numAttributes() -
		// 1).numValues();
		Instances removeClassInstances = new Instances(instances);
		removeClassInstances.deleteAttributeAt(removeClassInstances
				.numAttributes() - 1);

		EM clusterer = new EM();
		clusterer.setNumClusters(numClusters);

		clusterer.buildClusterer(removeClassInstances);
		List<Integer>[] heaps = new ArrayList[numClusters];
		int[] assignment = new int[instances.numInstances()];
		for (int i = 0; i < numClusters; i++) {
			heaps[i] = new ArrayList<>();
		}
		for (int i = 0; i < assignment.length; i++) {
			assignment[i] = clusterer.clusterInstance(removeClassInstances
					.instance(i));
		}
		// System.out.println(Arrays.toString(assignment));
		for (int i = 0; i < assignment.length; i++) {
			heaps[assignment[i]].add(i);
		}
		int correctClassNum = 0;
		for (int i = 0; i < heaps.length; i++) {
			int[] classInstancesInCluster = new int[numClusters];
			for (int j = 0; j < heaps[i].size(); j++) {
				int classIndex = (int) instances.instance(heaps[i].get(j))
						.value(instances.numAttributes() - 1);
				if (classIndex < classInstancesInCluster.length) {
					classInstancesInCluster[classIndex]++;
				}

			}
			correctClassNum += Utils.maxValueAndIndex(classInstancesInCluster)[0];
		}
		return (correctClassNum + 0.0) / instances.numInstances();

	}

	/**
	 * Obtain the distance matrix D of X, which is computed by given distance
	 * measure, which is replced by a matrix method.
	 * 
	 * @param X
	 *            X ∈ R^(n*d), The given sample matrix, where n is the number of
	 *            samples, and d is the number of features of a sample.
	 * @param dm
	 *            The distance measure, which can be ||X_i - X_j||_1, ||X_i -
	 *            X_j||_2 or ||X_i - X_j||_inf.
	 * @return D∈R^(n*n) where D_ij is the distance between X_i and X_j.
	 */
	@Deprecated
	public static Matrix distanceMatrix(Matrix X, DistanceMeasure dm) {
		double[][] xArray = X.toDoubleArray();
		int n = (int) X.getRowCount();
		Matrix distX = Matrix.Factory.zeros(n, n);
		for (int i = 0; i < n; i++) {
			for (int j = i; j < n; j++) {
				double distIJ = dm.getDistance(xArray[i], xArray[j]);
				distX.setAsDouble(distIJ, i, j);
				distX.setAsDouble(distIJ, j, i);
			}
		}
		return distX;
	}

	/**
	 * File content copy.
	 * 
	 * @param source
	 *            The source file
	 * @param target
	 *            The target file.
	 * @return true if file copy success.
	 */
	public static boolean fileCopy(File source, File target) {
		try {
			if (!target.exists()) {
				target.createNewFile();
			}

			FileInputStream inStream = new FileInputStream(source);
			FileChannel in = inStream.getChannel();
			FileOutputStream outStream = new FileOutputStream(target);
			FileChannel out = outStream.getChannel();
			in.transferTo(0, in.size(), out);
			inStream.close();
			in.close();
			outStream.close();
			out.close();
		} catch (Exception e) {
			e.printStackTrace();
			return false;
		}

		return true;
	}

	/**
	 **************************** 
	 * Random swap for a array by using the third type solution.
	 * 
	 * @param paraArray
	 *            the array you need to random swap.
	 **************************** 
	 */
	public static void randomSwap(int[] paraArray) {
		int tempRandNumber;
		int tempSwapVariable;
		for (int i = 1; i < paraArray.length; i++) {
			// tempRandNumber = tempRandomSeed.nextInt(paraArray.length);
			tempRandNumber = random.nextInt(i);
			// Swap
			tempSwapVariable = paraArray[tempRandNumber];
			paraArray[tempRandNumber] = paraArray[i];
			paraArray[i] = tempSwapVariable;
		} // Of for i
	}

	/**
	 * Generate a random permutation array by given lower bound and upper bound,
	 * the length of which is upperbound - lowerbound E.g. If the lowerBound is
	 * 5, and the upperBound is 10. The result is a random permutation of
	 * sequence [5, 6, 7, 8, 9], excluding 10.
	 * 
	 * @param lowerBound
	 * @param upperBound
	 * @return
	 */
	public static int[] randomPermutationArray(int lowerBound, int upperBound) {
		int[] array = new int[upperBound - lowerBound];
		for (int i = 0; i < array.length; i++) {
			array[i] = i + lowerBound;
		}
		randomSwap(array);
		return array;
	}

	public static void swap(int[] array, int i1, int i2) {
		int t = array[i1];
		array[i1] = array[i2];
		array[i2] = t;
	}

	public static long[] intArrayToLongType(int[] array) {
		long[] result = new long[array.length];
		for (int i = 0; i < result.length; i++) {
			result[i] = array[i];
		}
		return result;
	}

	public static void main(String[] args) throws Exception {
		// // Instances instances = new Instances(new FileReader(new
		// // File("src/data/common/iris.arff")));
		// //
		// // for (int i = 2; i < 11; i++) {
		// // System.out.println(kMeansClassifyAccuracy(instances, i));
		// // }
		// long c_time = System.currentTimeMillis();
		// Matrix X = Utils.load2DSparseMatrixFromTxt(new
		// File("src/data/basehock/review/train.data"), " ", 11269, 61188,
		// Integer.class);
		// Matrix W = Utils.load2DSparseMatrixFromTxt(
		// new
		// File("src/data/basehock/npfs/W/train_W,numNeighbors=500,rho=2.0,convergValue=0.01.data"),
		// " ",
		// 11269, 11269, Double.class);
		// int n = (int) X.getRowCount();
		// int d = (int) X.getColumnCount();
		// Matrix E_n_n_subW_T = SparseDoubleMatrix.Factory.eye(n,
		// n).minus(W).transpose();
		// double[] normValue = new double[(int) d];
		// for (int i = 0; i < d; i++) {
		// normValue[i] = E_n_n_subW_T.mtimes(X.selectColumns(Ret.LINK,
		// i)).norm2();
		// // normValue[i] = X_T.selectRows(Ret.LINK,
		// // i).mtimes(E_n_n_subW).norm2();
		// System.out.println("Norm value of " + i + "-th attribute");
		// }
		// int[] allIndices = Utils.argSort(normValue, Utils.Order.DESC);
		//
		// Utils.writeIntArrayToTxt(allIndices,
		// "src/data/basehock/features_desc/train_allfeatures_desc,numNeighbors=500,rho=2.0,convergValue=0.01.data");
		//
		// System.out.println("OK, Time costs: " + (System.currentTimeMillis() -
		// c_time) + "ms");
		System.out.println(testMatrix);
		System.out.println(centralize(testMatrix, 0));
		System.out.println(centralize(testMatrix, 1));

		System.out.println(Arrays.toString(argSort(new double[]{1, 2, 3, 4, 5}, Order.DESC)));
	}
}
