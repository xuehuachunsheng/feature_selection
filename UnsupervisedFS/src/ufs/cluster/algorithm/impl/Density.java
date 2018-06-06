package ufs.cluster.algorithm.impl;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Stack;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import ufs.cluster.algorithm.Cluster;
import ufs.cluster.evaluate.EvaluationIndexType;
import ufs.utils.Utils;
import ufs.utils.Utils.Order;

/**
 * The basic implementation of Density Peaks. Author: Yanxue <br>
 * E-mail: yeah_imwyx@163.com <br>
 * Organization: <a href=http://www.fansmale.com>Lab of Machine Learning</a>
 * Written Time: Jan. 12, 2017 <br>
 * Last Modified Time: Jan. 12, 2017 <br>
 * Progress: Done.<br>
 */
public class Density extends Cluster {

	/**
	 * The dc.
	 */
	double dc;

	/**
	 * The density of each sample.
	 */
	int[] rho;

	/**
	 * The distance between a sample and its master.
	 */
	double[] delta;

	/**
	 * masterIndices[i] stores the master index of i-th sample.
	 */
	int[] masterIndices;

	/**
	 * The center indices mapping in the original sample set. <SampleIndex,
	 * ClusterIndex>
	 */
	int[] centerIndices;

	public Density(Matrix pData, int k) {
		super(pData, k);

		delta = new double[(int) pData.getRowCount()];
		Arrays.fill(delta, Integer.MAX_VALUE);

		masterIndices = new int[(int) pData.getRowCount()];
		Arrays.fill(masterIndices, -1);

		rho = new int[(int) pData.getRowCount()];
	}

	@Override
	public void cluster() {
		// Compute rho
		computeRho();

		// Compute masters
		computeMasters();

		// Compute centers
		computeCenters();

		// Compute clusters
		predictLabels = new int[(int) data.getRowCount()];

		boolean[] tIsComputed = new boolean[predictLabels.length];
		// <SampleIndex, ClusterLabel>
		Map<Integer, Integer> tCenterIndicesMapping = new HashMap<>();

		for (int i = 0; i < centerIndices.length; i++) {
			tCenterIndicesMapping.put(centerIndices[i], i);
		}

		int[] tMasterIndices = Arrays.copyOf(masterIndices,
				masterIndices.length);
		for (int i = 0; i < centerIndices.length; i++) {
			if (tCenterIndicesMapping.containsKey(centerIndices[i])) {
				tMasterIndices[centerIndices[i]] = -1;
			}
		}

		// By using stack to replace recursive progress
		Stack<Integer> tStack = new Stack<>();
		Loop: for (int i = 0; i < predictLabels.length; i++) {
			if (tIsComputed[i]) {
				continue;
			}
			int tIndices = i;
			while (tIndices != -1) {
				tStack.push(tIndices);
				tIndices = tMasterIndices[tIndices];
				if (tIndices != -1 && tIsComputed[tIndices]) {
					while (!tStack.isEmpty()) {
						int tIndices2 = tStack.pop(); 
						predictLabels[tIndices2] = predictLabels[tIndices];
						tIsComputed[tIndices2] = true;
					}
					continue Loop;
				}
			}
			int tFinalMasterIndex = tStack.pop();
			int tClusterLabel = tCenterIndicesMapping.get(tFinalMasterIndex);
			while (!tStack.isEmpty()) {
				tIndices = tStack.pop();
				predictLabels[tIndices] = tClusterLabel;
				tIsComputed[tIndices] = true;
			}
		}

	}

	private void computeCenters() {
		double[] tValues = new double[(int) data.getRowCount()];
		for (int i = 0; i < tValues.length; i++) {
			tValues[i] = rho[i] * delta[i];
		}
		centerIndices = Utils.argSort(tValues, Order.DESC, k);
		centers = data.selectRows(Ret.NEW, Utils.intArrayToLongType(centerIndices));
	}

	private void computeRho() {
		for (int i = 0; i < data.getRowCount() - 1; i++) {
			for (int j = i + 1; j < data.getRowCount(); j++) {
				if (data.selectRows(Ret.LINK, i)
						.minus(data.selectRows(Ret.LINK, j)).normF() < dc) {
					rho[i]++;
					rho[j]++;
				}
			}
		}
	}

	private void computeMasters() {
		for (int i = 0; i < data.getRowCount(); i++) {
			for (int j = 0; j < data.getRowCount(); j++) {
				if (rho[i] < rho[j]) {
					double tDist = data.selectRows(Ret.LINK, i)
							.minus(data.selectRows(Ret.LINK, j)).normF();
					if (tDist < delta[i]) {
						delta[i] = tDist;
						masterIndices[i] = j;
					}
				}
			}
		}
	}

	public double getDc() {
		return dc;
	}

	public void setDc(double dc) {
		this.dc = dc;
	}

	@Override
	public String toString() {
		return "Density [\r\ndc=" + dc + ", \r\nrho=" + Arrays.toString(rho)
				+ ", \r\ndelta=" + Arrays.toString(delta)
				+ ", \r\nmasterIndices=" + Arrays.toString(masterIndices)
				+ ", \r\ncenterIndices=" + Arrays.toString(centerIndices)
				+ ", \r\nk=" + k
				+ ", \r\npredictLabels=" + Arrays.toString(predictLabels)
				+ ", \r\nrealLabels=" + Arrays.toString(realLabels)
				+ ", \r\ncenters=" + centers + "\r\n]";
	}

	public static void main(String[] args) throws Exception {
		weka.core.Instances dataInstances = new weka.core.Instances(
				new java.io.FileReader("src/data/arff/iris.arff"));
		dataInstances.setClassIndex(dataInstances.numAttributes() - 1);
		Matrix dataWithLabel = Utils.instancesToMatrixWithLabel(dataInstances);
		Matrix dataWithoutLabel = dataWithLabel.deleteColumns(Ret.NEW,
				dataInstances.numAttributes() - 1);
		int[] pRealLabels = dataWithLabel
				.selectColumns(Ret.LINK, dataInstances.numAttributes() - 1)
				.transpose().toIntArray()[0];
		Density density = new Density(dataWithoutLabel, 3);
		density.setDc(0.71);
		density.cluster();

		density.setRealLabels(pRealLabels);

		System.out.println(density);

		System.out.println(Arrays.toString(density.getPredictLabels()));

		System.out.println("JC: " + density.getEvaluationResult(EvaluationIndexType.JC));
		
		System.out.println("FMI: " + density.getEvaluationResult(EvaluationIndexType.FMI));
		
		System.out.println("purity: " + density.getEvaluationResult(EvaluationIndexType.PURITY));
	}
}
