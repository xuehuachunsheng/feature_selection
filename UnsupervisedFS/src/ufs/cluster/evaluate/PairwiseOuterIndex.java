package ufs.cluster.evaluate;

import java.util.HashSet;
import java.util.Set;

import org.ujmp.core.Matrix;

/**
 * This class is the basic implementation of the pairwise considered outer
 * indices. <br>
 * 
 * Author: Yanxue <br>
 * E-mail: yeah_imwyx@163.com <br>
 * Organization: <a href=http://www.fansmale.com>Lab of Machine Learning</a>
 * Written Time: Jan. 12, 2017 <br>
 * Last Modified Time: Jan. 12, 2017 <br>
 * Progress: Done.<br>
 */
public abstract class PairwiseOuterIndex extends OuterIndex {

	/**
	 * It is assumed that h_i, h_i* represents the class label of the i'th
	 * sample(x_i) given by the model of cluster and the real, respectively. we
	 * assign
	 * 
	 * <pre>
	 * 	ss = {(x_i, x_j) | h_i = h_j, h_i* = h_j*, i < j}
	 *  sd = {(x_i, x_j) | h_i = h_j, h_i* != h_j*, i < j}
	 *  ds = {(x_i, x_j) | h_i != h_j, h_i* = h_j*, i < j}
	 *  dd = {(x_i, x_j) | h_i != h_j, h_i* != h_j*, i < j}
	 * </pre>
	 * 
	 */
	protected Set<Pairwise> ss, sd, ds, dd;

	public PairwiseOuterIndex(Matrix pData, int[] pPredictLabels,
			int[] pRealLabels) {
		super(pData, pPredictLabels, pRealLabels);
		paramConstGenerated();
	}

	public void paramConstGenerated() {

		Set<Pairwise> tSS = new HashSet<>();
		Set<Pairwise> tSD = new HashSet<>();
		Set<Pairwise> tDS = new HashSet<>();
		Set<Pairwise> tDD = new HashSet<>();

		for (int i = 0; i < realLabels.length; i++) {
			for (int j = i + 1; j < realLabels.length; j++) {
				if (predictLabels[i] == predictLabels[j]) {
					if (realLabels[i] == realLabels[j]) {
						tSS.add(new Pairwise(i, j));
					} else {
						tSD.add(new Pairwise(i, j));
					}
				} else {
					if (realLabels[i] == realLabels[j]) {
						tDS.add(new Pairwise(i, j));
					} else {
						tDD.add(new Pairwise(i, j));
					}
				} // Of if
			} // Of for j
		} // Of for i
		ss = tSS;
		sd = tSD;
		ds = tDS;
		dd = tDD;
	}

	public Set<Pairwise> getSs() {
		return ss;
	}

	public Set<Pairwise> getSd() {
		return sd;
	}

	public Set<Pairwise> getDs() {
		return ds;
	}

	public Set<Pairwise> getDd() {
		return dd;
	}


	private static class Pairwise {
		int i;
		int j;

		public Pairwise(int pI, int pJ) {
			i = pI;
			j = pJ;
		}

		@Override
		public int hashCode() {
			final int prime = 31;
			int result = 1;
			result = prime * result + i;
			result = prime * result + j;
			return result;
		}

		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (getClass() != obj.getClass())
				return false;
			Pairwise other = (Pairwise) obj;
			if (i != other.i)
				return false;
			if (j != other.j)
				return false;
			return true;
		}
	}
}
