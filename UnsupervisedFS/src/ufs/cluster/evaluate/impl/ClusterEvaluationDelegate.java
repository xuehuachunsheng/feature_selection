package ufs.cluster.evaluate.impl;

import java.io.File;
import java.io.FileReader;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import ufs.cluster.algorithm.Cluster;
import ufs.cluster.evaluate.ClusterEvaluation;
import ufs.cluster.evaluate.EvaluationIndexType;
import ufs.utils.Utils;
import weka.core.Instances;

/**
 * It is a delegate method that compute the special evaluation index.
 * 
 * Author: Yanxue <br>
 * E-mail: yeah_imwyx@163.com <br>
 * Organization: <a href=http://www.fansmale.com>Lab of Machine Learning</a>
 * Written Time: Jan. 12, 2017 <br>
 * Last Modified Time: Jan. 12, 2017 <br>
 * Progress: Done.<br>
 * 
 */
public class ClusterEvaluationDelegate implements ClusterEvaluation {

	Cluster cluster;

	EvaluationIndexType evaluationType;

	public ClusterEvaluationDelegate(Cluster pCluster,
			EvaluationIndexType pEvaluationType) {
		cluster = pCluster;
		evaluationType = pEvaluationType;
	}

	@Override
	public double precision() {
		ClusterEvaluation ce = null;
		switch (evaluationType) {
		case JC:
			ce = new JaccardCoefficient(cluster.getData(), cluster.getPredictLabels(), cluster.getRealLabels());
			break;
		case FMI:
			ce = new FowlkesMallowsIndex(cluster.getData(), cluster.getPredictLabels(), cluster.getRealLabels());
			break;
		case RI:
			ce = new RandIndex(cluster.getData(), cluster.getPredictLabels(), cluster.getRealLabels());
			break;
		case DI:
			ce = new DunnIndex(cluster.getData(), cluster.getData().euklideanDistance(Ret.NEW, false), cluster.getPredictLabels(), cluster.getCenters());
			break;
		case DBI:
			ce = new DaviesBouldinIndex(cluster.getData(), cluster.getData().euklideanDistance(Ret.NEW, false), cluster.getPredictLabels(), cluster.getCenters());
			break;
		case ACC:
			ce = new Accuracy(cluster.getData(), cluster.getPredictLabels(), cluster.getRealLabels());
			break;
		case PURITY:
			ce = new Purity(cluster.getData(), cluster.getPredictLabels(), cluster.getRealLabels());
			break;
		case NMI:
			ce = new NormalizedMutualInformation(cluster.getData(), cluster.getPredictLabels(), cluster.getRealLabels());
			break;
		default:
			throw new RuntimeException(
					"Error occurred in ClusterEvaluationDelegate.precision(), the method is not implemented");
		}
		return ce.precision();
	}

}
