package ufs.utils;

import java.io.File;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

/**
 * There are some Constant values.
 * 
 * @author wyx
 *
 */
public final class ConstValues {

	/**
	 * The weight matrix constant. weightMatrix[i][j] = exp(-||x_i - x_j||^2 /
	 * LS_CONSTANT)
	 */
	public static final double LS_CONSTANT = 1e-3;

	// Iterative Parameter: The Coefficient of exterior penalty function. Rho.
	public static final double RHO = 2;

	// Iterative Parameter: Convergent value-Epsilon
	public static final double CONVERG_VALUE = 1e-2;

	// Iterative Parameter: max number of iterative.
	// If the number of iterative greater than this number, it stops.
	public static final int MAX_ITE_NUM = 1_000;

	// Matrix file Path
	public static final String DATA_MATRIX_PATH = "src/data/mat/";

	// NAME_TYPE_NumSample_NumFeature_NumClass
	public static final String STD_WARPAR_MATRIX_130$2400$10 = "WarpAR10P_std.mat";

	public static final String STD_TOX_MATRIX_171$5748$4 = "Tox171_std.mat";

	public static final String STD_ORL_MATRIX_400$1024$40 = "Orl32_std.mat";

	public static final String STD_WARPPIE_MATRIX_210$2420$10 = "WarpPIE10P_std.mat";

	public static final String STD_YALE_MATRIX_165$1024$15 = "Yale32_std.mat";

	public static final String STD_CARCINOM_MATRIX_174$9182$11 = "Carcinom_std.mat";

	public static final String STD_GLIOMA_MATRIX_50$4434$4 = "Glioma_std.mat";

	public static final String STD_COIL20_MATRIX_1440$1024$20 = "COIL20_std.mat";

	public static final String STD_ISOLET_MATRIX_1560$617$26 = "Isolet_std.mat";

	public static final String[] STD_DATA_MATRIX = {
			STD_WARPAR_MATRIX_130$2400$10, STD_TOX_MATRIX_171$5748$4,
			STD_ORL_MATRIX_400$1024$40, STD_WARPPIE_MATRIX_210$2420$10,
			STD_YALE_MATRIX_165$1024$15, STD_CARCINOM_MATRIX_174$9182$11,
			STD_GLIOMA_MATRIX_50$4434$4, STD_COIL20_MATRIX_1440$1024$20,
			STD_ISOLET_MATRIX_1560$617$26 };

	
	public static final String WARPAR_MATRIX_130$2400$10 = "WarpAR10P.mat";

	public static final String TOX_MATRIX_171$5748$4 = "Tox171.mat";

	public static final String ORL_MATRIX_400$1024$40 = "Orl32.mat";

	public static final String WARPPIE_MATRIX_210$2420$10 = "WarpPIE10P.mat";

	public static final String YALE_MATRIX_165$1024$15 = "Yale32.mat";

	public static final String CARCINOM_MATRIX_174$9182$11 = "Carcinom.mat";

	public static final String GLIOMA_MATRIX_50$4434$4 = "Glioma.mat";

	public static final String COIL20_MATRIX_1440$1024$20 = "COIL20.mat";

	public static final String ISOLET_MATRIX_1560$617$26 = "Isolet.mat";

	public static final String[] DATA_MATRIX = { WARPAR_MATRIX_130$2400$10,
			TOX_MATRIX_171$5748$4, ORL_MATRIX_400$1024$40,
			WARPPIE_MATRIX_210$2420$10, YALE_MATRIX_165$1024$15,
			CARCINOM_MATRIX_174$9182$11, GLIOMA_MATRIX_50$4434$4,
			COIL20_MATRIX_1440$1024$20, ISOLET_MATRIX_1560$617$26 };

	public static final int[] NUM_CLUSTERS;

	static {
		// Computing the number of clusters.
		// Note that the number of clusters must started with zero.
		NUM_CLUSTERS = new int[DATA_MATRIX.length];
		try {
			for (int i = 0; i < DATA_MATRIX.length; i++) {
				NUM_CLUSTERS[i] = Utils.loadMatrix2DFromMat(new File(DATA_MATRIX_PATH
						+ DATA_MATRIX[i]), "Y").max(Ret.NEW, Matrix.ALL).getAsInt(0, 0);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	// // Arff Path
	// public static final String DATA_ARFF_PATH = "src/data/mat/arff";
	//
	// // NAME_TYPE_NumSample_NumFeature_NumClass
	// public static final String WARPAR_ARFF_130$2400$10 =
	// "warpAR10P_original.arff";
	//
	// public static final String TOX_ARFF_171$5748$4 = "TOX_171_original.arff";
	//
	// public static final String ORL_ARFF_400$1024$40 = "ORL32_original.arff";
	//
	// public static final String WARPPIE_ARFF_210$2420$10 =
	// "warpPIE10P_original.arff";
	//
	// public static final String YALE_ARFF_165$1024$15 =
	// "Yale32_original.arff";
	//
	// public static final String COIL20_ARFF_1440$1024$20 =
	// "COIL20_original.arff";
	//
	// public static final String ISOLET_ARFF_1560$617$26 =
	// "Isolet_original.arff";
	//
	// public static final String BASEHOCK_ARFF_1993$4682$2 =
	// "Basehock_original.arff";
	//
	// public static final String[] DATA_ARFF = { WARPAR_ARFF_130$2400$10,
	// TOX_ARFF_171$5748$4, ORL_ARFF_400$1024$40,
	// WARPPIE_ARFF_210$2420$10, YALE_ARFF_165$1024$15,
	// COIL20_ARFF_1440$1024$20, ISOLET_ARFF_1560$617$26,
	// BASEHOCK_ARFF_1993$4682$2, };
}
