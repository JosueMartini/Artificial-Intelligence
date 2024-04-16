import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

public class PS4 {

	public static ArrayList<int[]> hotY = new ArrayList<>();
	public static ArrayList<ArrayList<Double>> x = new ArrayList<>();
	public static ArrayList<ArrayList<Double>> w1 = new ArrayList<>();
	public static ArrayList<ArrayList<Double>> w2 = new ArrayList<>();
	public static String inputFile;
	public static double[] predictions;

	public static ArrayList<ArrayList<Double>> dropFirstColumn(ArrayList<ArrayList<Double>> matrix) {
		ArrayList<ArrayList<Double>> result = new ArrayList<>();
		for (ArrayList<Double> row : matrix) {
			ArrayList<Double> newRow = new ArrayList<>(row.subList(1, row.size()));
			result.add(newRow);
		}
		return result;
	}

	public static ArrayList<ArrayList<Double>> transpose(ArrayList<ArrayList<Double>> matrix) {
		int numRows = matrix.size();
		int numCols = matrix.get(0).size();

		ArrayList<ArrayList<Double>> transposed = new ArrayList<>();
		for (int j = 0; j < numCols; j++) {
			ArrayList<Double> newRow = new ArrayList<>();
			for (int i = 0; i < numRows; i++) {
				newRow.add(matrix.get(i).get(j));
			}
			transposed.add(newRow);
		}
		return transposed;
	}

	public static ArrayList<ArrayList<Double>> multiply(ArrayList<ArrayList<Double>> A,
			ArrayList<ArrayList<Double>> B) {
		int rows1 = A.size();
		int cols1 = A.get(0).size();
		int rows2 = B.size();
		int cols2 = B.get(0).size();

		if (cols1 != rows2) {
			System.err.println("Error with multiplication!  Check the dimensions.");
			throw new IllegalArgumentException();
		}

		ArrayList<ArrayList<Double>> C = new ArrayList<>();
		for (int i = 0; i < rows1; i++) {
			ArrayList<Double> newRow = new ArrayList<>();
			for (int j = 0; j < cols2; j++) {
				double sum = 0.0;
				for (int k = 0; k < cols1; k++) {
					sum += A.get(i).get(k) * B.get(k).get(j);
				}
				newRow.add(sum);
			}
			C.add(newRow);
		}
		return C;
	}

	public static void printDimensions(ArrayList<ArrayList<Double>> matrix) {
		int numRows = matrix.size();
		int numCols = matrix.isEmpty() ? 0 : matrix.get(0).size();
		String dimensions = String.format("Matrix dimensions: %d x %d", numRows, numCols);
		System.out.println(dimensions);
	}

	public static void loadData(String Ydata, String XData, String w1data, String w2data) throws IOException {
		// Read in Y data
		BufferedReader br = new BufferedReader(new FileReader(Ydata));
		String line;

		while ((line = br.readLine()) != null) {
			int num = Integer.parseInt(line);
			int[] oneHotEncode = new int[10];

			if (num == 0) {
				oneHotEncode[9] = 1;
			} else {
				oneHotEncode[num - 1] = 1;
			}

			hotY.add(oneHotEncode);

		}

		br.close();

		// Read in X data
		BufferedReader br2 = new BufferedReader(new FileReader(XData));
		String line2;

		while ((line2 = br2.readLine()) != null) {
			String[] split = line2.split(",");
			ArrayList<Double> row = new ArrayList<>();
			row.add(1.0);
			
			for (String i : split) {
				row.add(Double.parseDouble(i));
			}

			x.add(row);

		}

		br2.close();

		// Read in W1
		BufferedReader br3 = new BufferedReader(new FileReader(w1data));
		String line3;

		ArrayList<Double> biasRow = new ArrayList<>();

		for (int i = 0; i < 785; i++) {
			biasRow.add(1.0);
		}

		w1.add(biasRow);

		while ((line3 = br3.readLine()) != null) {
			String[] split = line3.split(",");
			ArrayList<Double> row = new ArrayList<>();

			for (String i : split) {
				row.add(Double.parseDouble(i));
			}

			w1.add(row);
		}

		br3.close();

		// Read in W2
		BufferedReader br4 = new BufferedReader(new FileReader(w2data));
		String line4;

		while ((line4 = br4.readLine()) != null) {
			String[] split = line4.split(",");
			ArrayList<Double> row = new ArrayList<>();

			for (String i : split) {
				row.add(Double.parseDouble(i));
			}

			w2.add(row);
		}

		br4.close();

		int numEntries = x.size();
		int numFeatures = x.get(0).size() - 1;

		System.out.println("Training Phase:  " + inputFile);
		System.out.println("--------------------------------------------------------------");
		System.out.println("=> Number of Entries (n): " + numEntries);
		System.out.println("=> Number of Features (p): " + numFeatures);

	}

	public static double calculateLoss(double[] predictions, ArrayList<int[]> hotY, ArrayList<ArrayList<Double>> w1,
			ArrayList<ArrayList<Double>> w2, double lambda) {
		double loss = 0;
		int n = hotY.size();
		int numClasses = 10;

		for (int i = 0; i < n; i++) {
			int[] yi = hotY.get(i);
			for (int k = 0; k < numClasses; k++) {
				double yHat = predictions[k];
				if (yi[k] == 1) {
					loss += -Math.log(yHat);
				} else {
					loss += -Math.log(1 - yHat);
				}
			}
		}
		loss /= n;

		double regularizationTerm1 = 0;
		for (ArrayList<Double> neuronWeights : w1) {
			for (Double weight : neuronWeights) {
				regularizationTerm1 += weight * weight;
			}
		}
		regularizationTerm1 *= lambda / (2 * n);

		double regularizationTerm2 = 0;
		for (ArrayList<Double> neuronWeights : w2) {
			for (Double weight : neuronWeights) {
				regularizationTerm2 += weight * weight;
			}
		}
		regularizationTerm2 *= lambda / (2 * n);

		loss += regularizationTerm1 + regularizationTerm2;

		return loss;
	}

	public static ArrayList<ArrayList<Double>> sigmoid(ArrayList<ArrayList<Double>> matrix) {
		ArrayList<ArrayList<Double>> result = new ArrayList<>();
		for (ArrayList<Double> row : matrix) {
			ArrayList<Double> newRow = new ArrayList<>();
			for (Double val : row) {
				double sigmoid = 1.0 / (1.0 + Math.exp(-val));
				newRow.add(sigmoid);
			}
			result.add(newRow);
		}
		return result;
	}
	
	public static double[] forwardProp(ArrayList<ArrayList<Double>> x, ArrayList<ArrayList<Double>> w1,
			ArrayList<ArrayList<Double>> w2) {		
		ArrayList<ArrayList<Double>> h1 = multiply(x, transpose(w1));
		ArrayList<ArrayList<Double>> h1Activated = sigmoid(h1);

		ArrayList<ArrayList<Double>> yHat = multiply(h1Activated, transpose(w2));
		ArrayList<ArrayList<Double>> yHatActivated = sigmoid(yHat);

		ArrayList<Double> flattenedYHat = new ArrayList<>();
		for (ArrayList<Double> row : yHatActivated) {
			flattenedYHat.addAll(row);
		}

		double[] predictions = new double[flattenedYHat.size()];
		for (int i = 0; i < predictions.length; i++) {
			predictions[i] = flattenedYHat.get(i);
		}

		return predictions;
	}

	public static void backPropagation(ArrayList<ArrayList<Double>> x, ArrayList<int[]> hotY, double[] predictions,
	        ArrayList<ArrayList<Double>> w1, ArrayList<ArrayList<Double>> w2, double alpha, double lambda) {
	    int n = x.size();
	    int numClasses = 10;

	    // Compute delta2
	    ArrayList<ArrayList<Double>> delta2 = new ArrayList<>();
	    for (int i = 0; i < n; i++) {
	        int[] yi = hotY.get(i);
	        ArrayList<Double> delta2Row = new ArrayList<>();
	        for (int k = 0; k < numClasses; k++) {
	            delta2Row.add(predictions[k] - yi[k]);
	        }
	        delta2.add(delta2Row);
	    }

	    // Compute delta1
	    ArrayList<ArrayList<Double>> delta1 = new ArrayList<>();
	    for (int i = 0; i < n; i++) {
	        ArrayList<Double> delta2Row = delta2.get(i);
	        ArrayList<Double> delta1Row = new ArrayList<>();
	        for (int j = 0; j < w1.size(); j++) {
	            double sum = 0.0;
	            for (int k = 0; k < numClasses; k++) {
	                sum += delta2Row.get(k) * w2.get(k).get(j);
	            }
	            delta1Row.add(sum * sigmoidDerivative(x.get(i).get(j)));
	        }
	        delta1.add(delta1Row);
	    }

	    // Compute gradient of W2
	    ArrayList<ArrayList<Double>> h = computeHiddenLayerOutput(x, w1);
	    ArrayList<ArrayList<Double>> gradW2 = multiply(transpose(delta2), h);

	    // Adjust weights for W2
	    for (int k = 0; k < numClasses; k++) {
	        for (int j = 0; j < w2.get(0).size(); j++) {
	            w2.get(k).set(j, w2.get(k).get(j) - (alpha / n) * gradW2.get(k).get(j) + (lambda / n) * w2.get(k).get(j));
	        }
	    }

	    // Compute gradient of W1	    
	    ArrayList<ArrayList<Double>> gradW1 = multiply(transpose(delta1), x);

	    // Adjust weights for W1
	    for (int j = 0; j < w1.size(); j++) {
	        for (int i = 0; i < w1.get(j).size(); i++) {
	            w1.get(j).set(i, w1.get(j).get(i) - (alpha / n) * gradW1.get(j).get(i) + (lambda / n) * w1.get(j).get(i));
	        }
	    }
	}
	
	public static ArrayList<ArrayList<Double>> computeHiddenLayerOutput(ArrayList<ArrayList<Double>> x, ArrayList<ArrayList<Double>> w1) {
	    ArrayList<ArrayList<Double>> h = new ArrayList<>();
	    for (ArrayList<Double> row : x) {
	        ArrayList<Double> hRow = new ArrayList<>();
	        for (int j = 0; j < w1.size(); j++) {
	            double sum = 0.0;
	            for (int i = 0; i < row.size(); i++) {
	                sum += row.get(i) * w1.get(j).get(i);
	            }
	            hRow.add( (1.0 / (1.0 + Math.exp(-sum))));
	        }
	        h.add(hRow);
	    }
	    return h;
	}



	public static double sigmoidDerivative(double x) {
		double sigmoid = 1.0 / (1.0 + Math.exp(-x));
		return sigmoid * (1 - sigmoid);
	}

	public static ArrayList<ArrayList<ArrayList<Double>>> batchGradientDescent(ArrayList<ArrayList<Double>> X,
			ArrayList<int[]> hotY, double alpha) {
		System.out.println("\nStarting Gradient Descent:");
		System.out.println("--------------------------------------------------------------\n");

		int epoch = 0;
		double prevLoss = Double.MAX_VALUE;
		double epsilon = 0.001;
		double delta = 0;

		while (epoch < 10) {
			double totalLoss = 0.0;

			predictions = forwardProp(X, w1, w2);

			double loss = calculateLoss(predictions, hotY, w1, w2, 0.01);
			totalLoss += loss;

			backPropagation(X, hotY, predictions, w1, w2, alpha, 0.0001);

			double avgLoss = totalLoss / X.size();

			delta = Math.abs(prevLoss - avgLoss);

			System.out.printf("Epoch %d: Loss of %.7f Delta = %.2f%% Epsilon = %.2f%%\n", (epoch + 1), avgLoss, delta,
					epsilon);

			prevLoss = avgLoss;
			epoch++;
		}

		System.out.println("\nEpochs Required: " + (epoch));
		ArrayList<ArrayList<ArrayList<Double>>> learnedWeights = new ArrayList<>();
		learnedWeights.add(w1);
		learnedWeights.add(w2);

		return learnedWeights;
	}

	public static void printTestResults(ArrayList<ArrayList<Double>> testX, ArrayList<int[]> testY,
			ArrayList<ArrayList<Double>> w1, ArrayList<ArrayList<Double>> w2) {
		int correctPredictions = 0;
		System.out.println("\nTesting Phase (first " + 10 + " records):");
		System.out.println("--------------------------------------------------------------\n");

		int i = 0;
		for (i = 0; i < 10; i++) {
			int[] actualLabelOneHot = testY.get(i);
			double[] predictedProbabilities = predictions;
			System.out.println(predictedProbabilities[0]);
			int actualLabel = getLabel(actualLabelOneHot);
			int predictedLabel = getMaxIndex(predictedProbabilities);
			boolean isCorrect = (actualLabel == predictedLabel);
			String correctStr = isCorrect ? "TRUE" : "FALSE";
			System.out.println("Test Record " + (i + 1) + ": " + actualLabel + " Prediction: " + predictedLabel
					+ " Correct: " + correctStr);
			if (isCorrect) {
				correctPredictions++;
			}
		}

		int totalRecords = i;
		double accuracy = ((double) correctPredictions / totalRecords) * 100;
		System.out.println("\n=> Number of Test Entries (n): " + totalRecords);
		System.out.println("=> Accuracy: " + String.format("%.2f", accuracy) + "%");
	}

	public static int getMaxIndex(double[] arr) {
		int maxIndex = 0;
		double maxValue = arr[0];
		for (int i = 1; i < arr.length; i++) {
			if (arr[i] > maxValue) {
				maxValue = arr[i];
				maxIndex = i;
			}
		}
		return maxIndex + 1;
	}

	public static int getLabel(int[] oneHotLabel) {
		for (int i = 0; i < oneHotLabel.length; i++) {
			if (oneHotLabel[i] == 1) {
				return i + 1;
			}
		}
		return -1;
	}

	public static void main(String[] args) throws IOException {
		System.out.println("************************************************************");
		System.out.println("Problem Set 4: Neural Network");
		System.out.println("Name: Josue Martinez");
		System.out.println("Syntax: java PS4 w1.txt w2.txt xdata.txt ydata.txt");
		System.out.println("************************************************************\n");

		String w1data = args[0];
		String w2data = args[1];
		inputFile = args[2];
		String ydata = args[3];

		// Load data
		loadData(ydata, inputFile, w1data, w2data);

		// Train model
		double alpha = 0.25;
		ArrayList<ArrayList<ArrayList<Double>>> learnedWeights = batchGradientDescent(x, hotY, alpha);

		
		
		// Test model
		printTestResults(x, hotY, learnedWeights.get(0), learnedWeights.get(1));

		BufferedWriter writer = new BufferedWriter(new FileWriter("w1out.txt"));
		ArrayList<ArrayList<Double>> w1 = learnedWeights.get(0);
		for (int i = 0; i < w1.size(); i++) {
			ArrayList<Double> neuronWeights = w1.get(i);
			for (int j = 0; j < neuronWeights.size(); j++) {
				writer.write(neuronWeights.get(j) + " ");
			}
			writer.newLine();
		}
		writer.close();

		BufferedWriter writer2 = new BufferedWriter(new FileWriter("w2out.txt"));
		ArrayList<ArrayList<Double>> w2 = learnedWeights.get(1);
		for (int i = 0; i < w2.size(); i++) {
			ArrayList<Double> neuronWeights = w2.get(i);
			for (int j = 0; j < neuronWeights.size(); j++) {
				writer2.write(neuronWeights.get(j) + " ");
			}
			writer2.newLine();
		}
		writer2.close();

	}

}
