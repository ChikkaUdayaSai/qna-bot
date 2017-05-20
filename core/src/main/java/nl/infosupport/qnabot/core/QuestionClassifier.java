package nl.infosupport.qnabot.core;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;

/**
 * Classifies questions posted by the user
 */
public class QuestionClassifier {
    private final MultiLayerNetwork neuralNetwork;

    /**
     * Initializes a new instance of {@link QuestionClassifier}
     *
     * @param neuralNetwork Neural network to base the classifier on
     */
    public QuestionClassifier(MultiLayerNetwork neuralNetwork) {
        this.neuralNetwork = neuralNetwork;
    }

    /**
     * Predicts the class of the input
     *
     * @param features Vector containing the features to predict the class for
     * @return Returns the class
     */
    public int predict(INDArray features) {
        INDArray output = neuralNetwork.output(features, false);
        return Nd4j.argMax(output, 0).getInt(0);
    }

    /**
     * Trains the classifier
     *
     * @param features Matrix containing the features to fit on
     * @param labels   Matrix containing the labels for the feature vectors
     */
    public void fit(INDArray features, INDArray labels) {
        neuralNetwork.fit(features, labels);
    }

    /**
     * Saves the trained network on disk
     * @param outputFile    Output file to save
     * @throws IOException  Gets thrown when the model could not be saved
     */
    public void save(File outputFile) throws IOException {
        ModelSerializer.writeModel(neuralNetwork, outputFile, true);
    }
}
