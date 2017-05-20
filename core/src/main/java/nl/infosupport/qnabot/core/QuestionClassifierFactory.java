package nl.infosupport.qnabot.core;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

/**
 * Factory class to produce new instances of {@link QuestionClassifier}
 */
public class QuestionClassifierFactory {

    /**
     * Initializes a new instance of {@link QuestionClassifierFactory}
     */
    public QuestionClassifierFactory() {

    }

    /**
     * Creates the questions classifier
     *
     * @param vocabularySize Input size for the neural network
     * @param numLabels      The output size for the neural network
     * @return Returns the classifier for the user questions
     */
    public QuestionClassifier create(int vocabularySize, int numLabels, TrainingSettings trainingSettings,
                                     IterationListener... iterationListeners) {
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(42)
                .iterations(trainingSettings.getIterations())
                .learningRate(trainingSettings.getLearningRate())
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                .layer(0, relu(vocabularySize, 1024))
                .layer(1, relu(1024, 1024))
                .layer(2, softmax(numLabels))
                .backprop(true).pretrain(false)
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(configuration);
        network.setListeners(iterationListeners);
        network.init();

        return new QuestionClassifier(network);
    }

    /**
     * Creates a question classifier based on a model saved earlier
     *
     * @param inputFile Input file to load the model from
     * @return Returns the restored question classifier
     * @throws IOException Gets thrown when the input file could not be read
     */
    public QuestionClassifier restore(File inputFile, IterationListener... iterationListeners) throws IOException {
        MultiLayerNetwork network = ModelSerializer.restoreMultiLayerNetwork(inputFile);
        network.setListeners(iterationListeners);

        return new QuestionClassifier(network);
    }

    /**
     * A dense layer with a rectified linear activation function
     *
     * @param inputSize  Number of input neurons
     * @param outputSize Number of output neurons
     * @return Returns the dense layer
     */
    private static DenseLayer relu(int inputSize, int outputSize) {
        return new DenseLayer.Builder()
                .nIn(inputSize).nOut(outputSize)
                .activation(Activation.RELU)
                .build();
    }

    /**
     * An output layer with a softmax activation function and a Cross Entropy loss function
     *
     * @param outputSize Number of output neurons
     * @return Returns the output layer
     */
    private static OutputLayer softmax(int outputSize) {
        return new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                .nOut(outputSize)
                .activation(Activation.SOFTMAX)
                .build();
    }
}
