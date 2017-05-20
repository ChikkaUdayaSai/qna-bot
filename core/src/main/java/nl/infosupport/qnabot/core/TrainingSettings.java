package nl.infosupport.qnabot.core;

/**
 * Defines the settings to use for optimizing the neural network
 * that is contained within the {@link QuestionClassifier}
 */
public class TrainingSettings {
    private final int iterations;
    private final double learningRate;

    /**
     * Initializes a new instance of {@link TrainingSettings}
     * @param iterations    Number of iterations to train the network for
     * @param learningRate  Learning rate to use for training the network
     */
    public TrainingSettings(int iterations, double learningRate) {
        this.iterations = iterations;
        this.learningRate = learningRate;
    }

    /**
     * Initializes a new instance of {@link TrainingSettings}
     * By default this uses 500 iterations and a learning rate of 0.05
     */
    public TrainingSettings() {
        iterations = 500;
        learningRate = 0.05;
    }

    /**
     * Gets the number of iterations to train the neural network for
     * @return  Returns the number of iterations
     */
    public int getIterations() {
        return iterations;
    }

    /**
     * Gets the learning rate for the neural network optimizer
     * @return  Gets the learning rate
     */
    public double getLearningRate() {
        return learningRate;
    }
}
