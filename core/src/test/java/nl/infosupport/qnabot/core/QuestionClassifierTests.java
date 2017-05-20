package nl.infosupport.qnabot.core;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.not;

public class QuestionClassifierTests {
    @Test
    public void fitTrainsNetwork() {
        INDArray features = Nd4j.rand(new int[]{10, 250});
        INDArray labels = Nd4j.rand(new int[]{10, 10});

        QuestionClassifierFactory factory = new QuestionClassifierFactory();
        TrainingSettings trainingSettings = new TrainingSettings(1, 0.05);

        QuestionClassifier classifier = factory.create(250,10, trainingSettings);

        classifier.fit(features, labels);
    }

    @Test
    public void predictReturnsPredictedClass() {
        INDArray features = Nd4j.rand(new int[] { 1, 250 });
        INDArray trainingSamples = Nd4j.rand(new int[]{1, 250});
        INDArray labels = Nd4j.rand(new int[]{1, 10});

        QuestionClassifierFactory factory = new QuestionClassifierFactory();
        TrainingSettings trainingSettings = new TrainingSettings(1, 0.05);

        QuestionClassifier classifier = factory.create(250,10, trainingSettings);

        classifier.fit(trainingSamples,labels);
        int outputClass = classifier.predict(features);

        assertThat(outputClass, not(equalTo(-1)));
    }
}
