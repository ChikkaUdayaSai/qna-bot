package nl.infosupport.qnabot.core;


import nl.infosupport.qnabot.core.QuestionsVectorizer;
import org.hamcrest.Matchers;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;

import static org.hamcrest.MatcherAssert.*;
import static org.hamcrest.Matchers.*;

public class QuestionsVectorizerTests {

    private QuestionsVectorizer vectorizer;
    private File inputFile;

    @Before
    public void setup() {
        inputFile = new File("src/test/data/questions.txt");
        vectorizer = new QuestionsVectorizer();

        vectorizer.fit(inputFile);
    }

    @Test
    public void tokenizerProcessesCorrectly() throws Exception {
        INDArray matrix = vectorizer.transform(inputFile);
        INDArray emptyMatrix = Nd4j.zeros(matrix.rows(), matrix.columns());

        assertThat(matrix, Matchers.is(not(nullValue())));
        assertThat(emptyMatrix.squaredDistance(matrix), greaterThan(0.0));

    }

    @Test
    public void tokenizerTransformsSingleSentence() throws Exception {
        INDArray vector = vectorizer.transform("Hello world");

        assertThat(vector, is(not(nullValue())));
        assertThat(Nd4j.zeros(vector.columns()).squaredDistance(vector), greaterThan(0.0));
    }
}
