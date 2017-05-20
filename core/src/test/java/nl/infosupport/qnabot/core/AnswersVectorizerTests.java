package nl.infosupport.qnabot.core;

import nl.infosupport.qnabot.core.AnswersVectorizer;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;

import static org.hamcrest.MatcherAssert.*;
import static org.hamcrest.Matchers.*;

public class AnswersVectorizerTests {
    @Test
    public void transformsInputFileToMatrix() throws IOException {
        File inputFile = new File("src/test/data/questions.txt");
        AnswersVectorizer vectorizer = new AnswersVectorizer();

        INDArray output = vectorizer.transform(inputFile);

        assertThat(output, is(not(nullValue())));
        assertThat(Nd4j.zeros(3, 3).squaredDistance(output), not(equalTo(0.0)));
    }
}
