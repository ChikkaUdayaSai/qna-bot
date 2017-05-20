package nl.infosupport.qnabot.core;

import org.junit.Test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.*;

public class QuestionClassifierFactoryTests {
    @Test
    public void producesClassifierInstance() {
        QuestionClassifierFactory factory = new QuestionClassifierFactory();

        QuestionClassifier questionClassifier = factory.create(250, 10, new TrainingSettings());
        assertThat(questionClassifier, is(not(nullValue())));
    }
}
