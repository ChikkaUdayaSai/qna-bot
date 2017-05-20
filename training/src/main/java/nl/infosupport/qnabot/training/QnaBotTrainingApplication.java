package nl.infosupport.qnabot.training;

import nl.infosupport.qnabot.core.*;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

public class QnaBotTrainingApplication {
    public static void main(String... args) {
        Logger logger = LoggerFactory.getLogger(QnaBotTrainingApplication.class);

        StatsStorage statsStorage = new InMemoryStatsStorage();

        UIServer server = UIServer.getInstance();
        QuestionClassifierFactory questionClassifierFactory = new QuestionClassifierFactory();

        server.attach(statsStorage);

        QuestionsVectorizer questionsVectorizer = new QuestionsVectorizer();
        AnswersVectorizer answersVectorizer = new AnswersVectorizer();

        questionsVectorizer.fit(new File("data/questions.txt"));

        try {
            INDArray labels = answersVectorizer.transform(new File("data/answers.txt"));
            INDArray features = questionsVectorizer.transform(new File("data/questions.txt"));

            QuestionClassifier classifier = questionClassifierFactory.create(
                    questionsVectorizer.getVocabularySize(), labels.columns(), new TrainingSettings(),
                    new StatsListener(statsStorage));

            classifier.fit(features, labels);

            classifier.save(new File("model/classifier.bin"));
        } catch (IOException ex) {
            logger.error("Failed to read training data", ex);
        }
    }
}
