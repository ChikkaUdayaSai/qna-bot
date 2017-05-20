package nl.infosupport.qnabot;

import nl.infosupport.qnabot.core.*;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

import java.io.File;
import java.io.IOException;

@SpringBootApplication
public class QnaBotServiceApplication {
    public static void main(String... args) {
        SpringApplication.run(QnaBotServiceApplication.class, args);
    }

    @Bean
    public QuestionClassifier questionClassifier() throws IOException {
        return QuestionClassifierFactory.restore(new File("model/classifier.bin"));
    }

    @Bean
    public QuestionsVectorizer questionsVectorizer() throws IOException {
        return QuestionVectorizerFactory.restore(new File("model/vocabulary.bin"));
    }

    @Bean
    public AnswersSource answersSource() throws IOException {
        return AnswersSourceFactory.create(new File("data/answers.txt"));
    }
}
