package nl.infosupport.qnabot.controllers;

import nl.infosupport.qnabot.core.AnswersSource;
import nl.infosupport.qnabot.core.QuestionClassifier;
import nl.infosupport.qnabot.core.QuestionsVectorizer;
import nl.infosupport.qnabot.models.Answer;
import nl.infosupport.qnabot.models.AskQuestionForm;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class ChatController {
    private final QuestionClassifier classifier;
    private final AnswersSource answersSource;
    private final QuestionsVectorizer questionsVectorizer;

    @Autowired
    public ChatController(QuestionClassifier classifier,
                          AnswersSource answersSource,
                          QuestionsVectorizer questionsVectorizer) {
        this.classifier = classifier;
        this.answersSource = answersSource;
        this.questionsVectorizer = questionsVectorizer;
    }

    @RequestMapping(
            path="/api/chat",
            consumes="application/json",
            produces="application/json",
            method = RequestMethod.POST
    )
    public ResponseEntity<?> askQuestion(@RequestBody AskQuestionForm form) {
        INDArray features = questionsVectorizer.transform(form.getText());
        int label = classifier.predict(features);

        return ResponseEntity.ok(new Answer(answersSource.getAnswer(label)));
    }
}
