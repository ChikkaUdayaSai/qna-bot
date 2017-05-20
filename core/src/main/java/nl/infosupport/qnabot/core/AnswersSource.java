package nl.infosupport.qnabot.core;

import java.io.File;
import java.util.List;

public class AnswersSource {
    private List<String> answers;

    public AnswersSource(List<String> answers) {
        this.answers = answers;
    }

    /**
     * Gets an answer from the answers source
     * @param index Index of the answer
     * @return  Returns the answer
     */
    public String getAnswer(int index) {
        return answers.get(index);
    }
}
