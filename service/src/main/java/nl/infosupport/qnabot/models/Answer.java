package nl.infosupport.qnabot.models;

public class Answer {
    private final String text;

    public Answer(String text) {
        this.text = text;
    }

    public String getText() {
        return text;
    }
}
