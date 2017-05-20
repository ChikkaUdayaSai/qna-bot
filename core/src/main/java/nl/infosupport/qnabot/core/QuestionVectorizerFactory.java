package nl.infosupport.qnabot.core;

import org.deeplearning4j.bagofwords.vectorizer.BagOfWordsVectorizer;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.text.sentenceiterator.FileSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.File;
import java.io.IOException;

/**
 * Produces new instances of the {@link QuestionVectorizerFactory}
 */
public class QuestionVectorizerFactory {
    /**
     * Restores a pretrained question vectorizer from disk
     *
     * @param inputFile Input file to read the vocabulary for the vectorizer from
     * @return Returns the question vectorizer
     * @throws IOException Gets thrown when the input file could not be read
     */
    public static QuestionsVectorizer restore(File inputFile) throws IOException {
        BagOfWordsVectorizer vectorizer = new BagOfWordsVectorizer.Builder()
                .setVocab(WordVectorSerializer.readVocabCache(inputFile))
                .setTokenizerFactory(tokenizerFactory())
                .build();

        return new QuestionsVectorizer(vectorizer);
    }

    /**
     * Creates a new question vectorizer to be trained on the specified input file
     *
     * @param trainingDataFile Input file containing the training data for the question vectorizer
     * @return Returns the trained vectorizer
     */
    public static QuestionsVectorizer create(File trainingDataFile) {
        BagOfWordsVectorizer vectorizer = new BagOfWordsVectorizer.Builder()
                .setIterator(new FileSentenceIterator(trainingDataFile))
                .setTokenizerFactory(tokenizerFactory())
                .build();

        vectorizer.fit();

        return new QuestionsVectorizer(vectorizer);
    }

    /**
     * Creates the tokenizer factory for the question vectorizer
     *
     * @return Returns the new tokenizer factory
     */
    private static TokenizerFactory tokenizerFactory() {
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        return tokenizerFactory;
    }
}
