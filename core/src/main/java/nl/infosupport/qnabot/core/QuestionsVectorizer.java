package nl.infosupport.qnabot.core;

import org.deeplearning4j.bagofwords.vectorizer.BagOfWordsVectorizer;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.List;

/**
 * Utility to convert sentences into vectors using a bag of words algorithm
 */
public class QuestionsVectorizer {
    private final BagOfWordsVectorizer vectorizer;

    /**
     * Initializes a new instance {@link QuestionsVectorizer}
     */
    public QuestionsVectorizer(BagOfWordsVectorizer vectorizer) {
        this.vectorizer = vectorizer;
    }

    /**
     * Transforms a single sentence into a vector
     *
     * @param sentence Sentence to vectorize
     * @return Returns the vectorized sentence
     */
    public INDArray transform(String sentence) {
        return vectorizer.transform(sentence);
    }

    /**
     * Transform a set of sentences from an input file into a matrix
     *
     * @param inputFile Input file to transform
     * @return Returns the matrix containing the vectorized sentences
     * @throws IOException Gets thrown when the input file cannot be processed
     */
    public INDArray transform(File inputFile) throws IOException {
        List<String> lines = Files.readAllLines(inputFile.toPath(), Charset.defaultCharset());
        INDArray matrix = Nd4j.zeros(lines.size(), vectorizer.getVocabCache().numWords());

        for (int index = 0; index < lines.size(); index++) {
            String line = lines.get(index);
            INDArray lineVector = vectorizer.transform(line);

            matrix.putRow(index, lineVector);
        }

        return matrix;
    }

    /**
     * Gets the number of words in the vocabulary
     *
     * @return Returns the number of words in the trained vocabulary
     */
    public int getVocabularySize() {
        return vectorizer.getVocabCache().numWords();
    }

    /**
     * Saves the vocabulary cache for the vectorizer to disk
     * @param outputFile     Output file to save the vocabulary to
     * @throws IOException  Gets thrown when the file could not be written
     */
    public void save(File outputFile) throws IOException {
        WordVectorSerializer.writeVocabCache(vectorizer.getVocabCache(), outputFile);
    }
}
