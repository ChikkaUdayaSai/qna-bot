package nl.infosupport.qnabot.core;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;

/**
 * Vectorizes answers
 */
public class AnswersVectorizer {
    /**
     * Transforms the lines in the input file into a matrix that
     * contains the vectorized answers.
     *
     * @param inputFile Input file to read from
     * @return Returns the vectorized answers file
     * @throws IOException Gets thrown when the answers file could not be read
     */
    public INDArray transform(File inputFile) throws IOException {
        List<String> answers = Files.readAllLines(inputFile.toPath());
        INDArray matrix = Nd4j.zeros(answers.size(), answers.size());

        for (int index = 0; index < answers.size(); index++) {
            INDArray vector = Nd4j.zeros(answers.size());
            vector.putScalar(index, 1.0);
            matrix.putRow(index, vector);
        }

        return matrix;
    }
}
