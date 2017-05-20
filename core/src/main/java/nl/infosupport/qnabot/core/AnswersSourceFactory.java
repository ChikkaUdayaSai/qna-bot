package nl.infosupport.qnabot.core;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

/**
 * Produces answers sources
 */
public class AnswersSourceFactory {
    /**
     * Creates a new answers source from the specified input file
     * @param inputFile Input file to read from
     * @return  Returns the answers source
     * @throws IOException  Gets thrown when the input file could not be read
     */
    public static AnswersSource create(File inputFile) throws IOException {
        return new AnswersSource(Files.readAllLines(inputFile.toPath()));
    }
}
