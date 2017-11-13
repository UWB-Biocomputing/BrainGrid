package edu.uwb.braingrid.workbench.provvisualizer.controller;

import difflib.Delta;
import difflib.DiffUtils;
import difflib.Patch;
import edu.uwb.braingrid.workbench.provvisualizer.Utility.FileUtility;
import javafx.fxml.FXML;
import javafx.scene.control.Label;
import org.fxmisc.flowless.VirtualizedScrollPane;
import org.fxmisc.richtext.CodeArea;
import org.fxmisc.richtext.LineNumberFactory;
import org.fxmisc.richtext.model.StyleSpans;
import org.fxmisc.richtext.model.StyleSpansBuilder;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;


public class TextComparisonController {
    private static final Pattern XML_TAG = Pattern.compile("(?<ELEMENT>(</?\\h*)(\\w+)([^<>]*)(\\h*/?>))"
            +"|(?<COMMENT><!--[^<>]+-->)");

    private static final Pattern ATTRIBUTES = Pattern.compile("(\\w+\\h*)(=)(\\h*\"[^\"]+\")");

    private static final int GROUP_OPEN_BRACKET = 2;
    private static final int GROUP_ELEMENT_NAME = 3;
    private static final int GROUP_ATTRIBUTES_SECTION = 4;
    private static final int GROUP_CLOSE_BRACKET = 5;
    private static final int GROUP_ATTRIBUTE_NAME = 1;
    private static final int GROUP_EQUAL_SYMBOL = 2;
    private static final int GROUP_ATTRIBUTE_VALUE = 3;

    @FXML
    private Label fileInfoLblLeft;

    @FXML
    private VirtualizedScrollPane scrollPaneLeft;

    @FXML
    private CodeArea codeAreaLeft;

    @FXML
    private VirtualizedScrollPane scrollPaneRight;

    @FXML
    private Label fileInfoLblRight;

    @FXML
    private CodeArea codeAreaRight;

    /**
     * Initializes the controller.
     */
    @FXML
    public void initialize(){
        configCodeAreas();
    }

    private void configCodeAreas(){
        configCodeArea(true);
        configCodeArea(false);
    }

    private void configCodeArea(boolean left){
        CodeArea codeArea = left?codeAreaLeft:codeAreaRight;

        codeArea.setParagraphGraphicFactory(LineNumberFactory.get(codeArea));
        codeArea.textProperty().addListener((obs, oldText, newText) -> {
            codeArea.setStyleSpans(0, computeHighlighting(newText));
        });
    }

    public void loadFiles(String filePathLeft, String filePathRight){
        loadFile(filePathLeft,true);
        loadFile(filePathRight,false);

        //highlight differences
        List<String> left = Arrays.asList(codeAreaLeft.getText().split("\\R"));
        List<String> right  = Arrays.asList(codeAreaRight.getText().split("\\R"));
        Patch<String> patches = DiffUtils.diff(left,right);

        int leftCaret = 0;
        int rightCaret = 0;

        for(Delta<String> delta : patches.getDeltas()){
            if(delta.getType() == Delta.TYPE.INSERT){
                for(int i = 0; i < delta.getRevised().size(); i++){
                    codeAreaRight.setParagraphStyle(delta.getRevised().getPosition() + i, Collections.singleton("paragraph-inserted"));
                }
            }
            else if(delta.getType() == Delta.TYPE.CHANGE){
                for(int i = 0; i < delta.getOriginal().size(); i++){
                    codeAreaLeft.setParagraphStyle(delta.getOriginal().getPosition() + i, Collections.singleton("paragraph-changed"));
                }

                for(int i = 0; i < delta.getRevised().size(); i++){
                    codeAreaRight.setParagraphStyle(delta.getRevised().getPosition() + i, Collections.singleton("paragraph-changed"));
                }
            }
            else if(delta.getType() == Delta.TYPE.DELETE){
                for(int i = 0; i < delta.getOriginal().size(); i++){
                    codeAreaLeft.setParagraphStyle(delta.getOriginal().getPosition() + i, Collections.singleton("paragraph-deleted"));
                }
            }
        }

    }

    public void loadFile(String filePath, boolean left){
        CodeArea codeArea = left?codeAreaLeft:codeAreaRight;
        VirtualizedScrollPane scrollPane = left?scrollPaneLeft:scrollPaneRight;

        codeArea.clear();
        Scanner in =null;

        try {
            in= new Scanner(new File(filePath));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        in.useDelimiter("\\A");
        if(in.hasNext()){
            codeArea.appendText(in.next());
        }

        scrollPane.scrollYToPixel(0);
    }

    private static StyleSpans<Collection<String>> computeHighlighting(String text) {
        Matcher matcher = XML_TAG.matcher(text);
        int lastKwEnd = 0;
        StyleSpansBuilder<Collection<String>> spansBuilder = new StyleSpansBuilder<>();
        while(matcher.find()) {

            spansBuilder.add(Collections.emptyList(), matcher.start() - lastKwEnd);
            if(matcher.group("COMMENT") != null) {
                spansBuilder.add(Collections.singleton("comment"), matcher.end() - matcher.start());
            }
            else {
                if(matcher.group("ELEMENT") != null) {
                    String attributesText = matcher.group(GROUP_ATTRIBUTES_SECTION);

                    spansBuilder.add(Collections.singleton("tagmark"), matcher.end(GROUP_OPEN_BRACKET) - matcher.start(GROUP_OPEN_BRACKET));
                    spansBuilder.add(Collections.singleton("anytag"), matcher.end(GROUP_ELEMENT_NAME) - matcher.end(GROUP_OPEN_BRACKET));

                    if(!attributesText.isEmpty()) {

                        lastKwEnd = 0;

                        Matcher amatcher = ATTRIBUTES.matcher(attributesText);
                        while(amatcher.find()) {
                            spansBuilder.add(Collections.emptyList(), amatcher.start() - lastKwEnd);
                            spansBuilder.add(Collections.singleton("attribute"), amatcher.end(GROUP_ATTRIBUTE_NAME) - amatcher.start(GROUP_ATTRIBUTE_NAME));
                            spansBuilder.add(Collections.singleton("tagmark"), amatcher.end(GROUP_EQUAL_SYMBOL) - amatcher.end(GROUP_ATTRIBUTE_NAME));
                            spansBuilder.add(Collections.singleton("avalue"), amatcher.end(GROUP_ATTRIBUTE_VALUE) - amatcher.end(GROUP_EQUAL_SYMBOL));
                            lastKwEnd = amatcher.end();
                        }
                        if(attributesText.length() > lastKwEnd)
                            spansBuilder.add(Collections.emptyList(), attributesText.length() - lastKwEnd);
                    }

                    lastKwEnd = matcher.end(GROUP_ATTRIBUTES_SECTION);

                    spansBuilder.add(Collections.singleton("tagmark"), matcher.end(GROUP_CLOSE_BRACKET) - lastKwEnd);
                }
            }
            lastKwEnd = matcher.end();
        }
        spansBuilder.add(Collections.emptyList(), text.length() - lastKwEnd);

        return spansBuilder.create();
    }

    public Label getFileInfoLblLeft() {
        return fileInfoLblLeft;
    }

    public void setFileInfoLblLeft(Label fileInfoLblLeft) {
        this.fileInfoLblLeft = fileInfoLblLeft;
    }

    public Label getFileInfoLblRight() {
        return fileInfoLblRight;
    }

    public void setFileInfoLblRight(Label fileInfoLblRight) {
        this.fileInfoLblRight = fileInfoLblRight;
    }
}
