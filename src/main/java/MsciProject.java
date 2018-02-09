package main.java;
import java.io.File;
import java.lang.reflect.Array;
import java.math.BigDecimal;
import java.text.DecimalFormat;
import java.util.Arrays;

import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.renderer.category.BarPainter;
import org.jfree.chart.renderer.category.BarRenderer;
import org.jfree.chart.renderer.category.CategoryItemRenderer;
import org.jfree.chart.renderer.category.StandardBarPainter;
import org.jfree.chart.renderer.xy.XYSplineRenderer;
import org.jfree.data.category.CategoryDataset;
import org.jfree.data.statistics.HistogramDataset;
import org.jfree.data.statistics.HistogramType;
import org.jfree.data.xy.IntervalXYDataset;
import org.terrier.learning.FeaturedResultSet;
import org.terrier.matching.models.basicmodel.In;
import org.terrier.structures.Index;
import org.terrier.utility.ApplicationSetup;

import edu.uci.jforests.dataset.BitNumericArray;
import edu.uci.jforests.dataset.ByteNumericArray;
import edu.uci.jforests.dataset.Feature;
import edu.uci.jforests.dataset.NullNumericArray;
import edu.uci.jforests.dataset.NumericArray;
import edu.uci.jforests.dataset.RankingDataset;
import edu.uci.jforests.dataset.ShortNumericArray;
import edu.uci.jforests.input.FeatureAnalyzer;
import edu.uci.jforests.learning.LearningUtils;
import edu.uci.jforests.learning.trees.Ensemble;
import edu.uci.jforests.learning.trees.Tree;
import edu.uci.jforests.learning.trees.decision.DecisionTree;
import edu.uci.jforests.learning.trees.regression.RegressionTree;
import edu.uci.jforests.sample.RankingSample;
import edu.uci.jforests.sample.Sample;
import gnu.trove.TIntIntHashMap;
import edu.uci.jforests.dataset.*;
import edu.uci.jforests.input.FeatureAnalyzer;
import edu.uci.jforests.learning.LearningUtils;
import edu.uci.jforests.learning.trees.Ensemble;
import edu.uci.jforests.learning.trees.regression.RegressionTree;
import edu.uci.jforests.sample.RankingSample;
import edu.uci.jforests.sample.Sample;
import gnu.trove.TIntIntHashMap;

import gnu.trove.TByteArrayList;

import java.awt.*;
import java.io.BufferedReader;
import java.util.*;
import java.util.regex.Pattern;

import org.jfree.chart.ChartUtilities;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.terrier.utility.ApplicationSetup;
import org.terrier.utility.ArrayUtils;
import org.terrier.utility.Files;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.ui.ApplicationFrame;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.chart.plot.*;
import org.jfree.ui.RefineryUtilities;

import javax.swing.*;
import java.io.*;
import java.util.ArrayList;
import java.util.List;


public class MsciProject {
    static Ensemble ensemble = new Ensemble();

    final static FeatureAnalyzer featureAnalyzer = new FeatureAnalyzer();




    private DefaultCategoryDataset createDataset() {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();

        List<Double> x = new ArrayList<Double>();
        for (int j = 0; j < 10; j++) {
            x.add(gaussian(0.217949));
        }
        double[] target = new double[x.size()];
        for (int i = 0; i < target.length; i++) {
            target[i] = x.get(i).doubleValue();
            for (int k = 0; k < target.length; k++) {
                dataset.addValue((Number) target[k], "gaussian", k);
            }
        }

        return dataset;
    }

    protected static void loadModel(String model_filename) throws Exception {
        String featureStats_filename = ApplicationSetup.getProperty("fat.matching.learned.jforest.statistics", model_filename + ".features");
        featureAnalyzer.loadFeaturesFromFile("/home/rachel/Desktop/IR/Test/src/main/java/jforests-feature-stats.txt");
        ensemble.loadFromFile(RegressionTree.class, new File(model_filename));


    }

    public static double gaussian(double feature) {
        Random r = new Random();
        double vectorGaussian = feature + r.nextGaussian();
        if (0 > vectorGaussian || vectorGaussian > 1) {

            return gaussian(feature);
        } else {
            return vectorGaussian;
        }
    }


    protected static RankingDataset makeDatasetDF(int N, int featureCount, double[][] doubleFeatures) {
        //doubleFeatures: indexed by document then feature
        //intFeatures: indexed by document then feature
        final int[][] intFeatures = new int[N][featureCount];
        TIntIntHashMap[] _valueHashMap = new TIntIntHashMap[featureCount];
        for (int f = 0; f < featureCount; f++) {
            _valueHashMap[f] = new TIntIntHashMap();
        }

        //we are also rotating here.
        // doubleFeatures = Arrays.copyOfRange(doubleFeatures, 1, doubleFeatures.length);
        for (int d = 0; d < N; d++) {
            for (int f = 0; f < featureCount; f++) {
                double value = doubleFeatures[d][f];
                if (featureAnalyzer.onLogScale[f]) {
                    value = (Math.log(value - featureAnalyzer.min[f] + 1) * featureAnalyzer.factor[f]);
                } else {
                    value = (value - featureAnalyzer.min[f]) * featureAnalyzer.factor[f];
                }
                int intValue = intFeatures[d][f] = (int) Math.round(value);
                _valueHashMap[f].adjustOrPutValue(intValue, 1, 1);

            }
        }
        final int[][] valueDistributions = new int[featureCount][];
        TIntIntHashMap _curMap;
        for (int f = 0; f < featureCount; f++) {
            _curMap = _valueHashMap[f];
            if (!_curMap.containsKey(0)) {
                _curMap.put(0, 0);
            }
            valueDistributions[f] = _curMap.keys();
            Arrays.sort(valueDistributions[f]);
        }

        final NumericArray[] bins = new NumericArray[featureCount];
        for (int i = 0; i < featureCount; i++) {
            int numValues = valueDistributions[i].length;
            if (numValues == 1 && valueDistributions[i][0] == 0) {
                bins[i] = NullNumericArray.getInstance();
            } else if (numValues <= 2) {
                bins[i] = new BitNumericArray(N);
            } else if (numValues <= Byte.MAX_VALUE) {
                bins[i] = new ByteNumericArray(N);
            } else if (numValues <= Short.MAX_VALUE) {
                bins[i] = new ShortNumericArray(N);
            } else {
                throw new RuntimeException("One of your features have more than " + Short.MAX_VALUE
                        + " distinct values. The support for this feature is not implemented yet.");
            }

        }


        final double[] targets = new double[N];
        for (int d = 0; d < N; d++) {
            for (int f = 0; f < featureCount; f++) {
                // System.out.println(intFeatures[d][f]);
                int index = Arrays.binarySearch(valueDistributions[f], intFeatures[d][f]);
                bins[f].set(d, index);
            }
        }
        Feature[] features = new Feature[featureCount];
        for (int f = 0; f < featureCount; f++) {
            features[f] = new Feature(bins[f]);
            features[f].upperBounds = valueDistributions[f];
            features[f].setName(String.valueOf(f + 1));

        }
        RankingDataset rtr1 = new RankingDataset();
        rtr1.init(features, targets, new int[]{0}, N);
        return rtr1;
    }


    protected static void applyModel(int N, int featureCount,
                                     double[][] doubleFeatures, double[] out_scores) {
        boolean score_is_feature = Boolean.parseBoolean(ApplicationSetup.getProperty("fat.matching.model.score_is_feature", "true"));
        //doubleFeatures is indexed by feature then document

        RankingDataset dataset = makeDatasetDF(N, featureCount, doubleFeatures);
        Sample sample = new RankingSample(dataset);

        // System.out.println(ensemble.getNumTrees());
        LearningUtils.updateScores(sample, out_scores, ensemble);

    }


    static double[][] rotate(final double[][] in) {
        final int I = in.length;
        final int J = in[0].length;
        final double[][] out = new double[J][I];
        for (int i = 0; i < I; i++)
            for (int j = 0; j < J; j++)
                out[j][i] = in[i][j];
        return out;
    }


    static final Pattern SPLIT_SPACE_PATTERN = Pattern.compile("\\s+");
    static final Pattern SPLIT_COLON_PATTERN = Pattern.compile(":");
    static final Pattern REPLACE_START_SPACE_PATTERN = Pattern.compile("^\\s+");
    static final boolean ACCEPT_BAD_VALUES = false;
    static String[] queryno;
    static String[][] docno;
    static double[][][] feature_scores;
    static int numFeature = -1;
    static String feature_path;
    static byte[][] relevanceLabels;
    static boolean hasRelevance = false;
    static final Set<String> queryBlacklist = new HashSet<String>();


    public static void load(String feature_path) throws Exception {


        final BufferedReader br = Files.openFileReader(feature_path);

        String fline = null;
        String lastQid = null;
        int count_query = -1;
        int count_qrel = 0;
        int total_count_qrels = 0;

        List<String> _querynos = new ArrayList<String>();
        List<List<String>> _docnos = new ArrayList<List<String>>();
        List<List<double[]>> _feature_scores = new ArrayList<List<double[]>>();
        List<TByteArrayList> _relevanceLabels = new ArrayList<TByteArrayList>();

        List<double[]> currentFeatureValues = new ArrayList<double[]>();
        List<String> currentDocnos = new ArrayList<String>();
        TByteArrayList currentLabels = new TByteArrayList();

        while ((fline = br.readLine()) != null) {
            // ignore comment lines
            if (fline.startsWith("#")) {
                continue;
            }

            fline = fline.trim();
            // e.g., 1 qid:968 1:0.000215 2:-0.000215 3:0.000000 #docid = BLOG06-feed-001092
            final int commentIndex = fline.indexOf('#');
            final String featurePart = (commentIndex == -1) ? fline : fline.substring(0, commentIndex - 1);
            //pieces = {"1 qid:968 1:0.000215 2:-0.000215 3:0.000000", "docid = BLOG06-feed-001092"}

            //left = {"1", "qid:968", "1:0.000215", ..}
            final String[] left = SPLIT_SPACE_PATTERN.split(featurePart, 0);

            //qid = 968
            String qid = left[1].substring(left[1].indexOf(':') + 1);
            //String qid = left[1].split(":")[1];
            if (lastQid == null || !qid.equals(lastQid)) {
                currentDocnos = new ArrayList<String>();
                currentLabels = new TByteArrayList();
                currentFeatureValues = new ArrayList<double[]>();

                lastQid = qid;
                count_qrel = 0;
                if (!queryBlacklist.contains(qid)) {
                    _docnos.add(currentDocnos);
                    _querynos.add(qid);
                    _relevanceLabels.add(currentLabels);
                    _feature_scores.add(currentFeatureValues);
                    count_query++;
                }
            }


            //relevance label
            currentLabels.add(Byte.parseByte(left[0]));

            if (commentIndex != -1) {
                final String comment = REPLACE_START_SPACE_PATTERN.matcher(fline.substring(commentIndex + 1)).replaceAll("");
                final String[] commentPieces = SPLIT_SPACE_PATTERN.split(comment, 0);
                //#docid = clueweb09-en0000-00-01664 1664
                currentDocnos.add(commentPieces[2]);
            } else {
                currentDocnos.add('d' + String.valueOf(total_count_qrels));
            }

            //assume left[1] contains qid
            final double[] currentFeatures = new double[left.length - 2];
            for (int i = 2; i < left.length; i++) {
                final String l = left[i]; //1:0.000215
                final int colonIndex = l.indexOf(':');
                if (colonIndex == -1) {
                    System.out.println("Malformed line. only " + l + " found");
                    continue;
                }

                //assume features are in correct order
                try {
                    currentFeatures[i - 2] = Double.parseDouble(l.substring(colonIndex + 1));
                } catch (NumberFormatException e) {
                    if (ACCEPT_BAD_VALUES)
                        currentFeatures[i - 2] = 0;
                    else
                        throw e;
                }
            }
            currentFeatureValues.add(currentFeatures);


            lastQid = qid;
            count_qrel++;
            total_count_qrels++;
        }
        br.close();

        final int numQuery = _querynos.size();
        queryno = _querynos.toArray(new String[numQuery]);
        docno = new String[numQuery][];
        relevanceLabels = new byte[numQuery][];
        feature_scores = new double[numQuery][][];
        for (int qi = 0; qi < numQuery; qi++) {
            docno[qi] = _docnos.get(qi).toArray(new String[0]);
            relevanceLabels[qi] = _relevanceLabels.get(qi).toNativeArray();
            feature_scores[qi] = new double[docno[qi].length][];
            for (int di = 0; di < docno[qi].length; di++) {
                feature_scores[qi][di] = _feature_scores.get(qi).get(di);
                if (numFeature == -1) {
                    numFeature = feature_scores[qi][di].length;
                } else if (feature_scores[qi][di].length != numFeature) {
                    throw new IllegalArgumentException("Number of features for query " + queryno[qi] + " doc "
                            + docno[qi][di] + " was unexpected. Expected " + numFeature + " found "
                            + feature_scores[qi][di].length);
                }
            }
        }
        _relevanceLabels = null;
        _docnos = null;
        _feature_scores = null;

        //check if any relevance information
        OUTER:
        for (byte[] b : relevanceLabels)
            for (byte bi : b)
                if (bi != -1) {
                    hasRelevance = true;
                    break OUTER;
                }
        if (!hasRelevance)
            relevanceLabels = null;

    }

    public static String[][] getDocnos() {
        return docno;
    }

    public static double[][][] getFeatures() {
        return feature_scores;
    }

    public static String[] getQuerynos() {
        return queryno;
    }

    public static int getNumFeatures() {
        return numFeature;
    }

    public static byte[][] getRelevance() {
        return relevanceLabels;
    }


    public static boolean hasRelevance() {
        return hasRelevance;
    }


    public static void offsetGraph(double[] initscore, int runs, int oppositeDoc, int doc, double startFeature){
        double startScore = initscore[doc];
        boolean lower = true; //The "doc" score is lower than the "opposite doc" score initially
        double boundary = initscore[oppositeDoc];
        if (boundary < startScore){ //doc score is higher than the "opposite doc" score initially
            lower = false;
        }


        for (Map.Entry<Double, Double> entry : results.entrySet()) {
            System.out.println("!: " + entry.getKey() + " !: " + startFeature );
            double difference = entry.getKey() - startFeature ;

            Double[] input  = {difference, entry.getKey()};
            System.out.println(difference);
            if (lower) {
                if (entry.getValue() > boundary) {
                    System.out.println("dif l " + difference);
                    offset.put(input, "t");

                } else {
                    offset.put(input, "f");

                }
            }
            else{
                if (entry.getValue()<boundary) {
                    System.out.println("dif g " + input[0]);
                    offset.put(input, "t");

                } else {
                    offset.put(input, "f");

                }
            }




        }
        String val;
        System.out.println("                    " + offset.size());
        for (Map.Entry<Double[], String> entry : offset.entrySet()) {
            if (entry.getValue() == "t") {
                System.out.println(entry.getKey()[0]);
                DecimalFormat df = new DecimalFormat("#.#");
                val = df.format(entry.getKey()[0]);

                if (val.equalsIgnoreCase("-0")) {
                    val = "0.0";
                }

                if (val.equalsIgnoreCase("0")){
                    System.out.println("o");
                    val = "0.0";
                }
                if (val.equalsIgnoreCase("-1")) {
                    val = "-1.0";
                }

                if (val.equalsIgnoreCase("1")){
                    val = "1.0";
                }
                System.out.println(val);

                System.out.println(val=="0");
                offsetNos.put(val, offsetNos.get(val) + 1);
                System.out.println("VAL" + offsetNos.get(val));


            }
        }
        offset.clear();
        results.clear();
        threshold.clear();

    }

    public static void start(String name) throws Exception {
       /* Scanner reader = new Scanner(System.in);  // Reading from System.in
        System.out.println("Which document would you like to alter? (0 or 1): ");
        int doc = reader.nextInt();
        System.out.println("Which feature would you like to alter? (0 to 63): ");
            int feat = reader.nextInt();



        System.out.println("How many iterations to run: ");

            int runs = reader.nextInt();

        reader.close();*/
        int doc = 0;
        int feat = 53;
        int runs = 200;
       // int nodocpairs = 8;
        int oppositeDoc = 0;
        if (doc == 0){
            oppositeDoc = 1;
        }
        int lineno = 0;
        File file = new File("vectors.txt");
        BufferedReader reader = new BufferedReader(new FileReader(file));

        String line;
        int docsno = 0;
        String out = "";

        while ((line = reader.readLine()) != null) {

            if (lineno< 2){
                out = out + line + "\n";
            }
            else{
                File output1 = new File("temp/0vectors" + docsno + ".txt");
                PrintWriter writer = new PrintWriter(output1);
                writer.write(out);
                writer.close();
                lineno = 0;
                docsno++;
                out = "";
                out = out + line + "\n";

            }
            lineno++;

        }
        for(int x = 0; x < 7; x++) {
            load("temp/0vectors" + x + ".txt");
            //CombinedLETORFeatureLoader("/home/rachel/Desktop/IR/Test/src/main/java/testvectors.txt");
            loadModel("ensemble.txt");
            // for (int dn  = 0; dn < nodocpairs+1; dn++)
            System.out.println(Arrays.deepToString(getFeatures()));
            double startFeature = getFeatures()[0][doc][feat];
            double[] outscore = {0, 0};
            applyModel(2, 64, getFeatures()[0], outscore);


            File output = new File("results/results_" + name + ".txt");
            String init = ("Initial results: " + Arrays.toString(outscore));
            double[] initscore = outscore;


            //change this ten times.
            for (int times = 0; times < runs; times++) {
                outscore[0] = 0;
                outscore[0] = 0;
                outscore[1] = 0;
                double replacement = gaussian(getFeatures()[0][doc][feat]);
                replacement = (double) Math.round(replacement * 1000000d) / 1000000d;
                //replacement = .805588;
                getFeatures()[0][doc][feat] = replacement;

                String fileName = "temp/test" + times + ".txt";
                PrintWriter writer = new PrintWriter(fileName);
                String firstLine = "";
                for (int a = 1; a < getFeatures()[0][0].length + 1; a++) {
                    firstLine = firstLine + a + ":" + getFeatures()[0][0][a - 1] + " ";
                }
                String secondLine = "";
                for (int a = 1; a < getFeatures()[0][1].length + 1; a++) {
                    secondLine = secondLine + a + ":" + getFeatures()[0][1][a - 1] + " ";
                    System.out.println("                    !!!!!!");
                }
                writer.write("1 qid:10 " + firstLine + "#docid = G00-00-1000000\n");
                writer.write("0 qid:10 " + secondLine + "#docid = G00-00-0000001");
                writer.close();

                load(fileName);

                applyModel(2, 64, getFeatures()[0], outscore);

                threshold.put(getFeatures()[0][doc][feat], outscore[oppositeDoc]);
                results.put(getFeatures()[0][doc][feat], outscore[doc]);
                System.out.println("                            pass");
            }

            // resultsf.write("Feature value: " + getFeatures()[0][doc][feat] + " Result : " + outscore[doc]);


            PrintWriter writer1 = new PrintWriter(output);
            writer1.write(init);
            for (Map.Entry<Double, Double> e : results.entrySet()) {
                writer1.write("\nFeature value: " + e.getKey() + "    Result: " + e.getValue());
            }
            writer1.close();
            JFreeChart chart = test("x vs y",
                    "Example Plot", name);
            offsetGraph(initscore, runs, oppositeDoc, doc, startFeature);
            JFreeChart chart2 = offsetChart(name);
        }


        }


    public static Map<Double, Double> results = new HashMap<Double, Double>();

    public static Map<Double, Double> mean1 = new HashMap<Double, Double>();

    public static Map<Double, Double> threshold = new HashMap<Double, Double>();
    public static Map<Double[], String> offset = new HashMap<Double[], String>();
    public static Map<String, Integer> offsetNos = new LinkedHashMap<>();



    private static XYDataset createXYDataset() throws FileNotFoundException {
        XYSeriesCollection dataset = new XYSeriesCollection();

        XYSeries series1 = new XYSeries("results ");
        for (Map.Entry<Double, Double> entry : results.entrySet()) {
            series1.add(entry.getKey(), entry.getValue());

        }


        dataset.addSeries(series1);
        XYSeries series2 = new XYSeries("threshold ");
        for (Map.Entry<Double, Double> entry1 : threshold.entrySet()) {
            series2.add(entry1.getKey(), entry1.getValue());

        }


        dataset.addSeries(series2);

        return dataset;
    }

    public static JFreeChart test(String applicationTitle, String chartTitle, String name) throws IOException {
        // Create dataset
        XYDataset dataset = createXYDataset();

        // Create chart
        JFreeChart chart = ChartFactory.createScatterPlot(
                "feature score vs relevance",
                "X-Axis", "Y-Axis", dataset);


        //Changes background color
        XYPlot plot = (XYPlot) chart.getPlot();
        plot.setBackgroundPaint(new Color(255, 228, 196));


        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(560, 367));

        File image = new File("chart/chart" + name + ".png");
        FileOutputStream fout = new FileOutputStream(image);
        ChartUtilities.writeChartAsPNG(fout, chart, 600, 400);
        fout.close();

        return chart;

    }

    public static JFreeChart offsetChart(String name)throws IOException{
        System.out.println("In");
        JFreeChart chart1 = ChartFactory.createBarChart(
                "offset",
                "Category",
                "Score",
                offsetDataset(),
                PlotOrientation.VERTICAL,
                true, true, false);
        //Changes background color
        CategoryPlot plot = chart1.getCategoryPlot();
        plot.setBackgroundPaint(new Color(255, 228, 196));


        ChartPanel chartPanel = new ChartPanel(chart1);
        chartPanel.setPreferredSize(new java.awt.Dimension(560, 367));

        File image = new File("offset/chart" + name + ".png");
        FileOutputStream fileout = new FileOutputStream(image);
        ChartUtilities.writeChartAsPNG(fileout, chart1, 1200, 500);
        fileout.close();

        return chart1;
    }

    private static CategoryDataset offsetDataset( ) {
        final DefaultCategoryDataset dataset =
                new DefaultCategoryDataset( );
        System.out.println(offsetNos);

        for (Map.Entry<String, Integer> e : offsetNos.entrySet()) {
            System.out.println(e.getKey());
            System.out.println(e.getValue());

            dataset.addValue(e.getValue(), "test",  e.getKey());
        }
        return dataset;

    }

    public static void main(String[] args) throws IOException, Exception {
        /* We want this to take a file, load it and change feature 46
        * Load the model and rerun on this changed file
        * Plot feature value vs ranking score on a graph
         */

        offsetNos.put("-1.0", 0);
        offsetNos.put("-0.9", 0);
        offsetNos.put("-0.8", 0);
        offsetNos.put("-0.7", 0);
        offsetNos.put("-0.6", 0);
        offsetNos.put("-0.5", 0);
        offsetNos.put("-0.4", 0);
        offsetNos.put("-0.3", 0);
        offsetNos.put("-0.2", 0);
        offsetNos.put("-0.1", 0);
        offsetNos.put("0.0", 0);
        offsetNos.put("0.1", 0);
        offsetNos.put("0.2", 0);
        offsetNos.put("0.3", 0);
        offsetNos.put("0.4", 0);
        offsetNos.put("0.5", 0);
        offsetNos.put("0.6", 0);
        offsetNos.put("0.7", 0);
        offsetNos.put("0.8", 0);
        offsetNos.put("0.9", 0);
        offsetNos.put("1.0", 0);
        System.out.println(offsetNos);



        System.setProperty("terrier.home", "/home/rachel/Desktop/IR/terrier-core-4.2");
        Scanner reader = new Scanner(System.in);
        System.out.println("Please choose a file name : ");
        String name = reader.nextLine();
        reader.close();
        start(name);
        System.out.println(offsetNos);




    }
}













