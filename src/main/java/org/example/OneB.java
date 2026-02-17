package org.example;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class OneB {

    public static class KMapper
            extends Mapper<Object, Text, Text, Text> {

        private final List<double[]> centroids = new ArrayList<>();

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {

            Configuration conf = context.getConfiguration();
            String seedsPath = conf.get("seeds.path");

            if (seedsPath == null || seedsPath.isEmpty()) {
                throw new IOException("Missing seeds.path in configuration.");
            }

            Path p = new Path(seedsPath);
            FileSystem fs = FileSystem.getLocal(conf);

            // Load centroids
            try (BufferedReader br = new BufferedReader(
                    new InputStreamReader(fs.open(p)))) {

                String line;
                while ((line = br.readLine()) != null) {
                    line = line.trim();
                    if (line.isEmpty()) continue;

                    String[] toks = line.split("\\s+");
                    if (toks.length < 2) continue;

                    double x = Double.parseDouble(toks[0]);
                    double y = Double.parseDouble(toks[1]);

                    centroids.add(new double[]{x, y});
                }
            }

            if (centroids.isEmpty()) {
                throw new IOException("No centroids loaded.");
            }

            // Emit dummy value for each centroid, so we don't lose centroids with no assigned points
            for (int i = 0; i < centroids.size(); i++) {
                context.write(new Text("c" + i), new Text("DUMMY"));
            }
        }

        @Override
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {

            String line = value.toString().trim();
            if (line.isEmpty()) return;

            String[] toks = line.split("\\s+");
            if (toks.length < 2) return;

            double x = Double.parseDouble(toks[0]);
            double y = Double.parseDouble(toks[1]);

            // Find nearest centroid
            int bestIdx = -1;
            double bestDist = Double.MAX_VALUE;

            for (int i = 0; i < centroids.size(); i++) {
                double[] c = centroids.get(i);
                double dx = x - c[0];
                double dy = y - c[1];
                double dist = dx * dx + dy * dy;

                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx = i;
                }
            }

            context.write(new Text("c" + bestIdx), new Text(x + " " + y));
        }
    }

    public static class KReducer
            extends Reducer<Text, Text, Text, Text> {

        private final List<double[]> oldCentroids = new ArrayList<>();

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {

            Configuration conf = context.getConfiguration();
            String seedsPath = conf.get("seeds.path");

            Path p = new Path(seedsPath);
            FileSystem fs = FileSystem.getLocal(conf);

            try (BufferedReader br = new BufferedReader(
                    new InputStreamReader(fs.open(p)))) {

                String line;
                while ((line = br.readLine()) != null) {
                    line = line.trim();
                    if (line.isEmpty()) continue;

                    String[] toks = line.split("\\s+");
                    if (toks.length < 2) continue;

                    double x = Double.parseDouble(toks[0]);
                    double y = Double.parseDouble(toks[1]);

                    oldCentroids.add(new double[]{x, y});
                }
            }
        }

        @Override
        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            double sumX = 0.0;
            double sumY = 0.0;
            long count = 0L;

            for (Text val : values) {

                String v = val.toString();

                if (v.equals("DUMMY")) continue;

                String[] toks = v.trim().split("\\s+");
                if (toks.length < 2) continue;

                double x = Double.parseDouble(toks[0]);
                double y = Double.parseDouble(toks[1]);

                sumX += x;
                sumY += y;
                count++;
            }

            int centroidIndex = Integer.parseInt(key.toString().substring(1));

            if (count == 0) {
                double[] old = oldCentroids.get(centroidIndex);
                context.write(key, new Text(old[0] + " " + old[1] + " 0"));
            } else {
                double avgX = sumX / count;
                double avgY = sumY / count;
                context.write(key, new Text(avgX + " " + avgY + " " + count));
            }
        }
    }

    public static void main(String[] args) throws Exception {

        final String DATASET_PATH = "/Users/alexli/CS4433/HW2/src/data/dataset.txt";
        final String SEEDS_PATH   = "/Users/alexli/CS4433/HW2/src/data/seeds.txt";
        final String OUTPUT_PATH  = "/Users/alexli/CS4433/HW2/src/data/output";
        final int R = 10; // number of iterations

        Configuration conf = new Configuration();
        FileSystem localFs = FileSystem.getLocal(conf);

        // Read initial seeds to determine K and preserve order
        List<String> seeds = new ArrayList<>();
        Path seedsP = new Path(SEEDS_PATH);
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(localFs.open(seedsP)))) {

            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                seeds.add(line);
            }
        }

        if (seeds.isEmpty()) {
            throw new IOException("No centroids loaded from seeds file: " + SEEDS_PATH);
        }

        // Iterative loop
        for (int iter = 0; iter < R; iter++) {

            System.out.println("Starting iteration " + (iter + 1) + " / " + R);

            conf.set("seeds.path", SEEDS_PATH);

            Path outPath = new Path(OUTPUT_PATH + "/iter-" + iter);
            if (localFs.exists(outPath)) {
                localFs.delete(outPath, true);
            }

            Job job = Job.getInstance(conf, "kmeans-iter-" + iter);
            job.setJarByClass(OneB.class);

            job.setMapperClass(KMapper.class);
            job.setReducerClass(KReducer.class);

            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);

            job.setMapOutputKeyClass(Text.class);
            job.setMapOutputValueClass(Text.class);

            FileInputFormat.addInputPath(job, new Path(DATASET_PATH));
            FileOutputFormat.setOutputPath(job, outPath);

            boolean success = job.waitForCompletion(true);

            if (!success) {
                System.err.println("Job failed at iteration " + iter);
                System.exit(1);
            }

            // Prepare an ordered array for new seeds; fill with nulls initially
            String[] newSeeds = new String[seeds.size()];
            // Read reducer outputs (part-* files)
            FileStatus[] stats = localFs.listStatus(outPath);
            for (FileStatus stat : stats) {
                Path p = stat.getPath();
                if (!p.getName().startsWith("part-")) continue;

                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(localFs.open(p)))) {

                    String line;
                    while ((line = br.readLine()) != null) {
                        line = line.trim();
                        if (line.isEmpty()) continue;

                        // Expected format: c{idx}\t<x> <y> <count>
                        String[] toks = line.split("\\s+");
                        if (toks.length < 3) continue;

                        String key = toks[0]; // e.g., c0
                        int idx = Integer.parseInt(key.substring(1));
                        String x = toks[1];
                        String y = toks[2];

                        if (idx >= 0 && idx < newSeeds.length) {
                            newSeeds[idx] = x + " " + y;
                        } else {
                            System.err.println("Warning: centroid index out of range: " + idx);
                        }
                    }
                }
            }

            // For any centroid with no emitted value (null), keep previous centroid
            for (int i = 0; i < newSeeds.length; i++) {
                if (newSeeds[i] == null) {
                    newSeeds[i] = seeds.get(i);
                }
            }

            // Overwrite seeds file for next iteration
            try (FSDataOutputStream out = localFs.create(seedsP, true);
                 BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(out))) {

                for (String s : newSeeds) {
                    bw.write(s);
                    bw.newLine();
                }
            }

            // Update seeds list for next iteration
            seeds = Arrays.asList(newSeeds);
            System.out.println("Iteration " + (iter + 1) + " complete.");
        }

        // After R iterations print final centroids
        System.out.println("KMeans finished after " + R + " iterations. Final centroids:");
        for (String s : seeds) {
            System.out.println(s);
        }

        System.exit(0);
    }
}
