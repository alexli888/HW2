package org.example;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class OneA {

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

        Configuration conf = new Configuration();
        conf.set("seeds.path", SEEDS_PATH);

        FileSystem localFs = FileSystem.getLocal(conf);
        Path outPath = new Path(OUTPUT_PATH);

        if (localFs.exists(outPath)) {
            localFs.delete(outPath, true);
        }

        Job job = Job.getInstance(conf, "kmeans-one-iteration");
        job.setJarByClass(OneA.class);

        job.setMapperClass(KMapper.class);
        job.setReducerClass(KReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(DATASET_PATH));
        FileOutputFormat.setOutputPath(job, outPath);

        boolean success = job.waitForCompletion(true);

        if (success) {
            System.out.println("Job done! Output directory: " + OUTPUT_PATH);

            FileStatus[] stats = localFs.listStatus(outPath);
            for (FileStatus stat : stats) {
                Path p = stat.getPath();
                if (!p.getName().startsWith("part-")) continue;

                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(localFs.open(p)))) {

                    String line;
                    while ((line = br.readLine()) != null) {
                        System.out.println(line);
                    }
                }
            }
            System.exit(0);
        } else {
            System.err.println("Job failed.");
            System.exit(1);
        }
    }
}
