package com.example.firstsensorcomputing;

import android.app.AlertDialog;
import android.content.Context;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.util.Log;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";

    private List<List<Float>> columnData = new ArrayList<>();
    private int currentIndex = 0;
    private final int windowSize = 50;
    private final int delay = 100; // 100ms
    private Module module;

    private GraphView[] graphViews = new GraphView[6];
    private Handler handler = new Handler();
    private TextView stressLevelTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        try {
            module = Module.load(assetFilePath(this, "transformer_model_oldv.pt"));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        stressLevelTextView = findViewById(R.id.stressLevel);

        // Initialize graph views
        graphViews[0] = findViewById(R.id.graph1);
        graphViews[1] = findViewById(R.id.graph2);
        graphViews[2] = findViewById(R.id.graph3);
        graphViews[3] = findViewById(R.id.graph4);
        graphViews[4] = findViewById(R.id.graph5);
        graphViews[5] = findViewById(R.id.graph6);

        graphViews[0].setLabel("Accelerometer X");
        graphViews[1].setLabel("Accelerometer Y");
        graphViews[2].setLabel("Accelerometer Z");
        graphViews[3].setLabel("Electrodermal Activity (\u00b5S)");
        graphViews[4].setLabel("Heart Rate");
        graphViews[5].setLabel("Temperature (\u00b0C)");

        // Prompt user to enter file name
        promptFileName();
    }

    private String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    private void promptFileName() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Enter File Name");

        final EditText input = new EditText(this);
        builder.setView(input);

        builder.setPositiveButton("OK", (dialog, which) -> {
            String fileName = input.getText().toString().trim();
            if (!fileName.isEmpty()) {
                if (loadCSVData(fileName)) {
                    Toast.makeText(this, "File loaded successfully", Toast.LENGTH_SHORT).show();
                    updateGraphs();
                } else {
                    Toast.makeText(this, "File not found or invalid. Please try again.", Toast.LENGTH_LONG).show();
                    promptFileName(); // Retry if file is not found
                }
            } else {
                Toast.makeText(this, "File name cannot be empty.", Toast.LENGTH_SHORT).show();
                promptFileName();
            }
        });

        builder.setCancelable(false);
        builder.show();
    }

    private boolean loadCSVData(String fileName) {
        try {
            // Path to the external storage location of the CSV file
            File file = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), fileName);
            if (!file.exists()) {
                Log.e(TAG, "CSV file not found at: " + file.getAbsolutePath());
                return false;
            }

            BufferedReader reader = new BufferedReader(new FileReader(file));
            String line;

            boolean firstLine = true; // Use this flag to skip the header row
            columnData.clear(); // Clear any existing data

            while ((line = reader.readLine()) != null) {
                if (firstLine) {
                    firstLine = false; // Skip the first line
                    continue;
                }

                String[] values = line.split(",");
                if (columnData.isEmpty()) {
                    // Initialize columnData list for each column
                    for (int i = 0; i < values.length; i++) {
                        columnData.add(new ArrayList<>());
                    }
                }

                for (int i = 0; i < values.length; i++) {
                    columnData.get(i).add(Float.parseFloat(values[i].trim())); // Parse and add the float value
                }
            }

            reader.close();
            return true;
        } catch (Exception e) {
            Log.e(TAG, "Error loading CSV file: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }

    private void updateGraphs() {
        handler.postDelayed(new Runnable() {
            @Override
            public void run() {
                if (columnData.isEmpty()) {
                    Log.e(TAG, "No data available to plot.");
                    return;
                }
                float[] inputArray = new float[graphViews.length];
                for (int i = 0; i < graphViews.length; i++) {
                    List<Float> data = columnData.get(i);
                    List<Float> window = new ArrayList<>();

                    int startIndex = currentIndex;
                    for (int j = 0; j < windowSize; j++) {
                        int index = (startIndex + j) % data.size();
                        window.add(data.get(index));
                    }
                    int midIndex = windowSize / 2;
                    inputArray[i] = window.get(midIndex);

                    graphViews[i].setData(window, graphViews[i].getLabel());
                }

                Tensor inputTensor = Tensor.fromBlob(inputArray, new long[]{1, inputArray.length}); // 1 batch, features equal to graphViews.length
                Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
                float[] scores = outputTensor.getDataAsFloatArray();

                // Log the appropriate state based on the highest score index
                int maxIndex = getMaxIndex(scores);
                runOnUiThread(() -> {
                    switch (maxIndex) {
                        case 0:
                            stressLevelTextView.setText("Stress Level: High");
                            break;
                        case 1:
                            stressLevelTextView.setText("Stress Level: Normal");
                            break;
                        case 2:
                            stressLevelTextView.setText("Stress Level: Low");
                            break;
                        default:
                            stressLevelTextView.setText("Stress Level: Unknown");
                    }
                });

                currentIndex = (currentIndex + 1) % columnData.get(0).size();
                handler.postDelayed(this, delay);
            }
        }, delay);
    }

    private int getMaxIndex(float[] scores) {
        int maxIndex = 0;
        for (int i = 1; i < scores.length; i++) {
            if (scores[i] > scores[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}
