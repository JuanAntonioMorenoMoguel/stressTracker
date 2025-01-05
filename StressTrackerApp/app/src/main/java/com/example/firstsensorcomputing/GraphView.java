package com.example.firstsensorcomputing;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;

import java.util.List;

public class GraphView extends View {

    private Paint linePaint;
    private Paint textPaint;
    private List<Float> data;
    private String label = "";

    public GraphView(Context context, AttributeSet attrs) {
        super(context, attrs);

        // Initialize paints
        linePaint = new Paint();
        linePaint.setColor(Color.BLUE);
        linePaint.setStrokeWidth(5f);
        linePaint.setStyle(Paint.Style.STROKE);

        textPaint = new Paint();
        textPaint.setColor(Color.BLACK);
        textPaint.setTextSize(40f);
    }

    public void setData(List<Float> data, String label) {
        this.data = data;
        this.label = label;
        invalidate(); // Trigger redraw
    }

    public void setLabel(String label) {
        this.label = label;
        invalidate(); // Redraw the view with the new label
    }

    public String getLabel() {
        return this.label;
    }


    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        if (data == null || data.isEmpty()) {
            return; // No data to draw
        }

        float width = getWidth();
        float height = getHeight();
        float maxValue = Float.MIN_VALUE;
        float minValue = Float.MAX_VALUE;

        // Find the min and max values in the data
        for (Float value : data) {
            if (value > maxValue) maxValue = value;
            if (value < minValue) minValue = value;
        }

        // Leave some padding for better visualization
        float padding = 50;
        float graphHeight = height - padding * 3; // Additional space for the label
        float graphWidth = width - padding * 2;

        // Draw the border around the graph
        Paint borderPaint = new Paint();
        borderPaint.setColor(Color.BLACK);
        borderPaint.setStyle(Paint.Style.STROKE);
        borderPaint.setStrokeWidth(4f);

        canvas.drawRect(padding, padding * 2, padding + graphWidth, padding * 2 + graphHeight, borderPaint);

        // Draw the Y-axis with tick marks and labels
        int yTicks = 5; // Number of Y-axis tick marks
        float yStep = graphHeight / yTicks; // Spacing between tick marks
        float valueStep = (maxValue - minValue) / yTicks; // Value increment per tick

        Paint axisPaint = new Paint();
        axisPaint.setColor(Color.BLACK);
        axisPaint.setStrokeWidth(2f);

        for (int i = 0; i <= yTicks; i++) {
            float y = padding * 2 + graphHeight - (i * yStep);
            float value = minValue + i * valueStep;

            // Draw tick mark
            canvas.drawLine(padding, y, padding - 10, y, axisPaint);

            // Draw Y-axis label
            canvas.drawText(String.format("%.1f", value), padding - 40, y + 10, textPaint);
        }

        // Draw the graph label above the graph
        canvas.drawText(label, width / 2 - textPaint.measureText(label) / 2, padding, textPaint);

        // Scale data points to fit within the view
        float xStep = graphWidth / (data.size() - 1);
        float prevX = padding;
        float prevY = padding * 2 + graphHeight - ((data.get(0) - minValue) / (maxValue - minValue) * graphHeight);

        for (int i = 1; i < data.size(); i++) {
            float x = padding + i * xStep;
            float y = padding * 2 + graphHeight - ((data.get(i) - minValue) / (maxValue - minValue) * graphHeight);

            // Draw the line segment between the previous and current points
            canvas.drawLine(prevX, prevY, x, y, linePaint);
            prevX = x;
            prevY = y;
        }

        // Calculate the middle index
        int middleIndex = data.size() / 2;
        float middleX = padding + middleIndex * xStep;
        float middleY = padding * 2 + graphHeight - ((data.get(middleIndex) - minValue) / (maxValue - minValue) * graphHeight);

        // Draw the red dot at the middle value
        Paint dotPaint = new Paint();
        dotPaint.setColor(Color.RED);
        canvas.drawCircle(middleX, middleY, 10, dotPaint);

        // Draw a small box with the middle value
        String middleValueText = String.format("%.2f", data.get(middleIndex));
        float textWidth = textPaint.measureText(middleValueText);
        float boxPadding = 10;

        Paint boxPaint = new Paint();
        boxPaint.setColor(Color.WHITE);
        boxPaint.setStyle(Paint.Style.FILL);

        Paint boxBorderPaint = new Paint();
        boxBorderPaint.setColor(Color.BLACK);
        boxBorderPaint.setStyle(Paint.Style.STROKE);
        boxBorderPaint.setStrokeWidth(2f);

        float boxX = middleX - textWidth / 2 - boxPadding;
        float boxY = middleY - boxPadding - 50; // Adjust box position relative to the dot

        // Draw the value box with border
        canvas.drawRect(boxX, boxY, boxX + textWidth + boxPadding * 2, boxY + 50, boxPaint);
        canvas.drawRect(boxX, boxY, boxX + textWidth + boxPadding * 2, boxY + 50, boxBorderPaint);

        // Draw the value text inside the box
        canvas.drawText(middleValueText, boxX + boxPadding, boxY + 35, textPaint);

        // Draw Y-axis line
        canvas.drawLine(padding, padding * 2, padding, padding * 2 + graphHeight, axisPaint);
    }


}
