
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os
import json
from math import pi
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Create templates directory
os.makedirs('templates', exist_ok=True)

# Define feature columns
features = ["pm25", "pm10", "no", "no2", "nox", "nh3", "so2", "co", "o3", "benzene", 
            "humidity", "wind_speed", "wind_direction", "solar_radiation", "rainfall", "air_temperature"]

# Load models
scaler = joblib.load("scaler.pkl")
feature_names = features  # Use the feature names defined above
label_encoder = joblib.load("label_encoder.pkl")  # Load label encoder for classification results

# Load metrics data for visualization
metrics_df = pd.read_csv("model_metrics.csv", index_col=0)

@app.route('/')
def home():
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convert Accuracy column to percentage if it exists and is not already in percentage form
        if 'Accuracy' in metrics_df.columns and metrics_df["Accuracy"].max() <= 1:
            metrics_df["Accuracy"] = (metrics_df["Accuracy"] * 100).round(2)
        
        # Get input values from form or JSON
        if request.is_json:
            data = request.get_json()
            input_values = data.get('features', [])
        else:
            input_values = []
            for feature in features:
                value = request.form.get(feature, 0)
                input_values.append(float(value))
        
        # Check if we have the right number of features
        if len(input_values) != len(features):
            return jsonify({'error': f'Expected {len(features)} features, got {len(input_values)}'})
        
        # Scale input data - create a DataFrame with feature names to avoid the warning
        input_df = pd.DataFrame([input_values], columns=features)
        input_scaled = scaler.transform(input_df)
        
        # Define models with exact filenames
        models = {
            "RandomForest": ("RandomForest_reg.pkl", "RandomForest_clf.pkl"),
            "KNN": ("KNN_reg.pkl", "KNN_clf.pkl"),
            "NaiveBayes": (None, "NaiveBayes_clf.pkl"),  # No regression model
            "SVM": ("SVM_reg.pkl", "SVM_clf.pkl"),
            "LightGBM": ("LightGBM_reg.pkl", "LightGBM_clf.pkl"),
            "MLP": ("MLP_reg.pkl", "MLP_clf.pkl")
        }
        
        # Get predictions from all models
        results = {}
        for model_name, (reg_file, clf_file) in models.items():
            model_results = {}
            
            # Regression Model Prediction
            if reg_file and os.path.exists(reg_file):
                reg_model = joblib.load(reg_file)
                # For LightGBM regressor, we need to pass a DataFrame with feature names
                if model_name == "LightGBM":
                    y_reg_pred = reg_model.predict(pd.DataFrame(input_scaled, columns=features))[0]
                else:
                    y_reg_pred = reg_model.predict(input_scaled)[0]
                model_results["Predicted Efficiency"] = float(y_reg_pred)
            
            # Classification Model Prediction
            if os.path.exists(clf_file):
                clf_model = joblib.load(clf_file)
                # For LightGBM classifier, we need to pass a DataFrame with feature names
                if model_name == "LightGBM":
                    y_clf_pred = clf_model.predict(pd.DataFrame(input_scaled, columns=features))[0]
                else:
                    y_clf_pred = clf_model.predict(input_scaled)[0]
                y_clf_label = label_encoder.inverse_transform([y_clf_pred])[0]
                model_results["Predicted Category"] = y_clf_label
            
            # Store results only if we have any predictions
            if model_results:
                results[model_name] = model_results
        
        # Convert results to DataFrame for further processing
        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_df.to_csv("model_test_results.csv")
        
        # Prepare visualization data in JSON format
        visualization_data = {}
        
        # 1. Performance Bar Chart Data
        visualization_data['performance_barchart'] = {
            'data': metrics_df.to_dict('list'),
            'models': metrics_df.index.tolist(),
            'metrics': metrics_df.columns.tolist(),
            'title': 'Model Performance Analysis'
        }
        
        # 2. Metrics Heatmap Data
        correlation_matrix = metrics_df.corr().round(2)
        visualization_data['metrics_heatmap'] = {
            'data': correlation_matrix.to_dict('records'),
            'labels': correlation_matrix.columns.tolist(),
            'title': 'Performance Metrics Correlation Heatmap'
        }
        
        # 3. Radar Chart Data
        labels = metrics_df.columns.tolist()
        num_vars = len(labels)
        
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles_list = [round(angle, 4) for angle in angles]
        
        radar_data = {
            'labels': labels,
            'angles': angles_list,
            'models': {}
        }
        
        for model in metrics_df.index:
            values = metrics_df.loc[model].tolist()
            radar_data['models'][model] = values
        
        visualization_data['radar_chart'] = {
            'data': radar_data,
            'title': 'Model Performance Radar Chart'
        }
        
        # 4. Performance Trends Data
        visualization_data['performance_trends'] = {
            'data': metrics_df.T.to_dict('list'),
            'metrics': metrics_df.columns.tolist(),
            'models': metrics_df.index.tolist(),
            'title': 'Model Performance Trends'
        }
        
        # 5. Feature Importance Data
        feature_importance_files = [
            "RandomForest_feature_importance.csv",
            "KNN_feature_importance.csv",
            "NaiveBayes_feature_importance.csv", 
            "SVM_feature_importance.csv",
            "LightGBM_feature_importance.csv",
            "MLP_feature_importance.csv"
        ]
        
        # Check which files actually exist
        existing_files = [f for f in feature_importance_files if os.path.exists(f)]
        
        feature_importance_data = {}
        if existing_files:
            for file in existing_files:
                model_name = file.replace('_feature_importance.csv', '')
                df_importance = pd.read_csv(file)
                feature_importance_data[model_name] = {
                    'features': df_importance["Feature"].tolist(),
                    'importance': df_importance["Importance"].tolist()
                }
            
            visualization_data['feature_importance'] = {
                'data': feature_importance_data,
                'title': 'Feature Importance Across Models'
            }
        
        # 6. ROC Curves Data
        roc_data = {}
        has_roc_data = False
        
        for model_name, (_, clf_file) in models.items():
            if not os.path.exists(clf_file):
                continue
                
            clf_model = joblib.load(clf_file)
            if hasattr(clf_model, 'predict_proba'):
                has_roc_data = True
                
                # Use DataFrame with feature names for LightGBM
                if model_name == "LightGBM":
                    y_clf_prob = clf_model.predict_proba(pd.DataFrame(input_scaled, columns=features))[0]
                else:
                    y_clf_prob = clf_model.predict_proba(input_scaled)[0]
                
                model_roc_data = []
                for i in range(len(y_clf_prob)):
                    fpr, tpr, _ = roc_curve(
                        [1 if j == i else 0 for j in range(len(y_clf_prob))], 
                        [1 if j == i else 0 for j in range(len(y_clf_prob))]
                    )
                    roc_auc = auc(fpr, tpr)
                    
                    model_roc_data.append({
                        'class': i,
                        'auc': round(roc_auc, 2),
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist()
                    })
                
                roc_data[model_name] = model_roc_data
        
        if has_roc_data:
            visualization_data['roc_curves'] = {
                'data': roc_data,
                'title': 'ROC Curves Across Models and Classes'
            }
        
        # 7. Individual Model Radar Charts Data
        individual_radar_data = {}
        metrics = metrics_df.columns.tolist()
        num_vars = len(metrics)
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles_list = [round(angle, 4) for angle in angles]

        for model in metrics_df.index:
            values = metrics_df.loc[model].tolist()
            individual_radar_data[model] = {
                'metrics': metrics,
                'angles': angles_list,
                'values': values,
                'title': f"{model} Performance Radar"
            }
        
        visualization_data['individual_radar_charts'] = individual_radar_data
        
        # 8. Individual Metric Analysis Data
        metric_comparison_data = {}
        metrics = metrics_df.columns
        for metric in metrics:
            metric_comparison_data[metric] = {
                'models': metrics_df.index.tolist(),
                'values': metrics_df[metric].tolist(),
                'title': f"{metric} Comparison Across Models"
            }
        
        visualization_data['metric_comparisons'] = metric_comparison_data
        
        # 9. Individual Feature Analysis Data
        feature_analyses_data = {}
        if existing_files:
            for file in existing_files:
                model_name = file.replace('_feature_importance.csv', '')
                df_importance = pd.read_csv(file)
                feature_analyses_data[model_name] = {
                    'features': df_importance["Feature"].tolist(),
                    'importance': df_importance["Importance"].tolist(),
                    'title': f"Feature Importance for {model_name}"
                }
        
        visualization_data['feature_analyses'] = feature_analyses_data
        
        # Convert metrics DataFrame to a format suitable for the results template
        metrics_dict = metrics_df.to_dict('records')
        metrics_with_index = {}
        for i, row in enumerate(metrics_dict):
            metrics_with_index[metrics_df.index[i]] = row
        
        # Return response based on request type
        if request.is_json:
            return jsonify({
                'results': results,
                'visualization_data': visualization_data,
                'success': True
            })
        else:
            return render_template(
                'results.html', 
                results=results, 
                metrics=metrics_with_index,
                visualization_data=json.dumps(visualization_data)
            )
    
    except Exception as e:
        error_message = str(e)
        print(f"Error: {error_message}")
        if request.is_json:
            return jsonify({'error': error_message, 'success': False})
        else:
            return render_template('error.html', error=error_message)

# Create index.html
index_html = '''
<!DOCTYPE html>
<html>
<head>
    <title>Air Quality Analysis</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding: 20px; }
        .container { max-width: 960px; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center mb-4">Air Quality Analysis Model</h1>
        
        <div class="card">
            <div class="card-header">
                <h3>Enter Feature Values</h3>
            </div>
            <div class="card-body">
                <form action="/predict" method="post">
                    <div class="row">
                        {% for feature in features %}
                        <div class="col-md-4 mb-3">
                            <label for="{{ feature }}" class="form-label">{{ feature }}</label>
                            <input type="number" step="any" class="form-control" id="{{ feature }}" name="{{ feature }}" required>
                        </div>
                        {% endfor %}
                    </div>
                    
                    <div class="text-center mt-3">
                        <button type="submit" class="btn btn-primary">Analyze Data</button>
                    </div>
                </form>
            </div>
        </div>
    </div>
</body>
</html>
'''

# Create results.html with Chart.js visualization
results_html = '''
<!DOCTYPE html>
<html>
<head>
    <title>Analysis Results</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-chart-matrix@1.1.1/dist/chartjs-chart-matrix.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2.0.0"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@2.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
    <style>
        body { padding: 20px; }
        .container { max-width: 1200px; }
        .chart-container { 
            position: relative;
            height: 400px;
            width: 100%;
            margin-bottom: 30px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center mb-4">Analysis Results</h1>
        
        <!-- Model Predictions -->
        <div class="card mb-4">
            <div class="card-header">
                <h3>Model Predictions</h3>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-striped">
                        <thead>
                            <tr>
                                <th>Model</th>
                                {% for key in results[results.keys()|list|first].keys() %}
                                <th>{{ key }}</th>
                                {% endfor %}
                            </tr>
                        </thead>
                        <tbody>
                            {% for model, result in results.items() %}
                            <tr>
                                <td><strong>{{ model }}</strong></td>
                                {% for key, value in result.items() %}
                                <td>{{ value }}</td>
                                {% endfor %}
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <!-- Model Performance Metrics Table -->
        <div class="card mb-4">
            <div class="card-header">
                <h3>Model Performance Metrics Table</h3>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-striped">
                        <thead>
                            <tr>
                                <th>Model</th>
                                {% for metric in metrics[metrics.keys()|list|first].keys() %}
                                <th>{{ metric }}</th>
                                {% endfor %}
                            </tr>
                        </thead>
                        <tbody>
                            {% for model, metric_values in metrics.items() %}
                            <tr>
                                <td><strong>{{ model }}</strong></td>
                                {% for metric, value in metric_values.items() %}
                                <td>{{ value }}</td>
                                {% endfor %}
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <!-- Visualizations -->
        <div class="card mb-4">
            <div class="card-header">
                <h3>Model Performance Analysis</h3>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6 mb-4">
                        <h4>Performance Bar Chart</h4>
                        <div class="chart-container">
                            <canvas id="performanceBarChart"></canvas>
                        </div>
                    </div>
                    <div class="col-md-6 mb-4">
                        <h4>Metrics Heatmap</h4>
                        <div class="chart-container">
                            <canvas id="metricsHeatmap"></canvas>
                        </div>
                    </div>
                    <div class="col-md-6 mb-4">
                        <h4>Radar Chart</h4>
                        <div class="chart-container">
                            <canvas id="radarChart"></canvas>
                        </div>
                    </div>
                    <div class="col-md-6 mb-4">
                        <h4>Performance Trends</h4>
                        <div class="chart-container">
                            <canvas id="performanceTrends"></canvas>
                        </div>
                    </div>
                    <div class="col-md-6 mb-4">
                        <h4>Feature Importance</h4>
                        <div class="chart-container">
                            <canvas id="featureImportance"></canvas>
                        </div>
                    </div>
                    <div class="col-md-6 mb-4">
                        <h4>ROC Curves</h4>
                        <div class="chart-container">
                            <canvas id="rocCurves"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Individual Model Radar Charts -->
        <div class="card mb-4">
            <div class="card-header">
                <h3>Individual Model Radar Charts</h3>
            </div>
            <div class="card-body">
                <div class="row" id="individualRadarCharts">
                    <!-- Individual radar charts will be inserted here -->
                </div>
            </div>
        </div>

        <!-- Individual Metric Analysis -->
        <div class="card mb-4">
            <div class="card-header">
                <h3>Individual Metric Analysis</h3>
            </div>
            <div class="card-body">
                <div class="row" id="metricComparisonCharts">
                    <!-- Individual metric charts will be inserted here -->
                </div>
            </div>
        </div>

        <!-- Individual Feature Analysis -->
        <div class="card mb-4">
            <div class="card-header">
                <h3>Individual Feature Analysis</h3>
            </div>
            <div class="card-body">
                <div class="row" id="featureAnalysisCharts">
                    <!-- Individual feature charts will be inserted here -->
                </div>
            </div>
        </div>
        
        <div class="text-center">
            <a href="/" class="btn btn-primary">Back to Input Form</a>
        </div>
    </div>

    <script>
        // Parse visualization data from server
        const visualizationData = JSON.parse('{{ visualization_data|safe }}');
        
        // Helper function to generate random colors with transparency
        function generateColors(count, alpha = 1) {
            const colors = [];
            for(let i = 0; i < count; i++) {
                const r = Math.floor(Math.random() * 255);
                const g = Math.floor(Math.random() * 255);
                const b = Math.floor(Math.random() * 255);
                colors.push(`rgba(${r}, ${g}, ${b}, ${alpha})`);
            }
            return colors;
        }
        
        // Generate color palette for models
        const modelColors = {};
        const models = Object.keys(visualizationData.performance_barchart.data);
        const modelColorPalette = generateColors(models.length);
        models.forEach((model, index) => {
            modelColors[model] = modelColorPalette[index];
        });
        
        // 1. Performance Bar Chart
        const performanceBarChartData = visualizationData.performance_barchart;
        const performanceBarChart = new Chart(
            document.getElementById('performanceBarChart').getContext('2d'),
            {
                type: 'bar',
                data: {
                    labels: performanceBarChartData.models,
                    datasets: performanceBarChartData.metrics.map((metric, index) => ({
                        label: metric,
                        data: performanceBarChartData.models.map(model => performanceBarChartData.data[model][metric]),
                        backgroundColor: generateColors(1)[0],
                        borderColor: 'rgba(0, 0, 0, 1)',
                        borderWidth: 1
                    }))
                },
                options: {
                    plugins: {
                        title: {
                            display: true,
                            text: performanceBarChartData.title
                        },
                        legend: {
                            position: 'top',
                        }
                    },
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            }
        );
        
        // 2. Metrics Heatmap (using regular Chart.js)
        const metricsHeatmapData = visualizationData.metrics_heatmap;
        const heatmapLabels = metricsHeatmapData.labels;
        
        // Create a custom heatmap using regular Chart.js
        const heatmapDataValues = [];
        const heatmapBackgroundColors = [];
        
        for (let i = 0; i < heatmapLabels.length; i++) {
            for (let j = 0; j < heatmapLabels.length; j++) {
                const value = metricsHeatmapData.data[i][heatmapLabels[j]];
                heatmapDataValues.push(value);
                
                // Generate color based on correlation value (-1 to 1)
                const normalizedValue = (value + 1) / 2; // Convert from [-1,1] to [0,1]
                const r = normalizedValue < 0.5 ? 255 : Math.floor(255 * (1 - normalizedValue) * 2);
                const b = normalizedValue > 0.5 ? 255 : Math.floor(255 * normalizedValue * 2);
                const g = 128;
                heatmapBackgroundColors.push(`rgba(${r}, ${g}, ${b}, 0.8)`);
            }
        }
        
        const metricsHeatmap = new Chart(
            document.getElementById('metricsHeatmap').getContext('2d'),
            {
                type: 'bar',
                data: {
                    labels: Array.from({ length: heatmapLabels.length * heatmapLabels.length }, (_, i) => {
                        const row = Math.floor(i / heatmapLabels.length);
                        const col = i % heatmapLabels.length;
                        return `${heatmapLabels[row]} / ${heatmapLabels[col]}`;
                    }),
                    datasets: [{
                        data: heatmapDataValues,
                        backgroundColor: heatmapBackgroundColors,
                        barPercentage: 1.0,
                        categoryPercentage: 1.0
                    }]
                },
                options: {
                    plugins: {
                        title: {
                            display: true,
                            text: metricsHeatmapData.title
                        },
                        legend: {
                            display: false
                        },
                        tooltip: {
                            callbacks: {
                                title: function(context) {
                                    const index = context[0].dataIndex;
                                    const row = Math.floor(index / heatmapLabels.length);
                                    const col = index % heatmapLabels.length;
                                    return `${heatmapLabels[row]} / ${heatmapLabels[col]}`;
                                },
                                label: function(context) {
                                    return `Value: ${context.raw.toFixed(2)}`;
                                }
                            }
                        }
                    },
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'y',
                    scales: {
                        x: {
                            display: false
                        },
                        y: {
                            display: false
                        }
                    }
                }
            }
        );
        
        // 3. Radar Chart
        const radarChartData = visualizationData.radar_chart;
        const radarChart = new Chart(
            document.getElementById('radarChart').getContext('2d'),
            {
                type: 'radar',
                data: {
                    labels: radarChartData.data.labels,
                    datasets: Object.keys(radarChartData.data.models).map((model, index) => ({
                        label: model,
                        data: radarChartData.data.models[model],
                        backgroundColor: `${modelColors[model].replace('1)', '0.2)')}`,
                        borderColor: modelColors[model],
                        borderWidth: 2,
                        pointBackgroundColor: modelColors[model],
                        pointRadius: 3
                    }))
                },
                options: {
                    plugins: {
                        title: {
                            display: true,
                            text: radarChartData.title
                        }
                    },
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        r: {
                            angleLines: {
                                display: true
                            },
                            min: 0
                        }
                    }
                }
            }
        );
        
        // 4. Performance Trends
        const performanceTrendsData = visualizationData.performance_trends;
        const performanceTrends = new Chart(
            document.getElementById('performanceTrends').getContext('2d'),
            {
                type: 'line',
                data: {
                    labels: performanceTrendsData.metrics,
                    datasets: performanceTrendsData.models.map((model, index) => ({
                        label: model,
                        data: performanceTrendsData.metrics.map(metric => performanceTrendsData.data[metric][model]),
                        borderColor: modelColors[model],
                        backgroundColor: `${modelColors[model].replace('1)', '0.2)')}`,
                        borderWidth: 2,
                        tension: 0.3,
                        fill: false,
                        pointRadius: 4
                    }))
                },
                options: {
                    plugins: {
                        title: {
                            display: true,
                            text: performanceTrendsData.title
                        }
                    },
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: false
                        }
                    }
                }
            }
        );
        
        // 5. Feature Importance
        if (visualizationData.feature_importance) {
            const featureImportanceData = visualizationData.feature_importance;
            
            // Combine all models' feature importance data
            const allFeatures = new Set();
            Object.values(featureImportanceData.data).forEach(modelData => {
                modelData.features.forEach(feature => allFeatures.add(feature));
            });
            
            const featureList = Array.from(allFeatures);
            
            // Create datasets for each model
            const datasets = Object.keys(featureImportanceData.data).map((model, index) => {
                const modelData = featureImportanceData.data[model];
                
                // Create array with all features, filling in zeros where the model doesn't have data
                const importanceValues = featureList.map(feature => {
                    const featureIndex = modelData.features.indexOf(feature);
                    return featureIndex >= 0 ? modelData.importance[featureIndex] : 0;
                });
                
                return {
                    label: model,
                    data: importanceValues,
                    backgroundColor: modelColors[model],
                    borderColor: 'rgba(0, 0, 0, 1)',
                    borderWidth: 1
                };
            });
            
            const featureImportance = new Chart(
                document.getElementById('featureImportance').getContext('2d'),
                {
                    type: 'bar',
                    data: {
                        labels: featureList,
                        datasets: datasets
                    },
                    options: {
                        plugins: {
                            title: {
                                display: true,
                                text: featureImportanceData.title
                            }
                        },
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true
                            },
                            x: {
                                ticks: {
                                    autoSkip: false,
                                    maxRotation: 90,
                                    minRotation: 45
                                }
                            }
                        }
                    }
                }
            );
        }   
         
        // 6. ROC Curves
if (visualizationData.roc_curves) {
    const rocCurvesData = visualizationData.roc_curves;

    // Create datasets for each model's ROC curve
    const datasets = [];
    Object.keys(rocCurvesData.data).forEach((model) => {
        rocCurvesData.data[model].forEach((classData) => {
            datasets.push({
                label: `${model} (Class ${classData.class}, AUC = ${classData.auc})`,
                data: classData.fpr.map((fpr, i) => ({ x: fpr, y: classData.tpr[i] })),
                borderColor: `${modelColors[model].replace('1)', '0.8)')}`,
                backgroundColor: 'transparent',
                borderWidth: 2,
                pointRadius: 0,
                lineTension: 0.4
            });
        });
    });

    // Add reference line
    datasets.push({
        label: 'Reference Line',
        data: [
            { x: 0, y: 0 },
            { x: 1, y: 1 }
        ],
        borderColor: 'rgba(0, 0, 0, 0.3)',
        borderWidth: 1,
        borderDash: [5, 5],
        pointRadius: 0,
        backgroundColor: 'transparent'
    });

    // Initialize the ROC curve chart
    new Chart(
        document.getElementById('rocCurves').getContext('2d'),
        {
            type: 'scatter',
            data: { datasets: datasets },
            options: {
                plugins: {
                    title: {
                        display: true,
                        text: rocCurvesData.title
                    }
                },
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: { display: true, text: 'False Positive Rate' },
                        min: 0, max: 1
                    },
                    y: {
                        title: { display: true, text: 'True Positive Rate' },
                        min: 0, max: 1
                    }
                }
            }
        }
    );
}

// 7. Individual Model Radar Charts
const individualRadarChartContainer = document.getElementById('individualRadarCharts');
if (visualizationData.individual_radar_charts) {
    Object.keys(visualizationData.individual_radar_charts).forEach(model => {
        const modelData = visualizationData.individual_radar_charts[model];
        
        const colDiv = document.createElement('div');
        colDiv.className = 'col-md-4 mb-4';
        
        const canvasContainer = document.createElement('div');
        canvasContainer.className = 'chart-container';
        
        const canvas = document.createElement('canvas');
        canvas.id = `radarChart_${model}`;
        
        canvasContainer.appendChild(canvas);
        colDiv.appendChild(document.createElement('h5')).textContent = modelData.title;
        colDiv.appendChild(canvasContainer);
        individualRadarChartContainer.appendChild(colDiv);

        new Chart(
            canvas.getContext('2d'),
            {
                type: 'radar',
                data: {
                    labels: modelData.metrics,
                    datasets: [{
                        label: model,
                        data: modelData.values,
                        backgroundColor: `${modelColors[model].replace('1)', '0.2)')}`,
                        borderColor: modelColors[model],
                        borderWidth: 2,
                        pointBackgroundColor: modelColors[model],
                        pointRadius: 3
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        r: {
                            angleLines: { display: true },
                            min: 0
                        }
                    }
                }
            }
        );
    });
}

// 8. Individual Metric Analysis
const metricComparisonContainer = document.getElementById('metricComparisonCharts');
if (visualizationData.metric_comparisons) {
    Object.keys(visualizationData.metric_comparisons).forEach(metric => {
        const metricData = visualizationData.metric_comparisons[metric];
        
        const colDiv = document.createElement('div');
        colDiv.className = 'col-md-4 mb-4';
        
        const canvasContainer = document.createElement('div');
        canvasContainer.className = 'chart-container';
        
        const canvas = document.createElement('canvas');
        canvas.id = `metricChart_${metric}`;
        
        canvasContainer.appendChild(canvas);
        colDiv.appendChild(document.createElement('h5')).textContent = metricData.title;
        colDiv.appendChild(canvasContainer);
        metricComparisonContainer.appendChild(colDiv);

        new Chart(
            canvas.getContext('2d'),
            {
                type: 'bar',
                data: {
                    labels: metricData.models,
                    datasets: [{
                        label: metric,
                        data: metricData.values,
                        backgroundColor: metricData.models.map(model => modelColors[model]),
                        borderColor: 'rgba(0, 0, 0, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { beginAtZero: true }
                    }
                }
            }
        );
    });
}
// 9. Individual Feature Analysis
        const featureAnalysisContainer = document.getElementById('featureAnalysisCharts');
        
        if (visualizationData.feature_analyses) {
            Object.keys(visualizationData.feature_analyses).forEach(model => {
                const featureData = visualizationData.feature_analyses[model];
                
                // Create canvas element for this chart
                const colDiv = document.createElement('div');
                colDiv.className = 'col-md-4 mb-4';
                
                const canvasContainer = document.createElement('div');
                canvasContainer.className = 'chart-container';
                
                const canvas = document.createElement('canvas');
                canvas.id = `featureChart_${model}`;
                
                canvasContainer.appendChild(canvas);
                colDiv.appendChild(document.createElement('h5')).textContent = featureData.title;
                colDiv.appendChild(canvasContainer);
                featureAnalysisContainer.appendChild(colDiv);
                
                // Create horizontal bar chart for this model's features
                new Chart(
                    canvas.getContext('2d'),
                    {
                        type: 'bar',
                        data: {
                            labels: featureData.features,
                            datasets: [{
                                label: 'Importance',
                                data: featureData.importance,
                                backgroundColor: modelColors[model],
                                borderColor: 'rgba(0, 0, 0, 1)',
                                borderWidth: 1
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            indexAxis: 'y',
                            scales: {
                                x: {
                                    beginAtZero: true
                                }
                            }
                        }
                    }
                );
            });
        }
    </script>
</body>
</html>
'''

# Create error.html
error_html = '''
<!DOCTYPE html>
<html>
<head>
    <title>Error</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding: 20px; }
        .container { max-width: 960px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="alert alert-danger" role="alert">
            <h4 class="alert-heading">Error!</h4>
            <p>{{ error }}</p>
        </div>
        <div class="text-center">
            <a href="/" class="btn btn-primary">Back to Input Form</a>
        </div>
    </div>
</body>
</html>
'''
# Write templates to files
with open('templates/index.html', 'w') as f:
    f.write(index_html)

with open('templates/results.html', 'w') as f:
    f.write(results_html)

with open('templates/error.html', 'w') as f:
    f.write(error_html)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 9000))
    app.run(host='0.0.0.0', port=port, debug=False)