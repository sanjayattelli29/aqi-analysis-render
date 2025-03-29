from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # For server environments
import seaborn as sns
import os
import base64
from io import BytesIO
from math import pi
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Create static directory for plots
os.makedirs('static', exist_ok=True)

# Define feature columns
features = ["pm25", "pm10", "no", "no2", "nox", "nh3", "so2", "co", "o3", "benzene", 
            "humidity", "wind_speed", "wind_direction", "solar_radiation", "rainfall", "air_temperature"]

# Load models
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")  # Load feature names used during training
label_encoder = joblib.load("label_encoder.pkl")  # Load label encoder for classification results

# Load metrics data for visualization
metrics_df = pd.read_csv("model_metrics.csv", index_col=0)

@app.route('/')
def home():
    return render_template('index.html', features=feature_names)  # Use feature_names

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
            for feature in feature_names:
                value = request.form.get(feature, 0)
                input_values.append(float(value))
        
        # Check if we have the right number of features
        if len(input_values) != len(feature_names):
            return jsonify({'error': f'Expected {len(feature_names)} features, got {len(input_values)}'})
        
        # Scale input data
        input_array = np.array(input_values).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        
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
                y_reg_pred = reg_model.predict(input_scaled)[0]
                model_results["Predicted Efficiency"] = float(y_reg_pred)
            
            # Classification Model Prediction
            if os.path.exists(clf_file):
                clf_model = joblib.load(clf_file)
                y_clf_pred = clf_model.predict(input_scaled)[0]
                y_clf_label = label_encoder.inverse_transform([y_clf_pred])[0]
                model_results["Predicted Category"] = y_clf_label
            
            # Store results only if we have any predictions
            if model_results:
                results[model_name] = model_results
        
        # Convert results to DataFrame
        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_df.to_csv("model_test_results.csv")
        
        # Generate visualization plots
        # 1. Performance Bar Chart
        plt.figure(figsize=(12, 6))
        metrics_df.plot(kind='bar', figsize=(12, 6), title="Model Performance Analysis")
        plt.xlabel("Models")
        plt.ylabel("Metric Scores")
        plt.xticks(rotation=45)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig('static/performance_barchart.png')
        plt.close()
        
        # 2. Metrics Heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(metrics_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Performance Metrics Correlation Heatmap")
        plt.tight_layout()
        plt.savefig('static/metrics_heatmap.png')
        plt.close()
        
        # 3. Radar Chart
        labels = metrics_df.columns.tolist()
        num_vars = len(labels)
        
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        for model in metrics_df.index:
            values = metrics_df.loc[model].tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
            ax.fill(angles, values, alpha=0.3)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        plt.title("Model Performance Radar Chart")
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        plt.savefig('static/radar_chart.png')
        plt.close()
        
        # 4. Performance Trends
        plt.figure(figsize=(12, 6))
        metrics_df.T.plot(kind='line', figsize=(12, 6), marker='o', title="Model Performance Trends")
        plt.xlabel("Metrics")
        plt.ylabel("Scores")
        plt.xticks(rotation=45)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig('static/performance_trends.png')
        plt.close()
        
        # 5. Feature Importance Plots
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
        
        if existing_files:
            plt.figure(figsize=(12, 6))
            for file in existing_files:
                df_importance = pd.read_csv(file)
                sns.barplot(x=df_importance["Feature"], y=df_importance["Importance"], label=file.replace("_feature_importance.csv", ""))
            plt.xticks(rotation=45)
            plt.title("Feature Importance Across Models")
            plt.legend()
            plt.tight_layout()
            plt.savefig('static/feature_importance.png')
            plt.close()
        else:
            # Create an empty feature importance plot to avoid errors in template
            plt.figure(figsize=(12, 6))
            plt.text(0.5, 0.5, "No feature importance data available", 
                     horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
            plt.savefig('static/feature_importance.png')
            plt.close()
        
        # 6. ROC Curves
        plt.figure(figsize=(15, 10))
        
        # Color schemes for different models
        color_schemes = {
            'RandomForest': plt.cm.Blues,
            'KNN': plt.cm.Reds,
            'NaiveBayes': plt.cm.Greens,
            'SVM': plt.cm.Oranges,
            'LightGBM': plt.cm.Purples,
            'MLP': plt.cm.Greys
        }
        
        # Line styles for different classes
        class_styles = ['-', '--', '-.', ':']
        
        has_roc_data = False
        for model_name, (_, clf_file) in models.items():
            if not os.path.exists(clf_file):
                continue
                
            clf_model = joblib.load(clf_file)
            if hasattr(clf_model, 'predict_proba'):
                has_roc_data = True
                y_clf_prob = clf_model.predict_proba(input_scaled)[0]
                
                # Get colormap for this model
                cmap = color_schemes.get(model_name, plt.cm.viridis)
                
                for i in range(len(y_clf_prob)):
                    fpr, tpr, _ = roc_curve([1 if j == i else 0 for j in range(len(y_clf_prob))], 
                                          [1 if j == i else 0 for j in range(len(y_clf_prob))])
                    roc_auc = auc(fpr, tpr)
                    
                    # Calculate color based on model type
                    color = cmap(0.3 + 0.7 * ((i + 1) / len(y_clf_prob)))
                    
                    # Plot with consistent styling per class
                    plt.plot(fpr, tpr, 
                             label=f"{model_name} (Class {i}, AUC = {roc_auc:.2f})",
                             linestyle=class_styles[i % len(class_styles)],
                             color=color,
                             linewidth=2,
                             alpha=0.8)
        
        if has_roc_data:
            # Add reference line
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
            
            # Improve appearance
            plt.xlabel("False Positive Rate", fontsize=12)
            plt.ylabel("True Positive Rate", fontsize=12)
            plt.title("ROC Curves Across Models and Classes", fontsize=16)
            plt.grid(True, linestyle='--', alpha=0.3)
            
            plt.legend(
                loc='center left', 
                bbox_to_anchor=(1, 0.5), 
                fontsize=10,
                frameon=True,
                fancybox=True,
                framealpha=0.9,
                ncol=1
            )
            
            plt.xlim([-0.01, 1.01])
            plt.ylim([-0.01, 1.01])
            plt.tick_params(direction='out', labelsize=10)
            plt.tight_layout()
        else:
            # Create an empty ROC plot to avoid errors in template
            plt.text(0.5, 0.5, "No ROC data available", 
                     horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
        
        plt.savefig('static/roc_curves.png')
        plt.close()
        
        # 7. Create Model Performance Metrics Table
        plt.figure(figsize=(12, len(metrics_df) * 0.5))
        ax = plt.subplot(111, frame_on=False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        table = plt.table(cellText=metrics_df.values.round(3),
                         rowLabels=metrics_df.index,
                         colLabels=metrics_df.columns,
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        plt.title('Model Performance Metrics')
        plt.savefig('static/metrics_table.png', bbox_inches='tight')
        plt.close()

        # 8. Individual Model Radar Charts
        metrics = metrics_df.columns.tolist()
        num_vars = len(metrics)
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]

        for model in metrics_df.index:
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            values = metrics_df.loc[model].tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
            ax.fill(angles, values, alpha=0.3)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_title(f"{model} Performance Radar")
            plt.savefig(f'static/{model}_radar.png')
            plt.close()

        # 9. Individual Metric Analysis
        metric_plots = []
        metrics = metrics_df.columns
        for metric in metrics:
            plt.figure(figsize=(10, 5))
            sns.barplot(x=metrics_df.index, y=metrics_df[metric])
            plt.title(f"{metric} Comparison Across Models")
            plt.xlabel("Models")
            plt.ylabel(metric)
            plt.xticks(rotation=45)
            plt.tight_layout()
            filename = f'static/{metric}_comparison.png'
            plt.savefig(filename)
            plt.close()
            metric_plots.append((metric, f'{metric}_comparison.png'))

        # 10. Individual Feature Analysis
        feature_plots = []
        if existing_files:
            for file in existing_files:
                model_name = file.replace('_feature_importance.csv', '')
                df_importance = pd.read_csv(file)
                plt.figure(figsize=(10, 5))
                sns.barplot(x=df_importance["Feature"], y=df_importance["Importance"])
                plt.title(f"Feature Importance for {model_name}")
                plt.xlabel("Feature")
                plt.ylabel("Importance")
                plt.xticks(rotation=45)
                plt.tight_layout()
                filename = f'static/{model_name}_features.png'
                plt.savefig(filename)
                plt.close()
                feature_plots.append((model_name, f'{model_name}_features.png'))
        
        # Prepare response
        plot_paths = {
            'performance_barchart': 'performance_barchart.png',
            'metrics_heatmap': 'metrics_heatmap.png',
            'radar_chart': 'radar_chart.png',
            'performance_trends': 'performance_trends.png',
            'feature_importance': 'feature_importance.png',
            'roc_curves': 'roc_curves.png',
            'metrics_table': 'metrics_table.png',
            'individual_radar_charts': {model: f'{model}_radar.png' for model in metrics_df.index},
            'metric_comparisons': {metric: plot for metric, plot in metric_plots},
            'feature_analyses': {model: plot for model, plot in feature_plots}
        }
        
        # Return response based on request type
        if request.is_json:
            return jsonify({
                'results': results,
                'success': True
            })
        else:
            return render_template(
                'results.html', 
                results=results, 
                metrics=metrics_df.to_dict(),
                plot_paths=plot_paths
            )
    
    except Exception as e:
        error_message = str(e)
        print(f"Error: {error_message}")
        if request.is_json:
            return jsonify({'error': error_message, 'success': False})
        else:
            return render_template('error.html', error=error_message)

# Create basic templates directory and templates if they don't exist
os.makedirs('templates', exist_ok=True)

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

# Create results.html
results_html = '''
<!DOCTYPE html>
<html>
<head>
    <title>Analysis Results</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding: 20px; }
        .container { max-width: 1200px; }
        img { max-width: 100%; }
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
                <img src="{{ url_for('static', filename=plot_paths.metrics_table) }}" alt="Metrics Table" class="img-fluid">
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
                        <img src="{{ url_for('static', filename=plot_paths.performance_barchart) }}" alt="Performance Bar Chart">
                    </div>
                    <div class="col-md-6 mb-4">
                        <h4>Metrics Heatmap</h4>
                        <img src="{{ url_for('static', filename=plot_paths.metrics_heatmap) }}" alt="Metrics Heatmap">
                    </div>
                    <div class="col-md-6 mb-4">
                        <h4>Radar Chart</h4>
                        <img src="{{ url_for('static', filename=plot_paths.radar_chart) }}" alt="Radar Chart">
                    </div>
                    <div class="col-md-6 mb-4">
                        <h4>Performance Trends</h4>
                        <img src="{{ url_for('static', filename=plot_paths.performance_trends) }}" alt="Performance Trends">
                    </div>
                    <div class="col-md-6 mb-4">
                        <h4>Feature Importance</h4>
                        <img src="{{ url_for('static', filename=plot_paths.feature_importance) }}" alt="Feature Importance">
                    </div>
                    <div class="col-md-6 mb-4">
                        <h4>ROC Curves</h4>
                        <img src="{{ url_for('static', filename=plot_paths.roc_curves) }}" alt="ROC Curves">
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
                <div class="row">
                    {% for model, chart_path in plot_paths.individual_radar_charts.items() %}
                    <div class="col-md-6 mb-4">
                        <h4>{{ model }}</h4>
                        <img src="{{ url_for('static', filename=chart_path) }}" alt="{{ model }} Radar Chart">
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>

        <!-- Individual Metric Analysis -->
        <div class="card mb-4">
            <div class="card-header">
                <h3>Individual Metric Analysis</h3>
            </div>
            <div class="card-body">
                <div class="row">
                    {% for metric, chart_path in plot_paths.metric_comparisons.items() %}
                    <div class="col-md-6 mb-4">
                        <h4>{{ metric }}</h4>
                        <img src="{{ url_for('static', filename=chart_path) }}" alt="{{ metric }} Comparison">
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>

        <!-- Individual Feature Analysis -->
        <div class="card mb-4">
            <div class="card-header">
                <h3>Individual Feature Analysis</h3>
            </div>
            <div class="card-body">
                <div class="row">
                    {% for model, chart_path in plot_paths.feature_analyses.items() %}
                    <div class="col-md-6 mb-4">
                        <h4>{{ model }}</h4>
                        <img src="{{ url_for('static', filename=chart_path) }}" alt="{{ model }} Feature Importance">
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
        
        <div class="text-center">
            <a href="/" class="btn btn-primary">Back to Input Form</a>
        </div>
    </div>
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
</head>
<body>
    <div class="container mt-5">
        <div class="alert alert-danger">
            <h4>Error</h4>
            <p>{{ error }}</p>
        </div>
        <a href="/" class="btn btn-primary">Back to Home</a>
    </div>
</body>
</html>
'''

# Write template files if they don't exist
if not os.path.exists('templates/index.html'):
    with open('templates/index.html', 'w') as f:
        f.write(index_html)

if not os.path.exists('templates/results.html'):
    with open('templates/results.html', 'w') as f:
        f.write(results_html)

if not os.path.exists('templates/error.html'):
    with open('templates/error.html', 'w') as f:
        f.write(error_html)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 9000))
    app.run(host='0.0.0.0', port=port, debug=True)