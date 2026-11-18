#!/usr/bin/env python3
"""
Employee Attrition Prediction & Workforce Risk Intelligence Platform
Local HTTP & REST API Server
Runs completely standalone using Python 3 standard library (no pip dependencies required to serve)
"""

import http.server
import socketserver
import os
import json
import urllib.parse
import sys

PORT = 5000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 70)
print("EMPLOYEE ATTRITION PREDICTION & RISK INTELLIGENCE PLATFORM")
print("Enterprise Predictive HR Intelligence Platform")
print("=" * 70)

# Try loading joblib models if available
models_loaded = False
try:
    import joblib
    import numpy as np
    
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    gb_model = joblib.load(os.path.join(MODELS_DIR, "gradient_boosting.joblib"))
    rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest.joblib"))
    mlp_model = joblib.load(os.path.join(MODELS_DIR, "mlp_classifier.joblib"))
    svm_model = joblib.load(os.path.join(MODELS_DIR, "svm_classifier.joblib"))
    kmeans_model = joblib.load(os.path.join(MODELS_DIR, "kmeans_model.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    models_loaded = True
    print("[OK] Scikit-Learn trained models successfully loaded from disk!")
except Exception as e:
    print(f"[INFO] Using precomputed model parameters and client-side engine ({e})")

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        
        # API Endpoints
        if parsed.path == '/api/health':
            self.send_json({'status': 'healthy', 'models_loaded': models_loaded})
            return

        elif parsed.path == '/api/metrics':
            metrics_file = os.path.join(BASE_DIR, "models", "model_metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, "r") as f:
                    data = json.load(f)
                self.send_json(data['metrics'])
            else:
                self.send_json({'error': 'Metrics not found'}, status=404)
            return

        elif parsed.path == '/api/clustering':
            metrics_file = os.path.join(BASE_DIR, "models", "model_metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, "r") as f:
                    data = json.load(f)
                self.send_json(data['clustering'])
            else:
                self.send_json({'error': 'Clustering data not found'}, status=404)
            return

        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == '/api/predict':
            length = int(self.headers.get('content-length', 0))
            body = self.rfile.read(length)
            try:
                payload = json.loads(body.decode('utf-8'))
                
                # If scikit-learn models are loaded, compute live inference
                if models_loaded:
                    # Construct feature vector
                    # [Age, MonthlyIncome, TotalWorkingYears, YearsAtCompany, YearsInCurrentRole,
                    #  YearsSinceLastPromotion, YearsWithCurrManager, DistanceFromHome, JobSatisfaction,
                    #  EnvironmentSatisfaction, WorkLifeBalance, JobInvolvement, StockOptionLevel,
                    #  OverTime, MaritalStatus, Department, JobRole, BusinessTravel, Gender]
                    feat_vec = [
                        float(payload.get('age', 32)),
                        float(payload.get('monthlyIncome', 3500)),
                        float(payload.get('totalWorkingYears', 8)),
                        float(payload.get('yearsAtCompany', 3)),
                        float(payload.get('yearsInCurrentRole', 2)),
                        float(payload.get('yearsSinceLastPromotion', 1)),
                        float(payload.get('yearsWithCurrManager', 2)),
                        float(payload.get('distanceFromHome', 12)),
                        float(payload.get('jobSatisfaction', 2)),
                        float(payload.get('environmentSatisfaction', 2)),
                        float(payload.get('workLifeBalance', 2)),
                        float(payload.get('jobInvolvement', 2)),
                        float(payload.get('stockOptionLevel', 0)),
                        1.0 if payload.get('overTime') == 'Yes' else 0.0,
                        0.0 if payload.get('maritalStatus') == 'Divorced' else (1.0 if payload.get('maritalStatus') == 'Married' else 2.0),
                        1.0 if payload.get('department') == 'Research & Development' else (2.0 if payload.get('department') == 'Sales' else 0.0),
                        7.0, # JobRole proxy
                        2.0, # BusinessTravel proxy
                        1.0 if payload.get('gender') == 'Male' else 0.0
                    ]
                    
                    x_scaled = scaler.transform([feat_vec])
                    prob_gb = float(gb_model.predict_proba(x_scaled)[0, 1])
                    prob_rf = float(rf_model.predict_proba(x_scaled)[0, 1])
                    prob_mlp = float(mlp_model.predict_proba(x_scaled)[0, 1])
                    prob_svm = float(svm_model.predict_proba(x_scaled)[0, 1])
                    cluster = int(kmeans_model.predict(x_scaled)[0])

                    self.send_json({
                        'gradient_boosting': round(prob_gb * 100, 2),
                        'random_forest': round(prob_rf * 100, 2),
                        'mlp_classifier': round(prob_mlp * 100, 2),
                        'svm_classifier': round(prob_svm * 100, 2),
                        'cluster': cluster,
                        'source': 'live_joblib_inference'
                    })
                else:
                    self.send_json({'error': 'Live models not loaded'}, status=500)
            except Exception as e:
                self.send_json({'error': str(e)}, status=400)
            return

        self.send_json({'error': 'Not found'}, status=404)

    def send_json(self, data, status=200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

def run():
    handler = CustomHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"\n[SERVER READY] Web interface active at: http://localhost:{PORT}")
        print("Press Ctrl+C to terminate server.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")

if __name__ == '__main__':
    run()
