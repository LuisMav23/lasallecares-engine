import os
import hashlib
from uuid import uuid4
from flask import Flask, request, jsonify, send_file, abort
from flask_cors import CORS

from src.process import (
    validate_dataset, summarize_answers, upload_file, load_data_and_preprocess,
    pca, upload_results, kmeans, get_uploaded_result_by_uuid,
    summarize_answer_per_cluster, upload_student_data
)
from src.db import (
    get_student_data_by_uuid_and_name, create_db, get_db_connection, close_db_connection,
    authenticate, insert_user, insert_result_record, get_user_records, delete_record,
    test_db_connection, get_all_users, delete_user, update_student_cluster
)
from src.classification import (
    predict_risk_rating
)

def get_env_var(name, default=None):
    return os.environ.get(name, default)

app = Flask(__name__)
app.secret_key = get_env_var('SECRET_KEY', 'your_secret_key')
CORS(
    app,
    resources={r"*": {"origins": ["*"]}},
    supports_credentials=False,
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    expose_headers=["Content-Disposition"]
)

def setup_db():
    """Initialize the database and create superadmin if needed."""
    admin_password = get_env_var('ADMIN_PASSWORD', 'admin1234')
    password_hash = hashlib.sha256(admin_password.encode()).hexdigest()
    create_db(password_hash=password_hash)
    conn = get_db_connection()
    close_db_connection(conn)
    authenticate('superadmin', password_hash)

@app.route('/')
def hello_world():
    """Health check endpoint."""
    return 'Hello'

@app.route('/test')
def test():
    """Test DB connection."""
    result = test_db_connection()
    return jsonify({'db_test': result[0]})

# =========================
# Auth and User Endpoints
# =========================

@app.route('/api/auth', methods=['GET'])
def authenticate_user():
    """Authenticate a user."""
    username = request.args.get('username')
    password = request.args.get('password')
    if not username or not password:
        abort(400, description="Missing username or password")
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    user = authenticate(username, password_hash)
    if user:
        return jsonify({'message': 'Authentication successful', 'user': user}), 200
    return jsonify({'message': 'Authentication failed'}), 401

@app.route('/api/users', methods=['GET','POST'])
def create_user():
    if request.method == 'POST':
        data = request.get_json()
        required_fields = ['username', 'password', 'first_name', 'last_name']
        if not all(data.get(field) for field in required_fields):
            abort(400, description="Missing required fields")
        password_hash = hashlib.sha256(data['password'].encode()).hexdigest()
        user_type = data.get('user_type', 'viewer')
        if insert_user(data['username'], password_hash, data['first_name'], data['last_name'], user_type):
            return jsonify({'message': 'User created successfully'}), 200
        return jsonify({'message': 'User creation failed'}), 500
    
    elif request.method == 'GET':
        users = get_all_users()
        return jsonify({'users': users}), 200

@app.route('/api/users/<string:id>', methods=['DELETE'])
def delete_user_by_id(id):
    """Delete a user by ID."""
    if delete_user(id):
        return jsonify({'message': 'User deleted successfully'}), 200
    return jsonify({'message': 'User deletion failed'}), 500

# =========================
# Data Fetch Endpoints
# =========================

@app.route('/api/data', methods=['GET'])
def get_data():
    """Get user records."""
    user = request.args.get('username')
    if not user:
        abort(400, description="Missing username")
    records = get_user_records(user)
    return jsonify({'records': records}), 200

@app.route('/api/data/<string:type>/<string:uuid>', methods=['GET'])
def get_data_by_uuid(type, uuid):
    """Get uploaded result by UUID and type."""
    result = get_uploaded_result_by_uuid(uuid, type)
    if not result:
        abort(404, description='Result not found')
    return jsonify(result), 200

@app.route('/api/data/<string:uuid>', methods=['DELETE'])
def delete_data_by_uuid(uuid):
    """Delete a record by UUID."""
    if delete_record(uuid):
        return jsonify({'message': 'Record deleted successfully'}), 200
    return jsonify({'message': 'Record deletion failed'}), 500

@app.route('/api/student/data/<string:uuid>/<string:form_type>/<string:name>', methods=['GET'])
def get_student_data_by_name(uuid, form_type, name):
    """Get student data by UUID, form type, and name."""
    result = get_student_data_by_uuid_and_name(uuid, name, form_type)
    print(result)
    return jsonify(result), 200

@app.route('/api/student/data/<string:uuid>/<string:form_type>/<string:name>/<string:cluster>', methods=['PUT'])
def update_student_cluster_by_name(uuid, name, form_type, cluster):
    """Update student cluster assignment."""
    if update_student_cluster(uuid, name, cluster, form_type):
        return jsonify({'message': 'Student cluster updated successfully'}), 200
    return jsonify({'message': 'Student cluster update failed'}), 500

@app.route('/api/answer_summary', methods=['GET'])
def get_answer_summary():
    """Get answer summary with optional filters."""
    uuid = request.args.get('uuid')
    form_type = request.args.get('form_type')
    gender = request.args.get("gender", "all")
    grade = request.args.get("grade", "all")
    cluster = request.args.get("cluster", "all")
    summary = summarize_answers(uuid, form_type, gender, grade, cluster)
    return jsonify(summary), 200

@app.route('/api/data', methods=['POST'])
def fetch_data():
    """Upload and process a CSV file."""
    if 'file' not in request.files:
        abort(400, description='No file part in request')
    file = request.files['file']
    if file.filename == '':
        abort(400, description='No selected file')
    if not file.filename.lower().endswith('.csv'):
        abort(400, description='Invalid file type. Only CSV files are accepted.')

    uuid = str(uuid4())
    record_name = request.form.get('datasetName')
    form_type = request.form.get('kindOfData')
    user = request.form.get('user')
    if not user:
        abort(400, description='User not found')

    try:
        file_path = upload_file(file, uuid, form_type)
        df, df_questions_only, df_scaled = load_data_and_preprocess(file_path, form_type)
        columns = df.columns.to_list()
        is_valid = True  # Optionally: validate_dataset(columns, form_type)
        if not is_valid:
            abort(400, description='Invalid dataset')

        # Use pre-trained models for prediction
        # Pass both scaled (for compatibility) and unscaled (for models) data
        df_pca, optimal_pc = pca(df_scaled, df_questions_only)
        if df_pca is None:
            abort(500, description='Failed to apply PCA transformation. Check that uploaded data matches expected format.')
        
        df_pca, optimal_k, cluster_count, df_original_questions_only = kmeans(df_pca, df_questions_only)
        if df_pca is None:
            abort(500, description='Failed to predict clusters. Check that uploaded data matches expected format.')
        
        # Predict risk ratings using pre-trained TensorFlow model (ASSI-A only)
        risk_prediction = predict_risk_rating(df)
        
        # Add risk rating predictions to the dataframe
        df_pca['RiskRating'] = risk_prediction['predictions']
        df_pca['RiskConfidence'] = risk_prediction['confidence']
        
        # Upload student data with both cluster and risk rating
        upload_student_data(df_pca, uuid, form_type)
        summary = summarize_answers(uuid, form_type, 'all', 'all', 'all')

        results = {
            'id': uuid,
            'user': user,
            'type': form_type,
            'data_summary': {
                'answers_summary': summary,
                'pca_summary': {'optimal_pc': int(optimal_pc) if optimal_pc is not None else None},
                'cluster_summary': {
                    'optimal_k': int(optimal_k) if optimal_k is not None else None,
                    'cluster_count': cluster_count if cluster_count is not None else {}
                },
                'risk_rating_summary': {
                    'model_name': risk_prediction['model_name'],
                    'risk_distribution': risk_prediction['risk_distribution'],
                    'classes': risk_prediction['classes']
                }
            }
        }

        results_path = upload_results(results)
        if not insert_result_record(uuid, record_name, user, form_type) or not results_path:
            abort(500, description='Failed to insert result record')

        return jsonify({'message': 'File uploaded and processed successfully', 'data': results}), 200
    except ValueError as e:
        app.logger.error(f"Data processing error: {str(e)}")
        abort(400, description=f'Data processing error: {str(e)}')
    except Exception as e:
        app.logger.exception('Error processing uploaded file')
        abort(500, description=f'Internal server error: {str(e)}')

# Updated download endpoint to use student_data folder instead of persisted/student_data
@app.route('/api/download/<string:type>/<string:uuid>', methods=['GET'])
def download_results(type, uuid):
    """Download results file as JSON."""
    # build a safe path to the csv file
    dir_path = os.path.join('persisted', 'student_data', type)
    filename = os.path.join(dir_path, f"{uuid}.csv")
    if not os.path.exists(filename):
        abort(404, description='File not found')

    # send the file as an attachment so the browser downloads it directly
    try:
        # send_file will set correct headers and content-type
        return send_file(filename, mimetype='text/csv', as_attachment=True, download_name=f"{type}_{uuid}.csv")
    except Exception as e:
        app.logger.exception('Failed to send file')
        abort(500, description='Failed to send file')

if __name__ == '__main__':
    setup_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
