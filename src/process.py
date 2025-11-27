import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import io
import json
import os

def get_question_mapping(form_type):
    """
    Get mapping from full question text to Q1-Q28 format.
    Returns a dictionary mapping full question text to Q names.
    """
    if form_type == 'ASSI-A':
        questions = [
            'Because I need at least a high-school degree in order to find a high-paying job later on.',
            'Because I experience pleasure and satisfaction while learning new things.',
            'Because I think that a high-school education will help me better prepare for the career I have chosen.',
            'Because I really like going to school.',
            "Honestly, I don't know; I really feel that I am wasting my time in school.",
            'For the pleasure I experience while surpassing myself in my studies.',
            'To prove to myself that I am capable of completing my high-school degree.',
            'In order to obtain a more prestigious job later on.',
            'For the pleasure I experience when I discover new things never seen before.',
            'Because eventually it will enable me to enter the job market in a field that I like.',
            'Because for me, school is fun.',
            'I once had good reasons for going to school; however, now I wonder whether I should continue.',
            'For the pleasure that I experience while I am surpassing myself in one of my personal accomplishments.',
            'Because of the fact that when I succeed in school I feel\r\nimportant.',
            'Because I want to have "the good life" later on.',
            'For the pleasure that I experience in broadening my\r\nknowledge about subjects which appeal to me.',
            'Because this will help me make a better choice regarding my career orientation.',
            'For the pleasure that I experience when I am taken by\r\ndiscussions with interesting teachers.',
            "I can't see why I go to school and frankly, I couldn't care\r\nless.",
            'For the satisfaction I feel when I am in the process of\r\naccomplishing difficult academic activities.',
            'To show myself that I am an intelligent person.',
            'In order to have a better salary later on.',
            'Because my studies allow me to continue to learn about\r\nmany things that interest me.',
            'Because I believe that my high school education will\r\nimprove my competence as a worker.',
            'For the "high" feeling that I experience while reading about various interesting subjects.',
            "I don't know; I can't understand what I am doing in school.",
            'Because high school allows me to experience a personal satisfaction in my quest for excellence in my studies.',
            'Because I want to show myself that I can succeed in my\r\nstudies.'
        ]
        return {q: f'Q{i+1}' for i, q in enumerate(questions)}
    elif form_type == 'ASSI-C':
        questions = [
            'Complain of aches or pains', 'Spend more time alone', 'Tire easily, little energy',
            'Fidgety, unable to sit still', 'Have trouble with teacher', 'Less interested in school',
            'Act as if driven by motor', 'Daydream too much', 'Distract easily',
            'Are afraid of new situations', 'Feel sad, unhappy', 'Are irritable, angry',
            'Feel hopeless', 'Have trouble concentrating', 'Less interested in friends',
            'Fight with other children', 'Absent from school', 'School grades dropping',
            'Down on yourself', 'Visit doctor with doctor finding nothing\r\nwrong',
            'Have trouble sleeping', 'Worry a lot', 'Want to be with parent more than before',
            'Feel that you are bad', 'Take unnecessary risks', 'Get hurt frequently',
            'Seem to be having less fun', 'Act younger than children your age',
            'Do not listen to rules', 'Do not show feelings', "Do not understand other people's feelings",
            'Tease others', 'Blame others for your troubles', 'Take things that do not belong to you',
            'Refuse to share'
        ]
        return {q: f'Q{i+1}' for i, q in enumerate(questions)}
    return {}


def map_question_columns_to_q_format(df, form_type):
    """
    Map full question text columns to Q1, Q2, Q3... format
    to match the training data format.
    """
    question_mapping = get_question_mapping(form_type)
    df_renamed = df.copy()
    
    # Helper function to normalize strings for comparison
    def normalize_text(text):
        """Normalize text for comparison by removing extra whitespace and newlines."""
        if not isinstance(text, str):
            return str(text)
        return ' '.join(text.strip().replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ').split())
    
    # Create a mapping that handles whitespace/newline differences
    column_mapping = {}
    for col in df.columns:
        # Check if column is already in Q format (Q1, Q2, etc.)
        if col.startswith('Q') and len(col) > 1 and col[1:].isdigit():
            continue  # Already in correct format
        
        # Skip non-question columns
        if col in ['Name', 'Gender', 'Grade', 'GradeLevel', 'StudentNumber', 'RiskRating']:
            continue
        
        # Try to match with question mapping
        col_normalized = normalize_text(col)
        best_match = None
        best_score = 0
        
        for full_text, q_name in question_mapping.items():
            text_normalized = normalize_text(full_text)
            
            # Check for exact match
            if col_normalized == text_normalized:
                column_mapping[col] = q_name
                break
            
            # Check for substring match (one contains the other)
            if col_normalized in text_normalized or text_normalized in col_normalized:
                # Use the longer match as it's more specific
                match_length = min(len(col_normalized), len(text_normalized))
                if match_length > best_score:
                    best_score = match_length
                    best_match = q_name
        
        # If we found a good match but didn't break, use it
        if best_match and col not in column_mapping:
            column_mapping[col] = best_match
    
    # Apply the mapping
    if column_mapping:
        df_renamed = df_renamed.rename(columns=column_mapping)
        print(f"Mapped {len(column_mapping)} columns to Q format: {list(column_mapping.values())}")
    else:
        # Check if columns are already in Q format
        q_columns = [col for col in df.columns if col.startswith('Q') and len(col) > 1 and col[1:].isdigit()]
        if q_columns:
            print(f"Columns already in Q format ({len(q_columns)} Q columns found)")
    
    return df_renamed


def validate_dataset(columns, type):
    expected_columns = []
    if type == 'ASSI-A':
        expected_columns = ['Name', 'Gender', 'Grade', 'Because I need at least a high-school degree in order to find a high-paying job later on.', 'Because I experience pleasure and satisfaction while learning new things.', 'Because I think that a high-school education will help me better prepare for the career I have chosen.', 'Because I really like going to school.', "Honestly, I don't know; I really feel that I am wasting my time in school.", 'For the pleasure I experience while surpassing myself in my studies.', 'To prove to myself that I am capable of completing my high-school degree.', 'In order to obtain a more prestigious job later on.', 'For the pleasure I experience when I discover new things never seen before.', 'Because eventually it will enable me to enter the job market in a field that I like.', 'Because for me, school is fun.', 'I once had good reasons for going to school; however, now I wonder whether I should continue.', 'For the pleasure that I experience while I am surpassing myself in one of my personal accomplishments.', 'Because of the fact that when I succeed in school I feel\r\nimportant.', 'Because I want to have "the good life" later on.', 'For the pleasure that I experience in broadening my\r\nknowledge about subjects which appeal to me.', 'Because this will help me make a better choice regarding my career orientation.', 'For the pleasure that I experience when I am taken by\r\ndiscussions with interesting teachers.', "I can't see why I go to school and frankly, I couldn't care\r\nless.", 'For the satisfaction I feel when I am in the process of\r\naccomplishing difficult academic activities.', 'To show myself that I am an intelligent person.', 'In order to have a better salary later on.', 'Because my studies allow me to continue to learn about\r\nmany things that interest me.', 'Because I believe that my high school education will\r\nimprove my competence as a worker.', 'For the "high" feeling that I experience while reading about various interesting subjects.', "I don't know; I can't understand what I am doing in school.", 'Because high school allows me to experience a personal satisfaction in my quest for excellence in my studies.', 'Because I want to show myself that I can succeed in my\r\nstudies.']
    elif type == 'ASSI-C':
        expected_columns = ['Name', 'Gender', 'Grade', 'Complain of aches or pains', 'Spend more time alone', 'Tire easily, little energy', 'Fidgety, unable to sit still', 'Have trouble with teacher', 'Less interested in school', 'Act as if driven by motor', 'Daydream too much', 'Distract easily', 'Are afraid of new situations', 'Feel sad, unhappy', 'Are irritable, angry', 'Feel hopeless', 'Have trouble concentrating', 'Less interested in friends', 'Fight with other children', 'Absent from school', 'School grades dropping', 'Down on yourself', 'Visit doctor with doctor finding nothing\r\nwrong', 'Have trouble sleeping', 'Worry a lot', 'Want to be with parent more than before', 'Feel that you are bad', 'Take unnecessary risks', 'Get hurt frequently', 'Seem to be having less fun', 'Act younger than children your age', 'Do not listen to rules', 'Do not show feelings', "Do not understand other people's feelings", 'Tease others', 'Blame others for your troubles', 'Take things that do not belong to you', 'Refuse to share']
    
    missing_columns = [col for col in expected_columns if col not in columns]
    return False if missing_columns else True

def upload_file(file, id, form_type):
    stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
    root_save_folder = 'persisted/uploads'
    if not os.path.exists(root_save_folder):
        os.makedirs(root_save_folder)

    type_save_folder = os.path.join(root_save_folder, form_type)
    if not os.path.exists(type_save_folder):
        os.makedirs(type_save_folder)
    
    csv_content = stream.getvalue()
    
    file_path =  os.path.join(type_save_folder, f'{id}.csv')
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(csv_content)
    df = pd.read_csv(file_path)
    
    # Check if Name column exists, if not create it
    if 'Name' not in df.columns:
        # Create Name column from StudentNumber or index
        if 'StudentNumber' in df.columns:
            df['Name'] = df['StudentNumber'].astype(str)
            print("Created 'Name' column from 'StudentNumber'")
        else:
            df['Name'] = [f'Student{i+1}' for i in range(len(df))]
            print("Created 'Name' column from index")
    
    df['Name'] = df['Name'].astype(str)
    df.to_csv(file_path, index=False)
    return file_path

def upload_student_data(df, id, form_type):
    root_save_folder = 'persisted/student_data'
    df_original = pd.read_csv(f'persisted/uploads/{form_type}/{id}.csv')
    
    # Add predictions from df (which contains Cluster, RiskRating, RiskConfidence)
    df_original['Cluster'] = df['Cluster'].values
    
    # Add RiskRating and RiskConfidence if present
    if 'RiskRating' in df.columns:
        df_original['RiskRating'] = df['RiskRating'].values
    if 'RiskConfidence' in df.columns:
        df_original['RiskConfidence'] = df['RiskConfidence'].values
        
    if not os.path.exists(root_save_folder):
        os.makedirs(root_save_folder)

    type_save_folder = os.path.join(root_save_folder, form_type)
    if not os.path.exists(type_save_folder):
        os.makedirs(type_save_folder)

    file_path = os.path.join(type_save_folder, f'{id}.csv')
    df_original.to_csv(file_path, index=False)

    return file_path

def upload_results(results):
    id = results['id']
    form_type = results['type']
    root_save_folder = 'persisted/results'

    try:
        # Create root directory if it doesn't exist
        if not os.path.exists(root_save_folder):
            os.makedirs(root_save_folder, exist_ok=True)
            print(f"Created directory: {root_save_folder}")

        # Create type-specific directory if it doesn't exist
        type_save_folder = os.path.join(root_save_folder, form_type)
        if not os.path.exists(type_save_folder):
            os.makedirs(type_save_folder, exist_ok=True)
            print(f"Created directory: {type_save_folder}")

        file_path = os.path.join(type_save_folder, f'{id}.json')
        
        # Check write permissions
        if not os.access(os.path.dirname(file_path), os.W_OK):
            error_msg = f"ERROR: No write permission to directory: {os.path.dirname(file_path)}"
            print(error_msg)
            return None
        
        # Check if directory exists
        if not os.path.exists(os.path.dirname(file_path)):
            error_msg = f"ERROR: Directory does not exist: {os.path.dirname(file_path)}"
            print(error_msg)
            return None
        
        print(f"Attempting to save results to: {file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        print(f"Successfully saved results to: {file_path}")
        return file_path
    except PermissionError as e:
        error_msg = f"ERROR: Permission denied saving results: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None
    except OSError as e:
        error_msg = f"ERROR: OS error saving results: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None
    except Exception as e:
        error_msg = f"ERROR: Unexpected error saving results: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None

def summarize_answers(uuid, form_type, gender, grade, cluster):
    file_path = os.path.join('persisted','student_data', form_type, f'{uuid}.csv')
    df = pd.read_csv(file_path)
    df = df.dropna(axis=0)

    if form_type == 'ASSI-C':
        # Convert 'Never', 'Sometimes', 'Often' to numerical values
        answer_map = {'Never': 0, 'Sometimes': 1, 'Often': 2}
        for col in df.columns:
            if col not in ['Gender', 'Grade', 'Name']:
                df[col] = df[col].map(answer_map)

    summary = {}
    # Filter the dataframe by gender and grade if they're not set to 'all'
    if gender != 'all':
        df = df[df['Gender'] == gender]
    if grade != 'all':
        df = df[df['Grade'] == int(grade)]
    if cluster != 'all':
        df = df[df['Cluster'] == int(cluster)]
    print(df.head(1))
        
    for column in df.columns:
        if column not in ['Name', 'Grade', 'Gender']:
            summary[column] = df[column].value_counts().to_dict()
    
    return summary

def summarize_answer_per_cluster(df_clustered, form_type):
    summary = []
    for cluster in df_clustered['Cluster'].unique():
        summary_cluster = {}
        df_cluster = df_clustered[df_clustered['Cluster'] == cluster]
        summary_cluster['cluster'] = cluster
        for column in df_cluster.columns:
            if column != 'Cluster':
                summary_cluster[column] = df_cluster[column].value_counts().to_dict()
        summary.append(summary_cluster)

    # For ASSI-C form type, revert numerical values back to text answers
    if form_type == 'ASSI-C':
        # Define the reverse mapping from numerical values to text answers
        reverse_answer_map = {0: 'Never', 1: 'Sometimes', 2: 'Often'}
        
        # For each cluster in the summary
        for cluster_summary in summary:
            # For each question in the cluster
            for question, answers in cluster_summary.items():
                # Skip the cluster identifier
                if question != 'cluster':
                    # Create a new dictionary with text answers
                    text_answers = {}
                    for num_value, count in answers.items():
                        # Handle gender column specifically
                        if question == 'Gender':
                            if num_value == 0:
                                text_answers['Female'] = count
                            elif num_value == 1:
                                text_answers['Male'] = count
                            else:
                                text_answers[num_value] = count
                        # Convert the numerical key to text if it's in our mapping
                        elif isinstance(num_value, (int, float)) and num_value in reverse_answer_map:
                            text_answers[reverse_answer_map[num_value]] = count
                        else:
                            # Keep as is if not a numerical value we're mapping
                            text_answers[num_value] = count
                    
                    # Replace the numerical answers with text answers
                    cluster_summary[question] = text_answers
    return summary


def load_data_and_preprocess(file_path, form_type):
    # Full DataFrame
    df = pd.read_csv(file_path)
    
    # Map question columns to Q1-Q28 format if they're in full text format
    # This ensures compatibility with pre-trained models
    df = map_question_columns_to_q_format(df, form_type)
    
    # Standardize column names - handle Grade vs GradeLevel
    if 'Grade' in df.columns and 'GradeLevel' not in df.columns:
        df = df.rename(columns={'Grade': 'GradeLevel'})
    
    # Use GradeLevel consistently
    grade_col = 'GradeLevel' if 'GradeLevel' in df.columns else 'Grade'
    
    # Only questions (drop Name only - keep grade for clustering)
    # Note: GradeLevel is used as a feature in clustering
    df_questions_only = df.drop(columns=['Name']).copy()

    if form_type == 'ASSI-C':
        # Convert 'Never', 'Sometimes', 'Often' to numerical values
        answer_map = {'Never': 0, 'Sometimes': 1, 'Often': 2}
        for col in df_questions_only.columns:
            if col not in ['Gender', 'Grade', 'GradeLevel', 'Name']:
                df_questions_only[col] = df_questions_only[col].map(answer_map)

    # Transform Gender to Numeric
    df_questions_only['Gender'] = df_questions_only['Gender'].map({'Female': 0, 'Male': 1})
    
    # Remove Na
    df_questions_only = df_questions_only.dropna(axis=0)
    
    # Update df to match dropped rows
    df = df.loc[df_questions_only.index].copy()

    # Identify columns to exclude from scaling (non-numeric or metadata columns)
    # Keep only question columns (Q1-Q28) and numeric metadata (Gender, GradeLevel)
    columns_to_scale = []
    for col in df_questions_only.columns:
        # Include question columns (Q1, Q2, etc.)
        if col.startswith('Q') and len(col) > 1 and col[1:].isdigit():
            columns_to_scale.append(col)
        # Include numeric metadata columns
        elif col in ['Gender', 'Grade', 'GradeLevel']:
            columns_to_scale.append(col)
        # Skip any other columns (like RiskRating, Cluster, etc.)
    
    # Validate that we have columns to scale
    if not columns_to_scale:
        raise ValueError(f"No valid numeric columns found for scaling. Available columns: {list(df_questions_only.columns)}")
    
    # Create a DataFrame with only numeric columns for scaling
    df_for_scaling = df_questions_only[columns_to_scale].copy()
    
    # Ensure all columns are numeric
    for col in df_for_scaling.columns:
        df_for_scaling[col] = pd.to_numeric(df_for_scaling[col], errors='coerce')
    
    # Remove any rows that became NaN after conversion
    df_for_scaling = df_for_scaling.dropna(axis=0)
    
    # Validate that we have data after conversion
    if df_for_scaling.empty:
        raise ValueError("No valid numeric data found after conversion. Please check that question columns contain numeric values.")
    
    # Update dataframes to match dropped rows
    df_questions_only = df_questions_only.loc[df_for_scaling.index].copy()
    df = df.loc[df_for_scaling.index].copy()
    
    # Create a scaled version for compatibility (but models will use unscaled)
    # We still return df_scaled for backward compatibility, but models use df_questions_only
    scaler = StandardScaler()
    df_scaled_values = scaler.fit_transform(df_for_scaling)
    df_scaled = pd.DataFrame(df_scaled_values, columns=columns_to_scale, index=df_for_scaling.index)
    df_scaled['Name'] = df['Name'].values
    
    root_save_folder = 'persisted/uploads'
    if not os.path.exists(root_save_folder):
        os.makedirs(root_save_folder)
    filename = os.path.join(root_save_folder, 'full_df.csv')
    df.to_csv(filename, index=False)

    return df, df_questions_only, df_scaled 

def count_items_in_cluster(df_pca, cluster):
    return df_pca[df_pca['Cluster'] == cluster].shape[0]

def load_clustering_models(model_path='models/clustering'):
    """Load pre-trained clustering models."""
    import pickle
    
    print(f"Loading clustering models from {model_path}...")
    
    with open(os.path.join(model_path, 'clustering_models.pkl'), 'rb') as f:
        models_dict = pickle.load(f)
    
    print("Clustering models loaded successfully!")
    return models_dict


def predict_clusters(df_scaled, df_original_questions_only):
    """
    Predict clusters for new data using pre-trained KMeans model.
    
    Args:
        df_scaled: Scaled DataFrame with Name and Grade columns (for compatibility)
        df_original_questions_only: Original unscaled question data (used for models)
    
    Returns:
        Tuple of (df_pca with clusters, optimal_k, cluster_count, df_original with clusters)
    """
    try:
        # Load pre-trained models
        models = load_clustering_models()
        scaler = models['scaler']
        pca = models['pca']
        kmeans = models['kmeans']
        optimal_k = models['optimal_k']
        optimal_pc = models['optimal_pc']
        feature_names = models['feature_names']
        
        print(f"Using pre-trained models: {optimal_k} clusters, {optimal_pc} PCs")
        
        # Extract Name from df_scaled (which has Name column)
        df_name = df_scaled[['Name']].copy()
        
        # Use the unscaled data (df_original_questions_only) and ensure columns match
        # Reorder columns to match training data feature order
        X_unscaled = df_original_questions_only.copy()
        
        # Ensure all required features are present
        missing_features = [f for f in feature_names if f not in X_unscaled.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            # Add missing features with zeros
            for feat in missing_features:
                X_unscaled[feat] = 0
        
        # Select only the features used in training, in the correct order
        X_unscaled = X_unscaled[feature_names]
        
        # Scale using the pre-trained scaler
        X_scaled = scaler.transform(X_unscaled)
        
        # Apply PCA transformation
        X_pca = pca.transform(X_scaled)
        
        # Create DataFrame with PCA results
        df_pca = pd.DataFrame(X_pca)
        df_pca['Name'] = df_name['Name'].values
        
        # Extract grade from the original scaled data
        if 'GradeLevel' in df_scaled.columns:
            df_pca['Grade'] = df_scaled['GradeLevel'].values
        elif 'Grade' in df_scaled.columns:
            df_pca['Grade'] = df_scaled['Grade'].values
        
        # Predict clusters
        clusters = kmeans.predict(X_pca)
        df_pca['Cluster'] = clusters
        
        # Count items per cluster
        cluster_count = {}
        for cluster in range(optimal_k):
            cluster_count[f"Cluster {cluster + 1}"] = count_items_in_cluster(df_pca, cluster)
        
        # Add cluster labels to original data
        df_original_questions_only['Cluster'] = df_pca['Cluster'].apply(lambda x: f'cluster_{x+1}')
        
        print(f"Cluster prediction complete. Distribution: {cluster_count}")
        
        return df_pca, optimal_k, cluster_count, df_original_questions_only

    except Exception as e:
        print(f"Error in predict_clusters: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


# Keep old kmeans function for backward compatibility (can be removed later)
def kmeans(df_pca, df_original_questions_only):
    """Deprecated: Use predict_clusters instead."""
    print("Warning: kmeans() is deprecated. Using predict_clusters() instead.")
    return predict_clusters(df_pca, df_original_questions_only)


def apply_pca(df_scaled, df_original_questions_only=None):
    """
    Apply PCA transformation using pre-trained model.
    
    Note: This function is kept for compatibility but predict_clusters() 
    handles both PCA and clustering together.
    
    Args:
        df_scaled: Scaled DataFrame with Name (for compatibility)
        df_original_questions_only: Original unscaled question data (optional, will use df_scaled if not provided)
    """
    try:
        # Load pre-trained models
        models = load_clustering_models()
        scaler = models['scaler']
        pca = models['pca']
        optimal_pc = models['optimal_pc']
        feature_names = models['feature_names']
        
        # Extract Name
        df_name = df_scaled[['Name']].copy()
        
        # Use unscaled data if provided, otherwise extract from df_scaled
        if df_original_questions_only is not None:
            X_unscaled = df_original_questions_only.copy()
        else:
            # Extract and use unscaled version (we'll need to reconstruct)
            # This is a fallback - ideally df_original_questions_only should be passed
            X_unscaled = df_scaled.drop(columns=['Name']).copy()
        
        # Ensure all required features are present
        missing_features = [f for f in feature_names if f not in X_unscaled.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            for feat in missing_features:
                X_unscaled[feat] = 0
        
        # Select only the features used in training, in the correct order
        X_unscaled = X_unscaled[feature_names]
        
        # Scale using the pre-trained scaler
        X_scaled = scaler.transform(X_unscaled)
        
        # Apply PCA
        principal_components = pca.transform(X_scaled)
        
        # Create DataFrame
        df_pca = pd.DataFrame(principal_components)
        df_pca['Name'] = df_name['Name'].values
        
        # Extract grade from original data
        if 'GradeLevel' in df_scaled.columns:
            df_pca['Grade'] = df_scaled['GradeLevel'].values
        elif 'Grade' in df_scaled.columns:
            df_pca['Grade'] = df_scaled['Grade'].values
        
        return df_pca, int(optimal_pc)
    
    except Exception as e:
        print(f"Error in apply_pca: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# Keep old pca function for backward compatibility
def pca(df_scaled, df_original_questions_only=None):
    """Deprecated: Use apply_pca instead."""
    return apply_pca(df_scaled, df_original_questions_only)

def get_uploaded_result_by_uuid(id, type):
    root_save_folder = 'persisted/results'
    type_save_folder = os.path.join(root_save_folder, type)
    file_path = os.path.join(type_save_folder, f'{id}.json')
    
    try:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error reading results: {e}")
        return None



