import os
import pandas as pd
import numpy as np
import mysql.connector
from mysql.connector import errorcode
from dotenv import load_dotenv
load_dotenv()

# Database configuration - customize via environment variables
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'guidance_system'),
    'raise_on_warnings': False,
}

# Connection helper functions

def get_db_connection(with_db=True):
    """Creates and returns a MySQL database connection."""
    config = DB_CONFIG.copy()
    if not with_db:
        config.pop('database', None)
    conn = mysql.connector.connect(**config)
    print("Connected to MySQL database")
    return conn


def close_db_connection(connection):
    """Closes the database connection."""
    connection.close()


def create_db(password_hash):
    """Creates the database (if needed) and necessary tables."""
    # Connect to MySQL server without specifying database
    conn = get_db_connection(with_db=False)
    cursor = conn.cursor()

    # Create database if it doesn't exist
    try:
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']} DEFAULT CHARACTER SET 'utf8mb4' COLLATE 'utf8mb4_unicode_ci';")
        print(f"Database '{DB_CONFIG['database']}' ensured.")
    except mysql.connector.Error as err:
        print(f"Failed creating database: {err}")
        cursor.close()
        conn.close()
        return
    finally:
        cursor.close()
        conn.close()

        # Connect to the newly created database
        conn = get_db_connection()
        cursor = conn.cursor()

        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255) NOT NULL UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                first_name VARCHAR(255) NOT NULL,
                last_name VARCHAR(255) NOT NULL,
                user_type ENUM('admin', 'viewer') NOT NULL DEFAULT 'viewer',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB;
        """)

        # Create records table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id INT AUTO_INCREMENT PRIMARY KEY,
                uuid VARCHAR(36) NOT NULL UNIQUE,
                name VARCHAR(255) NOT NULL,
                username VARCHAR(255) NOT NULL,
                type ENUM('ASSI-A', 'ASSI-C') NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
            ) ENGINE=InnoDB;
        """)

        # Create archived_records table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS archived_records (
                id INT AUTO_INCREMENT PRIMARY KEY,
                uuid VARCHAR(36) NOT NULL UNIQUE,
                name VARCHAR(255) NOT NULL,
                username VARCHAR(255) NOT NULL,
                type ENUM('ASSI-A', 'ASSI-C') NOT NULL,
                archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
            ) ENGINE=InnoDB;
        """)

        # Insert default admin user if not exists
        cursor.execute("SELECT id FROM users WHERE username = 'superadmin'")
        if cursor.fetchone() is None:
            cursor.execute(
                "INSERT INTO users (username, password_hash, first_name, last_name, user_type) VALUES (%s, %s, %s, %s, 'admin')",
                ('superadmin', password_hash, 'Admin', 'User')
            )
            print("Inserted default superadmin user.")
        
        conn.commit()
        cursor.close()
        close_db_connection(conn)
        print("Database setup complete.")


def test_db_connection():
    """Tests the database connection by executing a simple query."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES;")
        tables = cursor.fetchall()
        cursor.close()
        close_db_connection(conn)
        return tables
    except mysql.connector.Error as err:
        print("MySQL Error:", err)
        return False


def authenticate(username, password):
    """Authenticates a user by verifying credentials against the database."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT * FROM users WHERE username = %s AND password_hash = %s", 
        (username, password)
    )
    user = cursor.fetchone()
    cursor.close()
    close_db_connection(conn)

    if user:
        print("Authentication successful for user: ", username)
        return {
            "id": str(user['id']),
            "username": user['username'],
            "first_name": user['first_name'],
            "last_name": user['last_name'],
            "user_type": user['user_type']
        }
    return None


def insert_user(username, password, first_name, last_name, user_type="viewer"):
    """Inserts a new user into the database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, password_hash, first_name, last_name, user_type) VALUES (%s, %s, %s, %s, %s)",
            (username, password, first_name, last_name, user_type)
        )
        conn.commit()
        cursor.close()
        close_db_connection(conn)
        return True
    except mysql.connector.Error as err:
        print("MySQL Error:", err)
        conn.rollback()
        cursor.close()
        close_db_connection(conn)
        return False


def get_all_users():
    """Retrieves all users from the database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users")
        users = cursor.fetchall()
        cursor.close()
        close_db_connection(conn)
        return [
            {
                "id": str(u['id']),
                "username": u['username'],
                "first_name": u['first_name'],
                "last_name": u['last_name'],
                "user_type": u['user_type']
            } for u in users
        ]
    except mysql.connector.Error as err:
        print("MySQL Error:", err)
        return False


def delete_user(user_id):
    """Deletes a user from the database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        conn.commit()
        cursor.close()
        close_db_connection(conn)
        return True
    except mysql.connector.Error as err:
        print("MySQL Error:", err)
        return False


def insert_result_record(uuid, name, username, record_type):
    """Inserts a record for a user."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO records (uuid, name, username, type) VALUES (%s, %s, %s, %s)",
            (uuid, name, username, record_type)
        )
        conn.commit()
        cursor.close()
        close_db_connection(conn)
        return True
    except mysql.connector.Error as err:
        print("MySQL Error:", err)
        conn.rollback()
        close_db_connection(conn)
        return False


def get_user_records(username):
    """Retrieves all records associated with a specific username."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM records WHERE username = %s", (username,))
        rows = cursor.fetchall()
        cursor.close()
        close_db_connection(conn)
        return rows
    except mysql.connector.Error as err:
        print("MySQL Error:", err)
        return False


def delete_record(uuid):
    """Archives and deletes a record by UUID."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Insert into archived_records then delete
        cursor.execute(
            "INSERT IGNORE INTO archived_records (uuid, name, username, type) "
            "SELECT uuid, name, username, type FROM records WHERE uuid = %s", (uuid,)
        )
        cursor.execute("DELETE FROM records WHERE uuid = %s", (uuid,))
        conn.commit()
        cursor.close()
        close_db_connection(conn)
        return True
    except mysql.connector.Error as err:
        print("MySQL Error:", err)
        return False


# Student CSV data handlers remain unchanged

def get_student_data_by_uuid_and_name(uuid, name, form_type):
    try:
        df = pd.read_csv(os.path.join('persisted', 'student_data', form_type, f'{uuid}.csv'))
        search_col = 'StudentNumber'
        name = str(name)
        if search_col not in df.columns:
            return None

        df[search_col] = df[search_col].astype(str)
        df_match = df[df[search_col] == name]
        if df_match.empty:
            return None

        row = df_match.iloc[0]

        print(row)
        # Convert numpy types to native Python types for JSON serialization
        def to_native(val):
            try:
                # If value is numpy type, convert to Python scalar
                return int(val) if isinstance(val, (np.int64, np.int32)) else float(val) if isinstance(val, (np.float64, np.float32)) else val
            except Exception:
                return val

        result = {
            'Name': row[search_col],
            'Grade': int(row['Grade']) if not pd.isna(row['Grade']) else None,
            'Gender': row['Gender'],
            'Cluster': int(row['Cluster']) if not pd.isna(row['Cluster']) else None,
            'RiskRating': row.get('RiskRating', None),
            'RiskConfidence': float(row['RiskConfidence']) if 'RiskConfidence' in row and not pd.isna(row['RiskConfidence']) else None,
            'Questions': {
                col: to_native(row[col])
                for col in df.columns if col not in [search_col, 'Grade', 'Gender', 'Cluster', 'RiskRating', 'RiskConfidence', '__name_norm']
            }
        }

        print(result)
        return result
    except Exception as e:
        print("Error:", e)
        return False


def update_student_cluster(uuid, name, cluster, form_type):
    try:
        df = pd.read_csv(f'student_data/{form_type}/{uuid}.csv')
        df['Name'] = df['Name'].astype(str)
        df.loc[df['Name'] == name, 'Cluster'] = int(cluster)
        df.to_csv(f'student_data/{form_type}/{uuid}.csv', index=False)
        return True
    except Exception as e:
        print("Error:", e)
        return False
