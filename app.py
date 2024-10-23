from flask import Flask, request, jsonify
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from pymongo import MongoClient
# import google.generativeai as genai
import os
import uuid
import tensorflow as tf
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)
bcrypt = Bcrypt(app)

# MongoDB connection
mongo_uri = os.getenv('MONGO_URI', 'mongodb+srv://gowthambalaji344:gowthambalaji344@mida.2osc8bi.mongodb.net/')
client = MongoClient(mongo_uri)
db = client.hospitalDB

# Check MongoDB connection
try:
    client.admin.command('ping')
    print("Connected to MongoDB successfully")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")

from flask import send_from_directory
from urllib.parse import unquote

@app.route('/')
def home():
    return "Welcome to the MIDA backend!"

@app.route('/uploads/<path:filename>')
def serve_image(filename):
    decoded_filename = unquote(filename).replace(' ', '_')  # Replace spaces with underscores
    print(decoded_filename)  # For debugging
    return send_from_directory('uploads', decoded_filename)




@app.route('/api/reports', methods=['POST'])
def save_report():
    try:
        report_data = request.json

        # Insert the report into the 'reports' collection
        db.reports.insert_one(report_data)

        return jsonify({'message': 'Report saved successfully'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/fetch-report/<report_id>', methods=['GET'])
def fetch_report(report_id):
    try:
        # Fetch the report associated with the given report_id from the 'reports' collection
        report = db.reports.find_one({'id': report_id})

        if not report:
            return jsonify({'message': 'Report not found'}), 404

        # Convert ObjectId to string if necessary (though not required here)
        report['_id'] = str(report['_id'])  # If you need to include _id as well

        return jsonify(report), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/fetch-reports/<user_id>', methods=['GET'])
def fetch_reports(user_id):
    try:
        # Fetch reports associated with the given user_id from the 'reports' collection
        reports = list(db.reports.find({'staffId': user_id}))

        if not reports:
            return jsonify({'message': 'No reports found for this user'}), 404

        # Convert ObjectId to string and prepare the response
        for report in reports:
            report['_id'] = str(report['_id'])

        return jsonify(reports), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete-report/<report_id>', methods=['DELETE'])
def delete_report(report_id):
    try:
        # Delete the report associated with the given report_id from the 'reports' collection
        result = db.reports.delete_one({'id': report_id})

        if result.deleted_count == 0:
            return jsonify({'message': 'Report not found'}), 404

        return jsonify({'message': 'Report deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/online-user', methods=['GET'])
def get_online_user():
    try:
        online_user = db.online_users.find_one()
        if not online_user:
            return jsonify({'error': 'No user is logged in'}), 401
        
        return jsonify({'id': online_user['id']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    



@app.route('/api/hospital/register', methods=['POST'])
def register_hospital():
    data = request.json
    try:
        hashed_password = bcrypt.generate_password_hash(data['password']).decode('utf-8')
        hospital_id = str(uuid.uuid4())  # Generate UUID

        hospital = {
            'id': hospital_id,
            'name': data['hospitalName'],
            'location': data['location'],
            'staffSize': data['staffSize'],
            'adminEmail': data['adminEmail'],
            'password': hashed_password
        }
        result = db.hospitals.insert_one(hospital)
        hospital['_id'] = str(result.inserted_id)
        return jsonify(hospital)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    user_type = data.get('userType')
    email = data.get('email')
    password = data.get('password')

    try:
        print(f"Login attempt: userType={user_type}, email={email}")  # Debugging log

        if user_type == 'admin':
            user = db.hospitals.find_one({'adminEmail': email})
        elif user_type == 'staff':
            user = db.staff.find_one({'email': email})
        else:
            return jsonify({'error': 'Invalid user type'}), 400

        if not user:
            print("User not found")  # Debugging log
            return jsonify({'error': 'Invalid credentials'}), 401

        if not bcrypt.check_password_hash(user['password'], password):
            print("Password mismatch")  # Debugging log
            return jsonify({'error': 'Invalid credentials'}), 401

        # Store user UUID and email in online users collection
        db.online_users.insert_one({
            'id': user['id'],
            'email': email,
            'userType': user_type
        })

        return jsonify({'message': f'{user_type.capitalize()} login successful'})
    except Exception as e:
        print(f"Exception during login: {e}")  # Debugging log
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/profile', methods=['GET'])
def get_admin_profile():
    try:
        # Fetch the logged-in user's ID from the online_users collection
        online_user = db.online_users.find_one()
        if not online_user:
            return jsonify({'error': 'No user is logged in'}), 401

        user_id = online_user['id']
        user = db.hospitals.find_one({'id': user_id})
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Return the user's profile details
        profile = {
            'hospitalId': user['id'],
            'name': user['name'],
            'location': user['location'],
            'staffSize': user['staffSize'],
            'adminEmail': user['adminEmail']
        }

        return jsonify(profile)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    try:
        # Clear all data in online_users collection
        db.online_users.delete_many({})
        return jsonify({'message': 'Logged out successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/add-staff', methods=['POST'])
def add_staff():
    data = request.json
    try:
        # Generate a UUID for the staff member
        staff_id = str(uuid.uuid4())

        # Fetch the current online user to get their hospital ID
        online_user = db.online_users.find_one()
        if not online_user:
            return jsonify({'error': 'No user is logged in'}), 401

        # Fetch the hospital details
        hospital = db.hospitals.find_one({'id': online_user['id']})
        if not hospital:
            return jsonify({'error': 'Hospital not found'}), 404

        # Hash the password
        hashed_password = bcrypt.generate_password_hash(data['password']).decode('utf-8')

        # Create the staff member record
        staff = {
            'id': staff_id,
            'name': data['staffName'],
            'email': data['email'],
            'password': hashed_password,
            'role': data['role'],
            'hospitalId': online_user['id']
        }
        result = db.staff.insert_one(staff)
        staff['_id'] = str(result.inserted_id)

        # Send email to staff member
        send_email(data['email'], data['staffName'], data['password'], hospital['name'])

        return jsonify(staff)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def send_email(to_email, staff_name, password, hospital_name):
    try:
        # Email configuration
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        smtp_user = "rescan.mida@gmail.com"  # Your Gmail address
        smtp_password = "dtau gfzm lrci fwbr"  # Your App Password

        # Email content
        subject = "Your Staff Account Details"
        body = f"""
        Hi {staff_name},

        Your account has been created on the Medical Imaging Diagnostic Assistant. Here are your login details:

        Email: {to_email}
        Password: {password}

        Hospital: {hospital_name}

        Please log in and change your password as soon as possible.

        Happy reporting!

        Regards,
        {hospital_name} Admin
        """

        # Setting up the MIME
        msg = MIMEMultipart()
        msg['From'] = smtp_user
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Connecting to the server and sending the email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        text = msg.as_string()
        server.sendmail(smtp_user, to_email, text)
        server.quit()
        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {str(e)}")

@app.route('/api/admin/staff', methods=['GET'])
def get_staff():
    try:
        # Fetch the logged-in user's ID from the online_users collection
        online_user = db.online_users.find_one()
        if not online_user:
            return jsonify({'error': 'No user is logged in'}), 401

        hospital_id = online_user['id']
        staff_members = list(db.staff.find({'hospitalId': hospital_id}))

        for staff in staff_members:
            staff['_id'] = str(staff['_id'])

        return jsonify(staff_members)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/delete-staff/<string:id>', methods=['DELETE'])
def delete_staff(id):
    try:
        result = db.staff.delete_one({'id': id})
        if result.deleted_count == 1:
            return jsonify({'message': 'Staff deleted successfully'})
        else:
            return jsonify({'error': 'Staff not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/staff/profile', methods=['GET'])
def get_staff_profile():
    try:
        # Fetch the logged-in user's email from the online_users collection
        online_user = db.online_users.find_one()
        if not online_user:
            return jsonify({'error': 'No user is logged in'}), 401

        email = online_user['email']
        user = db.staff.find_one({'email': email})
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Fetch hospital details
        hospital = db.hospitals.find_one({'id': user['hospitalId']})
        if not hospital:
            return jsonify({'error': 'Hospital not found'}), 404

        # Return the user's profile details
        profile = {
            'staffId': user['id'],
            'name': user['name'],
            'email': user['email'],
            'role': user['role'],
            'hospitalName': hospital['name']  # Include hospital name
        }

        return jsonify(profile)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
 
# Define where to save uploaded files
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to preprocess the image for brain MRI
def preprocess_brain_mri_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale as done during training
    return img_array

# Classification function specifically for brain MRI
def classify_brain_mri_image(img_path):
    model = load_model('./models/brain_tumor_classifier.h5')
    img_array = preprocess_brain_mri_image(img_path)
    prediction = model.predict(img_array)
    return "Tumor" if prediction[0] > 0.5 else "No Tumor"

# Function to preprocess the image (general use)
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image
    return img_array

# Classification function for CT scans
def classify_ct_brain_image(img_path, model_path):
    model = tf.keras.models.load_model(model_path)
    img_array = preprocess_image(img_path, target_size=(128, 128))  # Adjust size as per your model
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    
    class_labels = ['aneurysm', 'cancer', 'tumor']  # Update with actual class labels
    predicted_class_label = class_labels[predicted_class[0]]
    
    return predicted_class_label

# Classification function for chest X-ray images
def classify_chest_xray(img_path):
    model = tf.keras.models.load_model('./models/xray_model.h5')
    img_array = preprocess_image(img_path, target_size=(150, 150))  # Target size for chest X-ray
    prediction = model.predict(img_array)
    
    return "Pneumonia" if prediction[0][0] > 0.5 else "Normal"

# Classification function for chest CT images
def classify_chest_ct_image(img_path):
    model = tf.keras.models.load_model('./models/chest_ct_classifier_vgg16.h5')
    img_array = preprocess_image(img_path, target_size=(224, 224))  # VGG16 input size
    predictions = model.predict(img_array)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions[0])

    # Define the class labels (ensure they match the training class order)
    class_labels = ['adenocarcinoma', 'large cell carcinoma', 'normal', 'squamous carcinoma']

    # Get the predicted class label
    predicted_class_label = class_labels[predicted_class_index]

    return predicted_class_label

@app.route('/api/diagnosis', methods=['POST'])
def diagnosis():
    organ = request.form.get('organ')
    scan_type = request.form.get('scanType')
    file = request.files.get('image')

    if not organ or not scan_type:
        return jsonify({'error': 'Organ and scan type are required'}), 400

    if not file:
        return jsonify({'error': 'No image provided'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Initialize result and content variables
    result = None
    content = ""

    if organ == 'Brain' and scan_type == 'MRI':
        result = classify_brain_mri_image(file_path)
        # Generate content for Brain MRI
        content = generate_brain_mri_content(result)

    elif organ == 'Brain' and scan_type == 'CT':
        model_path = './models/ct_brain_classifier.h5'
        result = classify_ct_brain_image(file_path, model_path)
        # Generate content for Brain CT
        content = generate_brain_ct_content(result)

    elif organ == 'Chest' and scan_type == 'X-ray':
        result = classify_chest_xray(file_path)
        # Generate content for Chest X-ray
        content = generate_chest_xray_content(result)

    elif organ == 'Chest' and scan_type == 'CT':
        result = classify_chest_ct_image(file_path)
        # Generate content for Chest CT
        content = generate_chest_ct_content(result)

    else:
        return jsonify({'result': 'Invalid organ or scan type'}), 400

    return jsonify({'result': result, 'content': content})

# Function to generate content for Brain MRI
def generate_brain_mri_content(result):
    if result == 'Tumor':
        return (
            "Findings:\n"
            "- A mass lesion is detected in the left occipital lobe, showing heterogenous signal intensities on T1 and T2-weighted images.\n"
            "- The lesion has irregular borders and is associated with surrounding edema.\n"
            "- Mild enhancement is observed on FLAIR images, with slight compression of adjacent structures.\n"
            "- The gray-white matter junction is disrupted in the affected area.\n\n"
            "Impression:\n"
            "- MRI findings indicate a mass lesion in the left occipital lobe, suggestive of a neoplastic process. Contrast-enhanced imaging and biopsy are recommended for further assessment."
        )
    elif result=="No Tumor":
        return (
            "Findings:\n"
            "- The brain parenchyma shows no abnormal signal intensities or mass lesions.\n"
            "- The gray-white matter junction is preserved.\n"
            "- The ventricles and subarachnoid spaces are of normal size and configuration.\n"
            "- No abnormal enhancement is detected.\n\n"
            "Impression:\n"
            "- The MRI findings show no evidence of tumor. Follow-up imaging may be done periodically to monitor for any changes."
        )
# Function to generate content for Brain CT
def generate_brain_ct_content(result):
    if result == 'tumor':
        return (
            "Findings:\n"
            "- A mass lesion is identified in the right frontal lobe, with heterogeneous attenuation.\n"
            "- The lesion is associated with mild perilesional edema.\n"
            "- Slight effacement of the adjacent sulci is observed.\n"
            "- The ventricular system shows slight compression on the right side, but no midline shift.\n"
            "- No evidence of hemorrhage or calcification within the lesion.\n\n"
            "Impression:\n"
            "- The CT findings are consistent with a neoplastic process in the right frontal lobe. Further evaluation with MRI and potential biopsy is advised for precise diagnosis and treatment planning."
        )
    elif result == 'aneurysm':
        return (
            "Findings:\n"
            "- An aneurysmal dilation is noted in the anterior communicating artery.\n"
            "- The aneurysm measures approximately 4mm in diameter.\n"
            "- There is no evidence of hemorrhage or surrounding edema.\n"
            "- The remaining cerebral arteries appear normal, with no additional vascular abnormalities detected.\n\n"
            "Impression:\n"
            "- The CT findings suggest an aneurysm in the anterior communicating artery. Clinical correlation and further evaluation with CTA or MRA are recommended to assess the risk and potential intervention options."
        )
    elif result == 'cancer':
        return (
            "Findings:\n"
            "- A focal lesion is observed in the left temporal lobe, showing signs of irregular borders and hyperattenuation.\n"
            "- There is associated edema and mild compression of the nearby ventricles.\n"
            "- No significant calcifications or hemorrhage are present within the lesion.\n"
            "- The sulci and gyri on the affected side are slightly effaced.\n\n"
            "Impression:\n"
            "- The CT findings are suggestive of a malignant process in the left temporal lobe. Further imaging with MRI and biopsy are advised to confirm the diagnosis and to plan treatment."
        )
    else:
        return (
            "Findings:\n"
            "- The brain parenchyma exhibits normal gray and white matter differentiation.\n"
            "- No mass lesions, aneurysms, hemorrhages, or abnormal calcifications are detected.\n"
            "- The ventricles, sulci, and cisterns are within normal limits.\n"
            "- The brainstem and cerebellum appear normal, with no midline shift observed.\n\n"
            "Impression:\n"
            "- The CT findings indicate a normal brain appearance with no evidence of pathology. Routine follow-up imaging may be considered as needed based on clinical judgment."
        )

# Function to generate content for Chest X-ray
def generate_chest_xray_content(result):
    if result == 'Pneumonia':
        return (
            "Findings:\n"
            "- Patchy consolidation is present in the right lower lobe.\n"
            "- Mild blunting of the costophrenic angle is noted, suggestive of minimal effusion.\n"
            "- The cardiac silhouette appears normal, and the diaphragm is well-defined.\n\n"
            "Impression:\n"
            "- The findings are consistent with right lower lobe pneumonia. Clinical correlation and appropriate antibiotic therapy are recommended."
        )
    else:
        return (
            "Findings:\n"
            "- The lung fields are clear with no evidence of consolidation or effusion.\n"
            "- The cardiomediastinal silhouette is normal.\n"
            "- The bony structures and diaphragm are intact and normal in appearance.\n\n"
            "Impression:\n"
            "- The chest X-ray shows normal findings with no abnormalities."
        )

# Function to generate content for Chest CT
def generate_chest_ct_content(result):
    if result == 'adenocarcinoma':
        return (
            "Findings:\n"
            "- A spiculated mass lesion is present in the left lower lobe.\n"
            "- The lesion shows heterogeneous attenuation with irregular margins and surrounding ground-glass opacity.\n"
            "- No evidence of pleural effusion or mediastinal lymphadenopathy.\n"
            "- The airways are patent and normal in caliber.\n\n"
            "Impression:\n"
            "- The CT findings are suggestive of adenocarcinoma. Further investigation, including biopsy, is advised for confirmation."
        )
    elif result == 'large_cell':
        return (
            
            "Findings:\n"
            "- A large, well-defined mass lesion is noted in the right upper lobe.\n"
            "- The mass demonstrates a lobulated contour and mild central necrosis.\n"
            "- No associated lymphadenopathy or pleural effusion.\n"
            "- The visualized bones and soft tissues appear normal.\n\n"
            "Impression:\n"
            "- The lesion's appearance suggests large cell carcinoma. Histopathological confirmation is recommended for accurate diagnosis and treatment planning."
        )
    elif result == 'squamous carcinoma':
        return (
            "Findings:\n"
            "- An irregular mass is seen in the right hilum extending into the adjacent lung parenchyma.\n"
            "- The lesion is associated with mediastinal lymphadenopathy.\n"
            "- The airways appear narrowed due to the mass effect.\n"
            "- Mild pleural thickening is observed.\n\n"
            "Impression:\n"
            "- The CT findings are indicative of squamous cell carcinoma. Further evaluation, including biopsy, is necessary for a definitive diagnosis."
        )
    else:
        return (
            "Findings:\n"
            "- The lungs are clear with no evidence of focal lesions, consolidations, or nodules.\n"
            "- The pleural spaces and mediastinum appear normal.\n"
            "- No signs of pleural effusion or lymphadenopathy.\n"
            "- The airways are patent, and the heart and vascular structures are within normal limits.\n\n"
            "Impression:\n"
            "- The CT findings indicate a normal chest appearance with no signs of malignancy."
        )


# Route for changing the password
@app.route('/api/staff/change-password', methods=['POST'])
def change_password():
    data = request.json
    email = data.get('email')
    old_password = data.get('oldPassword')
    new_password = data.get('newPassword')

    try:
        # Fetch the user type from the online_staff collection
        user_record = db.online_users.find_one({'email': email})

        if not user_record:
            return jsonify({'error': 'User not found in online_staff'}), 404

        user_type = user_record.get('userType')
        print(user_type)
        # Fetch the user based on the retrieved userType
        if user_type == 'admin':
            user = db.hospitals.find_one({'adminEmail': email})
        elif user_type == 'staff':
            user = db.staff.find_one({'email': email})
        else:
            return jsonify({'error': 'Invalid user type'}), 400

        if not user:
            return jsonify({'error': 'User not found in the relevant collection'}), 404

        # Check if the old password is correct
        if not bcrypt.check_password_hash(user['password'], old_password):
            return jsonify({'error': 'Incorrect old password'}), 401

        # Hash the new password
        hashed_new_password = bcrypt.generate_password_hash(new_password).decode('utf-8')

        # Update the password in the database
        if user_type == 'admin':
            db.hospitals.update_one({'adminEmail': email}, {'$set': {'password': hashed_new_password}})
        elif user_type == 'staff':
            db.staff.update_one({'email': email}, {'$set': {'password': hashed_new_password}})

        # Send notification email to the user
        send_password_change_email(email, user['name'], user_type)

        return jsonify({'message': 'Password changed successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Function to send an email after a password change
def send_password_change_email(to_email, user_name, user_type):
    try:
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        smtp_user = "rescan.mida@gmail.com"  # Your Gmail address
        smtp_password = "dtau gfzm lrci fwbr"  # Your App Password

        # Email content
        subject = "Password Change Notification"
        body = f"""
        Hi {user_name},

        Your {user_type.capitalize()} account password has been successfully changed.

        If you did not request this change, please contact support immediately.

        Regards,
        Medical Imaging Diagnostic Assistant Team
        """

        # Setting up the MIME
        msg = MIMEMultipart()
        msg['From'] = smtp_user
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Connecting to the server and sending the email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        text = msg.as_string()
        server.sendmail(smtp_user, to_email, text)
        server.quit()
        print("Password change email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {str(e)}")


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, host='0.0.0.0', port=5000)  # Specify the port here
