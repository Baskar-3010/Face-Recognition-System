from flask import Flask, render_template, request,jsonify
import register,main

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/register', methods=['post'])
def register_user():
    # Call a function from register.py
    if 'rollNo' in request.form is not None and 'name' in request.form is not None:
        register.registration(request.form['rollNo'], request.form['name'])
        # Call your registration logic here
        return jsonify({'message': 'User registered successfully'})
    else:
        return jsonify({'error': 'Missing parameters: username and/or email'}), 400

@app.route('/main', methods=['post'])
def main_file_execution():
    main.main_fun()
    return "Attendance Marked successfully"

if __name__ == '__main__':
    app.run(debug=True)
