from flask import Flask
from studyCam import study_app
from freeCam import free_app

app = Flask(__name__)

# Register blueprints from studyCam and freeCam
app.register_blueprint(study_app, url_prefix='/study')
app.register_blueprint(free_app, url_prefix='/free')

if __name__ == '__main__':
    app.run(debug=True)