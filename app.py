import os

from flask_cors import CORS

import controller.video
import controller.ai_test
# import speech_recognition as sr

import nltk

from controller.core import app

nltk.download('punkt')

app.config['CORS_HEADERS'] = 'Content-Type'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def main():
    app.run(debug=True)


if __name__ == '__main__':
    main()