import os
import controller.video
import controller.ai_test
# import speech_recognition as sr

import nltk

from controller.core import app

nltk.download('punkt')

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def main():
    app.run(debug=True)


if __name__ == '__main__':
    main()