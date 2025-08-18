import os
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
from .config import Config

def create_app():
    load_dotenv()

    application = Flask(__name__)
    #application.config.from_object(Config)

    