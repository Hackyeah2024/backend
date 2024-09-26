from json import JSONEncoder

from flask import Flask
from enum import Enum
from flask import Flask

app = Flask(__name__)

# Custom JSON encoder for enums
class CustomJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value  # Convert enum to its value (string)
        return super().default(obj)

app.json_encoder = CustomJSONEncoder