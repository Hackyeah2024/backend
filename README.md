## Installing dependencies
```
python3 -m venv ./.venv
source ./.venv/bin/activate
pip install -f requirements.txt
```
## Adding environment variables
You have to set the environmental variables with path to Google Cloud config file and with tokens for OpenAI and Cohere.
```
export GOOGLE_APPLICATION_CREDENTIALS="/Users/user/projects/backend/google_keys.json"
export OPENAI_API_KEY=open-api-key
export COHERE_API_KEY=cohere-api-key
```
## Running
```
python3 app.py
```

## Video processing request:
```
curl --location 'http://localhost:5000/process_video' \
--form 'video_file=@"/home/tarjei/Downloads/Video.mp4"'
```
