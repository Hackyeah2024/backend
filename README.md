## Installing dependencies
```
python3 -m venv ./.venv
source ./.venv/bin/activate
pip install -f requirements.txt
```
## Running
```
python3 app.py
```

#Video processing request:
```
curl --location 'http://localhost:5000/process_video' \
--form 'video_file=@"/home/tarjei/Downloads/Video.mp4"'
```
