mkdir C:\TQT-DEV\ext\speech-parser-docker\output-docker
docker cp speech-parser-gpu:/mnt/data/audio_files C:/TQT-DEV/ext/speech-parser-docker/output-docker/
docker cp speech-parser-gpu:/mnt/data/output/summary.txt C:/TQT-DEV/ext/speech-parser-docker/output-docker/
docker cp speech-parser-gpu:/mnt/data/output/filtered_transcription.txt C:/TQT-DEV/ext/speech-parser-docker/output-docker/
