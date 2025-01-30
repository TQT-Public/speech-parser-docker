### **Proposed Restructuring**
To make the project more modular and scalable, let’s restructure it slightly:

1. **New Directory Structure**:

```bash
speech-parser-docker/
├── audio_files/            # Input audio files
├── output/                 # Processed outputs
├── models/                 # All AI models (we’ll add subdirectories for each model)
│   ├── ai/deepseek/DeepSeek-R1-Distill-Llama-8B  # Language models (LLaMA, Mistral, Falcon, DeepSeek, etc.)
│   ├── vosk/vosk-model-ru-0.42               # Vosk speech recognition model
│   └── stable_diffusion/   # Stable Diffusion model (if needed)
├── speech_parser/                # Scripts for processing
│   ├── speech_parser.py    # Main script for speech parsing
│   ├── audio_processing # All the previous folders - related to python and parsing the text from WAV (audio_processing, utils, tests)
│   └── utils.py            # Utility functions
├── speech_analyzer/                # Scripts for processing
│   ├── speech_analyzer.py    # Main script for speech parsing
│   ├── model_loader.py     # Script to load and manage models
│   └── dialogue_analyzer.py # Script for dialogue analysis and summarization
├── Dockerfile              # Docker setup
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── .env                    # Environment variables
```
