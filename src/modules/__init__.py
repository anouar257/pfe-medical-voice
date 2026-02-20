# Modules IA : Whisper, pyannote, Resemblyzer
from .speech_to_text import load_model, transcribe_audio
# Modules optionnels (nécessitent des tokens/packages supplémentaires)
try:
    from .speaker_diarization import load_diarization_pipeline, diarize_audio, combine_transcription_diarization
except ImportError:
    pass

try:
    from .voice_embeddings import init_voice_encoder, create_voice_embedding, compare_voices, save_voice_profile
except ImportError:
    pass
