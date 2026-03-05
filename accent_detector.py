import os
import torch
import torchaudio
import traceback
from typing import Optional, Tuple
import tempfile


class AccentDetector:
    def __init__(self, model_name: str = "auto"):
        self.model = None
        self.processor = None
        self.model_type = None
        self.model_name = model_name
        self._load_model()
    
    def _load_model(self):
        models_to_try = []
        
        if self.model_name == "auto":
            models_to_try = [
                ("milespurvis", self._load_milespurvis),
                ("bookbot", self._load_bookbot),
                ("jzuluaga", self._load_jzuluaga_ecapa)
            ]
        elif self.model_name == "milespurvis":
            models_to_try = [("milespurvis", self._load_milespurvis)]
        elif self.model_name == "bookbot":
            models_to_try = [("bookbot", self._load_bookbot)]
        elif self.model_name == "jzuluaga":
            models_to_try = [("jzuluaga", self._load_jzuluaga_ecapa)]
        
        for model_type, load_func in models_to_try:
            try:
                print(f"Trying to load {model_type} model...")
                load_func()
                self.model_type = model_type
                print(f"✓ Successfully loaded {model_type} model")
                return
            except Exception as e:
                print(f"✗ Failed to load {model_type}: {e}")
                continue
        
        print("✗ All models failed to load")
        self.model = None
    
    def _load_milespurvis(self):
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
        
        model_name = "MilesPurvis/english-accent-classifier"
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
    
    def _load_bookbot(self):
        from speechbrain.pretrained.interfaces import foreign_class
        
        self.model = foreign_class(
            source="bookbot/english-accent-classifier",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            run_opts={"device": "cpu"}
        )
    
    def _load_jzuluaga_ecapa(self):
        from speechbrain.inference.classifiers import EncoderClassifier
        
        self.model = EncoderClassifier.from_hparams(
            source="Jzuluaga/accent-id-commonaccent_ecapa",
            savedir="pretrained_models/accent-id-commonaccent_ecapa",
            run_opts={"device": "cpu"}
        )
    
    def _resample_if_needed(self, waveform: torch.Tensor, sr: int, 
                           target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, 
                new_freq=target_sr
            )
            waveform = resampler(waveform)
            sr = target_sr
        return waveform, sr
    
    def _load_audio_safe(self, path: str) -> Tuple[torch.Tensor, int]:
        try:
            waveform, sr = torchaudio.load(path)
            return waveform, sr
        except Exception as e:
            print(f"torchaudio failed, trying scipy...")
        
        try:
            import scipy.io.wavfile as wavfile
            import numpy as np
            
            sr, data = wavfile.read(path)
            
            if data.dtype.kind in ('i', 'u'):
                maxv = float(2 ** (8 * data.dtype.itemsize - 1))
                data = data.astype("float32") / maxv
            
            waveform = torch.from_numpy(data.astype("float32"))
            
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.shape[0] != 1:
                waveform = waveform.transpose(0, 1)
            
            return waveform, sr
        except Exception as e2:
            raise Exception(f"All audio loading methods failed: {e2}")
    
    def _prepare_audio(self, audio_path: str) -> torch.Tensor:
        waveform, sr = self._load_audio_safe(audio_path)
        waveform, sr = self._resample_if_needed(waveform, sr, target_sr=16000)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        return waveform
    
    def detect_from_file(self, audio_path: str) -> str:
        if self.model is None:
            return "Model not loaded"
        
        try:
            if self.model_type == "milespurvis":
                return self._detect_milespurvis(audio_path)
            elif self.model_type == "bookbot":
                return self._detect_bookbot(audio_path)
            elif self.model_type == "jzuluaga":
                return self._detect_jzuluaga(audio_path)
            else:
                return "Unknown model type"
                
        except Exception as e:
            print(f"Detection error: {e}")
            traceback.print_exc()
            return "Detection failed"
    
    def _detect_milespurvis(self, audio_path: str) -> str:
        import librosa

        audio, sr = librosa.load(audio_path, sr=16000)

        inputs = self.processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            pred_id = torch.argmax(probs).item()

        predicted_accent = self.model.config.id2label[pred_id]
        confidence = probs[pred_id].item()
        
        return f"{predicted_accent} ({confidence:.1%})"
    
    def _detect_bookbot(self, audio_path: str) -> str:
        out_prob, score, index, text_lab = self.model.classify_file(audio_path)
        return text_lab[0] if isinstance(text_lab, list) else str(text_lab)
    
    def _detect_jzuluaga(self, audio_path: str) -> str:
        waveform = self._prepare_audio(audio_path)

        prediction = self.model.classify_batch(waveform)
        
        if isinstance(prediction, tuple) and len(prediction) >= 4:
            text_lab = prediction[3]
            if isinstance(text_lab, list) and len(text_lab) > 0:
                return text_lab[0]
            elif isinstance(text_lab, str):
                return text_lab
        
        return "Unknown"
    
    def detect_from_audio_data(self, audio_data) -> str:
        if not hasattr(audio_data, "get_wav_data"):
            return "Invalid audio data"

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            wav_bytes = audio_data.get_wav_data()
            tmp_file.write(wav_bytes)
        
        try:
            result = self.detect_from_file(tmp_path)
            return result
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    def is_available(self) -> bool:
        return self.model is not None
    
    def get_supported_accents(self) -> list:
        if self.model_type == "milespurvis":
            return ["us", "england", "indian", "australia", "canada", "african"]
        elif self.model_type in ["bookbot", "jzuluaga"]:
            return [
                "us", "england", "australia", "indian", "canada", 
                "bermuda", "scotland", "african", "ireland", "newzealand", 
                "wales", "malaysia", "philippines", "singapore", 
                "hongkong", "southatlandtic"
            ]
        return []


_detector_instance: Optional[AccentDetector] = None


def get_accent_detector() -> AccentDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = AccentDetector()
    return _detector_instance


def detect_accent(audio_input) -> str:
    detector = get_accent_detector()
    
    if not detector.is_available():
        return "Detector not available"
    
    if isinstance(audio_input, str):
        return detector.detect_from_file(audio_input)
    else:
        return detector.detect_from_audio_data(audio_input)