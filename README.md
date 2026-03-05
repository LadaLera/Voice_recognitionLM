# Voice_recognitionLM
ZPD tēma: "ANGĻU VALODAS DIALEKTU UN AKCENTU ATPAZĪŠANAS PROGRAMMAS IZSTRĀDE"
Šī programma ir izstrādāta, lai atpazītu dažādus angļu valodas dialektus un akcentus, izmantojot XGBoost mašīnmācīšanās modeli. Programma tika trenēta ar 18,406 audio paraugiem no 25 dažādiem angļu valodas akcentiem un sasniedza 88,6% precizitāti TOP 3 klasifikācijā reālā laika režīmā.

Nepieciešamie Faili:
1. main.py
2. accent_detector.py
3. trained_accent_model_stable.pkl

Instalācija
1. Prasības:
- Python 3.8+
- Windows 10/11
- Mikrofons
- Interneta savienojums (spēles režīmam)

2. Instalē bibliotēkas:
pip install pygame, 
pip install librosa, 
pip install xgboost, 
pip install speechrecognition, 
pip install pyaudio, 
pip install scikit-learn, 
pip install numpy

Sistēmas Arhitektūra:
1. LIETOTĀJA SASKARNE
 • Galvenais izvēlnes ekrāns
 • Spēles režīms, Akcentu detektors

3. AUDIO IERAKSTĪŠANA (SpeechRecognition + PyAudio)
 • Mikrofona aktivizēšana (8 sekundes, WAV formāts)
 • Audio resampling uz 16kHz, mono konversija

4. PAZĪMJU EKSTRAKCIJA (Librosa bibliotēka)                 
 • MFCC (40 koeficienti) → Mean, Std, Delta, Delta-Delta
 • Pitch (F0) analīze
 • Spectral features
 • Chroma features
 • Spectral contrast

5. AKCENTU KLASIFIKĀCIJA (XGBoost ML)
 • XGBoost modelis: 300 lēmumu koki
 • 25 akcenta kategorijas
 • StandardScaler normalizācija

6. TEKSTA ATPAZĪŠANA (Google Speech Recognition API)
   • Tikai spēles un tulkotāja režīmiem
 • Nepieciešams internets

7.REZULTĀTU APSTRĀDE
 • TOP-3 klasifikācija ar varbūtības procentiem
 • Akcenta identifikācija
 • Punktu skaitīšana (spēles režīmā)

8. ATGRIEZENISKĀ SAITE
 • Pareizi/Nepareizi indikatori (zaļš/sarkans)
 • Vizuāla feedback
 • Akcenta parādīšana ekrānā

Programmas Režīmi
1. Spēles Režīms:
- Interaktīva akcentu atpazīšanas spēle
- Punktu skaitīšanas sistēma
- Nepieciešams internets (Google Speech Recognition)

2. Akcentu Detektors:
- Precīza akcenta noteikšana (TOP-3 rezultāti)
- Darbojas lokāli (bez interneta)
- Apstrādes laiks: 2-3 sekundes

Precizitāte:
| Eksperiments                  | Precizitāte (TOP-1) | TOP-3   | Dalībnieki |
|-------------------------------|---------------------|---------|------------|
|1. Eksperiments (audio faili)  | 19.16%              | 58.9 %  | 47         |
|2. Eksperiments (reālais laiks)| 25.0%               | 88.6 %  | 44         |
Secinājums: Reālā laika režīms sniedz labākus rezultātus (+29.7 % TOP-3 precizitātes pieaugums).

Kontakti:
lmijerska@edu.riga.lv
vpavlova12@edu.riga.lv
