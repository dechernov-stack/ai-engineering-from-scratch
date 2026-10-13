# Клонирование голоса и конверсия голоса

> Клонирование голоса читает текст в чужом голосе. Конверсия голоса переписывает ваш голос в чуждый, сохраняя то, что вы сказали.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 06 (Speaker Recognition), Phase 6 · 07 (TTS)
**Time:** ~75 minutes

## Проблема

В 2026 году 5 секундного аудиоклипа достаточно, чтобы произвести высококачественный клон голоса любого с потребительской ГПУ. ElevenLabs, F5-TTS, OpenVoice v2, VoiceBox все отправляют клонирование с нулевым выстрелом или несколькими выстрелами. Технология является благословением (доступность TTS, дублирование, вспомогательные голоса) и оружием (смахи, политические глубокие фальсификации, кража IP).

Две тесно связанные задачи:

- **Voice cloning (TTS-side):**текст + 5-секундный справочный голос → аудио в этом голосе.
- **Voice conversion (speech-side):**исходный аудио (человек A говорит X) + справочный голос человека B → аудио B говорит X.

Оба фактора формируют форму волны (контент, динамик, прозодия) и перекомбинируют контент из одного источника с динамиком из другого.

Ключевое ограничение , которое вы сейчас подвергаете в 2026 году:**watermarking and consent gates are legally required in the EU (AI Act, enforceable August 2026) and in California (AB 2905, effective 2025)**Ваш трубопровод должен выделять нечуванную водяную отметку и отказываться от несогласованных клонов.

## Концепция

![Voice cloning vs conversion: factorize, swap speaker, recombine](../assets/voice-cloning.svg)

**Zero-shot cloning.**Передавайте 5-секундный клип модели, которая была обучена на тысячах динамиков.

Используется F5-TTS (2024), YourTTS (2022), XTTS v2 (2024), OpenVoice v2 (2024).

**Few-shot fine-tuning.**Записывайте 5-30 минут голоса. LoRA-фини-тунирование базовой модели на час. Качество скачает от "хорошо" до "неотличимого". Коки и ElevenLabs поддерживают эту модель; сообщество использует ее с F5-TTS.

**Voice conversion (VC).**Две семьи:

- **Recognition-synthesis.**Запустить модель ASR для извлечения представления контента (например, мягкие фонемы, PPGs), затем ресинтезировать с встроением целевых динамиков.
- **Disentanglement.**Обучайте автокодер, который разделяет контент, динамик и просодию в скрытом пространстве на узле. Сваб динамик встроенный при выводе. Нижее качество, но быстрее. Используется AutoVC (2019), варианты VITS-VC.

**Neural codec-based cloning (2024+).**VALL-E, VALL-E 2, NaturalSpeech 3, VoiceBox  рассматривают аудио как дискретные токены SoundStream / EnCodec, обучают большую авторегрессивность или модели совпадения потоков по сравнению с токеном кодека.

### Этика, не зажигание.

**Watermarking.**PerTh (Perth) и SilentCipher (2024) внедряют ~16-32 битный идентификатор незаметно в аудио. Выживает перекодирование, потоковые и общие редактирования.

**Consent gates.**"Я, Рохит, в 2026-04-22 разрешаю этот голос для X цели". Сохраняйте в логике, который может быть подделан.

**Detection.**В качестве детекторов AASIST, RawNet2 и Wav2Vec2-AASIST. ASVspoof 2025 Challenge опубликовали EER 0,82,3% для современных детекторов против выходов ElevenLabs, VALL-E 2 и Bark.

### Число (2026)

| Model | Zero-shot? | SECS (target sim) | WER (intel.) | Params |
|-------|-----------|--------------------|--------------|--------|
| F5-TTS | Yes | 0.72 | 2.1% | 335M |
| XTTS v2 | Yes | 0.65 | 3.5% | 470M |
| OpenVoice v2 | Yes | 0.70 | 2.8% | 220M |
| VALL-E 2 | Yes | 0.77 | 2.4% | 370M |
| VoiceBox | Yes | 0.78 | 2.1% | 330M |

SECS > 0,70 обычно не отличается от целевой для большинства слушателей.

```figure
sp-voice-factorize
```

## Создайте его

### Шаг 1: разложите с помощью распознавания-синтеза (демо только с кодом в main.py)

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

Концептуально простой; масса реализации в `tts_model`и кодировщик динамиков.

### Шаг 2: клон с нулевым выстрелом с F5-TTS

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

Справка должна соответствовать звуковой записке; несоответствие нарушает настройку.

### Шаг 3: конверсия голоса с помощью KNN-VC

```python
import torch
from knnvc import KNNVC  # 2023 model, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC запускает WavLM для извлечения вкладки на кадры для источника и целевого бакала, затем заменяет каждую источникную картинку своим ближайшим соседом в бакле.

### Шаг 4: вставьте водный знак

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # returns payload bytes
```

~ 32 бит полезной нагрузки, обнаруживаемой после перешифровки MP3 и легкого шума.

### Шаг 5: вход в согласие

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## Используйте его

Склад 2026 года:

| Situation | Pick |
|-----------|------|
| 5-sec zero-shot clone, open-source | F5-TTS or OpenVoice v2 |
| Commercial production cloning | ElevenLabs Instant Voice Clone v2.5 |
| Voice conversion (rewriting) | KNN-VC or Diff-HierVC |
| Many-speaker fine-tune | StyleTTS 2 + speaker adapter |
| Cross-lingual cloning | XTTS v2 or VALL-E X |
| Deepfake detection | Wav2Vec2-AASIST |

## Ловушки

- **Misaligned reference transcript.**F5-TTS и аналогичные требования, чтобы текст ссылки соответствовал точно с аудио ссылки, включая пунктуацию.
- **Reverberant reference.**Эхо убивает клон.
- **Emotional mismatch.**Референсная тренировка "веселая" производит веселые клоны всего.
- **Language leakage.**Клонирование английского говорящего, а затем просьба модели говорить по-французски, часто носит акцент в любом случае; используйте межязычные модели (XTTS, VALL-E X).
- **No watermark.**Законно не может быть отправлено в ЕС с августа 2026 года.

## Погрузи его

Сделайте это .`outputs/skill-voice-cloner.md`. Разработка газопровода клонирования или конверсии с согласие на вход + водознака + целевой целью качества.

## Упражнения

1. **Easy.**Беги .`code/main.py`. Демонстрирует обмен вкладкой динамика путем вычисления косинуса между двумя "говорящими" до и после обмена.
2. **Medium.**Используйте OpenVoice v2 для клонирования собственного голоса. Измерите SECS между ссылкой и клонированием. Измерите CER через Whisper.
3. **Hard.**Нанесите водный знак SilentCipher на 20 клонов, прокрутите их через 128 кбит/с MP3 кодирование+декодирование, обнаружите полезную нагрузку.

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Zero-shot clone | 5 seconds is enough | Pretrained model + speaker embedding; no training. |
| PPG | Phonetic posteriorgram | Per-frame ASR posteriors used as language-agnostic content rep. |
| KNN-VC | Nearest-neighbor conversion | Replace each source frame with nearest target-pool frame. |
| Neural codec TTS | VALL-E style | AR model over EnCodec/SoundStream tokens. |
| Watermark | Inaudible signature | Bits embedded in audio, survive re-encode. |
| SECS | Cloning fidelity | Cosine between target and clone speaker embeddings. |
| AASIST | Deepfake detector | Anti-spoof model; detects synthesized speech. |

## Дальнейшее чтение

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) Клонирование с нулевым выстрелом SOTA с открытым исходным кодом.
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111)и [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) нейронно-кодексовый ТТС.
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879) Конверсия голоса на основе разделения.
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) ВК на основе извлечения.
- [SilentCipher (2024) — Audio Watermarking](https://github.com/sony/silentcipher) Готовая к производству 32-битная аудиоводометка.
- [ASVspoof 2025 results](https://www.asvspoof.org/) гонка вооружений детектора против синтезатора, обновленная в 2026 году.
