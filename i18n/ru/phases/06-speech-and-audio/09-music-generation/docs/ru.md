# Музыкальное поколение  Музыкальное поколение, стабильный аудио, Suno и лицензионное землетрясение

> 2026 музыкальное поколение: Suno v5 и Udio v4 доминируют в коммерческой; MusicGen, Stable Audio Open и ACE-Step ведут открытый исход.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 4 · 10 (Diffusion Models)
**Time:** ~75 minutes

## Проблема

Текст → 30-секундный до 4-минутный музыкальный клип с текстом, вокал и структурой.

1. **Instrumental generation.**Текст вроде "lo-fi хип-хоп барабаны с теплыми клавишами" → аудио.
2. **Song generation (with vocals + lyrics).**"Контри песня о дождливых ночах в Техасе" → полная песня.
3. **Conditional / controllable.**Удоление существующего клипа, регенерация моста, смена жанра, отдельность ствола или расцвет.

## Концепция

![Music generation: token-LM vs diffusion, the 2026 model map](../assets/music-generation.svg)

### Токен LM по сравнению с токенами нейронного кодека

Мета **MusicGen**(2023, MIT) и многие производные: условие на встроенности текста/мелодии, авторегрессивно предсказывают токены EnCodec (32 кГц, 4 кодовые книги), декодируют с EnCodec. 300M - 3.3B параметры.

**ACE-Step**(открытый код, 4B XL выпущен в апреле 2026) расширяет это для полной песни лирической кондиции поколения.

### Диффузия над плавками или латентами

**Stable Audio (2023)**и **Stable Audio Open (2024)**Успешное в циркуляции, звуковом дизайне, окружающей среде. Нехорошое в структурированных полнометражных песнях.

**AudioLDM / AudioLDM2**: текстовая-аудио через латентную диффузию типа T2I, обобщенную к музыке, звуковым эффектам, речи.

### Гибрид (производство)  Суно, Удио, Лирия

Снимок с замыкающими весами. Вероятно, AR-кодек LM + диффузионный вокодер с специализированными голосами голоса / барабана / мелодией. Suno v5 (2026) является лидером качества ELO 1293. Udio v4 добавляет инпейнтинг + стволовое отделение (бас, барабаны, вокал отдельные загрузки).

### Оценка

- **FAD (Fréchet Audio Distance).**Установка расстояния между генерируемым и реальным аудиораздаванием с использованием функций VGGish или PANN. Ниже лучше.
- **Musicality (subjective).**Человеческая предпочтение.
- **Text-audio alignment.**Скорь CLAP между проскотом и выходом.
- **Musicality artifacts.**Неподвижные переходы, дрейф вокальной фразы, потеря структуры после 30 секунд.

## Карта модели 2026

| Model | Params | Length | Vocals | License |
|-------|--------|--------|--------|---------|
| MusicGen-large | 3.3B | 30 s | no | MIT |
| Stable Audio Open | 1.2B | 47 s | no | Stability non-commercial |
| ACE-Step XL (Apr 2026) | 4B | &gt; 2 min | yes | Apache-2.0 |
| YuE | 7B | &gt; 2 min | yes, multilingual | Apache-2.0 |
| Suno v5 (closed) | ? | 4 min | yes, ELO 1293 | commercial |
| Udio v4 (closed) | ? | 4 min | yes + stems | commercial |
| Google Lyria 3 (closed) | ? | real-time | yes | commercial |
| MiniMax Music 2.5 | ? | 4 min | yes | commercial API |

## Правовой ландшафт (2025-2026)

- **Warner Music vs Suno settlement.**$500 млн. WMG теперь контролирует ИИ-подобство, музыкальные права и пользовательские треки на Suno.
- **EU AI Act**+ **California SB 942**Музыка, созданная ИИ, должна быть раскрыта.
- **Riffusion / MusicGen**В MIT не имеют никакого багажа соответствия, но также не имеют коммерческих вокалов.

Устройства безопасности на судне:

1. Производить только инструментальные (MusicGen, Stable Audio Open, MIT/CC0 выходы).
2. Используйте коммерческие API (Suno, Udio, ElevenLabs Music) с лицензией на поколение.
3. Поезд на собственном или лицензированном каталоге (большинство предприятий заканчиваются здесь).
4. Ознакомьте поколения с водяными знаками + метаданными.

```figure
sp-codec-tokens
```

## Создайте его

### Шаг 1: создать с помощью MusicGen

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

Три размера: `small`(300 М, быстро),`medium`(1.5В), `large`(3.3Б) Малого достаточно для "получает идею".

### Шаг 2: кондиционирование мелодии

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody берет хромограмму и сохраняет мелодию, меняя тембры.

### Шаг 3: оценка FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

Вычисляет дистанцию VGGish-всаждения. Полезный для регрессионных тестов на уровне жанра; не заменитель для человеческих слушателей.

### Шаг 4: добавление к рабочему процессу LLM-музыка

Сочетайте с идеями из уроков 7-8:

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## Используйте его

| Goal | Stack |
|------|-------|
| Instrumental sound design | Stable Audio Open |
| Game / adaptive music | Google Lyria RealTime (closed) |
| Full songs with vocals (commercial) | Suno v5 or Udio v4 with explicit license |
| Full songs with vocals (open) | ACE-Step XL or YuE |
| Short ad jingle | MusicGen melody-conditioned on a hummed reference |
| Music-video background | MusicGen + Stable Video Diffusion |

## Ловушки, которые по-прежнему будут в 2026 году

- **Copyright-laundering prompts.**"Песня в стиле Тейлор Свифт"  коммерческая Suno / Audio фильтр эти теперь, открытые модели не. Добавьте свой собственный список фильтров.
- **Repetition / drift past 30 s.**Модели AR - цепочка. Пересекайте несколько поколений или используйте ACE-Step для структурной согласованности.
- **Tempo drift.**Модели отходят от BPM. Используйте теги BPM в запроте и послефильтре с библиотекой `beat_track`- Я не знаю .
- **Vocal intelligibility.**Если слова важны, используйте коммерческий API или тонкую настройку.
- **Mono output.**Открытые модели генерируют моно- или фальшивые стерео.

## Погрузи его

Сделайте это .`outputs/skill-music-designer.md`. Выберите модель, лицензионную стратегию, длину / план структуры и метаданные раскрытия для развертывания музыкального поколения.

## Упражнения

1. **Easy.**Беги .`code/main.py`. Он производит "генеративную" аккордовую прогрессию + шаблоны барабанов в виде символов ASCII  музыкального мультфильма.
2. **Medium.**Установка`audiocraft`, генерировать 10-секундные клипы по 4 жанровым запросам с помощью MusicGen-small, измерять FAD против ссылочного жанрового набора.
3. **Hard.**Используя ACE-Step (или MusicGen-melody), генерируйте три варианта одной мелодии с различными запросами тембров.

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| FAD | Audio FID | Fréchet distance between embedding distributions of real vs generated. |
| Chromagram | Melody as pitches | 12-dim per-frame vector; input to melody conditioning. |
| Stems | Instrument tracks | Separated bass / drums / vocals / melody as WAV. |
| Inpainting | Regen a section | Mask a time window; model regenerates just that. |
| CLAP | Text-audio CLIP | Contrastive audio-text embedding; eval text-audio alignment. |
| EnCodec | Music codec | Meta's neural codec used by MusicGen; 32 kHz, 4 codebooks. |

## Дальнейшее чтение

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) открытый авторегрессивный показатель.
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) дефолтный звук.
- [ACE-Step](https://github.com/ace-step/ACE-Step)Открытый 4B полный певный генератор, апрель 2026 года.
- [Suno v5 platform docs](https://suno.com) лидер в области коммерческого качества.
- [AudioLDM2](https://arxiv.org/abs/2308.05734) латентная диффузия для музыки + звуковые эффекты.
- [WMG-Suno settlement coverage](https://www.musicbusinessworldwide.com/warner-music-group-settles-with-suno-strikes-first-of-its-kind-deal-with-ai-song-generator/) Ноябрь 2025 прецедент.
