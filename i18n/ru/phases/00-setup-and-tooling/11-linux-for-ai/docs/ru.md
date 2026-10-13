# Linux для AI

> Большая часть AI работает на Linux. Вам нужно знать достаточно, чтобы не застрять.

**Type:** Learn
**Languages:** --
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~30 минут

## Learning Objectives

- Ориентироваться в файловой системе Linux и выполнять базовые операции с файлами из командной строки
- Управлять правами доступа к файлам с помощью `chmod` и `chown` для устранения ошибок «Permission denied»
- Устанавливать системные пакеты с помощью `apt` и настраивать новую GPU-машину для работы с AI
- Распознавать различия между macOS и Linux, на которых обычно спотыкаются разработчики, работающие на удалённых машинах

## Проблема

Вы разрабатываете на macOS или Windows. Но в тот момент, когда вы заходите по SSH на облачную GPU-машину, арендуете инстанс Lambda или поднимаете машину на EC2, вы оказываетесь в Ubuntu. Терминал — ваш единственный интерфейс. Нет ни Finder, ни Explorer, ни GUI. Если вы не умеете ориентироваться в файловой системе, устанавливать пакеты и управлять процессами из командной строки, вы застрянете, оплачивая простаивающие часы GPU, пока гуглите «как разархивировать файл в Linux».

Это руководство по выживанию. В нём описано именно то, что нужно для работы на удалённой машине с Linux в контексте AI. Ничего лишнего.

## Структура файловой системы

Linux организует всё под единым корневым `/`. Нет ни `C:\`, ни `/Volumes`. Вот директории, с которыми вы реально будете иметь дело:

```mermaid
graph TD
    root["/"] --> home["home/your-username/<br/>Your files — clone repos, run training"]
    root --> tmp["tmp/<br/>Temporary files, cleared on reboot"]
    root --> usr["usr/<br/>System programs and libraries"]
    root --> etc["etc/<br/>Config files"]
    root --> varlog["var/log/<br/>Logs — check when something breaks"]
    root --> mnt["mnt/ or /media/<br/>External drives and volumes"]
    root --> proc["proc/ and /sys/<br/>Virtual files — kernel and hardware info"]
```

Ваша домашняя директория — это `~` или `/home/your-username`. Почти всё, что вы делаете, происходит здесь.

## Основные команды

Это 15 команд, которые покрывают 95% того, что вам придётся делать на удалённой GPU-машине.

### Перемещение по системе

```bash
pwd                         # Where am I?
ls                          # What's here?
ls -la                      # What's here, including hidden files with details?
cd /path/to/dir             # Go there
cd ~                        # Go home
cd ..                       # Go up one level
```

### Файлы и директории

```bash
mkdir my-project            # Create a directory
mkdir -p a/b/c              # Create nested directories in one shot

cp file.txt backup.txt      # Copy a file
cp -r src/ src-backup/      # Copy a directory (recursive)

mv old.txt new.txt          # Rename a file
mv file.txt /tmp/           # Move a file

rm file.txt                 # Delete a file (no trash, it's gone)
rm -rf my-dir/              # Delete a directory and everything inside
```

`rm -rf` — это безвозвратная операция. Отмены нет. Дважды проверьте путь, прежде чем нажать enter.

### Чтение файлов

```bash
cat file.txt                # Print entire file
head -20 file.txt           # First 20 lines
tail -20 file.txt           # Last 20 lines
tail -f log.txt             # Follow a log file in real time (Ctrl+C to stop)
less file.txt               # Scroll through a file (q to quit)
```

### Поиск

```bash
grep "error" training.log           # Find lines containing "error"
grep -r "learning_rate" .           # Search all files in current directory
grep -i "cuda" config.yaml          # Case-insensitive search

find . -name "*.py"                 # Find all Python files under current dir
find . -name "*.ckpt" -size +1G     # Find checkpoint files larger than 1GB
```

## Права доступа

У каждого файла в Linux есть владелец и биты прав доступа. Вы столкнётесь с этим, когда скрипты не будут запускаться или вы не сможете писать в директорию.

```bash
ls -l train.py
# -rwxr-xr-- 1 user group 2048 Mar 19 10:00 train.py
#  ^^^             owner permissions: read, write, execute
#     ^^^          group permissions: read, execute
#        ^^        everyone else: read only
```

Типичные решения:

```bash
chmod +x train.sh           # Make a script executable
chmod 755 deploy.sh         # Owner: full, others: read+execute
chmod 644 config.yaml       # Owner: read+write, others: read only

chown user:group file.txt   # Change who owns a file (needs sudo)
```

Когда появляется сообщение «Permission denied», это почти всегда проблема прав доступа. `chmod +x` или `sudo` решают большинство случаев.

## Управление пакетами (apt)

Ubuntu использует `apt`. Так устанавливается программное обеспечение на уровне системы.

```bash
sudo apt update             # Refresh the package list (always do this first)
sudo apt install -y htop    # Install a package (-y skips confirmation)
sudo apt install -y build-essential  # C compiler, make, etc. Needed by many Python packages
sudo apt install -y tmux    # Terminal multiplexer (keep sessions alive after disconnect)

apt list --installed        # What's installed?
sudo apt remove htop        # Uninstall
```

Типичные пакеты, которые вы будете устанавливать на новой GPU-машине:

```bash
sudo apt update && sudo apt install -y \
    build-essential \
    git \
    curl \
    wget \
    tmux \
    htop \
    unzip \
    python3-venv
```

## Пользователи и sudo

Обычно вы входите в систему под обычным пользователем. Некоторые операции требуют root-доступа (администратора).

```bash
whoami                      # What user am I?
sudo command                # Run a single command as root
sudo su                     # Become root (exit to go back, use sparingly)
```

На облачных GPU-инстансах вы, как правило, единственный пользователь и уже имеете доступ к sudo. Не запускайте всё от имени root. Используйте sudo только когда это необходимо.

## Процессы и systemd

Когда обучение зависает, или вам нужно проверить, что запущено:

```bash
htop                        # Interactive process viewer (q to quit)
ps aux | grep python        # Find running Python processes
kill 12345                  # Gracefully stop process with PID 12345
kill -9 12345               # Force kill (use when graceful doesn't work)
nvidia-smi                  # GPU processes and memory usage
```

systemd управляет сервисами (фоновыми демонами). Вы будете использовать его, если запускаете серверы инференса:

```bash
sudo systemctl start nginx          # Start a service
sudo systemctl stop nginx           # Stop it
sudo systemctl restart nginx        # Restart it
sudo systemctl status nginx         # Check if it's running
sudo systemctl enable nginx         # Start automatically on boot
```

## Дисковое пространство

На GPU-машинах часто ограниченное место на диске. Модели и датасеты быстро его заполняют.

```bash
df -h                       # Disk usage for all mounted drives
df -h /home                 # Disk usage for /home specifically

du -sh *                    # Size of each item in current directory
du -sh ~/.cache             # Size of your cache (pip, huggingface models land here)
du -sh /data/checkpoints/   # Check how big your checkpoints are

# Find the biggest space hogs
du -h --max-depth=1 / 2>/dev/null | sort -hr | head -20
```

Типичные способы освободить место:

```bash
# Clear pip cache
pip cache purge

# Clear apt cache
sudo apt clean

# Remove old checkpoints you don't need
rm -rf checkpoints/epoch_01/ checkpoints/epoch_02/
```

## Работа с сетью

Вам придётся скачивать модели, передавать файлы и обращаться к API из командной строки.

```bash
# Download files
wget https://example.com/model.bin                   # Download a file
curl -O https://example.com/data.tar.gz              # Same thing with curl
curl -s https://api.example.com/health | python3 -m json.tool  # Hit an API, pretty-print JSON

# Transfer files between machines
scp model.bin user@remote:/data/                     # Copy file to remote machine
scp user@remote:/data/results.csv .                  # Copy file from remote to local
scp -r user@remote:/data/checkpoints/ ./local-dir/   # Copy directory

# Sync directories (faster than scp for large transfers, resumes on failure)
rsync -avz --progress ./data/ user@remote:/data/
rsync -avz --progress user@remote:/results/ ./results/
```

Используйте `rsync` вместо `scp` для всего крупного. Он передаёт только изменённые байты и справляется с прерванными соединениями.

## tmux: сохраняйте сессии активными

Когда вы заходите по SSH на удалённую машину, закрытие ноутбука убивает запущенное обучение. tmux предотвращает это.

```bash
tmux new -s train           # Start a new session named "train"
# ... start your training, then:
# Ctrl+B, then D            # Detach (training keeps running)

tmux ls                     # List sessions
tmux attach -t train        # Reattach to session

# Inside tmux:
# Ctrl+B, then %            # Split pane vertically
# Ctrl+B, then "            # Split pane horizontally
# Ctrl+B, then arrow keys   # Switch between panes
```

Всегда запускайте длительные задачи обучения внутри tmux. Всегда.

## WSL2 для пользователей Windows

Если вы на Windows, WSL2 даёт вам полноценную среду Linux без двойной загрузки.

```bash
# In PowerShell (admin)
wsl --install -d Ubuntu-24.04

# After restart, open Ubuntu from Start menu
sudo apt update && sudo apt upgrade -y
```

WSL2 использует настоящее ядро Linux. Всё, что описано в этом уроке, работает внутри него. Ваши файлы Windows доступны по пути `/mnt/c/Users/YourName/` изнутри WSL.

Проброс GPU работает при установленных драйверах NVIDIA на стороне Windows. Установите драйвер NVIDIA для Windows (не для Linux), и CUDA станет доступна внутри WSL2.

## Подводные камни: от macOS к Linux

Вещи, на которых вы споткнётесь, если пришли с macOS:

| macOS | Linux | Примечания |
|-------|-------|-------|
| `brew install` | `sudo apt install` | Иногда разные названия пакетов. `brew install htop` и `sudo apt install htop` работают одинаково, но `brew install readline` и `sudo apt install libreadline-dev` — нет. |
| `open file.txt` | `xdg-open file.txt` | Но на удалённой машине у вас не будет GUI. Используйте `cat` или `less`. |
| `pbcopy` / `pbpaste` | Недоступно | Передача в/из буфера обмена через конвейер не работает по SSH. |
| `~/.zshrc` | `~/.bashrc` | По умолчанию в macOS используется zsh. На большинстве Linux-серверов — bash. |
| `/opt/homebrew/` | `/usr/bin/`, `/usr/local/bin/` | Бинарные файлы хранятся в разных местах. |
| `sed -i '' 's/a/b/' file` | `sed -i 's/a/b/' file` | macOS-версии sed нужна пустая строка после `-i`. В Linux — нет. |
| Файловая система без учёта регистра | Файловая система с учётом регистра | `Model.py` и `model.py` — это два разных файла в Linux. |
| Окончания строк `\n` | Окончания строк `\n` | То же самое. Но Windows использует `\r\n`, что ломает bash-скрипты. Запустите `dos2unix`, чтобы это исправить. |

## Краткий справочник

```
Navigation:     pwd, ls, cd, find
Files:          cp, mv, rm, mkdir, cat, head, tail, less
Search:         grep, find
Permissions:    chmod, chown, sudo
Packages:       apt update, apt install
Processes:      htop, ps, kill, nvidia-smi
Services:       systemctl start/stop/restart/status
Disk:           df -h, du -sh
Network:        curl, wget, scp, rsync
Sessions:       tmux new/attach/detach
```

```figure
s0-process-fork
```

## Упражнения

1. Подключитесь по SSH к любой машине с Linux (или откройте WSL2) и перейдите в свою домашнюю директорию. Создайте папку проекта, создайте в ней три пустых файла с помощью `touch`, затем выведите их список с помощью `ls -la`.
2. Установите `htop` через apt, запустите его и определите, какой процесс использует больше всего памяти.
3. Запустите сессию tmux, выполните внутри неё `sleep 300`, отсоединитесь, выведите список сессий и снова подключитесь к ней.
4. Используйте `df -h`, чтобы проверить доступное место на диске, затем используйте `du -sh ~/.cache/*`, чтобы найти, что занимает место в вашем кэше.
5. Передайте файл с локальной машины на удалённую с помощью `scp`, затем выполните ту же передачу с помощью `rsync` и сравните опыт использования.