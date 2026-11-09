# CI/CD: PhotoAdvisor

## Поведение

- Pull request: существующие Python-проверки, затем сборка linux/amd64 Docker-образа без публикации. PhotoAdvisor также запускает Node-тесты интерфейса.
- Push в основную ветку: те же проверки, затем публикация `ghcr.io/arm-auto-place/photoadvisor:sha-<полный SHA коммита>`.
- Push в develop (где настроен): проверки и сборка без публикации.
- Actions → ML service CI/CD → Run workflow на основной ветке → `deploy=true`: проверки, публикация и деплой **того же образа по digest** в environment `production`.
- Деплой выполняется только вручную с основной ветки, после успешных проверок и сборки. Обновляется один Compose-сервис через `--no-deps --no-build`; база, Kafka, S3 и другие сервисы не перезапускаются.

## Настройки GitHub

Создайте environment `production`. При необходимости включите reviewers и ограничьте deployment branches основной веткой. Добавьте environment variables:

| Variable | Значение |
| --- | --- |
| `DEPLOY_HOST` | DNS-имя или IPv4 хоста, доступного с GitHub-hosted runner |
| `DEPLOY_USER` | SSH-пользователь с доступом к Docker |
| `DEPLOY_PORT` | SSH-порт; по умолчанию 22 |
| `DEPLOY_PATH` | Абсолютная папка существующего Compose-проекта на Linux |
| `COMPOSE_FILE` | Один Compose-файл относительно этой папки; например `compose.yaml` |
| `COMPOSE_PROJECT_NAME` | **Точное имя уже запущенного Compose-проекта** (проверьте `docker compose ls`) |
| `COMPOSE_SERVICE` | Точное имя сервиса; в примере `advisor` |

Пути должны быть без пробелов. Если используются несколько Compose-файлов, подготовьте один deployment-файл с нужными настройками. Workflow не копирует конфигурацию приложения на хост.

Environment secrets:

- `DEPLOY_SSH_KEY`: приватный SSH-ключ без passphrase для этого пользователя.
- `DEPLOY_KNOWN_HOSTS`: заранее проверенная запись ключа хоста в формате OpenSSH known_hosts (для нестандартного порта — `[host]:port`). Проверка ключа хоста обязательна.

Для публикации используется штатный `GITHUB_TOKEN` с `packages: write`. Организация должна разрешать Actions и запись пакетов; существующему GHCR-пакету нужно дать доступ этому репозиторию. Если пакет приватный, заранее выполните `docker login ghcr.io` на хосте под deployment-пользователем с токеном `read:packages`. Токены приложения остаются в серверном env-файле.

## Подготовка хоста

Нужны Linux, Bash, `flock`, Docker и Compose v2 с `up --wait --wait-timeout`. Архитектура образа — amd64. Серверный Compose-файл, env-файлы, сети и зависимости должны уже существовать. Каталог должен быть доступен deployment-пользователю для записи.

PhotoAdvisor требует подготовленных моделей в `./models`; workflow не скачивает и не обучает модели. Его `/health/ready` должен стать успешным за 180 секунд после запуска. У cv-number-blur нет Docker healthcheck: `--wait` подтверждает только запуск контейнера, а не готовность Kafka/S3 или качество inference. `/health` assistant_bot — проверка HTTP-процесса, не доступности БД/DeepSeek.

## Повторный запуск и откат

При успешном деплое сохраняется `.deploy-<service>.yaml` с точным digest; предыдущий успешный override — `.deploy-<service>.yaml.previous`. Для последующих ручных команд используйте базовый файл **и** этот override, иначе Compose может вернуть старый image из базовой конфигурации.

Пример (подставьте реальное имя проекта):

```bash
docker compose -p PROJECT -f compose.yaml -f .deploy-advisor.yaml ps advisor
# Ручной откат, если предыдущий override существует:
docker compose -p PROJECT -f compose.yaml -f .deploy-advisor.yaml.previous up -d --no-deps --no-build --wait --wait-timeout 180 advisor
```

Деплои одного проекта сериализуются файловой блокировкой на хосте. Ошибка pull останавливает деплой до пересоздания контейнера. Ошибка startup/healthcheck завершает workflow с ошибкой; **автоматического отката нет**, контейнер уже мог быть заменён. При первом деплое предыдущая версия не сохраняется автоматически — перед первым запуском зафиксируйте текущий image и команду отката. Успешный override при ошибке не перезаписывается.

Сборка и деплой здесь подготовлены как файлы репозитория. Для запуска нужно загрузить изменения в GitHub и заполнить указанные настройки; локальная подготовка не означает выполненный деплой.
