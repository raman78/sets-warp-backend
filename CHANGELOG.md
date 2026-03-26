# Changelog

## [Unreleased]

### Fixed
- Naprawiono błąd `httpx.RemoteProtocolError: Server disconnected without sending a response` w `admin_train.py`.
- Zoptymalizowano funkcje `_list_staging_folders` oraz `_list_screen_type_files`, aby używały `api.list_repo_tree` zamiast pobierania pełnej listy plików z Hugging Face (`api.list_repo_files`). Zapobiega to timeoutom na dużych zbiorach danych.
