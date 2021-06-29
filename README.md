# Routesim

Во время разработки использовался Python версии 3.9.5. Нотбуки хранятся в
директории `notebooks`.

## Установка симулятора

Создать тестовое окружение:

```
$ python -m venv routesim_env
$ source routesim_env/bin/activate
```

Перейти в директорию `routesim/src` и выполнить:

```
$ python setup.py develop
```

## Запуск экспериментов

Добавить ядро с тестовым окружением в Jupyter:

```
$ pip install ipykernel
$ python -m ipykernel install --user --name=routesim_env
```

Установить зависимости:

```
$ pip install -r routesim/requirements.txt
```

Перейти в директорию с нотбуками и открыть `run_simulation`.

## Обучение модели

Готовые предобученные модели находятся в директории `routesim/torch_models`.
Их нужно разжать перед использованием:

```
$ gunzip -k model_name
```

Для самостоятельного обучения модели, нужно перейти в директорию с нотбуками и
открыть `dqn_pretrain`. Обучающая выборка уже сгенерирована и находится в файле
`routesim/src/pretrain_data.csv`. Чтобы самостоятельно сгенерировать выборку,
нужно выполнить `routesim/src/gen_pretrain_data.py`.
