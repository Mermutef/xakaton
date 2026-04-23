import requests

BASE_URL = 'http://www.cbr.ru/dataservice'

def find_indicator_ids(keywords_list):
    """
    Поиск publicationId и datasetId по ключевым словам.
    Возвращает список словарей с ключами: name, publicationId, datasetId, type
    """
    # Шаг 1: Получаем список всех публикаций
    print("Загружаем список публикаций...")
    resp = requests.get(f'{BASE_URL}/publications')
    publications = resp.json()

    found_indicators = []

    for publ in publications:
        publ_id = publ['id']
        publ_name = publ['category_name']

        # Шаг 2: Для каждой публикации получаем список её показателей
        resp_ds = requests.get(f'{BASE_URL}/datasets?publicationId={publ_id}')
        datasets = resp_ds.json()

        for ds in datasets:
            ds_name = ds.get('full_name') or ds.get('name')
            if not ds_name:
                continue

            # Шаг 3: Проверяем, содержатся ли в названии показателя ключевые слова
            for keyword in keywords_list:
                if keyword.lower() in ds_name.lower():
                    print(f'Найдено совпадение по ключевому слову "{keyword}":')
                    print(f'  Публикация: {publ_name} (ID: {publ_id})')
                    print(f'  Показатель: {ds_name} (ID: {ds['id']})')
                    print(f'  Тип: {ds.get('type')}')
                    print('-' * 40)

                    found_indicators.append({
                        'name': ds_name,
                        'publicationId': publ_id,
                        'datasetId': ds['id'],
                        'type': ds.get('type'),
                        'keyword': keyword
                    })
                    # Прерываем внутренний цикл, чтобы не дублировать один показатель
                    break

    return found_indicators

# Запускаем поиск
if __name__ == '__main__':
    keywords = ['Ключевая ставка', 'Индекс потребительских цен', 'ИПЦ', 'key rate', 'CPI']
    results = find_indicator_ids(keywords)

    if results:
        print(f'\n*** Найдено {len(results)} подходящих показателей ***')
        for item in results:
            print(f'По запросу "{item["keyword"]}": publicationId={item["publicationId"]}, datasetId={item["datasetId"]}, Название={item["name"]}')
    else:
        print('\nПодходящих показателей не найдено. Попробуйте другие ключевые слова.')