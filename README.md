# Статический анализ сайта ОмГТУ
## Установка

Для работы программы понадобятся следующие Python 3.7 или выше, и следующие модули:

```
pip install requests
pip install bs4
pip install pandas
pip install matplotlib
pip install pymorphy2
pip install nltk
pip install sklearn
pip install openpyxl
pip install lxml
```

in python console:
```
import nltk
nltk.download()

column: identifier 
cell: book 
download
```

## Запуск

Для запуска программы запустить файл main.py

## Функционал

- Загрузка и парсинг страниц новостей ОмГТУ
- Частотный анализ слов
- Анализ tf-idf релевантности заголовков
- Статический анализ статей по годам
- Матрица схожести статей определенная косинусным коэффицентом


Приложение имеет графический интерфейс. При нажатие на кнопку, следует подождать выполнения программы, это может занять некоторое время. При загрузке/обновление корпуса, может достигать часа (если это делается впервые)

## Выходные данные

После загрузки корпуса в основной папке появится файл news_omstu.xlsx, который содержит информацию (ссылка, заголовок, дата публикации, автор, количество слов) о всех статьях из корпуса. В папке news_pages находятся полные страницы статей, а в папке corpus тексты статей с этих страниц
