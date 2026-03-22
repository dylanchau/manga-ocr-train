from manga_ocr import MangaOcr

def main():
    mocr = MangaOcr()
    text = mocr('./data/raw/7.jpg')
    print(text)


if __name__ == "__main__":
    main()
