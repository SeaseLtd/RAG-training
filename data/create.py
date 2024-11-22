import regex as re


def clean_text(text):
    if id == 2799:
        print()
    tag_regex = re.compile('<.*?>')
    text_without_html = re.sub(tag_regex, '', text)
    test_without_whitespaces = re.sub("\s\s+", " ", text_without_html)
    test_without_display = re.sub("\\\displaystyle .*}", " ", test_without_whitespaces)
    return re.sub("\\\\", " ", test_without_display)


with open("./documents_10k.tsv") as data:
    id = 0
    doc = "["
    for row in data:
        cleaned_text = clean_text(row[:-1])
        doc = doc + "{\"id\": " + str(id) + ", " + "\"body\": " + "\"" + cleaned_text + "\"}, "
        id = id + 1
    doc = doc[:-2] + "]"

    with open("./solr_documents.json", "w") as doc_file:
        print(doc, file=doc_file)
