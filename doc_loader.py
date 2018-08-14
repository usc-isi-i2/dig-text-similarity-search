import json


def load_docs(filename: str, start: int = 0, end: int = float("inf")) -> list:
    cdr_docs = list()
    loaded_doc_ids = set()

    # Report indexing
    name = filename.split("/")[-1]
    if start == 0 and end == float("inf"):
        print("  Loading all documents in {}...".format(name))
    else:
        if end == float("inf"):
            ending = "end"
        else:
            ending = end
        print("  Loading documents {} through {} from {}...".format(start, ending, name))

    skipped = 0
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            if start <= i < end:
                raw_doc = json.loads(line)

                # Ensure document is unique and has content
                doc_id = raw_doc["doc_id"]
                if doc_id in loaded_doc_ids:
                    skipped += 1
                else:
                    content = raw_doc["lexisnexis"]["doc_description"]
                    if content != "" and content != "DELETED_STORY":
                        loaded_doc_ids.add(doc_id)
                        cdr_docs.append(raw_doc)
                    else:
                        skipped += 1

    print("  {} docs loaded".format(len(cdr_docs)))
    if skipped > 0:
        print("  Note: {} non-unique/empty docs skipped".format(skipped))

    return cdr_docs
