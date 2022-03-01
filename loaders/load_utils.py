import os
import bz2


def load_dataset(folder, doc_id_to_uri, uri_to_gold_classes, domain_to_id, id_to_domain, hierarchy=None):
    data = []
    for root, dirs, files in os.walk(folder):
        for filename in files:
            if (filename == "virtualdocument.txt.bz2"):
                key = os.path.basename(root)

                if doc_id_to_uri[key] in uri_to_gold_classes:
                    txt = " ".join(
                        [str(line.decode("utf-8")).strip("\n") for line in bz2.open(os.path.join(root, filename), "r")])

                    direct_klasses = uri_to_gold_classes[doc_id_to_uri[key]]
                    undirect_classes = [k for k in direct_klasses]

                    if hierarchy is not None:
                        for klass in undirect_classes:
                            if domain_to_id[klass] in hierarchy:
                                for super_klass in hierarchy[domain_to_id[klass]]:
                                    if id_to_domain[super_klass] not in undirect_classes:
                                        undirect_classes.append(id_to_domain[super_klass])
                    # data.append([key, doc_id_to_uri[key], undirect_classes, txt])
                    data.append([undirect_classes, txt])
    return data


def load_benchmark(virtual_documents, doc_id_to_uri, uri_to_gold_classes, headers, hierarchy, uri_to_doc_id, domain_to_id, id_to_domain):
    data = []

    for root, dirs, files in os.walk(virtual_documents):
        for filename in files:
            if (filename == "virtualdocument.txt.bz2"):
                key = os.path.basename(root)
                txt = " ".join(
                    [str(line.decode("utf-8")).strip("\n") for line in bz2.open(os.path.join(root, filename), "r")])

                if doc_id_to_uri[key] not in uri_to_gold_classes:
                    print(f"{key} {doc_id_to_uri[key]} not found in gold standard ")
                    if doc_id_to_uri[key] + '/' in uri_to_gold_classes:
                        print(f"{key} {doc_id_to_uri[key]} renamed to {key} {doc_id_to_uri[key] + '/'}")
                        old_uri = doc_id_to_uri[key]
                        doc_id_to_uri[key] = old_uri + '/'
                        uri_to_doc_id[old_uri + '/'] = key
                        uri_to_doc_id.pop(old_uri + '/', None)

                direct_klasses = [headers[id] for id, flag in enumerate(uri_to_gold_classes[doc_id_to_uri[key]]) if
                                  flag]

                # direct_klasses = uri_to_gold_classes[doc_id_to_uri[key]]
                undirect_classes = [k for k in direct_klasses]

                if hierarchy is not None:
                    for klass in undirect_classes:
                        if domain_to_id[klass] in hierarchy:
                            for super_klass in hierarchy[domain_to_id[klass]]:
                                if id_to_domain[super_klass] not in undirect_classes:
                                    undirect_classes.append(id_to_domain[super_klass])

                # data.append([doc_id_to_uri[key], klasses, txt])
                data.append([undirect_classes, txt])
    return data