from Bio import Entrez
import numpy as np
import ftplib
import dq
import gzip
import xml.etree.ElementTree as ET
import pandas as pd
import pickle
import os
import time


class TfIdf:
    def __init__(self, total_pubmed_docs):
        self.doc_frequency_cache = {}
        self.total_pubmed_docs = total_pubmed_docs

    def get_doc_frequency(self, token):
        token = token.lower()
        if token in self.doc_frequency_cache:
            return self.doc_frequency_cache[token]
        Entrez.email = 'shunit.agmon@gmail.com'
        handle = Entrez.egquery(dbfrom='pubmed', term=token, retmode='xml')
        record = Entrez.read(handle)
        handle.close()
        count = int([res for res in record['eGQueryResult'] if res['DbName'] == 'pubmed'][0]['Count'])
        doc_frequency_cache[token] = count
        return count

    def tfidf(self, token, doc_tokens):
        tf = doc_tokens.count(token)/len(doc_tokens)
        doc_freq = self.get_doc_frequency(token)
        if doc_freq == 0:
            return 0
        idf = np.log(self.total_pubmed_docs/doc_freq)
        return tf*idf


def get_abstracts_by_pmids(list_of_pmids):
    ids = ",".join(map(str, list_of_pmids))
    Entrez.email = 'shunit.agmon@gmail.com'
    handle = Entrez.efetch(db='pubmed', id=ids, retmode='xml')
    record = Entrez.read(handle)
    handle.close()
    return record


def search(term, retmax, date_range_tuple=None, only_clinical_trials=True):
    Entrez.email = 'shunit.agmon@gmail.com'
    date_query = ''
    if date_range_tuple:
        start, end = date_range_tuple
        date_query = '(' + start + '[PDAT]:' + end + '[PDAT])'
    clinical_query = ''
    if only_clinical_trials:
        clinical_query = 'Clinical Trial[ptyp]'
    lang_query = "English[Language]"
    query = " AND ".join([q for q in [date_query, clinical_query, term, lang_query] if q])
    handle = Entrez.esearch(db='pubmed',
                            sort='pub+date',
                            retmode='xml',
                            term=query,
                            retmax=retmax)
    results = Entrez.read(handle)
    return results


def search_in_batches(query, retmax=200):
    Entrez.email = 'shunit.agmon@gmail.com'
    handle = Entrez.esearch(db='pubmed',
                            sort='pub+date',
                            retmode='xml',
                            term=query,
                            usehistory='y')
    results = Entrez.read(handle)
    count = int(results['Count'])
    webenv = results['WebEnv']
    print("Starting fetch generator with count={}, retmax={}".format(count, retmax))
    for retstart in range(0, count, retmax):
        handle = Entrez.efetch(db='pubmed', WebEnv=webenv, term=query, retstart=retstart, retmax=retmax, retmode='xml')
        yield handle


def count_abstracts_in_daterange(start_date, end_date):
    Entrez.email = 'shunit.agmon@gmail.com'
    res = Entrez.egquery(dbfrom='pubmed', term='(' + start_date + '[PDAT]:' + end_date + '[PDAT])', retmode='xml')
    record = Entrez.read(res)
    count = int([res for res in record['eGQueryResult'] if res['DbName'] == 'pubmed'][0]['Count'])
    return count


def get_date_range_from_pmids(pmids):
    rec = get_abstracts_by_pmids(pmids)
    years = [int(date['Year']) for date in dq.query('..PubDate', rec) if 'Year' in date]
    return min(years), max(years)


def get_pmids_in_daterange(start_date, end_date):
    """
    :param start_date: example '2016/01/01'
    :param end_date: example '2017/01/01'
    :return:
    """
    id_results = []
    query = '(' + start_date + '[PDAT]:' + end_date + '[PDAT])'

    res = search(query)
    if len(res['IdList']) > 0:
        id_results += res['IdList']
    print("Found {} papers between {} and {}.".format(len(id_results), start_date, end_date))
    return id_results


def get_ncts_from_pmids(list_of_pmids, output_file):
    # with open(output_file, "w", encoding='utf-8') as out:
    #     out.write("pmid\tncts\n")
    res = []
    for num_request, pmid in enumerate(list_of_pmids):
        record = get_abstracts_by_pmids([pmid])
        if num_request % 3 == 2:
            with open(output_file, "a", encoding='utf-8') as out:
                for pmid1, ncts1 in res:
                    out.write("{}\t{}\n".format(pmid1, ncts1))
                res = []
            time.sleep(1)
        nct = dq.query("..AccessionNumberList", record)
        print("{},{}".format(pmid, nct))
        res.append((pmid, nct))
    if len(res) > 0:
        with open(output_file, "a", encoding='utf-8') as out:
            for pmid, ncts in res:
                out.write("{}\t{}\n".format(pmid, ncts))


if __name__ == "__main__":
    f = open("pubmed_papers_with_clinical_trial.txt", "r")
    list_of_pmids = [pmid.strip() for pmid in f.read().split() if pmid.strip() != ""]
    get_ncts_from_pmids(list_of_pmids, output_file="pmids_with_ncts_1.tsv")