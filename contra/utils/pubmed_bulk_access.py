import numpy as np
import ftplib
import dq
import gzip
import xml.etree.ElementTree as ET
import pandas as pd
import pickle
import os
import time

PUBMED_FILES = 1015


# Pubmed XML utilities
def get_first_element(xml_object, xpath):
    results = xml_object.findall(xpath)
    if len(results) == 0 or results[0].text is None:
        return ""
    return results[0].text.replace(",", " ")


def get_values_list(xml_object, xpath):
    results = xml_object.findall(xpath)
    if len(results) == 0:
        return []
    return results[0].getchildren()


def extract_pubmed_date_from_xml(xml_object):
    date_objs = xml_object.findall(".//PubMedPubDate")
    if len(date_objs) == 0:
        return ""
    chosen_date = None
    if len(date_objs) == 1:
        chosen_date = date_objs[0]
    if len(date_objs) > 1:
        for date in date_objs:
            if 'PubStatus' in date.attrib.keys() and date.attrib['PubStatus'] == 'pubmed':
                chosen_date = date
        if not chosen_date:
            chosen_date = date_objs[0]
    # Create a date string
    year = get_first_element(chosen_date, 'Year')
    month = get_first_element(chosen_date, 'Month')
    day = get_first_element(chosen_date, 'Day')
    return "{}/{}/{}".format(year, month, day)


def sanitize_abstract_text_for_csv(string, forbidden_chars=["\r", "\n", ",", ";"]):
    s = string
    if s is None:
        return ""
    for c in forbidden_chars:
        s = s.replace(c, " ")
    return s


class BulkPubmedAccess:

    def __init__(self, folder, xmlgz_fname_pattern="pubmed20n{:04}.xml.gz"):
        self.gz_folder = folder
        self.fname_pattern = xmlgz_fname_pattern

    def download_pubmed(self):
        host = 'ftp.ncbi.nlm.nih.gov'
        ftp = ftplib.FTP(host, passwd="shunit.agmon@gmail.com")
        ftp.login()
        ftp.cwd("pubmed/baseline/")
        for i in range(1, PUBMED_FILES):
            fname = self.fname_pattern.format(i)
            print("downloading file {}".format(fname))
            ftp.retrbinary('RETR {}'.format(fname), open(os.path.join(self.gz_folder, fname), "wb").write)

    def shuffle_df_and_split_to_buckets(self, df, n=50):
        shuffled = df.sample(frac=1)
        chunk_size = int(len(df) / n)
        for i in range(n):
            print("writing chunk {}".format(i))
            chunk = shuffled.iloc[range(i * chunk_size, min(len(shuffled), (i + 1) * chunk_size))]
            out = os.path.join(self.gz_folder, 'pubmed_v2_shard_{}.csv'.format(i))
            write_header = not os.path.exists(out)
            chunk.to_csv(os.path.join(self.gz_folder, 'pubmed_v2_shard_{}.csv'.format(i)), mode='a', header=write_header)

    def parse_pubmed_xml_to_dataframe(self, file_index):
        data = {}
        fname = os.path.join(self.gz_folder, self.fname_pattern.format(file_index))
        print("working on file: {}".format(fname))
        with gzip.open(fname, "rb") as g:
            content = g.read()
            root = ET.fromstring(content)
        for pubmed_article in root.getchildren():
            pmid = pd.to_numeric(get_first_element(pubmed_article, "MedlineCitation/PMID"))
            abstract_texts = get_values_list(pubmed_article, "MedlineCitation/Article/Abstract")
            if not abstract_texts:  # Skip empty abstracts.
                continue
            if len(abstract_texts) > 1:
                print("fname:{} pmid:{} has more than one abstracttexts".format(fname, pmid))
            abstract_text_as_string = ";".join([sanitize_abstract_text_for_csv(item.text) for item in abstract_texts
                                                if item is not None])
            abstract_labels = ";".join([item.attrib.get('Label', "") for item in abstract_texts
                                        if item is not None])
            title = get_first_element(pubmed_article, "MedlineCitation/Article/ArticleTitle")
            pub_types = ";".join([sanitize_abstract_text_for_csv(item.text) for item in
                                  pubmed_article.findall(".//PublicationType")
                                  if item is not None])
            date = pd.to_datetime(extract_pubmed_date_from_xml(pubmed_article))
            keywords = pubmed_article.findall(".//Keyword")
            kw_list = []
            for item in keywords:
                if item is None or item.text is None:
                    continue
                words = item.text.split(",")
                if len(words) > 1:
                    kw_list.extend(words)
                elif len(words[0]) > 0:
                    kw_list.append(words[0])
            kw_as_text = ";".join([sanitize_abstract_text_for_csv(word) for word in kw_list])
            desc = pubmed_article.findall(".//MeshHeading/DescriptorName")
            mesh = ";".join([sanitize_abstract_text_for_csv(d.text) for d in desc])
            data[pmid] = {'title': title, 'abstract': abstract_text_as_string, 'labels': abstract_labels,
                          'pub_types': pub_types, 'date': date, 'file': fname,
                          'mesh_headings': mesh, 'keywords': kw_as_text}
        df = pd.DataFrame.from_dict(data, orient='index')
        return df


if __name__ == "__main__":
    PUBMED_FOLDER = 'pubmed_2019'
    bpa = BulkPubmedAccess(PUBMED_FOLDER, 'pubmed20n{:04}.xml.gz')
    bpa.download_pubmed()
    for i in range(1, PUBMED_FILES+1):
        bpa.parse_pubmed_xml_to_dataframe()
