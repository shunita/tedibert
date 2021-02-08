import ftplib
import gzip
import xml.etree.ElementTree as ET
import pandas as pd
import os

PUBMED_FILES_2019 = 1015
PUBMED_FILES = 1062  # 2020

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
    copyright_sign = '©'
    if s is None:
        return ''
    s = s.split(copyright_sign)[0]
    if s.startswith('[This corrects') or s.strip(' ') == 'Copyright:':
        return ''
    for c in forbidden_chars:
        s = s.replace(c, ' ')
    return s


class BulkPubmedAccess:

    def __init__(self, gz_folder, output_folder, xmlgz_fname_pattern="pubmed20n{:04}.xml.gz"):
        self.gz_folder = gz_folder
        self.output_folder = output_folder
        self.fname_pattern = xmlgz_fname_pattern

    def download_pubmed(self):
        host = 'ftp.ncbi.nlm.nih.gov'
        ftp = ftplib.FTP(host, passwd="shunit.agmon@gmail.com")
        ftp.login()
        ftp.cwd("pubmed/baseline/")
        for i in range(1, PUBMED_FILES+1):
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

    def split_df_by_years(self, df):
        if len(df) == 0:
            return
        df = df.dropna(subset=['date'], axis=0)
        df['year'] = pd.DatetimeIndex(df['date']).year
        for year in df['year'].unique():
            subset = df[df['year'] == year]
            out = os.path.join(self.output_folder, 'pubmed_{}.csv'.format(year))
            write_header = not os.path.exists(out)
            subset.to_csv(os.path.join(self.output_folder, 'pubmed_{}.csv'.format(year)), mode='a', header=write_header)
        

    def parse_pubmed_xml_to_dataframe(self, file_index):
        data = {}
        fname = os.path.join(self.gz_folder, self.fname_pattern.format(file_index))
        print("Parsing file: {}".format(fname))
        with gzip.open(fname, "rb") as g:
            content = g.read()
            root = ET.fromstring(content)
        for pubmed_article in root.getchildren():
            pmid = pd.to_numeric(get_first_element(pubmed_article, "MedlineCitation/PMID"))
            abstract_texts = get_values_list(pubmed_article, "MedlineCitation/Article/Abstract")
            if not abstract_texts:  # Skip empty abstracts.
                continue
            #if len(abstract_texts) > 1:
            #    print("fname:{} pmid:{} has more than one abstracttexts".format(fname, pmid))
            abstract_texts_sanitized = [sanitize_abstract_text_for_csv(item.text) for item in abstract_texts
                                        if item is not None]
            abstract_non_empty_indices = [i for i, item in enumerate(abstract_texts_sanitized) if item!='']
            abstract_text_as_string = ";".join([abstract_texts_sanitized[i] for i in abstract_non_empty_indices])
            if abstract_text_as_string.strip(' ;') == '':
                continue
            abstract_labels = ";".join([item.attrib.get('Label', "") for i,item in enumerate(abstract_texts)
                                        if item is not None and i in abstract_non_empty_indices])
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
            ncts = ";".join([item.text for item in get_values_list(pubmed_article, './/AccessionNumberList') 
                             if item is not None 
                             and item.text is not None 
                             and item.text.startswith("NCT")])
            data[pmid] = {'title': title, 'abstract': abstract_text_as_string, 'labels': abstract_labels,
                          'pub_types': pub_types, 'date': date, 'file': file_index,
                          'mesh_headings': mesh, 'keywords': kw_as_text, 'ncts': ncts}
        df = pd.DataFrame.from_dict(data, orient='index')
        return df

    def split_df_into_papers(self, df_path, target_dir):
        df = pd.read_csv(df_path, index_col=0)
        for i in range(len(df)):
        #for i in range(10):
            df.iloc[i:i+1].to_csv(os.path.join(target_dir, f'{df.iloc[i].name}.csv'))
        


if __name__ == "__main__":
    PUBMED_FOLDER = os.path.expanduser('~/pubmed_2020')
    OUTPUT_FOLDER = os.path.expanduser('~/pubmed_2020_by_years')
    bpa = BulkPubmedAccess(PUBMED_FOLDER, OUTPUT_FOLDER, 'pubmed21n{:04}.xml.gz')
    bpa.download_pubmed()
    for i in range(1, PUBMED_FILES+1):
        df = bpa.parse_pubmed_xml_to_dataframe(i)
        bpa.split_df_by_years(df)

    #bpa.split_df_into_papers(os.path.join(OUTPUT_FOLDER, 'pubmed_2018.csv'), os.path.join(OUTPUT_FOLDER, '2018') )
