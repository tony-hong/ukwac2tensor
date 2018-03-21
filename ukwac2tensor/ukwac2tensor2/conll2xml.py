# conll2xml.py
__author__ = "xhong@coli.uni-sb.com"


import os
import sys
import re
import gzip
import operator
import argparse
import cPickle
from collections import OrderedDict as od
from itertools import izip_longest
import xml.etree.cElementTree as et
import xml.dom.minidom as mdom

import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn


# Configuration of environment
SRC_DIR = os.path.dirname((os.path.dirname(os.path.abspath(__file__))))
sys.path.append(SRC_DIR)

wnl = WordNetLemmatizer()



class CoNLLTools:
    """
    Class that represents a collection of static methods required to parse
    ukWaC corpus.
    <Class implementation in order to wrap a mess of various string processing
    functions into a single class>
    """
    @staticmethod
    def is_ascii(s):
        return all(ord(c) < 128 for c in s)


    @staticmethod
    def validate(token):
        '''validate whether the token is a valid word
        '''
        if re.search(r"[^a-zA-Z']+", token):
            return False
        return True


    @staticmethod
    def penn2wn(pos):
        '''Converts P.O.S. tag from Penn TreeBank style to WordNet style
        '''
        first = pos[0]
        if first == 'J':
            return wn.ADJ
        elif first == 'N':
            return wn.NOUN
        elif first == 'R':
            return wn.ADV
        elif first == 'V':
            return wn.VERB
        return wn.NOUN


    @staticmethod
    def gzip_reader(fname):
        """
        Read a .gz archive and return its contents as a string.
        If the file specified is not an archive it attempt to read it
        as a general file.

        Args:
            *fname* (str) -- file path

        Returns:
            *fdata* (str) -- file contents

        """
        try:
            with gzip.open(fname, 'r') as f:
                fdata = f.read()
        except (OSError, IOError):
            with open(fname, 'r') as f:
                fdata = f.read()
        # return clean_bad_chars(fdata)
        return fdata


    @staticmethod
    def gzip_xml(fname):
        """
        Read and compress specified file, remove original file.

        Args:
            *fname* (str) -- file name

        """
        with open(fname, 'r') as f:
            fdata = f.read()
        with gzip.open(fname + '.gz', 'w') as gf:
            gf.write(fdata)
            os.remove(fname)
        print fname + '.gz successfully archived'


    @staticmethod
    def append_to_xml(fname, root):
        """
        Create xml file header, prettify xml structure and write xml
        representation of the sentences using ``\\r\\n`` as a separator.

        <IMPORTANT! Take into account that output file shall contain sentences
        separated by ``\\r\\n``. Head searching will not work otherwise. This
        is an ugly hack for ``<text id></text>`` tags to contain correct
        sentences.>

        Args:
            | *fname* (str) -- file name to write the data to
            | *root* (xml.etree object) -- xml.etree root object

        """
        rxml_header = re.compile(r'<\?xml version="1.0" \?>')
        ugly = et.tostring(root, 'utf-8', method='xml')
        parsed_xml = mdom.parseString(ugly)
        nice_xml = parsed_xml.toprettyxml(indent=" " * 3)
        even_more_nice_xml = rxml_header.sub('', nice_xml)
        with open(fname, 'a') as f:
            f.write(even_more_nice_xml)
            f.write('\r\n')  # delimiter required by head_searcher


    @staticmethod
    def get_dependants(sent, prd_id):
        """
        Retrieve roles for a given governor.

        Args:
            | *sent* (list) -- a list of word, POS-tag, word index and role
            |  tuples:
                ``[('first/JJ/3', ('I-A1',)), ('album/NN/4', ('E-A1',))]``
            | *prd_id* (int) -- index to access correct ukwac column

        Returns:
            | *role_bag* (list) -- a list of dicts where dep role is key and
               words, POS-tags, word indeces are values:
                ``[{'V': 'justify/VB/20'},
                  {'A1': 'a/DT/21 full/JJ/22 enquiry/NN/23'}]``

        """
        # rarg = re.compile(r'(?![O])[A-Z0-9\-]+')
        # in case of bad parsing
        try:
            # dep_roles = [(rarg.match(d[1][prd_id]).group(), d[0]) for d in sent
            #              if rarg.match(d[1][prd_id])]
            dep_roles = [(d[1][prd_id], d[0]) for d in sent
                         if d[1][prd_id]]
        except:
            dep_roles = [('', 0)]
        role_bag = []
        role_chunk = ()
        for i in iter(dep_roles):
            if re.match(r'\(.*\*', i[0]):
                role = re.findall(r"\((.*)\*", i[0])[0]
                role_chunk = (i[1],)
                continue
            elif i[0] == '*':
                role_chunk += (i[1],)
                continue
            elif i[0] == '*)':
                role_chunk += (i[1],)
                role_bag.append({role : ' '.join(role_chunk)})
                continue
            else:
                role_bag.append({role : i[1]})
                continue
        # print role_bag
        return role_bag


    def extract_dataframe(self, conll_data, toLower=False):
        """
        Extract columns from conll files, create an ordered dict of
        ("word", "lemma") pairs and construct sentences for SENNA input.

        Args:
            *data* (str) -- file contents

        Returns:
            | *dataframe* (np.array) -- a dataframe of conll data. 
                Each line is a sentence in CoNLL format:
                    Word    POS    Parse    Predicate   Frameset 1, 2, ...

        """
        data_lines = [cl for cl in conll_data.split('\n')]
        # print (data_lines[8])
        # test wsj
        # assert (data_lines[8].strip() == "")
        
        line = np.array([])
        seentence = np.array([])
        dataframe = []
        first = True
        for dl in data_lines:
            if dl.strip() != "":
                if toLower:
                    dl = dl.lower()
                line = np.array(dl.split())
                # if len(line) <= self.COL_2_PROPS: 
                #     continue
                if first:
                    seentence = line
                    first = False
                else:
                    seentence = np.vstack((seentence, line))
            else:
                if len(seentence) != 0:
                    dataframe.append(seentence)
                seentence = np.array([])
                line = np.array([])
                first = True
        # exclude those without Props 
        dataframe = np.array([t for t in dataframe])
        # print (np.max([len(l[0]) for l in dataframe]))
        # print (dataframe[0])

        return dataframe


    def df2malt(self, df):
        new_data = ''
        imgId = 0
        sent_cnt = 0
        isFirst = True

        for line in df:
            # line = sentence[0]
            # print line
            assert line[2] == '_'
            idx = line[0]
            token = line[1]
            pos = line[4].lower()
            dep = line[6]
            syntag = line[7].lower()
            if not self.is_ascii(token):
                token = '<UNKNOWN>'
            lower_token = token.lower()
            # lemma = wnl.lemmatize(lower_token, self.penn2wn(pos))
            line[2] = '_'

            new_list = []
            new_list.append(token)
            new_list.append('_')
            new_list.append(pos)
            new_list.append(idx)
            new_list.append(dep)
            new_list.append(syntag)
            new_line = '\t'.join(new_list)

            new_data += new_line + '\n'
        
        # print new_sent
        # new_data.append(new_sent)
        # DEBUG block
        # if len(new_data) > 50:
        #     break

        return new_data


    def extract_info(self, parse_dfs, srl_dfs, result_fname):
        """
        Extract columns from conll files, create an ordered dict of
        ("word", "lemma") pairs and construct sentences for SENNA input.

        Args:
            *data* (str) -- file contents

        Returns:
            | *norm_sents* (str) -- sentences reconstructed from conll and
               separated by "\`\`"
            | *dict_lemmas* (dict) -- a dict of all words and their lemmas
            | *text_ids* (OrderedDict) -- an ordered dict of url string and a
               number of sentences, that belong to this url. Used in order to
               provide source reference for extracted sentences
            | *include_malt_data* (dict) -- dict of sentence number and malt parse
               data extracted from conll file

        """
        # print (dataframe[0])
        result = []
        unique_pairs = {}
        prd_cnt = 0
        sent_cnt = 0
        text_idx = 0
        for idx, srl in enumerate(srl_dfs): 
            # print srl
            
            # Read parse 
            parse = parse_dfs[idx]
            # print parse.shape

           # Process text id
            if srl.ndim == 1:
                text_idx = parse[1]
                # print text_idx
                text_e = et.Element("text")
                text_e.set("id", text_idx)
                self.append_to_xml(result_fname, text_e)
                continue

            if len(parse) != len(srl):
                continue

            # skip clause without PRD
            num_props = len(srl[0]) - self.COL_2_PROPS
            if num_props == 0:
                continue
            
            idxs = parse[:,0]
            tokens = parse[:,1]
            pos_tags = parse[:,4]
            deps = parse[:,6]
            syn_tags = parse[:,7]

            # Read srls
            predicates = srl[:,0]

            # Process data
            lemmas = []
            for i, t in enumerate(tokens, start=0):
                first = pos_tags[i][0].upper()
                lemma = wnl.lemmatize(t, CoNLLTools.penn2wn(first))
                lemmas.append(lemma)
            mask = predicates != '-'
            prd_list = predicates[mask]
            assert (num_props == len(prd_list))
            num_tokens = len(tokens)
            # print lemmas
            # print predicates
            # print num_props, len(prd_list)

            # construct text
            malt_text = self.df2malt(parse)
            sent_tuples = [(''.join([lemmas[i], "/", pos_tags[i], "/", idxs[i]]), srl[i, self.COL_2_PROPS:])
                    for i in range(num_tokens)]
            lemmas_sent = ' '.join([i[0].rsplit('/', 2)[0] for i in sent_tuples]).lower()
            # raw_sent = ' '.join(tokens)
            # print sent_tuples
            # print raw_sent

            root_e = et.Element("s")
            malt_e = et.SubElement(root_e, "malt")
            lemmsent_e = et.SubElement(root_e, "lemsent")
            # rawsent_e = et.SubElement(root_e, "rawsent")

            malt_e.text = malt_text
            lemmsent_e.text = lemmas_sent

            # equal

            # TODO            
            # retrieving and creating governor nodes
            for prd_col, prd in enumerate(prd_list, start=self.COL_2_PROPS):
                props = srl[:, prd_col]
                prd_token = prd[1:]
                prd_idx = np.where(tokens==prd_token)[0][0]

                prd_lemma = lemmas[prd_idx]

                pred_e = et.SubElement(root_e, "predicate")
                gov_e = et.SubElement(pred_e, "governor")
                deps_e = et.SubElement(pred_e, "dependencies")

                pred_e.set("id", text_idx)

                # print prd_lemma, '/', pos_tags[prd_idx], '/', prd_idx

                gov_e.text = ''.join([prd_lemma, '/', pos_tags[prd_idx], '/', str(prd_idx)])

                dep_dic = self.get_dependants(sent_tuples, prd_col - self.COL_2_PROPS)

                for d in dep_dic:
                    # print d
                    dep_e = et.SubElement(deps_e, "dep")
                    dep_e.set("type", d.keys()[0])
                    dep_e.text = d.values()[0]
                prd_cnt += 1

            self.append_to_xml(result_fname, root_e)
            sent_cnt += 1

        self.gzip_xml(result_fname)

                # print (prd, srl[:, prd_id])
            # print (result)

        print ("Stats:")
        print ("No. of predicates: ", prd_cnt)
        print ("No. of sentences: ", sent_cnt)

        # dict_lemmas = dict(pairs)
        # sents = ' '.join([w[0] for w in pairs])
        # norm_sents = UkwacTools.insert_sent_delims(UkwacTools.reduce_contr(sents))
        # return norm_sents, dict_lemmas, UkwacTools.include_malt(lines)


    def __init__(self):
        # number of columns until props
        self.COL_2_PROPS = 1
        self.ROLE_LIST = ['A0', 'A1', 'AM-LOC', 'AM-TMP', 'AM-MNR']
        self.ROLE_FILTER = [
            'AM-DIS', 'AM-MOD', 'AM-NEG', 
            'R-A0', 'R-A1', 'R-A2', 
            'R-AM-TMP', 'R-AM-LOC', 'R-AM-CAU', 'R-AM-MNR', 'R-AM-EXT',
            'C-A0', 'C-A1', 'C-V', ]
        # self.ROLE_FILTER = ['AM-MOD', 'AM-NEG', 'AM-DIS']


if __name__ == '__main__':
    prs = argparse.ArgumentParser(description="""
    This script converts conll format data from dependency parser and semantic role 
    labeller to xml format converted data.

    Parsed and SRL files must pair up correctly. 
    """)
    # prs.add_argument('-d', '--dir',
    #                  help='Specify directory where conll files are located.',
    #                  required=False)
    prs.add_argument('-p', '--parse',
                     help='Specify directory whereconll parsed file are located. ',
                     required=True)
    prs.add_argument('-s', '--srl',
                     help='Specify directory where conll SRL file are located. ', 
                     required=True)
    prs.add_argument('-o', '--out', default=os.getcwd(),
                     help='Specify output directory. If not specified, current '
                          'dir is used.',
                     required=False)
    args = prs.parse_args()

    parse_fns = [os.path.join(args.parse, name) for name in os.listdir(args.parse) if re.search(r'conll', name)]
    srl_fns = [os.path.join(args.srl, name) for name in os.listdir(args.srl) if re.search(r'conll', name)]
    parse_fns.sort()
    srl_fns.sort()
    
    ct = CoNLLTools()

    for i, parse_fn in enumerate(parse_fns):
        srl_fn = srl_fns[i]
        print (parse_fn, srl_fn)
        output_name = re.findall(r'split/(.*)_parsed', parse_fn)[-1]
        output_name += '_converted.xml'
        output_fn = os.path.join(args.out, output_name)
        # print (output_name)
        # print (output_path)

        print 'reading ...'
        parse_data = ct.gzip_reader(parse_fn)
        srl_data = ct.gzip_reader(srl_fn)
        print 'extracting dataframe...'
        parse_dfs = ct.extract_dataframe(parse_data, toLower=True)
        srl_dfs = ct.extract_dataframe(srl_data)

        print 'extracting information...'
        ct.extract_info(parse_dfs, srl_dfs, output_fn)

        # debug
        # print srl_df[0].shape
        # print srl_df[0]
        # print parse_df[0].shape
        # print parse_df[0]
        # print ct.df2malt(parse_df[1])
        
        # print 'creating dat...'
        # result_fname = os.path.join(os.getcwd(), plain_xml_fname)
        # with open(result_fname, 'w+b') as of:
        #     of.write(plain_output)
        # result_fname = os.path.join(os.getcwd(), unique_xml_fname)
        # with open(result_fname, 'w+b') as of:
        #     of.write(unique_pairs)
