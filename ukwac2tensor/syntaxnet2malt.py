# syntaxnet2malt.py
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
import xml.etree.cElementTree as ET
import xml.dom.minidom as mdom

import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn


# Configuration of environment
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(SRC_DIR)

wnl = WordNetLemmatizer()


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


class CoNLLTools:
    """
    Class that represents a collection of static methods required to parse
    ukWaC corpus.
    <Class implementation in order to wrap a mess of various string processing
    functions into a single class>
    """

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


    def extract_dataframe(self, conll_data):
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
                line = np.array(dl.split('\t'))
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

        # TODO***
        # print (np.max([len(l[0]) for l in dataframe]))
        # print (seentence[0])

        return dataframe



    def extract_info(self, dataframe):
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
        for sentence in dataframe:
            # print sentence
            num_props = len(sentence[0]) - self.COL_2_PROPS
            
            tokens = sentence[:,0]
            POStags = sentence[:,1]
            parses = sentence[:,2]
            predicates = sentence[:,3]
            
            # lemmas = []
            # for i, t in enumerate(tokens, start=0):
            #     pos = POStags[i]
            #     lemma = wnl.lemmatize(t, CoNLLTools.penn2wn(pos[0]))
            #     lemmas.append(lemma)

            mask = predicates != '-'
            prd_list = predicates[mask]
            # print num_props, len(prd_list)
            assert (num_props == len(prd_list))
            num_tokens = len(tokens)

            ## raw sentence
            # print (' '.join(tokens))
            for prd_id, prd in enumerate(prd_list, start=self.COL_2_PROPS):
                props = sentence[:, prd_id]
                prd_lemma = wnl.lemmatize(prd.lower(), wn.VERB)
                role_word_dict = {}
                role_word_dict['V'] = prd_lemma
                for prop_id, prop in enumerate(props, start=0):
                    # find start of role
                    if prop[0] == '(' and prop[1] != 'V':
                        role = re.findall(r"\((.*)\*", prop)[0]
                        if role not in self.ROLE_LIST:
                            continue
                        # if role in self.ROLE_FILTER:
                        #     continue
                        if prop[-1] == ')':
                            pos = POStags[prop_id]
                            token = tokens[prop_id]
                            head = wnl.lemmatize(token.lower(), self.penn2wn(pos))
                        elif role in ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']:
                            for parse in parses[prop_id:]:
                                head = re.findall(r"NPB\~([^(\*)]*)[(\*]", parse)
                                if head:
                                    head = wnl.lemmatize(head[0].lower(), wn.NOUN)
                                    break
                                head = re.findall(r"VP-A\~([^(\*)]*)[(\*]", parse)
                                if head:
                                    head = wnl.lemmatize(head[0].lower(), wn.VERB)
                                    break
                        else:
                            role_end = np.argmax(props[prop_id:] == '*)') + 1
                            this_parses = parses[prop_id:]
                            span = this_parses[:role_end]
                            # print prop_id, role_end
                            # print span
                            for parse in span: 
                                head = re.findall(r"ADVP\~([^(\*)]*)[(\*]", parse)
                                if head:
                                    head = wnl.lemmatize(head[0].lower(), wn.ADV)
                                    break
                                head = re.findall(r"ADJP\~([^(\*)]*)[(\*]", parse)
                                if head:
                                    head = wnl.lemmatize(head[0].lower(), wn.ADJ)
                                    break
                                head = re.findall(r"NPB\~([^(\*)]*)[(\*]", parse)
                                if head:
                                    head = wnl.lemmatize(head[0].lower(), wn.NOUN)
                                    break
                        if head:
                            role_word_dict[role] = head
                            # print (prd, head, role)
                if len(role_word_dict) > 1:
                    result.append(str(role_word_dict))
                    role_word_dict = {}

                # print (prd, sentence[:, prd_id])
            # print (result)

        plain_output = "\n".join(result)

        return plain_output

        # dict_lemmas = dict(pairs)
        # sents = ' '.join([w[0] for w in pairs])
        # norm_sents = UkwacTools.insert_sent_delims(UkwacTools.reduce_contr(sents))
        # return norm_sents, dict_lemmas, UkwacTools.include_malt(lines)



    def __init__(self):
        # number of columns until props
        self.COL_2_PROPS = 4
        self.ROLE_LIST = [
            'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 
            'AM-ADV', 'AM-CAU', 'AM-DIR', 'AM-DIS', 'AM-EXT', 
            'AM-LOC', 'AM-MNR', 'AM-MOD', 'AM-NEG', 'AM-PNC', 
            'AM-PRD', 'AM-REC', 'AM-TMP']
        self.ROLE_FILTER = [
            'AM-DIS', 'AM-MOD', 'AM-NEG', 
            'R-A0', 'R-A1', 'R-A2', 
            'R-AM-TMP', 'R-AM-LOC', 'R-AM-CAU', 'R-AM-MNR', 'R-AM-EXT',
            'C-A0', 'C-A1', 'C-V', ]
        # self.ROLE_FILTER = ['AM-MOD', 'AM-NEG', 'AM-DIS']


def syntaxnet2malt(file):
    # setname = 'SP_SRL.' + testset.split('.')[1]
    # conll_fname_path = '.'.join([os.path.basename(testset),
                            # 'dat'])
    ct = CoNLLTools()

    # print 'reading %s...' % filename
    # fdata = ct.gzip_reader(conll_fname_path)


    fdata = file.read()

    print 'extracting dataframe...'
    df = ct.extract_dataframe(fdata)
    # print len(df)
    # sent_length = df[0].shape[0]
    # num_column = df[0].shape[1]

    print 'constructing new sentences ...'
    new_data = []
    imgId = 0
    sent_cnt = 0
    isFirst = True
    for sentence in df:
        # sentence = df[0]
        new_sent = ''
        if sentence.ndim == 1:
            assert sentence[2] == '_'
            imgId = sentence[1]
            sent_cnt += 1
            if not isFirst:
                new_sent += '</text>\n'
            else: 
                isFirst = False
            new_sent += '<text id="' + imgId + '">\n'
        else:
            new_sent += '<s>\n'
            for line in sentence:
                # line = sentence[0]
                # print line
                assert line[2] == '_'
                idx = line[0]
                token = line[1]
                pos = line[4]
                dep = line[6]
                syntag = line[7].upper()
                if not is_ascii(token):
                    token = '<UNKNOWN>'
                lower_token = token.lower()
                lemma = wnl.lemmatize(lower_token, ct.penn2wn(pos))
                line[2] = lemma

                new_list = []
                new_list.append(token)
                new_list.append(lemma)
                new_list.append(pos)
                new_list.append(idx)
                new_list.append(dep)
                new_list.append(syntag)
                new_line = '\t'.join(new_list)

                new_sent += new_line + '\n'
            new_sent += '</s>\n'
        
        # print new_sent
        new_data.append(new_sent)
        # DEBUG block
        # if len(new_data) > 50:
        #     break
    new_data.append('</text>\n')

    # output_str = '<text>\n'
    output_str = ''
    for sent in new_data:
        output_str += sent
        
    # print output_str
    return output_str

    # plain_output, unique_pairs = ct.extract_info_SRL(df)

    # print 'creating dat...'
    # result_fname = os.path.join(os.getcwd(), plain_xml_fname)
    # with open(result_fname, 'w+b') as of:
    #     of.write(plain_output)
    # result_fname = os.path.join(os.getcwd(), unique_xml_fname)
    # with open(result_fname, 'w+b') as of:
    #     of.write(unique_pairs)

    # sents, word_lemma, id_index, malt_dic = extract_info(df)
    # build_xml(word_lemma, id_index, result_fname, malt_dic, args.pverbs)



if __name__ == '__main__':
    prs = argparse.ArgumentParser(description="""
    This script converts SyntaxNet parsed format to MALT parsed format.
    """)
    prs.add_argument('-d', '--dir',
                     help='Specify directory where conll files are located.',
                     required=False)
    prs.add_argument('-f', '--file',
                     help='Specify conll file to process. ',
                     required=False)
    prs.add_argument('-o', '--out',
                     help='Specify output directory. If not specified current '
                          'dir is used.',
                     required=False)
    args = prs.parse_args()

    input_file = os.path.join(args.dir, args.file)
    if args.out:
        output_file = os.path.join(args.dir, args.out)
    else:
        output_file = input_file.strip('txt') + '_result.xml'

    # filename = args.file

    # SICK_parsed_PATH = os.path.join(SRC_DIR, "/SICK_parsed")
    # all_files = [f for _, _, f in os.walk(SICK_parsed_PATH)]
    
    with open(input_file, 'r') as f_in, \
        open(output_file, 'w') as f_out:
        result = syntaxnet2malt(f_in)
        f_out.write(result)


