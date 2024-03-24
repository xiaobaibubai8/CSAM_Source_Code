import os
import pickle
import numpy as np
import torch
import re
import json
from data.tokenization_util import camel_and_snake_to_word_list, load_stopwords
from torch_geometric.data import Dataset, Data
import random
from spacy.lang.en.stop_words import STOP_WORDS
class CodeSearchNetDataSet(Dataset):
   
    def len(self):
        return len(self.cfg_dd_graphs)

    def get(self, idx):
        rand_idx = random.randint(0, len(self.cfg_dd_graphs) - 1)
        # generate pos sample and neg sample
        while rand_idx == idx:
            rand_idx = random.randint(0, len(self.cfg_dd_graphs) - 1)
        return self.cfg_dd_graphs[idx], self.descriptions[idx], self.descriptions[rand_idx], \
            self.descriptions_lens[idx], self.descriptions_lens[rand_idx]
    
  # root path is the NCFG-CS/proprecessed_date
    def __init__(self, root, split_name, transform=None, pre_transform=None):
        self.split_name = split_name
        self.stopwords = load_stopwords()
        self.replace_tokens = [
            '1',
            '2',
            '3',
            '4',
            '5',
            '6',
            '7',
            '8',
            '9',
            '0',
            '==',
            '!=',
            '<p>',
            '<p',
            '<img>',
            '<img',
            '<i>',
            '<i',
            '+=',
            '-=',
            '=-',
            '=+',
            '++',
            '--',
            '=',
            '!',
            '?',
            '？',
            ';',
            '；',
            '：',
            ':',
            '{}',
            '{',
            '}',
            '[]',
            '[',
            ']',
            '()',
            '(',
            ')',
            '+',
            '-',
            ',',
            '&&',
            '||',
            '&',
            '|',
            '*',
            '//',
            '/',
            '<<<',
            '<<',
            '>>>',
            '>>',
            '<',
            '>',
            '/**',
            '**/',
            '*/',
            '/*',
            '@link',
            '@code',
            '%=',
            '%',
            '@',
            '#',
            '《',
            '》',
            '\\',
            '\\\\',
            '"',
            '\'',
            '...',
            '.',
            '\t',
            '\r',
            '\n',
        ]
        # the length after tokenization
        self.max_desc_len = 40
        self.max_code_len = 50
        self.max_node_len = 50
        self.cfg_dd_graphs = []
        self.descriptions = []
        self.descriptions_lens = []
        # self.codes = []
        # self.codes_lens = []
        self.ast_Id = 0                                              
        current_dir = os.path.dirname(__file__)
        self.node_ast_vocab = pickle.load(open(os.path.join(current_dir, 'pkl_peng2/java_code_astNode.pkl'), 'rb'))
        self.code_cfg_vocab = pickle.load(open(os.path.join(current_dir, 'pkl_peng3/java_code_cfg.pkl'), 'rb'))
        self.code_vocab = pickle.load(open(os.path.join(current_dir, 'pkl_backup/java_code.pkl'), 'rb'))
        self.desc_vocab = pickle.load(open(os.path.join(current_dir, 'pkl_backup/java_query.pkl'), 'rb'))
        super(CodeSearchNetDataSet, self).__init__(root, transform, pre_transform)
        self.load_saved_data()
    
    def load_saved_data(self):
        print(f'从{self.processed_paths[0]}load data...')
        # if data is proprecessd then load directly
        dataset_path = os.path.join(self.processed_paths[0])
        if os.path.exists(dataset_path):
            save_dict = torch.load(dataset_path)
            self.cfg_dd_graphs = save_dict['cfg_dd_graphs']
            self.descriptions = save_dict['descriptions']
            self.descriptions_lens = save_dict['descriptions_lens']
        else:
            raise FileNotFoundError(f'could not find data set:{str(dataset_path)}.')

    @property
    def raw_file_names(self):
        raw_file_names = []
        if self.split_name == 'train':
            for i in range(0, 16):
                raw_file_names.append(f'java_train_{i}.jsonl')
            # raw_file_names.append(f'java_train_15.jsonl')
        elif self.split_name == 'valid':
            raw_file_names.append(f'java_valid_0.jsonl')
        elif self.split_name == 'test':
            raw_file_names.append(f'java_test_0.jsonl')  
        else:
            raise NotImplementedError('do not exist the partition')
        return raw_file_names

    @property
    def processed_file_names(self):
        processed_file_name_list = []
        if self.split_name == 'train':
            processed_file_name_list.append('proprecessed_train.bin')
        elif self.split_name == 'valid':
            processed_file_name_list.append('proprecessed_valid.bin')
        elif self.split_name == 'test':
            processed_file_name_list.append('proprecessed_test.bin')
        else:
            raise NotImplementedError('do not exist the partition')
        return processed_file_name_list

    # process the raw data
    def process(self):
        discard_count = 0
        big_graph_count = 0
        for file_path in self.raw_paths:
            print(f'Processing {file_path}...')
            with open(file_path, 'r', encoding='UTF-8') as f:
                lines = f.readlines()
                line_num = 0
                for line in lines:
                    # try:
                    line_num += 1
                    obj = json.loads(line)
########################################################### generate description vector ###########################################################
                    docstring = obj['docstring']
                    docstring_tokens = re.split(
                        r'@param.*|@return.*|@since.*|@throws.*|@see|{@code|@code|{@link|–|—|···|∈|║|'
                        r'@link|<V>|@|<p.*>.*<p>|<p.*>|<p|</p>|<img.*>|<img|</img>|<d.*>|<b>|</d.*>|</b>|'
                        r'<T>|<a.*>|</a>|<d.*>.*</d.*>|&lt|&gt|\n|\t|\r|\.|,|#|\)|\(|}|{| |-|'
                        r'§|∪|∩|/\*.*|\*.*/|‘|’s|’|`|≤|“|”|→|~|≥',
                        docstring)
                    docstring_tokens = [token for token in docstring_tokens if token.strip() != '']
                    docstring_tokens = docstring_tokens[:40]
                    # filter chinese 
                    if re.search('[\u4e00-\u9fa5]|[\u0800-\u4e00]', ' '.join(docstring_tokens)):
                        discard_count += 1
                        continue
                    # return tokens sequence after proprecessing
                    desc_tokens = self._preprocess_desc(docstring_tokens)
                    if len(desc_tokens) < 3:
                        discard_count += 1
                        continue
                    def tokens_gen():
                        yield iter([' '.join(desc_tokens)])
                    encoded_desc = list(self.desc_vocab.transform(tokens_gen(), fixed_length=self.max_desc_len))[0]
                    desc_len = 0
                    for one in encoded_desc:
                        if one != self.desc_vocab.word_vocab[self.desc_vocab.PAD]:
                            desc_len += 1
########################################################### proprecessing NCFG ###########################################################
                    node_no = dict()                             # Record the mapping relationship between nodes and their numbers
                    edge_record = dict()                         # Record edges, a directed graph constructed with key as the starting point
                    num = 0                                      # Record the CFG node number of the statement
                    src_node_list = []                           # The key number of the image taken
                    dest_node_list = []                          # Get the number corresponding to value
                    mini_x = []                                  # Record the semantics of each node of AST child nodes
                    mini_edge = []                               # Record the edges of AST child nodes
                    mini_x_batch = []                            # Identifies the node number of the subgraph according to 0-3.5 million
                    mini_edge_batch = []                         # Identify the edge number of the subgraph according to 0-3.5 million
                    node_features = []                           # Or record the statement information of each previous node?
                    node_code_lens = []                          # A list corresponding to the list length above -- recording the length of each statement
                    astNode = obj['astNode']
                
                    for key, values in obj['cfg_dependency'].items():
                        key, num = self._build_node_features(key, node_code_lens, node_no, num, astNode, mini_x, mini_edge, mini_x_batch, mini_edge_batch, node_features)
                        for val in values:
                            val, num = self._build_node_features(val, node_code_lens, node_no, num, astNode, mini_x, mini_edge, mini_x_batch, mini_edge_batch, node_features)
                            src_node_list.append(node_no[key])
                            dest_node_list.append(node_no[val])
                            #  If the number corresponding to the key does not exist, then let it exist and the value is 0
                            if edge_record.get(node_no[key], None) is None:
                                edge_record[node_no[key]] = []
                            # If the number corresponding to the value is not in the record list corresponding to the key, then add the number of the value node to the list of nodes corresponding to the key.
                            if node_no[val] not in edge_record[node_no[key]]:
                                edge_record[node_no[key]].append(node_no[val])
                    
                    # filter graph with no node
                    if len(src_node_list) < 1:
                        discard_count += 1
                        continue
                    # filter big graph
                    if num > 500:
                        big_graph_count += 1
                        discard_count += 1
                        continue
########################################################### define format | Encapsulated into the corresponding object ###########################################################
                    # define format corresponding CFG
                    x = torch.tensor(node_features, dtype=torch.int32) 
                    edge_tensor = torch.tensor([src_node_list, dest_node_list], dtype=torch.long)
                    # edge_tensor, _ = thg.utils.add_self_loops(edge_tensor, num_nodes=x.size(0))
                    cfg_dd = Data(x=x, edge_index=edge_tensor)
                    
                    # Define CFG node AST related information
                    # [12,12,45] -> [0,0,1] -- the setting of the batch vector after scrambling
                    mini_x_batch = torch.tensor(mini_x_batch, dtype=torch.long)
                    unique_values, inverse_indices = torch.unique(mini_x_batch, return_inverse=True)
                    mini_x_batch = inverse_indices
                    mini_x = np.vstack(mini_x)
                    mini_edge = np.concatenate(mini_edge, axis=1)
                    # save statement AST
                    cfg_dd.mini_x = torch.tensor(mini_x, dtype=torch.int32)
                    cfg_dd.mini_edge = torch.tensor(mini_edge, dtype=torch.long)
                    cfg_dd.mini_x_batch = mini_x_batch
                    cfg_dd.mini_edge_batch = torch.tensor(mini_edge_batch, dtype=torch.long)
                    assert x.shape[0] == mini_x_batch[len(mini_x_batch) - 1] + 1
                    self.cfg_dd_graphs.append(cfg_dd)
                    # self.graphs_edge_types.append(torch.tensor(edge_types, dtype=torch.long).numpy())
                    self.descriptions.append(torch.tensor(encoded_desc, dtype=torch.int32).numpy())
                    self.descriptions_lens.append(desc_len)
        print(f'discard num：{discard_count}')
        print(f'discard the num of big graph：{big_graph_count}')
        print('Saving dataset...')
        self.descriptions = torch.tensor(self.descriptions, dtype=torch.int32)
        self.descriptions_lens = torch.tensor(self.descriptions_lens, dtype=torch.int32)
        save_dict = {
                     'cfg_dd_graphs': self.cfg_dd_graphs,
                     'descriptions': self.descriptions,
                     'descriptions_lens': self.descriptions_lens,
                    }
        save_path = os.path.join(self.root, f'proprecessed_{self.split_name}.bin')
        torch.save(save_dict, save_path)
        print(f'Saved dataset to {str(save_path)}')

    def preprocess_code(self, code):
        code = code.replace('\n', '').replace('\t', '')
        code = self.replace_to_token(code)
        code_tokens = self._preprocess_code(code)
        def code_gen():
            yield iter([' '.join(code_tokens)])
        encoded_code = list(self.node_ast_vocab.transform(code_gen(), fixed_length=self.max_node_len))[0]
        code_len = 0
        for one in encoded_code:
                if one != self.code_vocab.word_vocab[self.node_ast_vocab.PAD]:
                    code_len += 1
        return encoded_code, code_len

    def process_ast_stmt(self, ast_stmt):
        src_node_list = []
        dest_node_list = []
        x = []   
        num = 0
        node_no = dict()
        for key, values in ast_stmt.items():
            encoded_code, code_len = self.preprocess_code(key)
            if key not in node_no:
                node_no[key] = num
                num += 1
                x.append(encoded_code)      
            for val in values:
                encoded_code, code_len = self.preprocess_code(val)
                if val not in node_no:
                    node_no[val] = num
                    num += 1
                    x.append(encoded_code)      
                src_node_list.append(node_no[key])
                dest_node_list.append(node_no[val])
        return src_node_list, dest_node_list, x
    
    def _build_node_features(self, key, node_code_lens, node_no, num, astNode,  mini_x, mini_edge, mini_x_batch, mini_edge_batch, node_features):
        astStmt = dict()
        # get statement AST
        is_node_ast = True
        if astNode.get(key) is not None:
            astStmt = astNode.get(key)
        else:
            # end-if node.etc.
            is_node_ast = False
        # clear \n and \t
        key = key.replace('\n', '').replace('\t', '')
        if key not in node_no:
            node_no[key] = num
            num += 1
            encoded_code, code_len = self.preprocess_code(key)
            node_features.append(encoded_code)
            node_code_lens.append(code_len)
            # If it cannot be parsed into node ast, such as endif node - self-looping ast - remember to check the case of 0
            if is_node_ast == False:
                temCode = []
                temCode.append(encoded_code)
                tem_mini_x = torch.tensor(temCode, dtype=torch.int32)
                tem_mini_edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                if len(tem_mini_x) > 0 and len(tem_mini_edge_index) > 0:           
                    mini_x.append(tem_mini_x)
                    mini_edge.append(tem_mini_edge_index)
                    for _ in range(tem_mini_x.shape[0]):
                        # This is what needs to be traversed Code block -- if it is 0, the corresponding ast node number will not be added 
                        mini_x_batch.append(self.ast_Id)
                    for _ in range(tem_mini_edge_index.shape[1]):
                        # Identify the type and number of edges
                        mini_edge_batch.append(self.ast_Id)
                else:
                    temKey = "empty"
                    tem_code_tokens = self._preprocess_code(temKey)
                    def tem_code_gen():
                        yield iter([' '.join(tem_code_tokens)])
                    tem_encoded_code = list(self.node_ast_vocab.transform(tem_code_gen(), fixed_length=self.max_code_len))[0]
                    temCode = []
                    temCode.append(tem_encoded_code)
                    tem_mini_x = torch.tensor(temCode, dtype=torch.int32)
                    tem_mini_edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                    mini_x.append(tem_mini_x)
                    mini_edge.append(tem_mini_edge_index)
                    for _ in range(tem_mini_x.shape[0]):
                        mini_x_batch.append(self.ast_Id)
                    for _ in range(tem_mini_edge_index.shape[1]):
                        mini_edge_batch.append(self.ast_Id)
                self.ast_Id = self.ast_Id + 1
                return key, num
            
            if astStmt is not None:
                if len(astStmt) != 0:       
                    src_node_list2 = []
                    dest_node_list2 = []
                    x = []
                    src_node_list2, dest_node_list2, x = self.process_ast_stmt(astStmt) 
                else:
                    # parse it null, then set 0
                    temCode = []
                    temCode.append(encoded_code)
                    tem_mini_x = torch.tensor(temCode, dtype=torch.int32)
                    tem_mini_edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                    if len(tem_mini_x) > 0 and len(tem_mini_edge_index) > 0:           
                        mini_x.append(tem_mini_x)
                        mini_edge.append(tem_mini_edge_index)
                        for _ in range(tem_mini_x.shape[0]):
                            mini_x_batch.append(self.ast_Id)
                        for _ in range(tem_mini_edge_index.shape[1]):
                            mini_edge_batch.append(self.ast_Id)
                    else:
                        temKey = "empty"
                        tem_code_tokens = self._preprocess_code(temKey)
                        def tem_code_gen():
                            yield iter([' '.join(tem_code_tokens)])
                        tem_encoded_code = list(self.node_ast_vocab.transform(tem_code_gen(), fixed_length=self.max_code_len))[0]
                        temCode = []
                        temCode.append(tem_encoded_code)
                        tem_mini_x = torch.tensor(temCode, dtype=torch.int32)
                        tem_mini_edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                        mini_x.append(tem_mini_x)
                        mini_edge.append(tem_mini_edge_index)
                        for _ in range(tem_mini_x.shape[0]):
                            mini_x_batch.append(self.ast_Id)
                        for _ in range(tem_mini_edge_index.shape[1]):
                            mini_edge_batch.append(self.ast_Id)
                    self.ast_Id = self.ast_Id + 1
                    return key, num
            
                tem_mini_x = torch.tensor(x, dtype=torch.int32)
                tem_mini_edge_index = torch.tensor([src_node_list2, dest_node_list2], dtype=torch.long)
                if len(tem_mini_x) > 0 and len(tem_mini_edge_index) > 0:
                    mini_x.append(tem_mini_x)
                    mini_edge.append(tem_mini_edge_index)
                    for _ in range(tem_mini_x.shape[0]):
                        mini_x_batch.append(self.ast_Id)
                    for _ in range(tem_mini_edge_index.shape[1]):
                        mini_edge_batch.append(self.ast_Id)
                self.ast_Id = self.ast_Id + 1
        return key, num
    
    # replace '+' to ADD
    def replace_to_token(self, code) -> str:
        symbol_mapping = {
            '+': ' ADD ',
            '-': ' SUB ',
            '*': ' MUL ',
            '/': ' DIV ',
            '=': ' ASSIGN ',
            '(': ' LPAREN ',
            ')': ' RPAREN ',
            '{': ' LBRACE ',
            '}': ' RBRACE ',
            '[': ' LBRACK ',
            ']': ' RBRACK ',
            ';': ' SEMI ',
            ',': ' COMMA ',
            '.': ' DOT ',
            '>': ' GT ',
            '<': ' LT ',
            '!': ' BANG ',
            '~': ' TILDE ',
            '?': ' QUESTION ',
            ':': ' COLON ',
            '==': ' EQUAL ',
            '<=': ' LE ',
            '>=': ' GE ',
            '!=': ' NOTEQUAL ',
            '&&': ' AND ',
            '||': ' OR ',
            '++': ' INC ',
            '--': ' DEC ',
            '&': ' BITAND ',
            '|': ' BITOR ',
            '^': ' CARET ',
            '%': ' MOD ',
            '+=': ' ADD_ASSIGN ',
            '-=': ' SUB_ASSIGN ',
            '*=': ' MUL_ASSIGN ',
            '/=': ' DIV_ASSIGN ',
            '&=': ' AND_ASSIGN ',
            '|=': ' OR_ASSIGN ',
            '^=': ' XOR_ASSIGN ',
            '%=': ' MOD_ASSIGN ',
            '<<=': ' LSHIFT_ASSIGN ',
            '>>=': ' RSHIFT_ASSIGN ',
            '>>>=': ' URSHIFT_ASSIGN ',
            '1': ' NUM ',
            '2': ' NUM ',
            '3': ' NUM ',
            '4': ' NUM ',
            '5': ' NUM ',
            '6': ' NUM ',
            '7': ' NUM ',
            '8': ' NUM ',
            '9': ' NUM ',
            '0': ' NUM '
        }
        for symbol, replacement in symbol_mapping.items():
             code = code.replace(symbol, replacement)
        return code
    
    # code proprecessing
    def _preprocess_code(self, code):
        filtered_tokens = []
        code = code.replace('\n', '').replace('\t', '')
        code_tokens = code.split(' ')
        for token in code_tokens:
            token = token.strip()
            if token.startswith('//') or len(token) <= 1:
                continue
            for rt in self.replace_tokens:
                token = token.replace(rt, ' ')
            processed_tokens = camel_and_snake_to_word_list(token).lower().split(' ')
            for t in processed_tokens:
                if t not in self.replace_tokens and len(t.strip()) >= 1:
                    filtered_tokens.append(t)
        return filtered_tokens
  
    # description proprecessing
    def _preprocess_desc(self, docstring_tokens):
        final_token = ''
        first_in = True
        for t in docstring_tokens:
            if t in STOP_WORDS:
                continue
            t = t.strip()
            for rt in self.replace_tokens:
                t = t.replace(rt, ' ')
            tokens = camel_and_snake_to_word_list(t).lower().split(' ')
            temp_t = []
            for t1 in tokens:
                if t1 not in STOP_WORDS:
                    temp_t.append(t1)
            if len(temp_t) == 0:
                continue
            t = ' '.join(temp_t).strip()
            if len(t) > 1:
                final_token += ' ' + t
                first_in = True
            else:
                if first_in:
                    final_token += ' ' + t
                    first_in = False
                else:
                    final_token += t
        final_token = final_token.strip()
        return final_token.split(' ')

    def padding_to_max_len(self, desc_vectors, max_len, padding_content):
        while len(desc_vectors) < max_len:
            desc_vectors.append(padding_content)
        return desc_vectors


if __name__ == '__main__':
    valid_set = CodeSearchNetDataSet('../NCFG-CS/proprecessed_data/valid', 'valid')
    print('valid set size：', len(valid_set))
    for i in range(5):
        print(valid_set[i])
