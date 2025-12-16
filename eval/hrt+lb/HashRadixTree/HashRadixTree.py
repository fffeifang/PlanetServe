from collections import deque
from pyexpat import model
from typing import Dict, List, Tuple
from transformers import AutoTokenizer
from .ModelNode import ModelNode
import random
import math

##############################################################################
#                       HASH RADIX TRIE CLASSES
##############################################################################

class HashRadixNode:
    """
    A node in the HashRadix trie, storing children keyed by hash values
    and a marker for the end of a sequence.
    """
    __slots__ = ["children", "is_end", "model_list", "parent"]

    def __init__(self) -> None:
        self.children: Dict[int, "HashRadixNode"] = {}
        self.model_list = []       
        self.is_end: bool = False
        self.parent = None


class HashRadixTree:
    def __init__(self, 
                 first_workload, 
                 first_mnode,
                 candidate_models: List[ModelNode], 
                 gate: int = 10, 
                 model_num: int = 8, 
                 chunk_size: int = 10) -> None:
        self.root = HashRadixNode()
        self.gate = gate
        self.chunk_size = chunk_size
        self.candidate_models = candidate_models
        # self.model_task = {model.name: model.pending_tasks for model in self.candidate_models}
 
        ### for toolbench case
        self.root1 = HashRadixNode()
        self.magic_ids = [] # set prefix based on task
        h_val = self.hash_chunk(self.magic_ids)
        self.root.children[h_val] = self.root1
        
        
        node = self.insert_workload(first_workload)
        self.assign_workload(first_mnode, node)

    def find_idle_node(self) -> str:
        task_list = {model.pending_tasks:model for model in self.candidate_models}
        sorted_models = sorted(self.candidate_models, key=lambda model: model.pending_tasks)
        return sorted_models[0]
    
    def assign_workload(self, modelnode, hrt_node) -> None:
        model_name = modelnode.name
        # modelnode.pending_tasks += 1
        # self.model_task[model_name] += 1
        
        if modelnode not in hrt_node.model_list:
            hrt_node.model_list.append(modelnode)
        # print(hrt_node.model_list)
        
    
    @staticmethod
    def hash_chunk(chunk: Tuple[int, ...], mod: int = 15) -> int:
        """
        Compute an integer hash for a tuple of token IDs. Collisions are possible
        for sufficiently large data, but for small chunk sizes and small mod,
        this is just a demonstration of the approach.
        """
        hash_val = 0
        for tid in chunk:
            hash_val = (hash_val * 31 + tid) % mod
        return hash_val

    def insert_workload(self, tokens) -> None:
        """
        Insert a sequence of chunks (each chunk a tuple of token IDs) into the trie.
        """
        current = self.root
        h_val = self.hash_chunk(self.magic_ids)
        current = current.children[h_val]
        
        # if isinstance(tokens, list):
        #     input_ids = tokens
        # else:
        if isinstance(tokens, list):
            input_ids = tokens
        else:
            input_ids = tokens["input_ids"].tolist()[0]

        total_tokens = len(input_ids)
        chunks: List[Tuple[int, ...]] = []
        
        i = 0
        
        while i < total_tokens:
            chunk = tuple(input_ids[i: i + self.chunk_size])
            chunks.append(chunk)
            i += self.chunk_size
            if int(i / self.chunk_size) > self.gate:
                break
            
            hval = self.hash_chunk(chunk)
            if hval not in current.children:
                current.children[hval] = HashRadixNode()
                current.children[hval].parent = current
            current = current.children[hval]
        current.is_end = True
        
        return current
    
    
    def find_match_model(self, tokens) -> ModelNode:
        current = self.root
        h_val = self.hash_chunk(self.magic_ids)
        current = current.children[h_val]
        
        i = 0 
        d = 0
        
        if isinstance(tokens, list):
            input_ids = tokens
        else:
            input_ids = tokens["input_ids"].tolist()[0]
        

        total_tokens = len(input_ids)
        while i < total_tokens:
            hval = self.hash_chunk(tuple(input_ids[i: i+self.chunk_size]))
            i += self.chunk_size
            if hval in current.children:
                current = current.children[hval]
                d += 1
                if d >= self.gate:
                    break

            else:
                break
        
        if d >= self.gate:
            # print("! matched")
            # self.assign_workload(current.model_list[0], current)
            # print(self.model_task)
            match_model = None
            min = 1e9
            for model in current.model_list:
                if min > model.pending_tasks:
                    match_model = model
                    min = model.pending_tasks
            return  current.model_list[0], current
        else:
            node = self.insert_workload(tokens)
            match_model = random.choice(self.candidate_models)
            self.assign_workload(match_model, node)
            return match_model, node
    
    def load_balance(self, hrt_node, l_list):
        # print("now load_balance...")s
        if hrt_node.parent is not None:
            current = hrt_node.parent
            n = len(l_list)
            cutoff_idx = max(int(math.ceil(0.8 * n)) - 1, 0) 
            cutoff = l_list[cutoff_idx][1]
            for model in current.model_list:
                if model.latency_factor() < cutoff:
                    return model
            # 未命中则向上
            return self.load_balance(current, l_list)
        else:
            return l_list[0][0]



    def print_tree_by_layers(self) -> List[List[str]]:
        """
        Returns a list of layers (each layer is a list of string descriptions).
        Layer 0 is the root, layer 1 are the root's children, etc.
        """
        from collections import deque
        result: List[List[str]] = []
        queue = deque([("Root", self.root, 0)])  # (prefix, node, level)
        current_level = 0
        current_level_nodes: List[str] = []

        while queue:
            prefix, node, level = queue.popleft()
            if level > current_level:
                result.append(current_level_nodes)
                current_level_nodes = []
                current_level = level
            node_info = f"{prefix}"
            if node.is_end:
                node_info += " (End)"
            current_level_nodes.append(node_info)
            for hash_val, child in sorted(node.children.items(), key=lambda x: x[0]):
                child_prefix = f"{prefix}-{hash_val}" if prefix != "Root" else f"{hash_val}"
                queue.append((child_prefix, child, level + 1))
        if current_level_nodes:
            result.append(current_level_nodes)
        return result
