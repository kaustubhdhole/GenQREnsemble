import pyterrier as pt
from pyterrier.measures import * # don't uncomment this
import os

print(f"Using GPUs : CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

import torch
import numpy as np
def print_seeds():
    print(f'torch seed = {torch.seed()}')
    print(f'numpy seed = {np.random.seed()}')

scratchpath = input('Provide your local path where your code exists? eg /local/scratch/username/ ') 
os.chdir(f'{scratchpath}/prompt-prf/terrier-prf')
os.environ['TRANSFORMERS_CACHE'] = f'{scratchpath}/prompt-prf/hf_cache'
os.environ['IR_DATASETS_HOME']=f'{scratchpath}/prompt-prf/ir_datasets_download'
if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])
    #pt.init()

wprint = lambda x: None
from transformers import set_seed
#from pyterrier_t5 import MonoT5ReRanker, DuoT5ReRanker

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
import torch

def generate_few_shot_prompt(kfs):
    few_shots_prompts = [
        {
            "query": "Artificial intelligence",
            "keywords": "Machine learning, Neural networks, AI research, Deep learning, Natural language processing"
        },
        {
            "query": "Tell me about climate change and its effects on the environment.",
            "keywords": "Climate change impacts, Environmental consequences, Global warming, Climate crisis, Ecosystem degradation"
        },
        {
            "query": "The Great Gatsby book review",
            "keywords": "F. Scott Fitzgerald, Literary analysis, American literature, Roaring Twenties, Jay Gatsby"
        },
        {
            "query": "A study on the benefits of meditation in reducing stress and anxiety",
            "keywords": "Meditation benefits, Stress reduction, Anxiety management, Mindfulness practice, Psychological well-being"
        },
        {
            "query": "Renewable energy sources",
            "keywords": "Solar power, Wind energy, Green technology, Sustainable energy, Clean energy"
        },
        {
            "query": "How do I bake a chocolate cake from scratch?",
            "keywords": "Homemade chocolate cake recipe, Baking instructions, Dessert preparation, Cocoa powder, Cake ingredients"
        },
        {
            "query": "The Impact of Social Media on Society",
            "keywords": "Social media effects, Online communities, Digital communication, Social networking sites, Internet influence"
        },
        {
            "query": "Research on the correlation between exercise and cardiovascular health",
            "keywords": "Physical activity, Heart health, Cardiovascular fitness, Exercise benefits, Cardiac health"
        },
        {
            "query": "Space exploration",
            "keywords": "Astronomy, NASA missions, Space travel, Planetary science, Astronauts"
        },
        {
            "query": "Recommend a good thriller movie to watch tonight.",
            "keywords": "Thriller film recommendation, Suspenseful movies, Movie night, Cinematic suspense, Must-see thrillers"
        }
    ]
    # Ensure that k is within the valid range
    kfs = min(kfs, len(few_shots_prompts))
    kfs = max(kfs, 1)
    # Select the first k prompts and concatenate them
    selected_prompts = few_shots_prompts[:kfs]
    concatenated_prompt = "\n\n".join([f"Query: {prompt['query']}\nKeywords: {prompt['keywords']}" for prompt in selected_prompts])
    return concatenated_prompt

SETTING=0

def generate1(model, tok_prompt, setting=SETTING):
    if setting == 0:
        return model.generate(input_ids=tok_prompt['input_ids'].to('cuda'),
                             attention_mask=tok_prompt['attention_mask'].to('cuda'))
    elif setting == 1:
        return model.generate(input_ids=tok_prompt['input_ids'].to('cuda'),
                              attention_mask=tok_prompt['attention_mask'].to('cuda'), num_beams=5, early_stopping=True)
    elif setting == 2:
        return model.generate(input_ids=tok_prompt['input_ids'].to('cuda'), attention_mask=tok_prompt['attention_mask'].to('cuda'),
                              num_beams=5, no_repeat_ngram_size = 1, early_stopping=True)
    elif setting == 3:
        return model.generate(input_ids=tok_prompt['input_ids'].to('cuda'),
                              attention_mask=tok_prompt['attention_mask'].to('cuda'),
                              do_sample=True,
                              top_k=0, early_stopping=True)
    elif setting == 4:
        return model.generate(input_ids=tok_prompt['input_ids'].to('cuda'),
                              attention_mask=tok_prompt['attention_mask'].to('cuda'),
                              do_sample=True,
                              top_k=0, temperature=0.6, early_stopping=True)
    elif setting == 5:
        return model.generate(input_ids=tok_prompt['input_ids'].to('cuda'),
                              attention_mask=tok_prompt['attention_mask'].to('cuda'),
                              do_sample=True,
                              top_k=50, early_stopping=True)
    elif setting == 6:
        return model.generate(input_ids=tok_prompt['input_ids'].to('cuda'),
                              attention_mask=tok_prompt['attention_mask'].to('cuda'),
                              do_sample=True,top_p=0.92,
                              top_k=50, early_stopping=(not IS_ALTERNATE_MODEL))
    elif setting == 7:
        return model.generate(input_ids=tok_prompt['input_ids'].to('cuda'),
                              attention_mask=tok_prompt['attention_mask'].to('cuda'), num_beams=100, early_stopping=True)
    else:
        # this is what was used in the 0-shot paper
        return model.generate(input_ids=tok_prompt['input_ids'].to('cuda'),
                              attention_mask=tok_prompt['attention_mask'].to('cuda'), num_beams=25, early_stopping=True)

def cleanup(s1):
    return "".join([x if x.isalnum() else " " for x in s1.strip()])

def load_zs_query_reformulator(model, tokenizer):
    MAX_LENGTH = tokenizer.model_max_length
    def zsq_3(query, instruction, only_keywords=True, kfs=5):
        few_shot_prompt = generate_few_shot_prompt(kfs)
        fsprompt = lambda query: instruction + "\n\n" + few_shot_prompt + "\n\n" + f"Query:{query}\nKeywords:"
        input_text = fsprompt(query)
        tok_prompt = tokenizer(input_text, padding=True, return_tensors="pt", truncation=True)
        outputs = model.generate(input_ids=tok_prompt['input_ids'].to('cuda'),
                                 attention_mask=tok_prompt['attention_mask'].to('cuda'),
                                 min_length=20, max_new_tokens=40,
                                 temperature=1.2, do_sample=True,
                                 top_k=200, num_return_sequences=5,
                                 top_p=0.92,
                                 repetition_penalty=2.1,
                                 early_stopping=True)
                                 #early_stopping=(not IS_ALTERNATE_MODEL))
        query2 = ""
        i = 0
        if IS_ALTERNATE_MODEL: # actually this if should be for causalLM
            outputs = outputs[:, tok_prompt['input_ids'].shape[1]:]
        for x in tokenizer.batch_decode(outputs, skip_special_tokens=True):
            #print(i);
            i += 1
            query2 += x + " "
        if only_keywords:
            q = query2
        else:
            q = query + " " + query2
        return "".join([x if x.isalnum() else " " for x in q.strip()])
    def zsq_2(query, instruction, only_keywords=True):
        zero_shot_prompt = f"{instruction}: {query}"
        tok_prompt = tokenizer(zero_shot_prompt, padding=True, return_tensors="pt", truncation=True)
        outputs = model.generate(input_ids=tok_prompt['input_ids'].to('cuda'),
                                 attention_mask=tok_prompt['attention_mask'].to('cuda'),
                                 min_length=20, max_new_tokens=40,
                                 temperature=1.2, do_sample=True,
                                 top_k=200, num_return_sequences=5,
                                 top_p=0.92,
                                 repetition_penalty=2.1,
                                 early_stopping=True)
                                 #early_stopping=(not IS_ALTERNATE_MODEL))
        query2 = ""
        i = 0
        if IS_ALTERNATE_MODEL: # actually this if should be for causalLM
            outputs = outputs[:, tok_prompt['input_ids'].shape[1]:]
        for x in tokenizer.batch_decode(outputs, skip_special_tokens=True):
            #print(i);
            i += 1
            query2 += x + " "
        if only_keywords:
            q = query2
        else:
            q = query + " " + query2
        return "".join([x if x.isalnum() else " " for x in q.strip()])
    def zsq(query, instruction, only_keywords=True, setting=SETTING):
        zero_shot_prompt = f"{instruction}: {query}"
        wprint(f'zero_shot_prompt = {zero_shot_prompt}')
        tok_prompt = tokenizer(zero_shot_prompt, padding=True, return_tensors="pt",  truncation=True)
        outputs = generate1(model, tok_prompt, setting=setting)
        query2 = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        wprint(f"Model Output = {query2}\n")
        q = query + " " + query2
        if only_keywords:
            kw = "".join([x if x.isalnum() else " " for x in query2])
            kw = kw.strip()
            return (kw if kw else query) # very minor cases
        return "".join([x if x.isalnum() else " " for x in q])
    # PRE-RETRIVAL combination
    def zsq_combo(query, instructions, only_keywords=True, setting=SETTING):
        query2 = "" if only_keywords else query
        for instruction in instructions:
            zero_shot_prompt = f"{instruction}: {query}"
            wprint(f'get_zsq_keyword_prompt = {zero_shot_prompt}')
            tok_prompt = tokenizer(zero_shot_prompt, padding=True, return_tensors="pt", truncation=True)
            outputs = generate1(model, tok_prompt, setting=setting)
            mo = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            wprint(f"Model Output = {mo}\n")
            query2 += " " + mo
        kw =  "".join([x if x.isalnum() else " " for x in query2.strip()])
        kw = kw.strip()
        return (kw if kw else query) # very minor cases
    def fsq(query, few_shot_prompt, setting=SETTING):
        few_shot_prompt += f"{query}\nExpansion Terms: "
        wprint(f'few_shot_prompt = {few_shot_prompt}')
        tok_prompt = tokenizer(few_shot_prompt, padding=True, return_tensors="pt", truncation=True)
        outputs = generate1(model, tok_prompt, setting=setting)
        query2 = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        wprint(f"Model Output = {query2}\n")
        q = query + " " + query2
        return "".join([x if x.isalnum() else " " for x in q])
    def zsq_reform(query, instruction, setting=SETTING):
        zero_shot_prompt = f"{instruction}:\nQuery: {query}\nReformulated Query: "
        wprint(f'zero_shot_prompt = {zero_shot_prompt}')
        tok_prompt = tokenizer(zero_shot_prompt, padding=True, return_tensors="pt", truncation=True)
        outputs = generate1(model, tok_prompt, setting=setting)
        query2 = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        wprint(f"Model Output = {query2}\n")
        q = query2
        return "".join([x if x.isalnum() else " " for x in q])
    def fsq_reform(query, few_shot_prompt, setting=SETTING):
        few_shot_prompt += f"{query}\nReformulated Query: "
        wprint(f'few_shot_prompt = {few_shot_prompt}')
        tok_prompt = tokenizer(few_shot_prompt, padding=True, return_tensors="pt", truncation=True)
        outputs = generate1(model, tok_prompt, setting=setting)
        query2 = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        wprint(f"Model Output = {query2}\n")
        q = query2
        return "".join([x if x.isalnum() else " " for x in q])
    def zsq_prf(query, instruction, context,only_keywords=True, setting=SETTING):
        #zero_shot_prompt = f"{instruction}: {query} based on the given context {context}"
        #zero_shot_prompt = f"{instruction}: [{query}] based on the given context: [{context}]" --> previous numbers in the excel sheet
        zero_shot_prompt = f"{instruction}: {query}, based on the given context information: {context}"
        wprint(f'zero_shot_prompt = {zero_shot_prompt}')
        tok_prompt = tokenizer(zero_shot_prompt, padding=True, return_tensors="pt", truncation=True)
        outputs = generate1(model, tok_prompt, setting=setting)
        query2 = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        wprint(f"Model Output = {query2}\n")
        q = query2 if only_keywords else query + " " + query2
        kw = "".join([x if x.isalnum() else " " for x in q])
        kw = kw.strip()
        return (kw if kw else query) # very minor cases
    def zsq_prf2(query, instruction, context, only_keywords=True, context_first=False):
        zero_shot_prompt = f"{instruction}: {query}, based on the given context information: {context}"
        if context_first:
            zero_shot_prompt = f"Based on the given context information: {context}, {instruction}: {query}"
        tok_prompt = tokenizer(zero_shot_prompt, padding=True, return_tensors="pt", truncation=True)
        outputs = model.generate(input_ids=tok_prompt['input_ids'].to('cuda'),
                                 attention_mask=tok_prompt['attention_mask'].to('cuda'),
                                 min_length=20, max_new_tokens=40,
                                 temperature=1.2, do_sample=True,
                                 top_k=200, num_return_sequences=5,
                                 top_p=0.92,
                                 repetition_penalty=2.1,
                                 early_stopping=True)
        query2 = ""
        i = 0
        for x in tokenizer.batch_decode(outputs, skip_special_tokens=True):
            #print(i);
            i += 1
            query2 += x + " "
        if only_keywords:
            q = query2
        else:
            q = query + " " + query2
        return "".join([x if x.isalnum() else " " for x in q.strip()])
    #PRE-RETRIEVAL COMBO
    def zsq_combo_prf(query, instructions, context,only_keywords=True, prompt_style=1, setting=SETTING):
        if only_keywords:
            query2 = ""
        else:
            query2 = query
        for instruction in instructions:
            # if prompt_style:
            #     zero_shot_prompt = f"{instruction}: {query} based on the given context {context}" # 0-shot PRF results are on this prompt
            # else:
            #     zero_shot_prompt = f"{instruction}: [{query}] based on the given context: [{context}]" #
            zero_shot_prompt = f"{instruction}: {query}, based on the given context information: {context}"
            wprint(f'get_zsq_keyword_prompt = {zero_shot_prompt}')
            tok_prompt = tokenizer(zero_shot_prompt, padding=True, return_tensors="pt", truncation=True)
            outputs = generate1(model, tok_prompt, setting=setting)
            mo = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            wprint(f"Model Output = {mo}\n")
            query2 += " " + mo
        kw = "".join([x if x.isalnum() else " " for x in query2.strip()])
        kw = kw.strip()
        return (kw if kw else query)  # very minor cases
    return zsq, fsq, zsq_reform, fsq_reform, zsq_prf, zsq_combo, zsq_combo_prf, zsq_2, zsq_prf2, zsq_3

EVAL_METRICS = ['num_q', 'map', R@10, R@100, nDCG@10, nDCG@100, RR(rel=2), AP(rel=2)]

def get_index(EVALUATION_NAME, index_name, field=None, colbert=False,):
    if index_name == "disks45_nocr":
        print(f"Loading Index {index_name}...")
        index_path = f'./indices/{index_name}'
        pt_index_path = index_path + '/data.properties'
        if not os.path.exists(pt_index_path):
            dataset = pt.get_dataset(EVALUATION_NAME)
            indexer = pt.IterDictIndexer(index_path)
            index_ref = indexer.index(dataset.get_corpus_iter(), fields=['body', 'title'])
        else:
            dataset = pt.get_dataset(EVALUATION_NAME)
            print('Using prebuilt index.')
            index_ref = pt.IndexRef.of(pt_index_path)
        index = pt.IndexFactory.of(index_ref)
        queries = dataset.get_topics()
        if field:
            queries['query'] = queries[field]
        else:
            queries['query'] = queries['title'] #
        queries['query'] = queries['query'].apply(cleanup)
        print('Completed indexing')
        if colbert:
            def corpus_iterator():
                for y in dataset.get_corpus_iter():
                    y['text'] = y['title'] + " " + y['body']
                    if y['text'].strip():
                        yield y
            return index, dataset, queries, corpus_iterator
        return index, dataset, queries
    elif index_name == "disks45_nocr_kd":
        print(f"Loading Index {index_name}...")
        index_path = f'./indices/{index_name}'
        pt_index_path = index_path + '/data.properties'
        if not os.path.exists(pt_index_path):
            dataset = pt.get_dataset(EVALUATION_NAME)
            indexer = pt.IterDictIndexer(index_path)
            index_ref = indexer.index(dataset.get_corpus_iter(), fields=['body', 'title'])
        else:
            dataset = pt.get_dataset(EVALUATION_NAME)
            print('Using prebuilt index.')
            index_ref = pt.IndexRef.of(pt_index_path)
        index = pt.IndexFactory.of(index_ref)
        queries = dataset.get_topics()
        if field:
            queries['query'] = queries[field]
        else:
            queries['query'] = queries['title'] #
        queries['query'] = queries['query'].apply(cleanup)
        print('Completed indexing')
        if colbert:
            def corpus_iterator():
                for y in dataset.get_corpus_iter():
                    y['text'] = y['title'] + " " + y['body']
                    if y['text'].strip():
                        yield y
            return index, dataset, queries, corpus_iterator
        return index, dataset, queries
    if index_name == "beir_dbpedia-entity":
        print(f"Loading Index {index_name}...")
        index_path = f'./indices/{index_name}'
        pt_index_path = index_path + '/data.properties'
        if not os.path.exists(pt_index_path):
            dataset = pt.get_dataset(EVALUATION_NAME)
            indexer = pt.IterDictIndexer(index_path, meta={"docno": 200})
            index_ref = indexer.index(dataset.get_corpus_iter(), fields=['text', 'title', 'url'])
        else:
            dataset = pt.get_dataset(EVALUATION_NAME)
            print('Using prebuilt index.')
            index_ref = pt.IndexRef.of(pt_index_path)
        index = pt.IndexFactory.of(index_ref)
        queries = dataset.get_topics()
        queries['query'] = queries['query'].apply(cleanup)
        print('Completed indexing')
        if colbert:
            def corpus_iterator():
                for y in dataset.get_corpus_iter():
                    y['text'] = y['title'] + " " + y['text']
                    if y['text'].strip():
                        yield y
            return index, dataset, dataset.get_topics(), corpus_iterator
        return index, dataset, queries
    if index_name == "beir_webis-touche2020_v2":
        print(f"Loading Index {index_name}...")
        index_path = f'./indices/{index_name}'
        pt_index_path = index_path + '/data.properties'
        if not os.path.exists(pt_index_path):
            dataset = pt.get_dataset(EVALUATION_NAME)
            indexer = pt.IterDictIndexer(index_path, meta={"docno": 39})
            index_ref = indexer.index(dataset.get_corpus_iter(), fields=['text', 'title', 'stance', 'url'])
        else:
            dataset = pt.get_dataset(EVALUATION_NAME)
            print('Using prebuilt index.')
            index_ref = pt.IndexRef.of(pt_index_path)
        index = pt.IndexFactory.of(index_ref)
        print('Completed indexing')
        queries = dataset.get_topics()
        queries['query'] = queries['description'].str.cat(queries['text'], sep=' ')
        queries['query'] = queries['query'].apply(cleanup)
        if colbert:
            def corpus_iterator():
                for y in dataset.get_corpus_iter():
                    y['text'] = y['title'] + " " + y['text']
                    if y['text'].strip():
                        yield y
            return index, dataset, queries, corpus_iterator
        return index, dataset, queries
    elif index_name == "msmarco_passage":
        print(f"Loading Index {index_name}...")
        index_path = f'./indices/{index_name}'
        pt_index_path = index_path + '/data.properties'
        if not os.path.exists(pt_index_path):
            dataset = pt.get_dataset(EVALUATION_NAME)
            indexer = pt.IterDictIndexer(index_path)
            index_ref = indexer.index(dataset.get_corpus_iter(), fields=['text'])
        else:
            dataset = pt.get_dataset(EVALUATION_NAME)
            print('Using prebuilt index.')
            index_ref = pt.IndexRef.of(pt_index_path)
        index = pt.IndexFactory.of(index_ref)
        print('Completed indexing')
        if colbert:
            return index, dataset, dataset.get_topics(), dataset.get_corpus_iter
        queries = dataset.get_topics()
        queries['query'] = queries['query'].apply(cleanup)
        return index, dataset, queries
    elif index_name == "msmarco_document":
        print(f"Loading Index {index_name}...")
        index_path = f'./indices/{index_name}'
        pt_index_path = index_path + '/data.properties'
        if not os.path.exists(pt_index_path):
            dataset = pt.get_dataset(EVALUATION_NAME)
            indexer = pt.IterDictIndexer(index_path)
            index_ref = indexer.index(dataset.get_corpus_iter(), fields=['url', 'title', 'body'])
        else:
            dataset = pt.get_dataset(EVALUATION_NAME)
            print('Using prebuilt index.')
            index_ref = pt.IndexRef.of(index_path)
        index = pt.IndexFactory.of(index_ref)
        print('Completed indexing')
        queries = dataset.get_topics()
        queries['query'] = queries['query'].apply(cleanup)
        return index, dataset, queries
    elif index_name == "trec-covid":
        print(f"Loading Index {index_name}...")
        EVALUATION_NAME = "irds:cord19/trec-covid"
        index_name = "cord19/trec-covid"
        index_path = f'./indices/{index_name}'
        pt_index_path = index_path + '/data.properties'
        if not os.path.exists(pt_index_path):
            dataset = pt.get_dataset(EVALUATION_NAME)
            indexer = pt.IterDictIndexer(index_path)
            index_ref = indexer.index(dataset.get_corpus_iter(), fields=['title', 'doi', 'date', 'abstract'])
        else:
            dataset = pt.get_dataset(EVALUATION_NAME)
            print('Using prebuilt index.')
            index_ref = pt.IndexRef.of(index_path)
        index = pt.IndexFactory.of(index_ref)
        print('Completed indexing')
        queries = dataset.get_topics()
        queries['query'] = queries['title'].str.cat(queries['description'], sep=' ')
        queries['query'] = queries['query'].apply(lambda text: text.replace("?", ""))
        queries['query'] = queries['query'].apply(cleanup)
        if colbert:
            def corpus_iterator():
                for y in dataset.get_corpus_iter():
                    y['text'] = y['title'] + " " + y['abstract']
                    if y['text'].strip():
                        yield y
            return index, dataset, queries, corpus_iterator
        return index, dataset, queries
    else:
        print(f"KD:No index selected of name {index_name}.")
        return None

def get_bm25_pipe(index_name, index):
    if index_name in ["trec-covid", "msmarco_passage", "msmarco_document"]:
        bm25 = pt.BatchRetrieve.from_dataset(index_name, 'terrier_stemmed', wmodel='BM25')
        #bm25_10000 = pt.BatchRetrieve.from_dataset(index_name, 'terrier_stemmed', wmodel='BM25', num_results=10000)
    else:
        bm25 = pt.BatchRetrieve(index, wmodel='BM25')
        #bm25_10000 = pt.BatchRetrieve.from_dataset(index_name, 'terrier_stemmed', wmodel='BM25', num_results=10000)
    return bm25

# EVALUATION_NAME="irds:disks45/nocr/trec-robust-2004"
# index_name="disks45_nocr"
# field='title'
def evaluate_others(EVALUATION_NAME, index_name, field, mono=True):
    print("Running others")
    index, dataset, queries = get_index(EVALUATION_NAME, index_name)
    # The different retrievers
    bm25 = get_bm25_pipe(index_name, index)
        #bm25_terrier_stemmed_docT5query = pt.BatchRetrieve.from_dataset(index_name, 'terrier_stemmed_docT5query', wmodel='BM25')
    # With PRF approaches
    rm3_pipe = bm25 >> pt.rewrite.RM3(index) >> bm25
    bo1_pipe = bm25 >> pt.rewrite.Bo1QueryExpansion(index) >> bm25
    kl_pipe = bm25 >> pt.rewrite.KLQueryExpansion(index) >> bm25
    # from pyterrier_t5 import MonoT5ReRanker
    # monoT5 = MonoT5ReRanker()  # loads castorini/monot5-base-msmarco by default
    # mono_pipeline = bm25 % 100 >> pt.text.get_text(dataset, field) >> monoT5
    results = pt.Experiment(
        [bm25, rm3_pipe,bo1_pipe, kl_pipe, bm25,], #mono_pipeline
        queries,
        dataset.get_qrels(),
        eval_metrics=EVAL_METRICS,
        names=["BM25", "BM25+RM3", "BM25+Bo1", "BM25+KL","BM25",],#
        verbose=True,
        batch_size=1,
        baseline=0,
    )
    print(f"Results on {EVALUATION_NAME}")
    print(f"Other Approaches")
    print(results)
    dfs_to_concat.append(results)
    save_setting()
    return results


def evaluate_generative_expansion(EVALUATION_NAME, index_name,field,  beta=0.2):
    set_seed(0)
    print("Running evaluate_zero_shot_prompt_baseline")
    #index, dataset, queries = get_index(EVALUATION_NAME, index_name, field)
    index, dataset, queries, corpus_iterator = get_index(EVALUATION_NAME, index_name, field, colbert=True)
    bm25 = get_bm25_pipe(index_name, index)
    # (0) Baselines
    bm25_baseline = bm25
    zeroshot = pt.apply.query(lambda row: zs_qr_xxl(row.query, KEYWORD_GENERATION_INSTRUCTIONS[0], only_keywords=True, setting=1)) >> pt.rewrite.linear(beta, 1 - beta, format="terrierql") >> bm25
    # (1) Unified Query Setting
    #unified_query = pt.apply.query(lambda row: zsq_keyword_combo(row.query, KEYWORD_GENERATION_INSTRUCTIONS, only_keywords=True, setting=5)) >> pt.rewrite.linear(beta, 1 - beta, format="terrierql") >> bm25
    unified_query = pt.apply.query(
        lambda row: zsq_keyword_combo(row.query, KEYWORD_GENERATION_INSTRUCTIONS, only_keywords=True,
                                      setting=5)) >> pt.rewrite.linear(beta, 1 - beta, format="terrierql") >> bm25
    # (2) Document Fusion Setting
    ps = [(pt.apply.query(lambda row: (zs_qr_xxl(row.query, instruction, only_keywords=True, setting=5))) >> pt.rewrite.linear(beta, 1 - beta, format="terrierql") >> bm25) for instruction in KEYWORD_GENERATION_INSTRUCTIONS]
    assert len(ps) == 10
    document_fusion = (ps[0] + ps[1] + ps[2] + ps[3] + ps[4] + ps[5] + ps[6] + ps[7] + ps[8] + ps[9]) #>> pt.text.get_text(dataset, field)
    dfs_to_concat = []
    results = pt.Experiment(
        [bm25_baseline],
        queries,
        dataset.get_qrels(),
        # eval_metrics=EVAL_METRICS,
        eval_metrics=['num_q', 'map', 'recip_rank', R @ 1000, nDCG @ 10, nDCG @ 20],
        names=["BM25" ],
        verbose=True,
        batch_size=24,
    );
    dfs_to_concat.append(results); save_this(dfs_to_concat, index_name, fname="standalone")
    results = pt.Experiment(
        [zeroshot, unified_query, document_fusion],
        queries,
        dataset.get_qrels(),
        # eval_metrics=EVAL_METRICS,
        eval_metrics=['num_q', 'map', 'recip_rank', R @ 1000, nDCG @ 10, nDCG @ 20],
        names=[f"ZeroShot", f"UnifiedQuery", f"DocumentFusion"],
        verbose=True,
        batch_size=24,
        baseline=0,  # zeroshot
        correction='bonferroni'
    );
    dfs_to_concat.append(results); save_this(dfs_to_concat, index_name, fname="standalone")
    print(results)
    return results

# verify "only_keywords=" and "setting"
def evaluate_generative_expansion_models_with_monoT5ranker(EVALUATION_NAME, index_name, field, doc_field, beta=0.2):
    set_seed(0)
    print("Running evaluate_generative_expansion_models_with_monoT5ranker")
    index, dataset, queries, corpus_iterator = get_index(EVALUATION_NAME, index_name, field, colbert=True)
    #queries['text'] = queries['query']
    bm25 = get_bm25_pipe(index_name, index)
    monoT5 = MonoT5ReRanker(text_field=doc_field)
    # (0) Baselines
    #bm25_baseline = bm25 >> pt.text.get_text(dataset, field) >> monoT5
    bm25_baseline = bm25 >> pt.text.get_text(dataset, doc_field) >> monoT5
    #zeroshot = pt.apply.query(lambda row: zs_qr_xxl(row.query, KEYWORD_GENERATION_INSTRUCTIONS[0], only_keywords=True, setting=1)) >> pt.rewrite.linear(beta, 1 - beta, format="terrierql") >> bm25 >> pt.text.get_text(dataset, doc_field) >> monoT5
    zeroshot = pt.apply.query(lambda row: zs_qr_xxl(row.query, KEYWORD_GENERATION_INSTRUCTIONS[0], only_keywords=False,
                                                    setting=7)) >> bm25 >> pt.text.get_text(
        dataset, doc_field) >> monoT5 #  map  recip_rank    R@1000   nDCG@10   nDCG@20 --> 0.148246    0.565901  0.435757  0.354477  0.319002
    # (1) Unified Query Setting
    #unified_query = pt.apply.query(lambda row: zsq_keyword_combo(row.query, KEYWORD_GENERATION_INSTRUCTIONS, only_keywords=True, setting=5)) >> pt.rewrite.linear(beta, 1 - beta, format="terrierql") >> bm25 >> pt.text.get_text(dataset, doc_field) >> monoT5
    unified_query = pt.apply.query(
        lambda row: zsq_keyword_combo(row.query, KEYWORD_GENERATION_INSTRUCTIONS, only_keywords=False,
                                      setting=5))  >> bm25 >> pt.text.get_text(dataset, doc_field) >> monoT5
    # (2) Document Fusion Setting
    #ps = [(pt.apply.query(lambda row: (zs_qr_xxl(row.query, instruction, only_keywords=True, setting=5))) >> pt.rewrite.linear(beta, 1 - beta, format="terrierql") >> bm25) for instruction in KEYWORD_GENERATION_INSTRUCTIONS]
    ps = [(pt.apply.query(lambda row: (zs_qr_xxl(row.query, instruction, only_keywords=True, setting=5))) >> bm25) for instruction in KEYWORD_GENERATION_INSTRUCTIONS]
    assert len(ps) == 10
    document_fusion = (ps[0] + ps[1] + ps[2] + ps[3] + ps[4] + ps[5] + ps[6] + ps[7] + ps[8] + ps[9]) >> pt.text.get_text(dataset, doc_field) >> monoT5
    dfs_to_concat = []
    results = pt.Experiment(
        [zeroshot],
        queries,
        dataset.get_qrels(),
        # eval_metrics=EVAL_METRICS,
        eval_metrics=['num_q', 'map', 'recip_rank', R @ 1000, nDCG @ 10, nDCG @ 20],
        names=["ZeroShot"],
        verbose=True,
        batch_size=1,
    );results
    dfs_to_concat.append(results); save_this(dfs_to_concat, index_name, fname="monoT5reranked")
    # results = pt.Experiment(
    #     [bm25_baseline],
    #     queries,
    #     dataset.get_qrels(),
    #     # eval_metrics=EVAL_METRICS,
    #     eval_metrics=['num_q', 'map', 'recip_rank', R @ 1000, nDCG @ 10, nDCG @ 20],
    #     names=[f"bm25",],
    #     #verbose=True,
    #     batch_size=24,
    # );results
    results = pt.Experiment(
        [ zeroshot, unified_query, document_fusion],
        queries,
        dataset.get_qrels(),
        # eval_metrics=EVAL_METRICS,
        eval_metrics=['num_q', 'map', 'recip_rank', R @ 1000, nDCG @ 10, nDCG @ 20],
        names=[f"ZeroShot", f"UnifiedQuery",f"DocumentFusion"],
        verbose=True,
        batch_size=24,
        baseline=0,  # zeroshot
        correction='bonferroni'
    );
    dfs_to_concat.append(results); save_this(dfs_to_concat, index_name, fname="monoT5reranked")
    print(results)
    print(f"Completed evaluate_generative_expansion_models_with_monoT5ranker on {index_name}")
    return results

def save_this(dfs_to_concat, index_name, fname):
    combined_df = pd.concat(dfs_to_concat, ignore_index=True)
    # combined_df = combined_df.applymap(lambda x: f'{x:.2f}'.replace('.', ','))
    combined_df.to_csv(
        f'{scratchpath}/prompt-prf/experiment_fall23/final/results_{fname}_{index_name}.csv',
        index=False)

import json
def get_reformed_queries(queries, EVALUATION_NAME, i=0, prf_k=None, gold=False, return_map=False, context_first=True):
    queries = queries.copy()
    eval_name = EVALUATION_NAME.replace(":", "_").replace("/", "_").replace("_irds", "")
    alt = ONLY_MODEL_NAME + "/" if IS_ALTERNATE_MODEL else ""
    cf = "_cf" if context_first else ""
    fs = f"fs{KFS}_" if ENABLE_FEW_SHOT else ""
    if prf_k:
        s = "_gold" if gold else ""
        file_name = f'{scratchpath}/prompt-prf/experiment_fall23/{alt}expansions_{eval_name}/prf_{prf_k}/{fs}keywords_zsq_s0{cf}_prf_i{i}{s}.json'
    else:
        file_name = f'{scratchpath}/prompt-prf/experiment_fall23/{alt}expansions_{eval_name}/{fs}keywords_zsq_i{i}.json'
    queries['query_0'] = queries['query']
    with open(file_name, "r") as outfile:
        query_to_keyword = json.load(outfile)
    queries['query'] = queries['query_0'] + " " + queries['query'].apply(lambda q: query_to_keyword[q])
    if return_map:
        for q in query_to_keyword:
            #query_to_keyword[q] = q + " " + query_to_keyword[q]
            query_to_keyword[q] = q + " " + query_to_keyword[q]
        return query_to_keyword
    return queries

def load_all_files(EVALUATION_NAME, prf_k=False, s6=False):
    query_to_keyword_i = []
    for i in range(10):
        eval_name = EVALUATION_NAME.replace(":", "_").replace("/", "_").replace("_irds", "")
        s = "s6_" if s6 else ""
        if prf_k:
            file_name = f'{scratchpath}/prompt-prf/experiment_fall23/expansions_{eval_name}/prf_{prf_k}/{s}keywords_zsq_prf_i{i}.json'
        else:
            file_name = f'{scratchpath}/prompt-prf/experiment_fall23/expansions_{eval_name}/{s}keywords_zsq_i{i}.json'
        with open(file_name, "r") as outfile:
            query_to_keyword = json.load(outfile)
            query_to_keyword_i.append(query_to_keyword)
    assert len(query_to_keyword_i) == 10
    return query_to_keyword_i

def get_reformed_queries_ensemble(queries, EVALUATION_NAME, prf_k=None, return_map=False, context_first=True, gold=False):
    queries = queries.copy()
    eval_name = EVALUATION_NAME.replace(":", "_").replace("/", "_").replace("_irds", "")
    alt = ONLY_MODEL_NAME + "/" if IS_ALTERNATE_MODEL else ""
    folder_name = f'{scratchpath}/prompt-prf/experiment_fall23/{alt}expansions_{eval_name}'
    queries['query_0'] = queries['query']
    cf = "_cf" if context_first else ""
    fs = f"fs{KFS}_" if ENABLE_FEW_SHOT else ""
    dict_list = []
    Q2K = {}
    g = "_gold" if gold else ""
    for i in range(10):
        if prf_k:
            file_name = f'{scratchpath}/prompt-prf/experiment_fall23/{alt}expansions_{eval_name}/prf_{prf_k}/{fs}keywords_zsq_s0{cf}_prf_i{i}{g}.json'
        else:
            file_name = f'{scratchpath}/prompt-prf/experiment_fall23/{alt}expansions_{eval_name}/{fs}keywords_zsq_i{i}.json'
        with open(file_name, "r") as outfile:
            query_to_keyword = json.load(outfile)
            dict_list.append(query_to_keyword)
        queries['query'] = queries['query'] + " " + queries['query_0'].apply(lambda q: query_to_keyword[q])
    if return_map:
        for key in dict_list[0].keys():
            #Q2K[key] = key + " " + " ".join([d[key] for d in dict_list])
            Q2K[key] = key + " " + " ".join([d[key] for d in dict_list])
        return Q2K
    return queries


def save_this_here(dfs_to_concat, EVALUATION_NAME, prf_k=None, ensemble=False, suffix=None, gold=False):
    eval_name = EVALUATION_NAME.replace(":", "_").replace("/", "_").replace("_irds", "")
    eval_folder = f"{scratchpath}/prompt-prf/experiment_fall23/expansions_{eval_name}"
    if IS_ALTERNATE_MODEL:
        eval_folder = f"{scratchpath}/prompt-prf/experiment_fall23/{ONLY_MODEL_NAME}/expansions_{eval_name}"
    if prf_k:
        folder_name = f'{eval_folder}/prf_{prf_k}'
    else:
        folder_name = eval_folder
    combined_df = pd.concat(dfs_to_concat, ignore_index=True)
    s = "2" if ensemble else ""
    s1 = suffix if suffix else ""
    fsk = ("fs_" + str(KFS)) if ENABLE_FEW_SHOT else ""
    g = "_gold" if gold else ""
    fname = f'{folder_name}/results{s}{s1}{fsk}{g}.csv'
    combined_df.to_csv(fname, index=False)
    print(f"Saved to {fname}")

def evaluate_with_and_without_generative_expansion(EVALUATION_NAME, index_name, field, doc_field, prf_k=None, eval_all_instrucitons=False):
    dfs_to_concat = []
    index, dataset, queries, corpus_iterator = get_index(EVALUATION_NAME, index_name, field, colbert=True)
    # (0) Raw Queries
    bm25 = get_bm25_pipe(index_name, index)
    mono = MonoT5CustomReRanker(text_field=doc_field, query_field ='query')
    bm25_n = bm25 >> pt.text.get_text(dataset, doc_field) >> mono
    results1 = pt.Experiment([bm25, bm25_n], queries, dataset.get_qrels(), eval_metrics=['num_q', 'map',"P_10", 'recip_rank', R @ 1000, nDCG @ 10, nDCG @ 20, RR(rel=2), AP(rel=2)], names=["BM25", "BM25>>Neural"], verbose=True, batch_size=24);
    dfs_to_concat.append(results1); save_this_here(dfs_to_concat, EVALUATION_NAME, prf_k=prf_k)
    # # (1) 0-shot baseline
    if prf_k:
        rm3_baseline = bm25 >> pt.rewrite.RM3(index, fb_docs=prf_k) >> bm25
        rm3_baseline1 = rm3_baseline
        rm3_baseline_n = bm25 >> pt.rewrite.RM3(index, fb_docs=prf_k) >> bm25 >> pt.text.get_text(dataset, doc_field) >> mono
        results2 = pt.Experiment([rm3_baseline1, rm3_baseline_n], queries, dataset.get_qrels(),
                                 eval_metrics=['num_q', 'map', "P_10", 'recip_rank', R @ 1000, nDCG @ 10, nDCG @ 20, RR(rel=2),
                                               AP(rel=2)], names=["RM3", "RM3>>Neural"], verbose=True,
                                 batch_size=24);
        # rms = [(bm25 >> pt.rewrite.RM3(index, fb_docs=prf_k) >> bm25) for prf_k in [1,2,5]]
        # results2 = pt.Experiment(rms, queries, dataset.get_qrels(),
        #                          eval_metrics=['num_q', 'map', 'recip_rank', R @ 1000, nDCG @ 10, nDCG @ 20, RR(rel=2),
        #                                        AP(rel=2)], verbose=True,
        #                          batch_size=24);results2
        dfs_to_concat.append(results2);
        save_this_here(dfs_to_concat, EVALUATION_NAME, prf_k=prf_k)
    mono = MonoT5CustomReRanker(text_field=doc_field, query_field='query_0')
    reformed_queries = get_reformed_queries(queries,EVALUATION_NAME, i=0, prf_k=prf_k)
    bm25_n = bm25 >> pt.text.get_text(dataset, doc_field) >> mono
    results2 = pt.Experiment([bm25, bm25_n], reformed_queries, dataset.get_qrels(),
                             eval_metrics=['num_q', 'map', "P_10", 'recip_rank', R @ 1000, nDCG @ 10, nDCG @ 20,
                                           RR(rel=2), AP(rel=2)], names=["0-shot(BM25)", "0-shot(BM25)>>Neural"],
                             verbose=True, batch_size=24);
    dfs_to_concat.append(results2);
    save_this_here(dfs_to_concat, EVALUATION_NAME, prf_k=prf_k)
    if eval_all_instrucitons:
        for j in range(1, 10):
            reformed_queries_i = get_reformed_queries(queries, EVALUATION_NAME, i=j, prf_k=prf_k)
            results2 = pt.Experiment([bm25, bm25_n], reformed_queries_i, dataset.get_qrels(),
                                     eval_metrics=['num_q', 'map', "P_10", 'recip_rank', R @ 1000, nDCG @ 10, nDCG @ 20,
                                                   RR(rel=2), AP(rel=2)],
                                     names=[f"0-shot(BM25)-{j}", f"0-shot(BM25)-{j}>>Neural"], verbose=True, batch_size=24);
            dfs_to_concat.append(results2);
            save_this_here(dfs_to_concat, EVALUATION_NAME, prf_k=prf_k)
    # (2) ensemble baseline
    mono = MonoT5CustomReRanker(text_field=doc_field, query_field='query_0')
    reformed_queries = get_reformed_queries_ensemble(queries, EVALUATION_NAME, prf_k=prf_k)
    bm25_n = bm25 >> pt.text.get_text(dataset, doc_field) >> mono
    results2 = pt.Experiment([bm25, bm25_n], reformed_queries, dataset.get_qrels(),
                             eval_metrics=['num_q', 'map',"P_10", 'recip_rank', R @ 1000, nDCG @ 10, nDCG @ 20, RR(rel=2),
                                           AP(rel=2)], names=["Ensemble(BM25)", "Ensemble(BM25)>>Neural"], verbose=True,
                             batch_size=24);
    dfs_to_concat.append(results2);
    # (3) Document Fusion Baseline
    save_this_here(dfs_to_concat, EVALUATION_NAME, prf_k=prf_k, ensemble=True)
    return dfs_to_concat

def evaluate_with_and_without_generative_expansion_with_significance(EVALUATION_NAME, index_name, field, doc_field, prf_k=None, eval_all_instrucitons=False, context_first=True, gold=False, query_analysis=False):
    dfs_to_concat = []
    index, dataset, queries, corpus_iterator = get_index(EVALUATION_NAME, index_name, field, colbert=True)
    # (0) Raw Queries
    bm25 = get_bm25_pipe(index_name, index)
    mono2 = MonoT5CustomReRanker(text_field=doc_field, query_field='query_0')
    # # (1) 0-shot baseline
    if False:
        rm3_baseline = bm25 >> pt.rewrite.RM3(index, fb_docs=prf_k) >> bm25
        rm3_baseline1 = rm3_baseline
        bo1 = bm25 >> pt.rewrite.Bo1QueryExpansion(index, fb_docs=prf_k) >> bm25
        kl = bm25 >> pt.rewrite.KLQueryExpansion(index, fb_docs=prf_k) >> bm25
        rm3_baseline_n = bm25 >> pt.rewrite.RM3(index, fb_docs=prf_k) >> bm25 >> pt.text.get_text(dataset, doc_field) >> mono2
        results2 = pt.Experiment([rm3_baseline1, bo1, kl, rm3_baseline_n], queries, dataset.get_qrels(), perquery=query_analysis,
                                 eval_metrics=['num_q', 'map', "P_10", 'recip_rank', R @ 1000, nDCG @ 10, nDCG @ 20,
                                               RR(rel=2),
                                               AP(rel=2)], names=["RM3", "Bo1", "KL", "RM3>>Neural"], verbose=True,
                                 batch_size=24);
        dfs_to_concat.append(results2);
        save_this_here(dfs_to_concat, EVALUATION_NAME, prf_k=prf_k, suffix="_baselines_with_sig")
    mono1 = MonoT5CustomReRanker(text_field=doc_field, query_field='query')
    bm25_n = bm25 >> pt.text.get_text(dataset, doc_field) >> mono1
    reformed_queries = get_reformed_queries(queries,EVALUATION_NAME, i=0, prf_k=prf_k, gold=gold, return_map=True, context_first=context_first)
    bm25_rf = pt.apply.query(lambda row: reformed_queries[row.query]) >> bm25
    bm25_rf_n = bm25_rf >> pt.text.get_text(dataset, doc_field) >> mono2
    ensemble_queries = get_reformed_queries_ensemble(queries, EVALUATION_NAME, prf_k=prf_k, gold=gold, return_map=True, context_first=context_first)
    bm25_en = pt.apply.query(lambda row: ensemble_queries[row.query]) >> bm25
    bm25_en_n = bm25_en >> pt.text.get_text(dataset, doc_field) >> mono2
    # non-neural
    pp = f"PRF={prf_k}" if prf_k else ""
    pp = f"RF={prf_k}" if gold else pp
    sfx = "_query_analysis" if query_analysis else "_with_sig_b_cf"
    if query_analysis:
        results2 = pt.Experiment([bm25, bm25_rf, bm25_en], queries, dataset.get_qrels(),
                                 eval_metrics=['num_q', 'map', "P_10", 'recip_rank', R @ 1000, nDCG @ 10, nDCG @ 20,
                                               RR(rel=2), AP(rel=2)],
                                 names=["BM25", f"0-shot(BM25){pp}", f"PromptQR(BM25){pp}"],
                                 verbose=True, batch_size=24, perquery = query_analysis);
    else:
        results2 = pt.Experiment([bm25, bm25_rf, bm25_en], queries, dataset.get_qrels(),
                                 eval_metrics=['num_q', 'map', "P_10", 'recip_rank', R @ 1000, nDCG @ 10, nDCG @ 20,
                                               RR(rel=2), AP(rel=2)], names=["BM25",  f"0-shot(BM25){pp}", f"PromptQR(BM25){pp}"],
                                 verbose=True, baseline=1,batch_size=24, correction='bonferroni');
    dfs_to_concat.append(results2);
    save_this_here(dfs_to_concat, EVALUATION_NAME, prf_k=prf_k, gold=gold, suffix=sfx)
    # neural
    if query_analysis:
        results2 = pt.Experiment([bm25_n, bm25_rf_n, bm25_en_n], queries, dataset.get_qrels(),
                                 eval_metrics=['num_q', 'map', "P_10", 'recip_rank', R @ 1000, nDCG @ 10, nDCG @ 20,
                                               RR(rel=2), AP(rel=2)], names=["BM25>>Neural",
                                                                             f"0-shot(BM25){pp}>>Neural",
                                                                             f"PromptQR(BM25){pp}>>Neural"],
                                 verbose=True, batch_size=24, perquery = query_analysis);
    else:
        results2 = pt.Experiment([ bm25_n,  bm25_rf_n, bm25_en_n], queries, dataset.get_qrels(),
                                 eval_metrics=['num_q', 'map', "P_10", 'recip_rank', R @ 1000, nDCG @ 10, nDCG @ 20,
                                               RR(rel=2), AP(rel=2)], names=[ "BM25>>Neural",
                                                                             f"0-shot(BM25){pp}>>Neural",
                                                                             f"PromptQR(BM25){pp}>>Neural"],
                                 verbose=True, baseline=1, batch_size=24, perquery=query_analysis, correction='bonferroni');
    dfs_to_concat.append(results2);
    save_this_here(dfs_to_concat, EVALUATION_NAME, prf_k=prf_k, gold=gold, suffix=sfx)
    return dfs_to_concat

def evaluate_prf_special_with_significance(EVALUATION_NAME, index_name, field, doc_field, prf_k=None, eval_all_instrucitons=False):
    dfs_to_concat = []
    index, dataset, queries, corpus_iterator = get_index(EVALUATION_NAME, index_name, field, colbert=True)
    bm25 = get_bm25_pipe(index_name, index)
    mono2 = MonoT5CustomReRanker(text_field=doc_field, query_field='query_0')
    if prf_k:
        rm3_baseline = bm25 >> pt.rewrite.RM3(index, fb_docs=prf_k) >> bm25
        reformed_queries = get_reformed_queries(queries, EVALUATION_NAME, i=0, prf_k=prf_k, return_map=True, context_first=True)
        bm25_rf = pt.apply.query(lambda row: reformed_queries[row.query]) >> bm25
        ensemble_queries = get_reformed_queries_ensemble(queries, EVALUATION_NAME, prf_k=prf_k, return_map=True, context_first=True)
        bm25_en = pt.apply.query(lambda row: ensemble_queries[row.query]) >> bm25

    return dfs_to_concat


eval_metrics1 = ['num_q', 'map', "P_10", 'recip_rank', R @ 1000, nDCG @ 10, nDCG @ 20, RR(rel=2), AP(rel=2)]
def evaluate_topk_from_neural(EVALUATION_NAME, index_name, field, doc_field, prf_k=None, neural_reranker=False, enable_reformulation=False, rerank_via_reform=False, context_first=False):
    set_seed(0)
    dfs_to_concat = []
    index, dataset, queries, corpus_iterator = get_index(EVALUATION_NAME, index_name, field, colbert=True)
    bm25 = get_bm25_pipe(index_name, index)
    suffix = "_special_n2" if neural_reranker else "_special2"
    pp=""
    if enable_reformulation:
        queries = get_reformed_queries_ensemble(queries, EVALUATION_NAME) # query_0, query
        suffix = suffix + "promptQR_reformulated"
        if context_first:
            suffix += "_cf"
        pp="PromptQR>>"
        if rerank_via_reform:
            suffix += "_ref_reranked"
    assert prf_k is not None
    docno2doctext = {doc['docno']: doc[field] for doc in corpus_iterator()}
    # "Top-1 From Neural >> PromptPRF >> BM25"
    def ensemble_prf(search_results):
        x = search_results.sort_values(by='score', ascending=False)
        #print(f" search_results = \n{x}")
        q0 = list(x['query'].values)[0]
        top_docs = [docno2doctext[di].strip() for di in list(x['docno'].values)[0:prf_k]]
        assert len(top_docs) == prf_k
        context = "/".join([d.strip() for d in top_docs])
        q1 = q0
        for instruction in KEYWORD_GENERATION_INSTRUCTIONS:
            q1 += " " + zsq_prf2(q0, instruction, context, context_first=context_first)
        if enable_reformulation:
            x.rename(columns={'query': 'query_x'}, inplace=True)
        else:
            x.rename(columns={'query': 'query_0'}, inplace=True)
        #print(f"Reformulated Query = \n{q1}")
        x['query'] = q1
        x = x.drop(columns=["docid", "docno", 'rank', 'score'])
        x = x.head(1)
        return x
    mono = MonoT5CustomReRanker(text_field=doc_field, query_field='query')
    pipeline2 = bm25 >> pt.apply.by_query(ensemble_prf, add_ranks=False) >> bm25
    pipeline3 = bm25 >> pt.text.get_text(dataset, doc_field) >> mono >> pt.apply.by_query(ensemble_prf, add_ranks=False) >> bm25
    if neural_reranker:
        mono1 = MonoT5CustomReRanker(text_field=doc_field, query_field='query' if rerank_via_reform else 'query_0')
        pipeline3 = pipeline3 >> pt.text.get_text(dataset, doc_field) >> mono1
        pipeline2 = pipeline2 >> pt.text.get_text(dataset, doc_field) >> mono1
        results = pt.Experiment([pipeline2, pipeline3], queries, dataset.get_qrels(), eval_metrics1, [f"{pp}BM25(Top-{prf_k})>>PromptPRF>>(BM25+Neural)", f"{pp}BM25+Neural(Top-{prf_k})>>PromptPRF>>(BM25+Neural)"], verbose=True, batch_size=6);
    else:
        results = pt.Experiment([pipeline2, pipeline3], queries, dataset.get_qrels(), eval_metrics1, [f"{pp}BM25(Top-{prf_k})>>PromptPRF>>(BM25)", f"{pp}BM25+Neural(Top-{prf_k})>>PromptPRF>>(BM25)"], verbose=True, batch_size=6);
    dfs_to_concat.append(results)
    save_this_here(dfs_to_concat, EVALUATION_NAME, prf_k=prf_k, ensemble=True, suffix=suffix)
    return dfs_to_concat

def evaluate_special_pipelines(EVALUATION_NAME, index_name, field, doc_field, prf_k=None, enable_reformulation=False, rerank_via_reform=False, context_first=False):
    evaluate_topk_from_neural(EVALUATION_NAME, index_name, field, doc_field, prf_k, neural_reranker=False, enable_reformulation=enable_reformulation, rerank_via_reform=rerank_via_reform, context_first=context_first)
    evaluate_topk_from_neural(EVALUATION_NAME, index_name, field, doc_field, prf_k, neural_reranker=True, enable_reformulation=enable_reformulation, rerank_via_reform=rerank_via_reform, context_first=context_first)
    return None

def evaluate_doc_fusion(EVALUATION_NAME, index_name, field, doc_field, prf_k=None, s6=False, with_q0=False):
    dfs_to_concat = []
    index, dataset, queries, corpus_iterator = get_index(EVALUATION_NAME, index_name, field, colbert=True)
    bm25 = get_bm25_pipe(index_name, index)
    query_to_keyword_i = load_all_files(EVALUATION_NAME, s6=s6)
    # (0) doc fusion pipeline
    reciprocal_rank = pt.apply.doc_score(lambda row: (1 / (row["rank"] + 60)))
    ps = [(pt.apply.query(lambda row: (row.query + " " + query_to_keyword_i[i][row.query])) >> bm25) for i in range(10)]
    ok = [(pt.apply.query(lambda row: (query_to_keyword_i[i][row.query])) >> bm25) for i in range(10)]
    ps_r = [(ps[i] >> reciprocal_rank) for i in range(10)]
    ok_r = [(ok[i] >> reciprocal_rank) for i in range(10)]
    assert len(ps) == 10
    if not with_q0:
        doc_fusion_bm25 = (ps[0] + ps[1] + ps[2] + ps[3] + ps[4] + ps[5] + ps[6] + ps[7] + ps[8] + ps[9])
        doc_fusion_bm25_rr = (ps_r[0] + ps_r[1] + ps_r[2] + ps_r[3] + ps_r[4] + ps_r[5] + ps_r[6] + ps_r[7] + ps_r[8] + ps_r[9])
        doc_fusion_bm25_ok = (ok[0] + ok[1] + ok[2] + ok[3] + ok[4] + ok[5] + ok[6] + ok[7] + ok[8] + ok[9])
        doc_fusion_bm25_rr_ok = (ok_r[0] + ok_r[1] + ok_r[2] + ok_r[3] + ok_r[4] + ok_r[5] + ok_r[6] + ok_r[7] + ok_r[8] + ok_r[9])
        #doc_fusion_bm25_rr = ()
        results2 = pt.Experiment([doc_fusion_bm25, doc_fusion_bm25_rr, doc_fusion_bm25_ok, doc_fusion_bm25_rr_ok], queries, dataset.get_qrels(),
                                 eval_metrics=['num_q', 'map', 'P_10', 'recip_rank', R @ 1000, nDCG @ 10, nDCG @ 20,
                                               RR(rel=2), AP(rel=2)], names=["Document_Fusion", "DocumentFusion(ReciprocalRanked)",
                                                                             "DocumentFusionOnlyKeywords", "DocumentFusionOnlyKeywords(ReciprocalRanked)"], verbose=True,
                                 batch_size=24);
        dfs_to_concat.append(results2);
        save_this_here(dfs_to_concat, EVALUATION_NAME, prf_k=prf_k, suffix="_docfusion_s6" if s6 else "_docfusion")
    else:
        # (1) doc_fusion with q0
        bm25rr = pt.apply.query(lambda row: row.query) >> bm25 >> reciprocal_rank # so that another mono is not needed
        doc_fusion_bm25 = (bm25rr + ps[0] + ps[1] + ps[2] + ps[3] + ps[4] + ps[5] + ps[6] + ps[7] + ps[8] + ps[9])
        doc_fusion_bm25_rr = (bm25rr +
                    ps_r[0] + ps_r[1] + ps_r[2] + ps_r[3] + ps_r[4] + ps_r[5] + ps_r[6] + ps_r[7] + ps_r[8] + ps_r[9])
        doc_fusion_bm25_ok = (bm25rr + ok[0] + ok[1] + ok[2] + ok[3] + ok[4] + ok[5] + ok[6] + ok[7] + ok[8] + ok[9])
        doc_fusion_bm25_rr_ok = (bm25rr +
                    ok_r[0] + ok_r[1] + ok_r[2] + ok_r[3] + ok_r[4] + ok_r[5] + ok_r[6] + ok_r[7] + ok_r[8] + ok_r[9])
        # doc_fusion_bm25_rr = ()
        results2 = pt.Experiment([doc_fusion_bm25, doc_fusion_bm25_rr, doc_fusion_bm25_ok, doc_fusion_bm25_rr_ok], queries,
                                 dataset.get_qrels(),
                                 eval_metrics=['num_q', 'map', 'P_10', 'recip_rank', R @ 1000, nDCG @ 10, nDCG @ 20,
                                               RR(rel=2), AP(rel=2)],
                                 names=["Document_Fusion_withq0", "DocumentFusion_withq0(ReciprocalRanked)",
                                        "DocumentFusionOnlyKeywords_withq0", "DocumentFusionOnlyKeywords_withq0(ReciprocalRanked)"],
                                 verbose=True,
                                 batch_size=24);
        dfs_to_concat.append(results2);
        save_this_here(dfs_to_concat, EVALUATION_NAME, prf_k=prf_k, suffix="_docfusion_s6_with_q0" if s6 else "_docfusion_with_q0")
    # (2) doc fusion with mono-reranker
    mono = MonoT5CustomReRanker(text_field=doc_field, query_field='query_0')
    pipeline2 = doc_fusion_bm25 >> pt.text.get_text(dataset, doc_field) >> mono
    pipeline3 = doc_fusion_bm25_rr >> pt.text.get_text(dataset, doc_field) >> mono
    pipeline4 = doc_fusion_bm25_ok >> pt.text.get_text(dataset, doc_field) >> mono
    pipeline5 = doc_fusion_bm25_rr_ok >> pt.text.get_text(dataset, doc_field) >> mono
    results2 = pt.Experiment([pipeline2, pipeline3, pipeline4, pipeline5], queries, dataset.get_qrels(),
                             eval_metrics=['num_q', 'map','P_10', 'recip_rank', R @ 1000, nDCG @ 10, nDCG @ 20, RR(rel=2),
                                           AP(rel=2)], names=["Document_Fusion>>Neural",
                                                              "DocumentFusion(ReciprocalRanked)>>Neural",
                                                              "DocumentFusionOnlyKeywords(ReciprocalRanked)>>Neural",
                                                              "DocumentFusionOnlyKeywords>>Neural"], verbose=True, batch_size=24);
    dfs_to_concat.append(results2);
    suffix = ("_docfusion_s6" if s6 else "_docfusion")
    suffix = suffix + ("_with_q0" if with_q0 else "")
    save_this_here(dfs_to_concat, EVALUATION_NAME, prf_k=prf_k, suffix=suffix)
    return dfs_to_concat


def evaluate_generative_prf_models(EVALUATION_NAME, index_name, field, doc_field, beta=0.2):
    set_seed(0)
    print("Running evaluate_zero_shot_prompt_baseline")
    index, dataset, queries, corpus_iterator = get_index(EVALUATION_NAME, index_name, field, colbert=True)
    bm25 = get_bm25_pipe(index_name, index)
    docno2doctext = {doc['docno']: doc[field] for doc in corpus_iterator()}
    # Custom PRF searcher
    def prompt_model1(q0, docs, fn=zsq_prf):
        # Some logic here
        context = "/".join([d.strip() for d in docs])
        return fn(q0, KEYWORD_GENERATION_INSTRUCTIONS[0], context, setting=1)
    def prf_func1(search_results):
        x = search_results
        print(f" prf_func1 search_results = \n{search_results}")
        q0 = list(x['query'].values)[0]
        top_docs = [docno2doctext[di].strip() for di in list(x['docno'].values)]
        q1 = prompt_model1(q0, top_docs)
        x.rename(columns={'query': 'query_0'}, inplace=True)
        x['query'] = q1
        x = x.drop(columns=["docid", "docno", 'rank', 'score'])
        x = x.head(1)
        return x
    # (0) Baselines
    rm3_baseline1 = bm25 >> pt.rewrite.RM3(index, fb_docs=1) >> bm25
    #ks = [1, 2, 5, 10]
    ks = [2]
    zero_shot_prfs = [(bm25 % k >> pt.apply.by_query(prf_func1, add_ranks=False) >> bm25) for k in ks]
    # (1) Unified Query Setting
    def prompt_model2(q0, docs):
        # Some logic here
        context = "/".join([d.strip() for d in docs])
        return zsq_combo_prf(q0, KEYWORD_GENERATION_INSTRUCTIONS, context, setting=5, prompt_style=2)
    def prf_func2(search_results):
        x = search_results
        print(f" prf_func1 search_results = \n{search_results}")
        q0 = list(x['query'].values)[0]
        top_docs = [docno2doctext[di].strip() for di in list(x['docno'].values)]
        q1 = prompt_model2(q0, top_docs)
        x.rename(columns={'query': 'query_0'}, inplace=True)
        x['query'] = q1
        x = x.drop(columns=["docid", "docno", 'rank', 'score'])
        x = x.head(1)
        return x
    unified_query_pipelines = [(bm25 % k >> pt.apply.by_query(prf_func2, add_ranks=False) >> bm25) for k in ks]
    # (2) Document Fusion Setting
    def prompt_model3(q0, docs):
        # Some logic here
        context = "/".join([d.strip() for d in docs])
        return zsq_prf(q0, instruction, context, setting=5)
    def prf_func3(search_results):
        x = search_results
        print(f" prf_func1 search_results = \n{search_results}")
        q0 = list(x['query'].values)[0]
        top_docs = [docno2doctext[di].strip() for di in list(x['docno'].values)]
        q1 = prompt_model3(q0, top_docs)
        x.rename(columns={'query': 'query_0'}, inplace=True)
        x['query'] = q1
        x = x.drop(columns=["docid", "docno", 'rank', 'score'])
        x = x.head(1)
        return x
    doc_fusion_pipelines = []
    for k in ks:
        ps = []
        for instruction in KEYWORD_GENERATION_INSTRUCTIONS:
            p1 = (bm25 % k >> pt.apply.by_query(prf_func3, add_ranks=False) >> bm25)
            ps.append(p1)
        assert len(ps) == 10
        pipeline = ps[0] + ps[1] + ps[2] + ps[3] + ps[4] + ps[5] + ps[6] + ps[7] + ps[8] + ps[9]
        doc_fusion_pipelines.append(pipeline)
    dfs_to_concat = []
    results = pt.Experiment(
        [rm3_baseline1],
        queries,
        dataset.get_qrels(),
        # eval_metrics=EVAL_METRICS,
        eval_metrics=['num_q', 'map', 'recip_rank', R @ 1000, nDCG @ 10, nDCG @ 20],
        names=["RM3_fb2",],
        verbose=True,
        batch_size=1,
    );
    dfs_to_concat.append(results); save_this(dfs_to_concat, index_name, "standalone_prf")
    for j in range(len(ks)):
        k = ks[j]
        results = pt.Experiment(
            [zero_shot_prfs[j]] + [unified_query_pipelines[j]] + [doc_fusion_pipelines[j]],
            queries,
            dataset.get_qrels(),
            # eval_metrics=EVAL_METRICS,
            eval_metrics=['num_q', 'map', 'recip_rank', R @ 1000, nDCG @ 10, nDCG @ 20],
            names=[f"ZeroShot_fb_{k}"] + [f"UnifiedQueryPRF_fb_{k}"] + [f"DocumentFusionPRF_fb_{k}"],
            verbose=True,
            batch_size=1,
            baseline=0,  # zeroshot
            correction='bonferroni'
        );
        dfs_to_concat.append(results); save_this(dfs_to_concat, index_name, "standalone_prf")
    print(results)
    return results

def evaluate_generative_prf_models_with_monoreranker(EVALUATION_NAME, index_name, field, doc_field, beta=0.2):
    set_seed(0)
    print("Running evaluate_zero_shot_prompt_baseline")
    index, dataset, queries, corpus_iterator = get_index(EVALUATION_NAME, index_name, field, colbert=True)
    queries['text'] = queries['query']
    docno2doctext = {doc['docno']: doc[field] for doc in corpus_iterator()}
    monoT5 = MonoT5ReRanker(text_field=doc_field)
    # Custom PRF searcher
    def prompt_model1(q0, docs, fn=zsq_prf):
        # Some logic here
        context = "/".join([d.strip() for d in docs])
        return fn(q0, KEYWORD_GENERATION_INSTRUCTIONS[0], context, setting=1)
    def prf_func1(search_results):
        x = search_results
        q0 = list(x['query'].values)[0]
        top_docs = [docno2doctext[di].strip() for di in list(x['docno'].values)]
        q1 = prompt_model1(q0, top_docs)
        x.rename(columns={'query': 'query_0'}, inplace=True)
        x['query'] = q1
        x = x.drop(columns=["docid", "docno", 'rank', 'score'])
        x = x.head(1)
        return x
    bm25 = get_bm25_pipe(index_name, index)
    # (0) Baselines
    rm3_baseline1 = bm25 >> pt.rewrite.RM3(index, fb_docs=2) >> bm25 >> pt.text.get_text(dataset, doc_field) >> monoT5# default params of pyterrier
    #rm3_baseline2 = bm25 >> pt.rewrite.RM3(index, fb_terms=20, fb_docs=10) >> bm25 >> pt.text.get_text(dataset, field) >> monoT5 # default params of pyterrier
    #ks = [1, 2, 5, 10]
    ks = [2]
    zero_shot_prfs = [(bm25 % k >> pt.apply.by_query(prf_func1, add_ranks=False) >> bm25 >> pt.text.get_text(dataset, doc_field) >> monoT5) for k in ks]
    # (1) Unified Query Setting
    def prompt_model2(q0, docs):
        # Some logic here
        context = "/".join([d.strip() for d in docs])
        return zsq_combo_prf(q0, KEYWORD_GENERATION_INSTRUCTIONS, context, setting=5, prompt_style=2)
    def prf_func2(search_results):
        x = search_results
        q0 = list(x['query'].values)[0]
        top_docs = [docno2doctext[di].strip() for di in list(x['docno'].values)]
        q1 = prompt_model2(q0, top_docs)
        x.rename(columns={'query': 'query_0'}, inplace=True)
        x['query'] = q1
        x = x.drop(columns=["docid", "docno", 'rank', 'score'])
        x = x.head(1)
        return x
    unified_query_pipelines = [(bm25 % k >> pt.apply.by_query(prf_func2, add_ranks=False) >> bm25 >> pt.text.get_text(dataset, doc_field) >> monoT5) for k in ks]
    # (2) Document Fusion Setting
    def prompt_model3(q0, docs):
        # Some logic here
        context = "/".join([d.strip() for d in docs])
        return zsq_prf(q0, instruction, context, setting=5)
    def prf_func3(search_results):
        x = search_results
        q0 = list(x['query'].values)[0]
        top_docs = [docno2doctext[di].strip() for di in list(x['docno'].values)]
        q1 = prompt_model3(q0, top_docs)
        x.rename(columns={'query': 'query_0'}, inplace=True)
        x['query'] = q1
        x = x.drop(columns=["docid", "docno", 'rank', 'score'])
        x = x.head(1)
        return x
    doc_fusion_pipelines = []
    for k in ks:
        ps = []
        for instruction in KEYWORD_GENERATION_INSTRUCTIONS:
            p1 = (bm25 % k >> pt.apply.by_query(prf_func3, add_ranks=False) >> bm25)
            ps.append(p1)
        assert len(ps) == 10
        pipeline = ps[0] + ps[1] + ps[2] + ps[3] + ps[4] + ps[5] + ps[6] + ps[7] + ps[8] + ps[9]
        pipeline = pipeline >> pt.text.get_text(dataset, doc_field) >> monoT5
        doc_fusion_pipelines.append(pipeline)
    dfs_to_concat = []
    results = pt.Experiment(
        [rm3_baseline1],
        queries,
        dataset.get_qrels(),
        # eval_metrics=EVAL_METRICS,
        eval_metrics=['num_q', 'map', 'recip_rank', R @ 1000, nDCG @ 10, nDCG @ 20],
        names=["RM3_fb3_mt"],
        verbose=True,
        batch_size=1,
    );
    dfs_to_concat.append(results); save_this(dfs_to_concat, index_name, "monot5ranked_prf")
    for j in range(len(ks)):
        k = ks[j]
        results = pt.Experiment(
            [zero_shot_prfs[j]] + [unified_query_pipelines[j]] + [doc_fusion_pipelines[j]],
            queries,
            dataset.get_qrels(),
            # eval_metrics=EVAL_METRICS,
            eval_metrics=['num_q', 'map', 'recip_rank', R @ 1000, nDCG @ 10, nDCG @ 20],
            names=[f"ZeroShot_fb_{k}_mt"] + [f"UnifiedQueryPRF_fb_{k}_mt"]+ [f"DocumentFusionPRF_fb_{k}_mt"],
            verbose=True,
            batch_size=1,
            baseline=0,  # zeroshot
            correction='bonferroni'
        );
        dfs_to_concat.append(results); save_this(dfs_to_concat, index_name, "monot5ranked_prf")
    print(results)
    return results

# For Query Reformulation (without context)
KEYWORD_GENERATION_INSTRUCTIONS = [
"Improve the search effectiveness by suggesting expansion terms for the query",
"Recommend expansion terms for the query to improve search results",
"Improve the search effectiveness by suggesting useful expansion terms for the query",
"Maximize search utility by suggesting relevant expansion phrases for the query",
"Enhance search efficiency by proposing valuable terms to expand the query",
"Elevate search performance by recommending relevant expansion phrases for the query",
"Boost the search accuracy by providing helpful expansion terms to enrich the query",
"Increase the search efficacy by offering beneficial expansion keywords for the query",
"Optimize search results by suggesting meaningful expansion terms to enhance the query",
"Enhance search outcomes by recommending beneficial expansion terms to supplement the query"
]

def read_qr_pairs():
    with open("/local/scratch/kdhole/prompt-prf/experiment_fall23/qr_pairs.tsv", "r+") as f:
        sps = [sp.split("\t") for sp in f.read().strip().split("\n")]
        return sps

def save_setting(m=""):
    combined_df = pd.concat(dfs_to_concat, ignore_index=True)
    #combined_df = combined_df.applymap(lambda x: f'{x:.2f}'.replace('.', ','))
    combined_df.to_csv(f'/local/scratch/kdhole/prompt-prf/experiment_fall23/results_{index_name}/setting_{SETTING}{m}.csv', index=False)

import tqdm

def create_and_save_new_reformulations_2(EVALUATION_NAME, index_name, field, setting=None, enable_few_shot=False, kfs=5):
    set_seed(0)
    index, dataset, queries, corpus_iterator = get_index(EVALUATION_NAME, index_name, field, True)
    # (0) store original queries
    eval_name = EVALUATION_NAME.replace(":","_").replace("/","_").replace("_irds","")
    eval_folder = f"{scratchpath}/prompt-prf/experiment_fall23/expansions_{eval_name}"
    if IS_ALTERNATE_MODEL:
        eval_folder = f"{scratchpath}/prompt-prf/experiment_fall23/{ONLY_MODEL_NAME}/expansions_{eval_name}"
    if not os.path.exists(eval_folder):
        os.mkdir(eval_folder)
    # (1) compute ZSQ keywords for each instruction
    for i in range(10):
        query_to_keyword = {}
        instruction = KEYWORD_GENERATION_INSTRUCTIONS[i]
        j = 0
        for _, query in tqdm.tqdm(queries['query'].iteritems()):
            print(f"i={i} and query number {j}"); j +=1
            if setting is None:
                if enable_few_shot:
                    keywords = zsq_3(query, instruction, kfs=5)
                else:
                    keywords = zsq_2(query, instruction)
            else:
                keywords = zs_qr_xxl(query, instruction, only_keywords=True ,setting=setting)
            query_to_keyword[query] = keywords
            print(f"\tQuery: {query} Generated Keywords: {keywords}")
        s = f"s{setting}_" if setting else ""
        fs = f"fs{kfs}_" if enable_few_shot else ""
        file_name = f'{eval_folder}/{fs}{s}keywords_zsq_i{i}.json'
        import json
        with open(file_name, "w") as outfile:
            json.dump(query_to_keyword, outfile, indent=4)

def create_and_save_new_reformulations_prf2(EVALUATION_NAME, index_name, field, gold=False, context_first=False):
    set_seed(0)
    index, dataset, queries, corpus_iterator = get_index(EVALUATION_NAME, index_name, field, True)
    bm25 = get_bm25_pipe(index_name, index)
    docno2doctext = {doc['docno']: doc[field] for doc in corpus_iterator()}
    def prf_func1(search_results, instruction, docno2doctext, k):
        x = search_results.sort_values(by='score', ascending=False)
        #print(f"Search results = {x}")
        q0 = list(x['query'].values)[0]
        top_docs = [docno2doctext[di].strip() for di in list(x['docno'].values)[0:k]]
        context = "/".join([d.strip() for d in top_docs])
        #print(f"topk-context = {context}")
        q1 = zsq_prf2(q0, instruction, context, context_first=context_first)
        return q1
    if gold:
        K = 5 # just the max out of 1,2,5
        all_qrels = dataset.get_qrels()
        all_qrels_sorted = all_qrels.sort_values(by=['qid', 'label'], ascending=[True, False])
        query2doc = {}
        for qid, group in all_qrels_sorted.groupby('qid'):
            top_k_docnos = group['docno'].head(K).tolist()
            query = queries[queries.qid==qid]['query'].values[0]
            query2doc[query] = top_k_docnos
        def prf_gold_func(q, instruction, docno2doctext, k):
            top_docs = [docno2doctext[di].strip() for di in query2doc[q][0:k]]
            context = "/".join([d.strip() for d in top_docs])
            print(f"Feeding context = {context} for query {q}")
            # TODO: redo Gold experiments: wrong input sent below
            q1 = zsq_prf2(q, instruction, context, context_first=context_first)
            return q1
    # (0) store original queries
    eval_name = EVALUATION_NAME.replace(":","_").replace("/","_").replace("_irds","")
    eval_folder = f"{scratchpath}/prompt-prf/experiment_fall23/expansions_{eval_name}"
    if IS_ALTERNATE_MODEL:
        eval_folder = f"{scratchpath}/prompt-prf/experiment_fall23/{ONLY_MODEL_NAME}/expansions_{eval_name}"
    if not os.path.exists(eval_folder):
        os.mkdir(eval_folder)
    # (1) compute ZSQ keywords for each instruction
    for k in [4]:
        for i in range(10):
            query_to_keyword = {}
            instruction = KEYWORD_GENERATION_INSTRUCTIONS[i]
            j = 0
            for _, query in tqdm.tqdm(queries['query'].iteritems()):
                print(f"i={i} and query number {j} k = {k}"); j +=1
                if gold:
                    keywords = prf_gold_func(query, instruction, docno2doctext, k)
                else:
                    keywords = prf_func1(bm25.search(query), instruction, docno2doctext, k)
                query_to_keyword[query] = keywords
            sfx = "_gold" if gold else ""
            prf_folder = f'{eval_folder}/prf_{k}'
            cf = "_cf" if context_first else ""
            file_name = prf_folder + f'/keywords_zsq_s0{cf}_prf_i{i}{sfx}.json'
            if not os.path.exists(prf_folder):
                os.mkdir(prf_folder)
            import json
            with open(file_name, "w") as outfile:
                json.dump(query_to_keyword, outfile, indent=4)

def colbert_over_reformulations(SETTING, EVALUATION_NAME, index_name):
    index, dataset, original_queries = get_index(EVALUATION_NAME, index_name)

import pandas as pd

triplets = [
['irds:msmarco-passage/trec-dl-2019/judged',  'msmarco_passage', 'text', 'text'],
['irds:disks45/nocr/trec-robust-2004', "disks45_nocr_kd", "title", "body"],
["irds:beir/webis-touche2020/v2", "beir_webis-touche2020_v2", "text", "text"],
["irds:beir/dbpedia-entity/test", "beir_dbpedia-entity", 'text', 'text'],
["irds:cord19/trec-covid", "trec-covid", 'text', 'abstract'],]

ENABLE_FEW_SHOT=False
MODEL_NAME="google/flan-t5-xxl"
IS_ALTERNATE_MODEL = MODEL_NAME != "google/flan-t5-xxl"

if __name__ == '__main__':
#if True:
    MODEL_NAME="google/flan-t5-xxl"
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, device_map='auto')
    #MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
    #ONLY_MODEL_NAME=MODEL_NAME.split("/")[1]
    IS_ALTERNATE_MODEL = MODEL_NAME != "google/flan-t5-xxl"
    ENABLE_FEW_SHOT=False
    # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map='auto');
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # if IS_ALTERNATE_MODEL:
    #     tokenizer.pad_token = "[PAD]"
    #     tokenizer.padding_side = "left"
    zs_qr_xxl, fs_qr_xxl, zsq_reform, fsq_reform, zsq_prf, zsq_keyword_combo, zsq_combo_prf, zsq_2, zsq_prf2, zsq_3 = load_zs_query_reformulator(model, tokenizer)
    #for triplet in triplets[0:1]:
        # KFS = 5
        #ENABLE_FEW_SHOT = KFS > 0
        #create_and_save_new_reformulations_2(triplet[0], triplet[1], triplet[2],enable_few_shot=ENABLE_FEW_SHOT, kfs=KFS)
        #create_and_save_new_reformulations_prf2(triplet[0], triplet[1], triplet[2], context_first=True, gold=True)
        #create_and_save_new_reformulations_prf2(triplet[0], triplet[1], triplet[2], context_first=True)
        #evaluate_with_and_without_generative_expansion_with_significance(triplet[0], triplet[1], triplet[2], triplet[3])
        # ENABLE_FEW_SHOT = False
        # evaluate_with_and_without_generative_expansion(triplet[0], triplet[1], triplet[2], triplet[3],
        #                                                eval_all_instrucitons=True)
        #evaluate_with_and_without_generative_expansion(triplet[0], triplet[1], triplet[2], triplet[3])
# TODO: pass appripriate field in monoT5

