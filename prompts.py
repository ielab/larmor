# PROMPT are triples: (system prompt, user prompt, system response)
PROMPT_LLAMA = {
    'beir/trec-covid': ('You are a question generator.\n',
                        'Generate a short question that the following scientific paper can answer.\n'
                        'Paper: {title} {text}\n'
                        'Do not provide reasons; just generate the question.',
                        'Question:'),

    'beir/nfcorpus': ('You are a question generator.\n',
                      'Generate a short question that the following scientific paper can answer.\n'
                      'Paper: {title} {text}\n'
                      'Do not provide reasons; just generate the question.',
                      'Question:'),
}

ICL_PROMPT_LLAMA = {
    'beir/trec-covid': ('You are a question generator.\n'
                        'Your generated questions might look like the following:\n'
                        '{queries}',
                        'Generate a question that the following scientific paper can answer.\n'
                        'Paper: {title} {text}\n'
                        'Do not provide reasons; just generate the question.',
                        'Question:'),

    'beir/nfcorpus': ('You are a question generator. Your generated questions might look like the following:\n'
                      '{queries}',
                      'Generate a question that the following scientific paper can answer.\n'
                      'Paper: {title} {text}\n'
                      'Do not provide reasons; just generate the question.',
                      'Question:'),
}

PROMPT_QG_FLAN_T5 = {
    'beir/robust04': {
        'system': '',
        'user': 'Generate a question that the following news article can answer. Avoid generating general questions.\nArticle: {title} {text}\n',
        'assistant': 'Question:'
    },

    'beir/trec-news': {
        'system': '',
        'user': 'Generate a headline for the following news article.\nArticle: {title} {text}\n',
        'assistant': 'Headline:'
    },

    'beir/signal1m': {
        'system': '',
        'user': 'Generate a news article title for the following tweet.\nTweet: {title} {text}\n',
        'assistant': 'Title:'
    },

    'beir/trec-covid': {
        'system': '',
        'user': 'Generate a question that the following scientific paper can answer. Avoid generating general questions.\nPaper: {title} {text}\n',
        'assistant': 'Question:'
    },

    'beir/nfcorpus': {
        'system': '',
        'user': 'Generate a question that the following scientific paper can answer. Avoid generating general questions.\nPaper: {title} {text}\n',
        'assistant': 'Question:'
    },

    'beir/fiqa': {
        'system': '',
        'user': 'Generate a question that the following financial web article can answer. Avoid generating general questions.\nArticle: {title} {text}\n',
        'assistant': 'Question:'
    },

    'beir/arguana': {
        'system': '',
        'user': 'Generate an argument that counters the following argument.\nArgument: {title} {text}\n',
        'assistant': 'Counter argument:'
    },

    'beir/scidocs': {
        'system': '',
        'user': 'Generate a scientific paper title that is related to the following paper.\nPaper: {title} {text}\n',
        'assistant': 'Title:'
    },

    'beir/scifact': {
        'system': '',
        'user': 'Generate a claim that is supported by the following scientific evidence.\nEvidence: {title} {text}\n',
        'assistant': 'Claim:'
    },

    'beir/quora': {
        'system': '',
        'user': 'Generate a question that differs from the following question but essentially asks the same thing.\nQuestion: {title} {text}\n',
        'assistant': 'An equivalent question:'
    },

    'beir/nq': {
        'system': '',
        'user': 'Generate a question that the following Wikipedia page can answer. Avoid generating general questions.\nWikipedia page: {title} {text}\n',
        'assistant': 'Question:'
    },

    'beir/hotpotqa': {
        'system': '',
        'user': 'Generate a question that the following passage can answer. Avoid generating general questions.\nPassage: {title} {text}\n',
        'assistant': 'Question:'
    },

    'beir/dbpedia-entity': {
        'system': '',
        'user': 'Generate an entity-based query that the following Wikipedia introduction paragraph can answer. Avoid generating general queries.\nWikipedia paragraph: {title} {text}\n',
        'assistant': 'Query:'
    },
}

PROMPT_POINTWISE_FLAN_T5 = {
    'beir/robust04': {
        'system': '',
        'user': "For the following question and news article, judge whether they are 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant'.\nQuestion: {query}\nArticle: {title} {text}",
        'assistant': 'Judgement:',
        'key': 'Highly Relevant'
    },

    'beir/trec-news': {
        'system': '',
        'user': "For the following headline and news article, judge whether they are 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant'.\nHeadline: {query}\nArticle: {title} {text}",
        'assistant': 'Judgement:',
        'key': 'Highly Relevant',
    },

    'beir/signal1m': {
        'system': '',
        'user': "For the following news article title and tweet, judge whether they are 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant'.\nTitle: {query}\nTweet: {title} {text}",
        'assistant': 'Judgement:',
        'key': 'Highly Relevant'
    },

    'beir/trec-covid': {
        'system': '',
        'user': "For the following query and document, judge whether they are 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant'.\nQuery: {query}\nDocument: {title} {text}",
        'assistant': 'Judgement:',
        'key': 'Highly Relevant'
    },

    'beir/nfcorpus': {
        'system': '',
        'user': "For the following query and document, judge whether they are 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant'.\nQuery: {query}\nDocument: {title} {text}",
        'assistant': 'Judgement:',
        'key': 'Highly Relevant'
    },

    'beir/fiqa': {
        'system': '',
        'user': "For the following query and document, judge whether they are 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant'.\nQuery: {query}\nDocument: {title} {text}",
        'assistant': 'Judgement:',
        'key': 'Highly Relevant'
    },

    'beir/arguana': {
        'system': '',
        'user': "For the following two arguments, judge whether the Argument 1 is 'Similar', 'Somewhat Similar', or 'Counter Argument' to the Argument 2.\nArgument 1: {query}\nArgument 2: {title} {text}",
        'assistant': 'Judgement:',
        'key': 'Counter Argument'
    },

    'beir/scidocs': {
        'system': '',
        'user': "For the following paper title and document, judge whether they are 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant'.\nTitle: {query}\nDocument: {title} {text}",
        'assistant': 'Judgement:',
        'key': 'Highly Relevant'
    },

    'beir/scifact': {
        'system': '',
        'user': "For the following claim and scientific evidence, judge whether the evidence is 'Supports', 'Partially Supports', or 'Not Support' the claim.\nClaim: {query}\nEvidence: {title} {text}",
        'assistant': 'Judgement:',
        'key': 'Supports'
    },

    'beir/quora': {
        'system': '',
        'user': "For the following two questions, judge whether they are asking the same thing. Answer with 'Same', 'Partially Same', or 'Not Same'.\nQuestion 1: {query}\nQuestion 2: {title} {text}",
        'assistant': 'Judgement:',
        'key': 'Same'
    },

    'beir/nq': {
        'system': '',
        'user': "For the following query and document, judge whether they are 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant'.\nQuery: {query}\nDocument: {title} {text}",
        'assistant': 'Judgement:',
        'key': 'Highly Relevant'
    },

    'beir/hotpotqa': {
        'system': '',
        'user': "For the following query and document, judge whether they are 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant'.\nQuery: {query}\nDocument: {title} {text}",
        'assistant': 'Judgement:',
        'key': 'Highly Relevant'
    },

    'beir/dbpedia-entity': {
        'system': '',
        'user': "For the following query and document, judge whether they are 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant'.\nQuery: {query}\nDocument: {title} {text}",
        'assistant': 'Judgement:',
        'key': 'Highly Relevant'
    },
}


PROMPT_QG_OpenAI = {
    'beir/robust04': {
        'system': 'You are a question generator. Your generated questions should be in JSON format with the following structure: {"generated_question": "xxx"}.',
        'user': 'Generate a question that the following news article can answer. Avoid generating general questions.\nArticle: {title} {text}\n',
        'key': "generated_question"
    },

    'beir/trec-news': {
        'system': 'You are a news article headline generator. Your generated headlines should be in JSON format with the following structure: {"headline": "xxx"}.',
        'user': 'Generate a headline for the following news article.\nArticle: {title} {text}\n',
        'key': "headline"
    },

    'beir/signal1m': {
        'system': 'You are a news article title generator. Your generated titles should be in JSON format with the following structure: {"title": "xxx"}.',
        'user': 'Generate a news article title for the following tweet.\nTweet: {title} {text}\n',
        'key': "title"
    },

    'beir/trec-covid': {
        'system': 'You are a question generator. Your generated questions should be in JSON format with the following structure: {"generated_question": "xxx"}.',
        'user': 'Generate a question that the following scientific paper can answer. Avoid generating general questions.\nPaper: {title} {text}\n',
        'key': "generated_question"
    },

    'beir/nfcorpus': {
        'system': 'You are a question generator. Your generated questions should be in JSON format with the following structure: {"generated_question": "xxx"}.',
        'user': 'Generate a question that the following scientific paper can answer. Avoid generating general questions.\nPaper: {title} {text}\n',
        'key': "generated_question"
    },

    'beir/fiqa': {
        'system': 'You are a question generator. Your generated questions should be in JSON format with the following structure: {"generated_question": "xxx"}.',
        'user': 'Generate a question that the following financial web article can answer. Avoid generating general questions.\nArticle: {title} {text}\n',
        'key': "generated_question"
    },

    'beir/arguana': {
        'system': 'You are a counter argument generator. Your generated counter arguments should be in JSON format with the following structure: {"counter_argument": "xxx"}.',
        'user': 'Generate an argument that counters the following argument.\nArgument: {title} {text}\n',
        'key': "counter_argument"
    },

    'beir/scidocs': {
        'system': 'You are a paper title generator. Your generated titles should be in JSON format with the following structure: {"generated_title": "xxx"}.',
        'user': 'Generate a scientific paper title that is related to the following paper.\nPaper: {title} {text}\n',
        'key': "generated_title"
    },

    'beir/scifact': {
        'system': 'You are a claim generator. Your generated claims should be in JSON format with the following structure: {"generated_claim": "xxx"}.',
        'user': 'Generate a claim that is supported by the following scientific evidence.\nEvidence: {title} {text}\n',
        'key': "generated_claim"
    },

    'beir/quora': {
        'system': 'You are a similar question generator. Your generated questions should be in JSON format with the following structure: {"equivalent_question": "xxx"}.',
        'user': 'Generate a question that differs from the following question but essentially asks the same thing.\nQuestion: {title} {text}\n',
        'key': "equivalent_question"
    },

    'beir/nq': {
        'system': 'You are a question generator. Your generated questions should be in JSON format with the following structure: {"generated_question": "xxx"}.',
        'user': 'Generate a question that the following Wikipedia page can answer. Avoid generating general questions.\nWikipedia page: {title} {text}\n',
        'key': "generated_question"
    },

    'beir/hotpotqa': {
        'system': 'You are a question generator. Your generated questions should be in JSON format with the following structure: {"generated_question": "xxx"}.',
        'user': 'Generate a question that the following passage can answer. Avoid generating general questions.\nPassage: {title} {text}\n',
        'key': "generated_question"
    },

    'beir/dbpedia-entity': {
        'system': 'You are a query generator. Your generated queries should be in JSON format with the following structure: {"generated_query": "xxx"}.',
        'user': 'Generate an entity-based query that the following Wikipedia introduction paragraph can answer. Avoid generating general queries.\nWikipedia paragraph: {title} {text}\n',
        'key': "generated_query"
    },
}


PROMPT_POINTWISE_OPENAI = {
    'beir/trec-covid': {
        'system': "You are RankGPT, an intelligent assistant specialized in judging whether a query and a document are 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant' to each other. Your generated judgement should be in JSON format with the following structure: {'judgement': 'xxx'}, where 'xxx' is one of 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant'.",
        'user': "Please judge the following query and document:\nQuery: {query}\nDocument: {title} {text}",
        'key': 'Highly Relevant'
    },

    'beir/trec-nfcorpus': {
        'system': "You are RankGPT, an intelligent assistant specialized in judging whether a query and a document are 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant' to each other. Your generated judgement should be in JSON format with the following structure: {'judgement': 'xxx'}, where 'xxx' is one of 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant'.",
        'user': "Please judge the following query and document:\nQuery: {query}\nDocument: {title} {text}",
        'key': 'Highly Relevant'
    },

    'beir/fiqa': {
        'system': "You are RankGPT, an intelligent assistant specialized in judging whether a query and a document are 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant' to each other. Your generated judgement should be in JSON format with the following structure: {'judgement': 'xxx'}, where 'xxx' is one of 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant'.",
        'user': "Please judge the following query and document:\nQuery: {query}\nDocument: {title} {text}",
        'key': 'Highly Relevant'
    },

    # 'beir/arguana': {
    #     'system': '',
    #     'user': "For the following two arguments, judge whether the Argument 1 is 'Similar', 'Somewhat Similar', or 'Counter Argument' to the Argument 2.\nArgument 1: {query}\nArgument 2: {title} {text}",
    #     'assistant': 'Judgement:',
    #     'key': 'Counter Argument'
    # },
    #
    # 'beir/scidocs': {
    #     'system': '',
    #     'user': "For the following paper title and document, judge whether they are 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant'.\nTitle: {query}\nDocument: {title} {text}",
    #     'assistant': 'Judgement:',
    #     'key': 'Highly Relevant'
    # },

    # 'beir/scifact': {
    #     'system': '',
    #     'user': "For the following claim and scientific evidence, judge whether the evidence is 'Supports', 'Partially Supports', or 'Not Support' the claim.\nClaim: {query}\nEvidence: {title} {text}",
    #     'assistant': 'Judgement:',
    #     'key': 'Supports'
    # },
    #
    # 'beir/quora': {
    #     'system': '',
    #     'user': "For the following two questions, judge whether they are asking the same thing. Answer with 'Same', 'Partially Same', or 'Not Same'.\nQuestion 1: {query}\nQuestion 2: {title} {text}",
    #     'assistant': 'Judgement:',
    #     'key': 'Same'
    # },

    'beir/nq': {
        'system': "You are RankGPT, an intelligent assistant specialized in judging whether a query and a document are 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant' to each other. Your generated judgement should be in JSON format with the following structure: {'judgement': 'xxx'}, where 'xxx' is one of 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant'.",
        'user': "Please judge the following query and document:\nQuery: {query}\nDocument: {title} {text}",
        'key': 'Highly Relevant'
    },

    'beir/hotpotqa': {
        'system': "You are RankGPT, an intelligent assistant specialized in judging whether a query and a document are 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant' to each other. Your generated judgement should be in JSON format with the following structure: {'judgement': 'xxx'}, where 'xxx' is one of 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant'.",
        'user': "Please judge the following query and document:\nQuery: {query}\nDocument: {title} {text}",
        'key': 'Highly Relevant'
    },

    'beir/dbpedia-entity': {
        'system': "You are RankGPT, an intelligent assistant specialized in judging whether a query and a document are 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant' to each other. Your generated judgement should be in JSON format with the following structure: {'judgement': 'xxx'}, where 'xxx' is one of 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant'.",
        'user': "Please judge the following query and document:\nQuery: {query}\nDocument: {title} {text}",
        'key': 'Highly Relevant'
    },
}


ICL_PROMPT_T5 = {
    'beir/trec-covid': (
        'You are a question generator.\n'
        'Your generated questions might look like the following:\n'
        '{queries}\n',
        'Generate a question that the following scientific paper can answer. '
        'Avoid to generate general questions.\n',
        'Paper: {title} {text}',
        'Question:'
    ),

    'beir/nfcorpus': ('You are a question generator. '
                      'Your generated questions might look like the following:\n'
                      '{queries}\n',
                      'Generate a question that the following scientific paper can answer. '
                      'Avoid to generate general questions.\n',
                      'Paper: {title} {text}',
                      'Question:'),

    'beir/fiqa': ('You are a question generator. '
                  'Your generated questions might look like the following:\n'
                  '{queries}\n',
                  'Generate a question that the following financial web article can answer. '
                  'Avoid to generate general questions.\n'
                  'Article: {title} {text}',
                  'Question:'),

    'beir/arguana': ('You are an counter argument generator. '
                     'Your generated counter argument might look like the following:\n'
                     '{queries}\n',
                     'Generate an argument that counter argues the following argument.\n'
                     'Argument: {title} {text}',
                     'Counter argument:'),

    'beir/scidocs': ('You are a paper title generator. '
                     'Your generated titles might look like the following:\n'
                     '{queries}\n',
                     'Generate a scientific paper title that are related to the following paper.\n'
                     'Paper: {title} {text}\n',
                     'Question:'),

    'beir/scifact': ('You are a claim generator. '
                     'Your generated claims might look like the following:\n'
                     '{queries}\n',
                     'Generate a claim that are supported by the following scientific evidence.\n'
                     'Evidence: {title} {text}\n',
                     'Claim:'),

    'beir/quora': ('You are a duplicate question generator. '
                   'Your generated questions might look like the following:\n'
                   '{queries}\n',
                   'Generate a question that differs from the following question but essentially asks the same thing.\n'
                   'Question: {title} {text}\n',
                   'A equivalent question:'
                   ),

    'beir/nq': (
        'You are a question generator.\n'
        'Your generated questions might look like the following:\n'
        '{queries}\n',
        'Generate a question that the following Wikipedia page can answer. '
        'Avoid to generate general questions.\n'
        'Wikipedia: {title} {text}',
        'Question:'
    ),

    'beir/hotpotqa': (
        'You are a question generator.\n'
        'Your generated questions might look like the following:\n'
        '{queries}\n',
        'Generate a question that the following Wikipedia passage can answer. '
        'Avoid to generate general questions.\n'
        'Passage: {title} {text}',
        'Question:'
    ),

    'beir/dbpedia-entity': (
        'You are a query generator.\n'
        'Your generated queries might look like the following:\n'
        '{queries}\n',
        'Generate a entity-based query that the following Wikipedia introduction paragraph can answer. '
        'Avoid to generate general queries.\n'
        'Wikipedia paragraph: {title} {text}\n',
        'Query:'
    ),

}
