PROMPT_QG_FLAN_T5 = {
    'robust04': {
        'system': '',
        'user': 'Generate a question that the following news article can answer. Avoid generating general questions.\nArticle: {title} {text}\n',
        'assistant': 'Question:'
    },

    'trec-news': {
        'system': '',
        'user': 'Generate a headline for the following news article.\nArticle: {title} {text}\n',
        'assistant': 'Headline:'
    },

    'signal1m': {
        'system': '',
        'user': 'Generate a news article title for the following tweet.\nTweet: {title} {text}\n',
        'assistant': 'Title:'
    },

    'trec-covid': {
        'system': '',
        'user': 'Generate a question that the following scientific paper can answer. Avoid generating general questions.\nPaper: {title} {text}\n',
        'assistant': 'Question:'
    },

    'nfcorpus': {
        'system': '',
        'user': 'Generate a question that the following scientific paper can answer. Avoid generating general questions.\nPaper: {title} {text}\n',
        'assistant': 'Question:'
    },

    'fiqa': {
        'system': '',
        'user': 'Generate a question that the following financial web article can answer. Avoid generating general questions.\nArticle: {title} {text}\n',
        'assistant': 'Question:'
    },

    'arguana': {
        'system': '',
        'user': 'Generate an argument that counters the following argument.\nArgument: {title} {text}\n',
        'assistant': 'Counter argument:'
    },

    'scidocs': {
        'system': '',
        'user': 'Generate a scientific paper title that is related to the following paper.\nPaper: {title} {text}\n',
        'assistant': 'Title:'
    },

    'scifact': {
        'system': '',
        'user': 'Generate a claim that is supported by the following scientific evidence.\nEvidence: {title} {text}\n',
        'assistant': 'Claim:'
    },

    'quora': {
        'system': '',
        'user': 'Generate a question that differs from the following question but essentially asks the same thing.\nQuestion: {title} {text}\n',
        'assistant': 'An equivalent question:'
    },

    'nq': {
        'system': '',
        'user': 'Generate a question that the following Wikipedia page can answer. Avoid generating general questions.\nWikipedia page: {title} {text}\n',
        'assistant': 'Question:'
    },

    'hotpotqa': {
        'system': '',
        'user': 'Generate a question that the following passage can answer. Avoid generating general questions.\nPassage: {title} {text}\n',
        'assistant': 'Question:'
    },

    'dbpedia-entity': {
        'system': '',
        'user': 'Generate an entity-based query that the following Wikipedia introduction paragraph can answer. Avoid generating general queries.\nWikipedia paragraph: {title} {text}\n',
        'assistant': 'Query:'
    },
}

PROMPT_POINTWISE_FLAN_T5 = {
    'robust04': {
        'system': '',
        'user': "For the following question and news article, judge whether they are 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant'.\nQuestion: {query}\nArticle: {title} {text}",
        'assistant': 'Judgement:',
        'key': 'Highly Relevant'
    },

    'trec-news': {
        'system': '',
        'user': "For the following headline and news article, judge whether they are 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant'.\nHeadline: {query}\nArticle: {title} {text}",
        'assistant': 'Judgement:',
        'key': 'Highly Relevant',
    },

    'signal1m': {
        'system': '',
        'user': "For the following news article title and tweet, judge whether they are 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant'.\nTitle: {query}\nTweet: {title} {text}",
        'assistant': 'Judgement:',
        'key': 'Highly Relevant'
    },

    'trec-covid': {
        'system': '',
        'user': "For the following query and document, judge whether they are 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant'.\nQuery: {query}\nDocument: {title} {text}",
        'assistant': 'Judgement:',
        'key': 'Highly Relevant'
    },

    'nfcorpus': {
        'system': '',
        'user': "For the following query and document, judge whether they are 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant'.\nQuery: {query}\nDocument: {title} {text}",
        'assistant': 'Judgement:',
        'key': 'Highly Relevant'
    },

    'fiqa': {
        'system': '',
        'user': "For the following query and document, judge whether they are 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant'.\nQuery: {query}\nDocument: {title} {text}",
        'assistant': 'Judgement:',
        'key': 'Highly Relevant'
    },

    'arguana': {
        'system': '',
        'user': "For the following two arguments, judge whether the Argument 1 is 'Similar', 'Somewhat Similar', or 'Counter Argument' to the Argument 2.\nArgument 1: {query}\nArgument 2: {title} {text}",
        'assistant': 'Judgement:',
        'key': 'Counter Argument'
    },

    'scidocs': {
        'system': '',
        'user': "For the following paper title and document, judge whether they are 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant'.\nTitle: {query}\nDocument: {title} {text}",
        'assistant': 'Judgement:',
        'key': 'Highly Relevant'
    },

    'scifact': {
        'system': '',
        'user': "For the following claim and scientific evidence, judge whether the evidence is 'Supports', 'Partially Supports', or 'Not Support' the claim.\nClaim: {query}\nEvidence: {title} {text}",
        'assistant': 'Judgement:',
        'key': 'Supports'
    },

    'quora': {
        'system': '',
        'user': "For the following two questions, judge whether they are asking the same thing. Answer with 'Same', 'Partially Same', or 'Not Same'.\nQuestion 1: {query}\nQuestion 2: {title} {text}",
        'assistant': 'Judgement:',
        'key': 'Same'
    },

    'nq': {
        'system': '',
        'user': "For the following query and document, judge whether they are 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant'.\nQuery: {query}\nDocument: {title} {text}",
        'assistant': 'Judgement:',
        'key': 'Highly Relevant'
    },

    'hotpotqa': {
        'system': '',
        'user': "For the following query and document, judge whether they are 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant'.\nQuery: {query}\nDocument: {title} {text}",
        'assistant': 'Judgement:',
        'key': 'Highly Relevant'
    },

    'dbpedia-entity': {
        'system': '',
        'user': "For the following query and document, judge whether they are 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant'.\nQuery: {query}\nDocument: {title} {text}",
        'assistant': 'Judgement:',
        'key': 'Highly Relevant'
    },
}


PROMPT_QG_OpenAI = {
    'robust04': {
        'system': 'You are a question generator. Your generated questions should be in JSON format with the following structure: {"generated_question": "xxx"}.',
        'user': 'Generate a question that the following news article can answer. Avoid generating general questions.\nArticle: {title} {text}\n',
        'key': "generated_question"
    },

    'trec-news': {
        'system': 'You are a news article headline generator. Your generated headlines should be in JSON format with the following structure: {"headline": "xxx"}.',
        'user': 'Generate a headline for the following news article.\nArticle: {title} {text}\n',
        'key': "headline"
    },

    'signal1m': {
        'system': 'You are a news article title generator. Your generated titles should be in JSON format with the following structure: {"title": "xxx"}.',
        'user': 'Generate a news article title for the following tweet.\nTweet: {title} {text}\n',
        'key': "title"
    },

    'trec-covid': {
        'system': 'You are a question generator. Your generated questions should be in JSON format with the following structure: {"generated_question": "xxx"}.',
        'user': 'Generate a question that the following scientific paper can answer. Avoid generating general questions.\nPaper: {title} {text}\n',
        'key': "generated_question"
    },

    'nfcorpus': {
        'system': 'You are a question generator. Your generated questions should be in JSON format with the following structure: {"generated_question": "xxx"}.',
        'user': 'Generate a question that the following scientific paper can answer. Avoid generating general questions.\nPaper: {title} {text}\n',
        'key': "generated_question"
    },

    'fiqa': {
        'system': 'You are a question generator. Your generated questions should be in JSON format with the following structure: {"generated_question": "xxx"}.',
        'user': 'Generate a question that the following financial web article can answer. Avoid generating general questions.\nArticle: {title} {text}\n',
        'key': "generated_question"
    },

    'beir/arguana': {
        'system': 'You are a counter argument generator. Your generated counter arguments should be in JSON format with the following structure: {"counter_argument": "xxx"}.',
        'user': 'Generate an argument that counters the following argument.\nArgument: {title} {text}\n',
        'key': "counter_argument"
    },

    'scidocs': {
        'system': 'You are a paper title generator. Your generated titles should be in JSON format with the following structure: {"generated_title": "xxx"}.',
        'user': 'Generate a scientific paper title that is related to the following paper.\nPaper: {title} {text}\n',
        'key': "generated_title"
    },

    'scifact': {
        'system': 'You are a claim generator. Your generated claims should be in JSON format with the following structure: {"generated_claim": "xxx"}.',
        'user': 'Generate a claim that is supported by the following scientific evidence.\nEvidence: {title} {text}\n',
        'key': "generated_claim"
    },

    'quora': {
        'system': 'You are a similar question generator. Your generated questions should be in JSON format with the following structure: {"equivalent_question": "xxx"}.',
        'user': 'Generate a question that differs from the following question but essentially asks the same thing.\nQuestion: {title} {text}\n',
        'key': "equivalent_question"
    },

    'nq': {
        'system': 'You are a question generator. Your generated questions should be in JSON format with the following structure: {"generated_question": "xxx"}.',
        'user': 'Generate a question that the following Wikipedia page can answer. Avoid generating general questions.\nWikipedia page: {title} {text}\n',
        'key': "generated_question"
    },

    'hotpotqa': {
        'system': 'You are a question generator. Your generated questions should be in JSON format with the following structure: {"generated_question": "xxx"}.',
        'user': 'Generate a question that the following passage can answer. Avoid generating general questions.\nPassage: {title} {text}\n',
        'key': "generated_question"
    },

    'dbpedia-entity': {
        'system': 'You are a query generator. Your generated queries should be in JSON format with the following structure: {"generated_query": "xxx"}.',
        'user': 'Generate an entity-based query that the following Wikipedia introduction paragraph can answer. Avoid generating general queries.\nWikipedia paragraph: {title} {text}\n',
        'key': "generated_query"
    },
}