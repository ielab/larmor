
from beir.util import cos_sim, dot_score


def get_score(query_encoding, doc_encoding, score_function):

    if score_function == "cos_sim":
        scores = cos_sim(query_encoding, doc_encoding)
    elif score_function == "dot":
        scores = dot_score(query_encoding, doc_encoding)
    else:
        raise "Unknown scoring function"

    return scores